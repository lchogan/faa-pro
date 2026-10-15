"""On-demand FAA data fetch for the airport-diagram classifier.

Ports the download logic from the standalone faa-downloader project so a
single run fetches the latest FAA files (airport-diagram PDF, approach
chart, NASR airspace class + runway data) for one FAA LID. Pure helpers
(AIRAC cycle math, d-TPP parsing, folder de-duplication) are unit-tested;
network helpers are thin wrappers around `requests`.

Key concepts:
  - AIRAC cycle: the 28-day FAA publication cycle; the d-TPP and PDF URLs
    are keyed by the YYNN cycle string.
  - d-TPP metafile: XML index mapping each airport to its chart PDFs.
  - NASR 28-day subscription: a zip of airport CSVs; we read airspace
    class (for the macOS Finder label) and runway designations (for the
    classifier's runway detection).
"""
from __future__ import annotations

import csv
import io
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ── FAA endpoints ─────────────────────────────────────────────────────────
BASE_PDF_URL = "https://aeronav.faa.gov/d-tpp/{cycle}/{pdf_name}"
METAFILE_URL = "https://aeronav.faa.gov/d-tpp/{cycle}/xml_data/d-TPP_Metafile.xml"
NASR_URL = (
    "https://nfdc.faa.gov/webContent/28DaySub/"
    "28DaySubscription_Effective_{date}.zip"
)

# AIRAC reference: cycle 2401 started 2024-01-25.
AIRAC_EPOCH = datetime(2024, 1, 25)

# Airspace class -> macOS Finder color label index.
# 0=None 1=Gray 2=Green 3=Purple 4=Blue 5=Yellow 6=Red 7=Orange
AIRSPACE_LABEL_COLORS: dict[str, int] = {"B": 4, "C": 2, "D": 3, "E": 5, "unknown": 6}

APPROACH_PRIORITY = ["ILS", "RNAV", "GPS", "VOR", "NDB", "LOC"]


def get_current_cycle(today: datetime | None = None) -> tuple[str, str]:
    """Return the current AIRAC cycle string (YYNN) and effective date.

    Verified: today=2026-02-21 -> ('2602', '2026-02-19').

    :param today: override "now" for testing; defaults to UTC now.
    :returns: (cycle_str, effective_date_str as YYYY-MM-DD)
    """
    if today is None:
        today = datetime.now(timezone.utc).replace(tzinfo=None)
    n = (today - AIRAC_EPOCH).days // 28
    cycle_start = AIRAC_EPOCH + timedelta(days=n * 28)
    year = cycle_start.year

    jan1 = datetime(year, 1, 1)
    n_jan1 = max(0, (jan1 - AIRAC_EPOCH).days // 28)
    first_start = AIRAC_EPOCH + timedelta(days=n_jan1 * 28)
    if first_start.year < year:
        n_jan1 += 1

    cycle_in_year = (n - n_jan1) + 1
    cycle_str = f"{str(year)[-2:]}{cycle_in_year:02d}"
    return cycle_str, cycle_start.strftime("%Y-%m-%d")


def parse_tpp_records(
    root: ET.Element,
) -> tuple[dict[str, str], dict[str, list[tuple[str, str]]]]:
    """Walk the d-TPP XML and return (apd_map, iap_map).

    :param root: parsed d-TPP metafile root element.
    :returns: apd_map {apt_ident: apd_pdf_name} for chart_code APD, and
        iap_map {apt_ident: [(chart_name, pdf_name), ...]} for chart_code IAP.
    """
    apd_map: dict[str, str] = {}
    iap_map: dict[str, list[tuple[str, str]]] = {}

    for airport_elem in root.iter("airport_name"):
        apt_ident = airport_elem.get("apt_ident", "").strip()
        if not apt_ident:
            continue
        for record in airport_elem.findall("record"):
            chart_code = record.findtext("chart_code", "").strip()
            chart_name = record.findtext("chart_name", "").strip()
            pdf_name = record.findtext("pdf_name", "").strip()
            if not pdf_name:
                continue
            if chart_code == "APD":
                apd_map[apt_ident] = pdf_name
            elif chart_code == "IAP":
                iap_map.setdefault(apt_ident, []).append((chart_name, pdf_name))

    return apd_map, iap_map


def select_best_approach(
    iap_list: list[tuple[str, str]], priority: list[str]
) -> tuple[str, str] | None:
    """Return the highest-priority approach from iap_list, else the first, else None.

    :param iap_list: [(chart_name, pdf_name), ...] for one airport.
    :param priority: keywords in descending preference (e.g. ["ILS", "RNAV"]).
    """
    if not iap_list:
        return None
    for keyword in priority:
        for chart_name, pdf_name in iap_list:
            if keyword.upper() in chart_name.upper():
                return chart_name, pdf_name
    return iap_list[0]


def dedup_folder(parent: Path, lid: str) -> Path:
    """Return the first non-existing of parent/lid, parent/lid-1, parent/lid-2, ...

    Never returns a path that already exists, so a re-run of an
    already-classified airport does not overwrite prior work.
    """
    candidate = parent / lid
    if not candidate.exists():
        return candidate
    i = 1
    while True:
        candidate = parent / f"{lid}-{i}"
        if not candidate.exists():
            return candidate
        i += 1


def filter_runway_rows(rows: Iterable[dict], lid: str) -> list[dict]:
    """Return only the rows whose ARPT_ID matches lid (case-insensitive).

    :param rows: iterable of dicts with ARPT_ID key.
    :param lid: airport identifier to filter on.
    :returns: list of matching runway rows.
    """
    want = lid.strip().upper()
    return [r for r in rows if (r.get("ARPT_ID") or "").strip().upper() == want]


def extract_runway_rows(
    zf: zipfile.ZipFile, lid: str, filename: str = "APT_RWY.csv"
) -> list[dict]:
    """Read APT_RWY.csv from an open NASR CSV zip and return this airport's rows.

    :param zf: open zipfile.ZipFile from NASR 28-day subscription.
    :param lid: airport identifier to filter on.
    :param filename: name of CSV file within the zip (default "APT_RWY.csv").
    :returns: list of runway rows for this airport.
    """
    with zf.open(filename) as f:
        reader = csv.DictReader(io.TextIOWrapper(f, encoding="latin-1"))
        return filter_runway_rows(list(reader), lid)


def write_runway_csv(rows: list[dict], dest: Path) -> None:
    """Write runway rows to a CSV the classifier's load_nasr_runways can read.

    Writes all columns present on the rows (load_nasr_runways reads only
    ARPT_ID and RWY_ID, but preserving the full schema keeps the file a
    faithful subset of NASR APT_RWY.csv). A no-op when rows is empty.

    :param rows: list of runway dicts to write.
    :param dest: path where to write the CSV file.
    """
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(dest, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_session(timeout: int = 60) -> requests.Session:
    """Return a requests.Session with retry/backoff and a browser-like UA."""
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1.5,
                  status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {"User-Agent": "Mozilla/5.0 (compatible; FAA-Airport-Fetch/1.0)"}
    )
    return session


def fetch_metafile(cycle: str, session: requests.Session) -> ET.Element:
    """Fetch and parse the d-TPP metafile for the given cycle."""
    resp = session.get(METAFILE_URL.format(cycle=cycle), timeout=90)
    resp.raise_for_status()
    return ET.fromstring(resp.content)


def _parse_apt_base_csv(zf: zipfile.ZipFile, filename: str = "APT_BASE.csv") -> dict[str, dict]:
    """Parse APT_BASE.csv -> {ARPT_ID: row} for US airports."""
    airports: dict[str, dict] = {}
    with zf.open(filename) as f:
        reader = csv.DictReader(io.TextIOWrapper(f, encoding="latin-1"))
        for row in reader:
            apt_id = row.get("ARPT_ID", "").strip()
            if apt_id and row.get("COUNTRY_CODE", "US").strip() == "US":
                airports[apt_id] = dict(row)
    return airports


def _merge_airspace_class(airports: dict[str, dict], zf: zipfile.ZipFile,
                          filename: str = "CLS_ARSP.csv") -> None:
    """Inject '_AIRSPACE_CLASS' (B/C/D/E or None) into each airport row."""
    priority = ["CLASS_B_AIRSPACE", "CLASS_C_AIRSPACE", "CLASS_D_AIRSPACE", "CLASS_E_AIRSPACE"]
    letter = {"CLASS_B_AIRSPACE": "B", "CLASS_C_AIRSPACE": "C",
              "CLASS_D_AIRSPACE": "D", "CLASS_E_AIRSPACE": "E"}
    with zf.open(filename) as f:
        reader = csv.DictReader(io.TextIOWrapper(f, encoding="latin-1"))
        for row in reader:
            apt_id = row.get("ARPT_ID", "").strip()
            if apt_id not in airports:
                continue
            assigned = None
            for col in priority:
                if row.get(col, "").strip().upper() == "Y":
                    assigned = letter[col]
                    break
            airports[apt_id]["_AIRSPACE_CLASS"] = assigned


def download_nasr(date_str: str, session: requests.Session,
                  lid: str | None = None, timeout: int = 300) -> tuple[dict[str, dict], list[dict]]:
    """Download the 28-day NASR zip; return (airports, runway_rows).

    airports: {ARPT_ID: row} with '_AIRSPACE_CLASS' merged in.
    runway_rows: this airport's APT_RWY.csv rows when `lid` is given, else [].
    Handles the nested CSV_Data/<date>_CSV.zip layout; falls back to a
    flat APT_BASE.csv in the outer zip. Runway rows are only available
    from the inner CSV zip.
    """
    resp = session.get(NASR_URL.format(date=date_str), timeout=timeout)
    resp.raise_for_status()
    runway_rows: list[dict] = []

    with zipfile.ZipFile(io.BytesIO(resp.content)) as outer_zf:
        inner_names = [n for n in outer_zf.namelist()
                       if re.search(r"CSV_Data/.*CSV\.zip$", n, re.I)]
        if inner_names:
            with outer_zf.open(inner_names[0]) as inner_raw:
                inner_bytes = io.BytesIO(inner_raw.read())
            with zipfile.ZipFile(inner_bytes) as inner_zf:
                airports = _parse_apt_base_csv(inner_zf)
                _merge_airspace_class(airports, inner_zf)
                if lid:
                    runway_rows = extract_runway_rows(inner_zf, lid)
            return airports, runway_rows

        base_csv = [n for n in outer_zf.namelist() if re.search(r"APT_BASE\.csv$", n, re.I)]
        if base_csv:
            airports = _parse_apt_base_csv(outer_zf, filename=base_csv[0])
            cls_csv = [n for n in outer_zf.namelist() if re.search(r"CLS_ARSP\.csv$", n, re.I)]
            if cls_csv:
                _merge_airspace_class(airports, outer_zf, filename=cls_csv[0])
            return airports, runway_rows

    print(
        "WARN: NASR zip layout not recognized (no CSV_Data/*CSV.zip or APT_BASE.csv); "
        "returning no airport data",
        file=sys.stderr,
    )
    return {}, runway_rows


def determine_airspace_class(lid: str, row: dict) -> str | None:
    """Return the surface airspace class: CLS_ARSP value, else tower->D, else None."""
    if "_AIRSPACE_CLASS" in row:
        return row["_AIRSPACE_CLASS"]
    if row.get("TWR_TYPE_CODE", "").strip().upper() in ("ATCT", "ATCT-R", "ATCT-M"):
        return "D"
    return None


def set_finder_label(folder: Path, airspace_class: str | None,
                     label_map: dict[str, int]) -> None:
    """Apply a macOS Finder color label to a folder via AppleScript (best-effort)."""
    key = airspace_class if airspace_class in label_map else "unknown"
    label_idx = label_map.get(key, 6)
    # Escape backslash first, then double-quote, so the path stays a valid
    # AppleScript string literal.
    escaped_path = str(folder.resolve()).replace("\\", "\\\\").replace('"', '\\"')
    script = (
        f'tell application "Finder" to set label index of '
        f'(POSIX file "{escaped_path}" as alias) to {label_idx}'
    )
    # Best-effort: a hung Finder or missing osascript must not abort the fetch.
    try:
        result = subprocess.run(
            ["osascript", "-e", script], capture_output=True, text=True, timeout=10
        )
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"WARN: could not set Finder label on {folder}: {exc}", file=sys.stderr)
        return
    if result.returncode != 0:
        print(
            f"WARN: osascript failed to set Finder label on {folder}: "
            f"{result.stderr.strip()}",
            file=sys.stderr,
        )


def download_pdf(url: str, dest: Path, session: requests.Session, timeout: int) -> None:
    """Download a PDF; raise if the response is implausibly small (error page)."""
    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()
    if len(resp.content) < 500:
        raise ValueError(f"Response too small ({len(resp.content)} bytes) — likely an error page")
    dest.write_bytes(resp.content)
