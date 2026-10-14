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

import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

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
