# On-Demand FAA Fetch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the classifier download the latest FAA files on demand from only an FAA LID code, replacing the dependence on a pre-built bulk-download directory.

**Architecture:** Port the proven download logic from the standalone `faa-downloader` into a new, unit-tested module `python/pipeline/faa_fetch.py` with a thin CLI `python/fetch_airport.py`. Rewrite `classify-airports.sh` to prompt for a destination and LID code(s), fetch fresh FAA data per code into a de-duplicated folder, then call the existing `classify.sh`. All risky logic (dedup, cycle math, d-TPP/NASR parsing, orchestration) lives in Python and is tested; the bash wrappers stay thin.

**Tech Stack:** Python 3.10+, `requests`, stdlib `xml.etree`/`csv`/`zipfile`, pytest, bash.

**Spec:** `docs/superpowers/specs/2026-10-14-on-demand-faa-fetch-design.md`

## Global Constraints

- Python `requires-python >= 3.10` (existing `pyproject.toml`).
- New runtime dependency: `requests>=2.31` (add to `pyproject.toml` `dependencies`).
- Tests import via `from pipeline.X import Y` / `from <module> import Y`; `python/tests/conftest.py` already puts `python/` on `sys.path`. Run tests with `cd python && python -m pytest`.
- FAA endpoints (copy verbatim): d-TPP metafile `https://aeronav.faa.gov/d-tpp/{cycle}/xml_data/d-TPP_Metafile.xml`; PDF `https://aeronav.faa.gov/d-tpp/{cycle}/{pdf_name}`; NASR `https://nfdc.faa.gov/webContent/28DaySub/28DaySubscription_Effective_{date}.zip`.
- AIRAC epoch: `datetime(2024, 1, 25)`.
- Finder label color map (airspace class → macOS label index): `{"B": 4, "C": 2, "D": 3, "E": 5, "unknown": 6}`; `None`/unknown → `6` (Red).
- Approach priority: `["ILS", "RNAV", "GPS", "VOR", "NDB", "LOC"]`.
- Downloaded files are named by the **true lowercased LID** (`ord-faa.pdf`, `ord-approach.pdf`, `ord-nasr-rwy.csv`) even when the containing folder is de-duplicated (`ord-1`), so the classifier derives the correct airport code and NASR key.
- The separate `faa-downloader` repo is **not** modified. Deprecation is a note in this repo's `README.md` only.
- The interactive **"Place approach lights? [y/N]"** prompt in `classify-airports.sh` is preserved. There is **no** "Proceed?" confirmation and **no** source-folder prompt.

## Review Focus

- **LID with no FAA airport diagram** (e.g. a small field, or a typo): the fetch must exit non-zero with a clear message and write nothing; the wrapper skips it and continues other codes. → Task 4.
- **Re-run of an already-classified airport**: must create `ord-1`, `ord-2`, … and never overwrite the existing folder. → Task 1 (`dedup_folder`) + Task 4 (end-to-end).
- **LID casing / whitespace** (`ord`, `ORD`, ` Ord `): normalized to uppercase for FAA lookups and lowercase for filenames. → Task 4.
- **Airport has a diagram but no approach chart**: approach download warns, the diagram PDF is still produced. → Task 4.
- **De-duplicated folder must still classify under the true LID**: inside `ord-1/`, the PDF is `ord-faa.pdf` and the runway CSV is `ord-nasr-rwy.csv` (not `ord-1-…`). → Task 4.

---

### Task 1: faa_fetch pure helpers + constants

**Files:**
- Create: `python/pipeline/faa_fetch.py`
- Modify: `python/pyproject.toml` (add `requests` dependency)
- Test: `python/tests/test_faa_fetch.py`

**Interfaces:**
- Produces:
  - `AIRAC_EPOCH: datetime`, `AIRSPACE_LABEL_COLORS: dict[str, int]`, `APPROACH_PRIORITY: list[str]`, `BASE_PDF_URL: str`, `METAFILE_URL: str`, `NASR_URL: str` (module constants)
  - `get_current_cycle(today: datetime | None = None) -> tuple[str, str]`
  - `parse_tpp_records(root: xml.etree.ElementTree.Element) -> tuple[dict[str, str], dict[str, list[tuple[str, str]]]]`
  - `select_best_approach(iap_list: list[tuple[str, str]], priority: list[str]) -> tuple[str, str] | None`
  - `dedup_folder(parent: pathlib.Path, lid: str) -> pathlib.Path`

- [ ] **Step 1: Write the failing tests**

```python
# python/tests/test_faa_fetch.py
import xml.etree.ElementTree as ET
from datetime import datetime

from pipeline import faa_fetch


def test_get_current_cycle_known_date():
    # AIRAC epoch 2024-01-25; 2026-02-21 falls in cycle 2602 (eff 2026-02-19).
    assert faa_fetch.get_current_cycle(datetime(2026, 2, 21)) == ("2602", "2026-02-19")


DTPP_XML = """<?xml version="1.0"?>
<digital_tpp>
  <state_code ID="IL">
    <city_name ID="CHICAGO">
      <airport_name apt_ident="ORD" ID="CHICAGO OHARE INTL">
        <record><chart_code>APD</chart_code><chart_name>AIRPORT DIAGRAM</chart_name><pdf_name>00166AD.PDF</pdf_name></record>
        <record><chart_code>IAP</chart_code><chart_name>RNAV (GPS) RWY 28R</chart_name><pdf_name>00166R28R.PDF</pdf_name></record>
        <record><chart_code>IAP</chart_code><chart_name>ILS OR LOC RWY 10L</chart_name><pdf_name>00166IL10L.PDF</pdf_name></record>
      </airport_name>
    </city_name>
  </state_code>
</digital_tpp>
"""


def test_parse_tpp_records_builds_apd_and_iap_maps():
    apd, iap = faa_fetch.parse_tpp_records(ET.fromstring(DTPP_XML))
    assert apd == {"ORD": "00166AD.PDF"}
    assert iap["ORD"] == [
        ("RNAV (GPS) RWY 28R", "00166R28R.PDF"),
        ("ILS OR LOC RWY 10L", "00166IL10L.PDF"),
    ]


def test_select_best_approach_prefers_priority_then_falls_back():
    iap = [("RNAV (GPS) RWY 28R", "b.PDF"), ("ILS OR LOC RWY 10L", "a.PDF")]
    assert faa_fetch.select_best_approach(iap, faa_fetch.APPROACH_PRIORITY) == (
        "ILS OR LOC RWY 10L", "a.PDF",
    )
    # No priority keyword present -> first entry.
    assert faa_fetch.select_best_approach([("VISUAL 9", "v.PDF")], ["ILS"]) == (
        "VISUAL 9", "v.PDF",
    )
    assert faa_fetch.select_best_approach([], faa_fetch.APPROACH_PRIORITY) is None


def test_dedup_folder_increments_suffix(tmp_path):
    assert faa_fetch.dedup_folder(tmp_path, "ord") == tmp_path / "ord"
    (tmp_path / "ord").mkdir()
    assert faa_fetch.dedup_folder(tmp_path, "ord") == tmp_path / "ord-1"
    (tmp_path / "ord-1").mkdir()
    assert faa_fetch.dedup_folder(tmp_path, "ord") == tmp_path / "ord-2"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd python && python -m pytest tests/test_faa_fetch.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline.faa_fetch'`.

- [ ] **Step 3: Write the module with constants + pure helpers**

```python
# python/pipeline/faa_fetch.py
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

import io
import re
import subprocess
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
        first_start = AIRAC_EPOCH + timedelta(days=n_jan1 * 28)

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
```

- [ ] **Step 4: Add the `requests` dependency**

In `python/pyproject.toml`, add `"requests>=2.31",` to the `dependencies` list (after `"shapely>=2.0",`).

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd python && python -m pytest tests/test_faa_fetch.py -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add python/pipeline/faa_fetch.py python/pyproject.toml python/tests/test_faa_fetch.py
git commit -m "feat: add faa_fetch pure helpers (cycle, d-TPP parse, dedup)"
```

---

### Task 2: NASR runway extraction

**Files:**
- Modify: `python/pipeline/faa_fetch.py`
- Test: `python/tests/test_faa_fetch.py`

**Interfaces:**
- Consumes: `pipeline.extract_pdf_text.load_nasr_runways` (round-trip check).
- Produces:
  - `filter_runway_rows(rows: Iterable[dict], lid: str) -> list[dict]`
  - `extract_runway_rows(zf: zipfile.ZipFile, lid: str, filename: str = "APT_RWY.csv") -> list[dict]`
  - `write_runway_csv(rows: list[dict], dest: pathlib.Path) -> None`

- [ ] **Step 1: Write the failing tests**

```python
# append to python/tests/test_faa_fetch.py
import csv
import io
import zipfile

from pipeline.extract_pdf_text import load_nasr_runways

RWY_HEADER = ["ARPT_ID", "RWY_ID", "RWY_LEN"]
RWY_ROWS = [
    {"ARPT_ID": "ORD", "RWY_ID": "10L/28R", "RWY_LEN": "13000"},
    {"ARPT_ID": "ORD", "RWY_ID": "04L/22R", "RWY_LEN": "7500"},
    {"ARPT_ID": "MDW", "RWY_ID": "13C/31C", "RWY_LEN": "6500"},
]


def _apt_rwy_bytes() -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=RWY_HEADER)
    w.writeheader()
    w.writerows(RWY_ROWS)
    return buf.getvalue().encode("latin-1")


def test_filter_runway_rows_matches_lid_case_insensitively():
    rows = faa_fetch.filter_runway_rows(RWY_ROWS, "ord")
    assert [r["RWY_ID"] for r in rows] == ["10L/28R", "04L/22R"]
    assert faa_fetch.filter_runway_rows(RWY_ROWS, "zzz") == []


def test_extract_runway_rows_reads_from_zip():
    zbytes = io.BytesIO()
    with zipfile.ZipFile(zbytes, "w") as zf:
        zf.writestr("APT_RWY.csv", _apt_rwy_bytes())
    zbytes.seek(0)
    with zipfile.ZipFile(zbytes) as zf:
        rows = faa_fetch.extract_runway_rows(zf, "ORD")
    assert {r["RWY_ID"] for r in rows} == {"10L/28R", "04L/22R"}


def test_write_runway_csv_round_trips_through_loader(tmp_path):
    dest = tmp_path / "ord-nasr-rwy.csv"
    faa_fetch.write_runway_csv(faa_fetch.filter_runway_rows(RWY_ROWS, "ORD"), dest)
    # The classifier's loader must read our output back, keyed by lowercase LID.
    loaded = load_nasr_runways(dest)
    assert loaded == {"ord": {"10L", "28R", "4L", "22R"}}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd python && python -m pytest tests/test_faa_fetch.py -k runway -v`
Expected: FAIL — `AttributeError: module 'pipeline.faa_fetch' has no attribute 'filter_runway_rows'`.

- [ ] **Step 3: Implement the runway helpers**

```python
# add to python/pipeline/faa_fetch.py
import csv


def filter_runway_rows(rows: Iterable[dict], lid: str) -> list[dict]:
    """Return only the rows whose ARPT_ID matches lid (case-insensitive)."""
    want = lid.strip().upper()
    return [r for r in rows if (r.get("ARPT_ID") or "").strip().upper() == want]


def extract_runway_rows(
    zf: zipfile.ZipFile, lid: str, filename: str = "APT_RWY.csv"
) -> list[dict]:
    """Read APT_RWY.csv from an open NASR CSV zip and return this airport's rows."""
    with zf.open(filename) as f:
        reader = csv.DictReader(io.TextIOWrapper(f, encoding="latin-1"))
        return filter_runway_rows(list(reader), lid)


def write_runway_csv(rows: list[dict], dest: Path) -> None:
    """Write runway rows to a CSV the classifier's load_nasr_runways can read.

    Writes all columns present on the rows (load_nasr_runways reads only
    ARPT_ID and RWY_ID, but preserving the full schema keeps the file a
    faithful subset of NASR APT_RWY.csv). A no-op when rows is empty.
    """
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(dest, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
```

Move the `import csv` to the top-of-file import block rather than mid-file if the module style prefers it (it does — group with the other stdlib imports).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd python && python -m pytest tests/test_faa_fetch.py -k runway -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add python/pipeline/faa_fetch.py python/tests/test_faa_fetch.py
git commit -m "feat: extract per-airport NASR runway rows for the classifier"
```

---

### Task 3: NASR + network helpers

**Files:**
- Modify: `python/pipeline/faa_fetch.py`
- Test: `python/tests/test_faa_fetch.py`

**Interfaces:**
- Produces:
  - `make_session(timeout: int = 60) -> requests.Session`
  - `fetch_metafile(cycle: str, session: requests.Session) -> ET.Element`
  - `download_nasr(date_str: str, session: requests.Session, lid: str | None = None, timeout: int = 300) -> tuple[dict[str, dict], list[dict]]`
  - `determine_airspace_class(lid: str, row: dict) -> str | None`
  - `set_finder_label(folder: pathlib.Path, airspace_class: str | None, label_map: dict[str, int]) -> None`
  - `download_pdf(url: str, dest: pathlib.Path, session: requests.Session, timeout: int) -> None`

- [ ] **Step 1: Write the failing test (pure logic only)**

```python
# append to python/tests/test_faa_fetch.py
def test_determine_airspace_class_prefers_cls_arsp_then_tower():
    assert faa_fetch.determine_airspace_class("ORD", {"_AIRSPACE_CLASS": "B"}) == "B"
    # CLS_ARSP present but None -> None even if a tower exists.
    assert faa_fetch.determine_airspace_class(
        "XXX", {"_AIRSPACE_CLASS": None, "TWR_TYPE_CODE": "ATCT"}
    ) is None
    # No CLS_ARSP entry at all -> infer Class D from tower presence.
    assert faa_fetch.determine_airspace_class("YYY", {"TWR_TYPE_CODE": "ATCT"}) == "D"
    assert faa_fetch.determine_airspace_class("ZZZ", {}) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd python && python -m pytest tests/test_faa_fetch.py -k airspace -v`
Expected: FAIL — `AttributeError: ... has no attribute 'determine_airspace_class'`.

- [ ] **Step 3: Implement network + NASR helpers**

```python
# add to python/pipeline/faa_fetch.py


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
    script = (
        f'tell application "Finder" to set label index of '
        f'(POSIX file "{folder.resolve()}" as alias) to {label_idx}'
    )
    subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=10)


def download_pdf(url: str, dest: Path, session: requests.Session, timeout: int) -> None:
    """Download a PDF; raise if the response is implausibly small (error page)."""
    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()
    if len(resp.content) < 500:
        raise ValueError(f"Response too small ({len(resp.content)} bytes) — likely an error page")
    dest.write_bytes(resp.content)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd python && python -m pytest tests/test_faa_fetch.py -v`
Expected: PASS (all tests so far).

- [ ] **Step 5: Commit**

```bash
git add python/pipeline/faa_fetch.py python/tests/test_faa_fetch.py
git commit -m "feat: add NASR download + airspace/label/pdf network helpers"
```

---

### Task 4: fetch_airport.py CLI

**Files:**
- Create: `python/fetch_airport.py`
- Test: `python/tests/test_fetch_airport.py`

**Interfaces:**
- Consumes: all of `pipeline.faa_fetch` (Tasks 1–3).
- Produces:
  - `run(lid: str, dest_parent: pathlib.Path, priority: list[str] | None = None, *, session=None) -> tuple[pathlib.Path, pathlib.Path]` — returns (folder, apd_pdf_path), prints `FOLDER=<path>` and `PDF=<path>`.
  - `main(argv: list[str] | None = None) -> int` — CLI entry; exit code 2 when the LID has no FAA airport diagram.

- [ ] **Step 1: Write the failing tests (network functions monkeypatched)**

```python
# python/tests/test_fetch_airport.py
import xml.etree.ElementTree as ET

import pytest

import fetch_airport
from pipeline import faa_fetch

APD_XML = """<?xml version="1.0"?>
<digital_tpp><state_code><city_name>
  <airport_name apt_ident="ORD">
    <record><chart_code>APD</chart_code><chart_name>AIRPORT DIAGRAM</chart_name><pdf_name>APD.PDF</pdf_name></record>
    <record><chart_code>IAP</chart_code><chart_name>ILS RWY 10L</chart_name><pdf_name>ILS.PDF</pdf_name></record>
  </airport_name>
</city_name></state_code></digital_tpp>
"""

RWY_ROWS = [{"ARPT_ID": "ORD", "RWY_ID": "10L/28R"}]


@pytest.fixture
def patched(monkeypatch):
    """Stub every network call so run() exercises only orchestration."""
    monkeypatch.setattr(faa_fetch, "get_current_cycle", lambda today=None: ("2602", "2026-02-19"))
    monkeypatch.setattr(faa_fetch, "make_session", lambda *a, **k: object())
    monkeypatch.setattr(faa_fetch, "fetch_metafile", lambda cycle, session: ET.fromstring(APD_XML))
    monkeypatch.setattr(faa_fetch, "download_nasr",
                        lambda date, session, lid=None, timeout=300: ({"ORD": {"_AIRSPACE_CLASS": "B"}}, RWY_ROWS))
    monkeypatch.setattr(faa_fetch, "set_finder_label", lambda *a, **k: None)
    monkeypatch.setattr(faa_fetch, "download_pdf",
                        lambda url, dest, session, timeout: dest.write_bytes(b"%PDF-1.4 fake" + b"x" * 600))


def test_run_downloads_and_reports_paths(patched, tmp_path, capsys):
    folder, pdf = fetch_airport.run("ord", tmp_path)
    assert folder == tmp_path / "ord"
    assert pdf == tmp_path / "ord" / "ord-faa.pdf" and pdf.exists()
    assert (folder / "ord-approach.pdf").exists()
    assert (folder / "ord-nasr-rwy.csv").exists()
    out = capsys.readouterr().out
    assert f"FOLDER={folder}" in out and f"PDF={pdf}" in out


def test_run_dedups_and_keeps_true_lid_filenames(patched, tmp_path):
    (tmp_path / "ord").mkdir()
    folder, pdf = fetch_airport.run("ORD", tmp_path)
    assert folder == tmp_path / "ord-1"           # folder deduped
    assert pdf.name == "ord-faa.pdf"              # file keeps true LID
    assert (folder / "ord-nasr-rwy.csv").exists()  # not ord-1-...


def test_run_no_approach_still_writes_diagram(patched, tmp_path, monkeypatch):
    no_iap = """<?xml version="1.0"?><digital_tpp><airport_name apt_ident="ORD">
    <record><chart_code>APD</chart_code><chart_name>AD</chart_name><pdf_name>APD.PDF</pdf_name></record>
    </airport_name></digital_tpp>"""
    monkeypatch.setattr(faa_fetch, "fetch_metafile", lambda c, s: ET.fromstring(no_iap))
    folder, pdf = fetch_airport.run("ord", tmp_path)
    assert pdf.exists()
    assert not (folder / "ord-approach.pdf").exists()


def test_main_exits_nonzero_when_no_diagram(patched, tmp_path, monkeypatch, capsys):
    empty = """<?xml version="1.0"?><digital_tpp></digital_tpp>"""
    monkeypatch.setattr(faa_fetch, "fetch_metafile", lambda c, s: ET.fromstring(empty))
    rc = fetch_airport.main(["--lid", "zzz", "--dest-parent", str(tmp_path)])
    assert rc == 2
    assert list(tmp_path.iterdir()) == []          # wrote nothing
    assert "no FAA airport diagram" in capsys.readouterr().err.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd python && python -m pytest tests/test_fetch_airport.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fetch_airport'`.

- [ ] **Step 3: Write the CLI**

```python
# python/fetch_airport.py
#!/usr/bin/env python3
"""Fetch the latest FAA data for one airport LID and lay it out for classify.sh.

Downloads the airport-diagram PDF, best approach chart, and this airport's
NASR runway rows into a de-duplicated folder under a destination parent,
and applies a macOS Finder color label by airspace class. Prints two
machine-readable lines the bash wrapper parses: FOLDER=<path>, PDF=<path>.

Depends on pipeline.faa_fetch. Network calls go through that module so
tests can stub them.

Usage:
    python fetch_airport.py --lid ORD --dest-parent /path/to/artwork/airports
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline import faa_fetch


def run(lid: str, dest_parent: Path, priority: list[str] | None = None,
        *, session=None) -> tuple[Path, Path]:
    """Download FAA data for one LID. Returns (folder, apd_pdf_path).

    :raises LookupError: when the LID has no published FAA airport diagram.
    """
    priority = priority or faa_fetch.APPROACH_PRIORITY
    lid_upper = lid.strip().upper()
    lid_lower = lid_upper.lower()

    cycle, date_str = faa_fetch.get_current_cycle()
    session = session or faa_fetch.make_session()

    root = faa_fetch.fetch_metafile(cycle, session)
    apd_map, iap_map = faa_fetch.parse_tpp_records(root)
    apd_pdf = apd_map.get(lid_upper)
    if not apd_pdf:
        raise LookupError(f"no FAA airport diagram for {lid_upper}")

    airports, runway_rows = faa_fetch.download_nasr(date_str, session, lid_upper)
    airspace = faa_fetch.determine_airspace_class(lid_upper, airports.get(lid_upper, {}))

    folder = faa_fetch.dedup_folder(dest_parent, lid_lower)
    folder.mkdir(parents=True)
    faa_fetch.set_finder_label(folder, airspace, faa_fetch.AIRSPACE_LABEL_COLORS)

    apd_dest = folder / f"{lid_lower}-faa.pdf"
    faa_fetch.download_pdf(
        faa_fetch.BASE_PDF_URL.format(cycle=cycle, pdf_name=apd_pdf),
        apd_dest, session, 120)

    iap = faa_fetch.select_best_approach(iap_map.get(lid_upper, []), priority)
    if iap:
        try:
            faa_fetch.download_pdf(
                faa_fetch.BASE_PDF_URL.format(cycle=cycle, pdf_name=iap[1]),
                folder / f"{lid_lower}-approach.pdf", session, 120)
        except Exception as exc:  # approach is optional; the diagram is required
            print(f"WARN: approach download failed for {lid_upper}: {exc}", file=sys.stderr)

    faa_fetch.write_runway_csv(runway_rows, folder / f"{lid_lower}-nasr-rwy.csv")

    print(f"FOLDER={folder}")
    print(f"PDF={apd_dest}")
    return folder, apd_dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fetch latest FAA data for one airport LID.")
    ap.add_argument("--lid", required=True, help="FAA location identifier, e.g. ORD")
    ap.add_argument("--dest-parent", required=True, type=Path,
                    help="Parent folder; the airport folder is created (deduped) inside it.")
    ap.add_argument("--priority", default="",
                    help="Comma-separated approach priority override.")
    args = ap.parse_args(argv)

    priority = [p.strip() for p in args.priority.split(",") if p.strip()] or None
    try:
        run(args.lid, args.dest_parent, priority)
    except LookupError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Note: `python/` is on `sys.path` for the CLI because Python auto-adds the script's own directory (`python/`), so `from pipeline import faa_fetch` resolves. Tests get the same path via `conftest.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd python && python -m pytest tests/test_fetch_airport.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add python/fetch_airport.py python/tests/test_fetch_airport.py
git commit -m "feat: add fetch_airport CLI (dedup, download, runway csv, stdout contract)"
```

---

### Task 5: classify.sh — NASR_RWY_CSV pass-through

**Files:**
- Modify: `classify.sh` (the `classify_pipeline.py` invocation around lines 149-159, and a comment near the `PIPELINE_EXTRA` block ~line 49-56)

**Interfaces:**
- Consumes: optional env var `NASR_RWY_CSV` (absolute path to a per-airport NASR runway CSV), set by `classify-airports.sh`.

- [ ] **Step 1: Add the pass-through**

In `classify.sh`, immediately before the Step 2 pipeline call (`"$PYTHON" "$PROJECT/python/classify_pipeline.py" \`), add:

```bash
    # Per-run fresh NASR runway data from fetch_airport.py, if present.
    # classify_pipeline.py defaults --nasr-rwy to the static data/ snapshot;
    # override it with the freshly-downloaded per-airport CSV when provided.
    local nasr_rwy_arg=()
    if [[ -n "${NASR_RWY_CSV:-}" && -f "${NASR_RWY_CSV}" ]]; then
        nasr_rwy_arg=(--nasr-rwy "$NASR_RWY_CSV")
        echo "   • using fresh NASR runway data: $NASR_RWY_CSV"
    fi
```

Then change the pipeline invocation to include the array:

```bash
    "$PYTHON" "$PROJECT/python/classify_pipeline.py" \
        --paths "$paths_csv" \
        --pdf "$pdf" \
        --out "$json_out" \
        "${nasr_rwy_arg[@]}" \
        $PIPELINE_EXTRA
```

- [ ] **Step 2: Verify bash syntax and the conditional**

Run:
```bash
bash -n classify.sh && echo "syntax ok"
# Confirm the flag is appended only when the file exists:
NASR_RWY_CSV=/does/not/exist bash -c 'f=(); [[ -n "${NASR_RWY_CSV:-}" && -f "$NASR_RWY_CSV" ]] && f=(--nasr-rwy "$NASR_RWY_CSV"); echo "args=${f[*]:-none}"'
```
Expected: `syntax ok`, then `args=none`. (With a real existing file the array would contain `--nasr-rwy <path>`.)

- [ ] **Step 3: Commit**

```bash
git add classify.sh
git commit -m "feat: let classify.sh use fresh per-airport NASR runway data via NASR_RWY_CSV"
```

---

### Task 6: classify-airports.sh — on-demand entry flow

**Files:**
- Modify: `classify-airports.sh` (full rewrite of the prompt + loop body; keep the config-loading + lighting-prompt structure)

**Interfaces:**
- Consumes: `fetch_airport.py` (`FOLDER=`/`PDF=` stdout contract), `classify.sh` (`NASR_RWY_CSV`, `PLACE_LIGHTS` env), `classify-wrapper.conf` (`DEFAULT_DEST_DIR`).

- [ ] **Step 1: Rewrite the script**

Replace the entire contents of `classify-airports.sh` with:

```bash
#!/usr/bin/env bash
# classify-airports.sh — on-demand FAA airport classifier.
#
# Prompts for a destination folder and one or more FAA LID codes, then for
# EACH code:
#   1. Downloads the latest FAA data (airport-diagram PDF, approach chart,
#      NASR airspace class + runway rows) via fetch_airport.py into a
#      de-duplicated folder under the destination (ord, ord-1, ord-2, ...).
#   2. Runs classify.sh on the freshly-downloaded PDF.
#
# Re-running on an already-classified airport is safe: the folder is
# de-duplicated, never overwritten. The "Place approach lights?" prompt is
# forwarded to classify.sh via PLACE_LIGHTS. Fresh per-airport NASR runway
# data is forwarded via NASR_RWY_CSV.
#
# The destination default comes from classify-wrapper.conf (DEFAULT_DEST_DIR);
# press Enter to accept it or type another path for this run only.

set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$PROJECT/.venv/bin/python3"
CONFIG="$PROJECT/classify-wrapper.conf"

# Hardcoded fallback default. Overridden by classify-wrapper.conf if present.
DEFAULT_DEST_DIR="/Users/lukehogan/Documents/startups/aoa/products/artwork/airports"

if [[ -f "$CONFIG" ]]; then
    # shellcheck disable=SC1090
    source "$CONFIG"
fi

DEST_DIR="$DEFAULT_DEST_DIR"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
read -e -p "Destination folder [$DEST_DIR]: " input
if [[ -n "${input:-}" ]]; then
    DEST_DIR="$input"
fi

read -e -p "Airport LID code(s), comma-separated: " codes_input
if [[ -z "${codes_input:-}" ]]; then
    echo "ERROR: no airport codes provided" >&2
    exit 1
fi

read -e -p "Place approach lights? [y/N] " place_lights_in
case "$(echo "$place_lights_in" | tr '[:upper:]' '[:lower:]')" in
    y|yes) export PLACE_LIGHTS=1 ;;
    *)     export PLACE_LIGHTS=0 ;;
esac
echo

# ---------------------------------------------------------------------------
# Normalize codes: split on comma, trim whitespace, lowercase.
# ---------------------------------------------------------------------------
IFS=',' read -ra RAW_CODES <<< "$codes_input"
CODES=()
for raw in "${RAW_CODES[@]}"; do
    trimmed="$(echo "$raw" | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')"
    if [[ -n "$trimmed" ]]; then
        CODES+=("$trimmed")
    fi
done
if [[ ${#CODES[@]} -eq 0 ]]; then
    echo "ERROR: no valid airport codes after trimming" >&2
    exit 1
fi

mkdir -p "$DEST_DIR"

echo "Destination: $DEST_DIR"
echo "Codes:       ${CODES[*]}"
echo

# ---------------------------------------------------------------------------
# Process each code: fetch latest FAA data, then classify.
# ---------------------------------------------------------------------------
failed=0
for code in "${CODES[@]}"; do
    echo "▶  $code: fetch latest FAA data"
    # Capture fetch_airport.py stdout so we can read FOLDER= / PDF=.
    if ! fetch_out="$("$PYTHON" "$PROJECT/python/fetch_airport.py" \
            --lid "$code" --dest-parent "$DEST_DIR")"; then
        echo "   FAILED: could not fetch $code (no diagram or download error)" >&2
        (( failed++ )) || true
        continue
    fi

    folder="$(echo "$fetch_out" | sed -n 's/^FOLDER=//p')"
    pdf="$(echo "$fetch_out" | sed -n 's/^PDF=//p')"
    if [[ -z "$folder" || -z "$pdf" || ! -f "$pdf" ]]; then
        echo "   FAILED: fetch produced no usable PDF for $code" >&2
        (( failed++ )) || true
        continue
    fi

    # Forward fresh per-airport NASR runway data to classify.sh if written.
    export NASR_RWY_CSV="$folder/$code-nasr-rwy.csv"

    echo "▶  $code: classify ($folder)"
    if ! bash "$PROJECT/classify.sh" "$pdf"; then
        echo "   FAILED: classify $code" >&2
        (( failed++ )) || true
    fi
    echo
done

if [[ $failed -eq 0 ]]; then
    echo "Done. Processed: ${CODES[*]}"
else
    echo "$failed airport(s) failed."
    exit 1
fi
```

- [ ] **Step 2: Verify bash syntax and the stdout parsing**

Run:
```bash
bash -n classify-airports.sh && echo "syntax ok"
# Confirm FOLDER=/PDF= parsing works on sample fetch output:
printf 'FOLDER=/tmp/airports/ord-1\nPDF=/tmp/airports/ord-1/ord-faa.pdf\n' \
  | { out="$(cat)"; echo "folder=$(echo "$out" | sed -n 's/^FOLDER=//p')"; \
      echo "pdf=$(echo "$out" | sed -n 's/^PDF=//p')"; }
```
Expected: `syntax ok`, then `folder=/tmp/airports/ord-1` and `pdf=/tmp/airports/ord-1/ord-faa.pdf`.

- [ ] **Step 3: Live smoke test (requires network + a `.venv`)**

This is the real end-to-end check (no automated test hits the live FAA). If a `.venv` is not present, create one first:
```bash
python3 -m venv .venv && .venv/bin/pip install -e python/
```
Then run against a small airport into a scratch dir, twice, to confirm dedup:
```bash
printf '/tmp/faa-smoke\napf\nn\n' | bash classify-airports.sh
printf '/tmp/faa-smoke\napf\nn\n' | bash classify-airports.sh
ls /tmp/faa-smoke        # expect: apf  apf-1
ls /tmp/faa-smoke/apf    # expect: apf-faa.pdf apf-approach.pdf apf-nasr-rwy.csv apf-diagram.svg (+ .ai if Illustrator ran)
```
Expected: two folders `apf` and `apf-1`; each contains `apf-faa.pdf` and a rendered `apf-diagram.svg`.

- [ ] **Step 4: Commit**

```bash
git add classify-airports.sh
git commit -m "feat: rewrite classify-airports.sh to fetch latest FAA data per LID"
```

---

### Task 7: Docs — document new flow, deprecate bulk downloader

**Files:**
- Modify: `README.md`
- Modify: `classify-wrapper.conf` (drop the now-unused `DEFAULT_SOURCE_DIR`)

- [ ] **Step 1: Trim the wrapper config**

In `classify-wrapper.conf`, delete the `DEFAULT_SOURCE_DIR=...` line and update the header comment so it describes only `DEFAULT_DEST_DIR` (the source-folder concept no longer exists).

- [ ] **Step 2: Update the README**

Add a section documenting the on-demand flow and mark the bulk downloader deprecated. Place it near the existing "Quick start" section:

```markdown
### On-demand classify (recommended)

`classify-airports.sh` downloads the latest FAA data for an airport on
demand — you no longer need a pre-built bulk-download directory:

```bash
bash classify-airports.sh
# Destination folder [.../artwork/airports]:   (Enter to accept)
# Airport LID code(s), comma-separated: ord, apf
# Place approach lights? [y/N]
```

For each LID it downloads the airport-diagram PDF, best approach chart,
and this airport's NASR runway data via `python/fetch_airport.py`, into a
de-duplicated folder (`ord`, then `ord-1`, `ord-2`, … on re-runs), then
runs the classifier. Fresh runway data is passed to the pipeline via
`--nasr-rwy`; the airspace class sets the folder's macOS Finder label.

> **Deprecated:** the standalone bulk downloader at
> `aoa-code/faa-downloader` is superseded by this on-demand path. Its
> download logic now lives in `python/pipeline/faa_fetch.py`.
```

- [ ] **Step 3: Verify the docs render and links are accurate**

Run: `grep -n "fetch_airport\|faa_fetch\|On-demand" README.md`
Expected: the new section and module references are present.

- [ ] **Step 4: Commit**

```bash
git add README.md classify-wrapper.conf
git commit -m "docs: document on-demand FAA fetch; deprecate bulk downloader"
```

---

## Self-Review

**Spec coverage:** faa_fetch module (T1–T3) ✔; fetch CLI + dedup + stdout contract (T4) ✔; classify.sh NASR pass-through (T5) ✔; classify-airports.sh rewrite with dest prompt + multi-code + lighting prompt + no Proceed/source prompt (T6) ✔; requests dependency (T1) ✔; README deprecation (T7) ✔; runway refresh from fresh NASR (T2 extract + T4 write + T5 consume) ✔; full-parity download incl. approach chart + Finder label (T3/T4) ✔.

**Placeholder scan:** no TBD/TODO; every code and test step carries concrete content.

**Type consistency:** `dedup_folder(parent, lid)`, `download_nasr(...) -> (airports, runway_rows)`, `write_runway_csv(rows, dest)`, `run(lid, dest_parent, priority=None, *, session=None) -> (folder, pdf)`, `main(argv) -> int` used consistently across tasks; filenames always `<lid_lower>-faa.pdf` / `-approach.pdf` / `-nasr-rwy.csv`.

**Review Focus coverage:** no-diagram LID → `test_main_exits_nonzero_when_no_diagram` (T4); re-run dedup → `test_dedup_folder_increments_suffix` (T1) + `test_run_dedups_and_keeps_true_lid_filenames` (T4); LID casing → `run()` normalization asserted in T4 tests (lowercase filenames from `"ORD"`); missing approach → `test_run_no_approach_still_writes_diagram` (T4); deduped folder keeps true-LID filenames → `test_run_dedups_and_keeps_true_lid_filenames` (T4).
