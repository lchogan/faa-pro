import csv
import io
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime

from pipeline import faa_fetch
from pipeline.extract_pdf_text import load_nasr_runways


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


def test_determine_airspace_class_prefers_cls_arsp_then_tower():
    assert faa_fetch.determine_airspace_class("ORD", {"_AIRSPACE_CLASS": "B"}) == "B"
    # CLS_ARSP present but None -> None even if a tower exists.
    assert faa_fetch.determine_airspace_class(
        "XXX", {"_AIRSPACE_CLASS": None, "TWR_TYPE_CODE": "ATCT"}
    ) is None
    # No CLS_ARSP entry at all -> infer Class D from tower presence.
    assert faa_fetch.determine_airspace_class("YYY", {"TWR_TYPE_CODE": "ATCT"}) == "D"
    assert faa_fetch.determine_airspace_class("ZZZ", {}) is None
