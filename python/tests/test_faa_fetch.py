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
