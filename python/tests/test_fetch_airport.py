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
    assert "no faa airport diagram" in capsys.readouterr().err.lower()


def _write_fake_pdf(dest):
    dest.write_bytes(b"%PDF-1.4 fake" + b"x" * 600)


def test_run_approach_failure_warns_but_keeps_diagram(patched, tmp_path, monkeypatch, capsys):
    def flaky_download(url, dest, session, timeout):
        if "approach" in dest.name:
            raise RuntimeError("boom")
        _write_fake_pdf(dest)

    monkeypatch.setattr(faa_fetch, "download_pdf", flaky_download)
    folder, pdf = fetch_airport.run("ord", tmp_path)
    assert pdf.exists()
    assert not (folder / "ord-approach.pdf").exists()
    assert "WARN" in capsys.readouterr().err


def test_run_cleans_up_folder_when_diagram_download_fails(patched, tmp_path, monkeypatch):
    def failing_download(url, dest, session, timeout):
        raise RuntimeError("boom")

    monkeypatch.setattr(faa_fetch, "download_pdf", failing_download)
    with pytest.raises(RuntimeError):
        fetch_airport.run("ord", tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_run_survives_nasr_failure(patched, tmp_path, monkeypatch, capsys):
    def failing_nasr(date, session, lid=None, timeout=300):
        raise RuntimeError("nasr 503")

    monkeypatch.setattr(faa_fetch, "download_nasr", failing_nasr)
    folder, pdf = fetch_airport.run("ord", tmp_path)
    assert pdf.exists()
    assert not (folder / "ord-nasr-rwy.csv").exists()
    assert "NASR fetch failed" in capsys.readouterr().err
