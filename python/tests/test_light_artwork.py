import json
from pathlib import Path

import pytest

from pipeline.light_artwork import load_light_artwork

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "lights_sample.json"


def test_load_returns_symbols_and_native_width():
    lib = load_light_artwork(FIXTURE)
    assert lib.native_width == 100.0
    assert set(lib.symbols) == {"malsr", "odals"}
    assert lib.symbols["malsr"]["paths"][0]["d"] == "M0 0 L10 0"


def test_missing_native_width_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"symbols": {}}))
    with pytest.raises(ValueError, match="native_width"):
        load_light_artwork(bad)
