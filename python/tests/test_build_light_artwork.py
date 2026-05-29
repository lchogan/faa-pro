import json
from pathlib import Path

import pytest

from pipeline.build_light_artwork import build_light_artwork

ARTBOARDS = Path(__file__).resolve().parent / "fixtures" / "artboards"


def test_build_writes_normalized_library(tmp_path):
    out = tmp_path / "lights.json"
    build_light_artwork(ARTBOARDS, out)
    data = json.loads(out.read_text())
    assert data["native_width"] == 100.0          # shared artboard width W
    assert set(data["symbols"]) == {"malsr", "odals"}
    # malsr centerline M50 120 -> attach origin (0,0); L50 20 -> forward 100.
    # canonical (X,Y) = (H - py, px - W/2): (120-120, 0)=(0,0); (120-20,0)=(100,0)
    d = data["symbols"]["malsr"]["paths"][0]["d"]
    assert d.startswith("M0")          # attach point at origin
    assert "100" in d                  # forward extent along +X


def test_build_rejects_mismatched_artboard_widths(tmp_path):
    bad = tmp_path / "artboards"
    bad.mkdir()
    (bad / "a.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="120" '
        'viewBox="0 0 100 120"><path d="M50 120 L50 20" fill="none" '
        'stroke="#000"/></svg>'
    )
    (bad / "b.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="80" height="120" '
        'viewBox="0 0 80 120"><path d="M40 120 L40 20" fill="none" '
        'stroke="#000"/></svg>'
    )
    with pytest.raises(ValueError, match="width"):
        build_light_artwork(bad, tmp_path / "out.json")
