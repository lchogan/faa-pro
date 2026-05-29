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
    # Use format-independent parse so we're not sensitive to "M0" vs "M 0,0".
    from svgelements import Path as SvgPath
    seg = SvgPath(data["symbols"]["malsr"]["paths"][0]["d"])
    fp = seg.first_point
    assert abs(fp.x) < 1e-6 and abs(fp.y) < 1e-6        # attach point at origin
    xs = [s.end.x for s in seg if s.end is not None]
    assert max(xs) > 99.0                               # forward extent ~100 along +X


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


def test_curves_are_preserved_not_flattened(tmp_path):
    """Bézier control points must survive the canonical transform (regression)."""
    ab = tmp_path / "ab"
    ab.mkdir()
    (ab / "curve.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="120" '
        'viewBox="0 0 100 120"><path d="M50 120 C40 90 60 50 50 20" '
        'fill="none" stroke="#000"/></svg>'
    )
    out = tmp_path / "lights.json"
    build_light_artwork(ab, out)
    ds = json.loads(out.read_text())["symbols"]["curve"]["paths"][0]["d"]
    assert "C" in ds.upper()   # cubic preserved, not flattened to straight lines


def test_svg_primitives_are_not_dropped(tmp_path):
    """SVG primitives (rect/circle/line/polygon) must be converted, not silently dropped."""
    ab = tmp_path / "ab"
    ab.mkdir()
    (ab / "dot.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="120" '
        'viewBox="0 0 100 120"><circle cx="50" cy="60" r="10" fill="#000"/></svg>'
    )
    out = tmp_path / "lights.json"
    build_light_artwork(ab, out)
    paths = json.loads(out.read_text())["symbols"]["dot"]["paths"]
    assert paths and paths[0]["d"].strip()   # circle converted to a path, not dropped
