# Approach Lighting Symbols Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render an approach-lighting symbol at each runway end that has an approach lighting system (ALS), oriented to the runway, scaled uniformly per diagram to the widest runway's depicted width, placed a visible gap off the pavement end.

**Architecture:** A new `approach_light_layout` module mirrors `runway_label_layout` but drops the collision-march (lights may overlap pavement). It reuses `_runway_extents` for axis/width and the existing step-3 runway-number↔end pairing to map NASR per-end ALS codes to geometric ends — no lat/long or compass needed. Placement is emitted as `light_groups` in the predictions JSON; `render_svg_layers` injects them as new `<path>` artwork (loaded from a normalized JSON artwork library) into a new "Approach Lights" SVG layer.

**Tech Stack:** Python 3.12, `csv` (stdlib), `pytest` (added here — no tests exist yet), `svgelements` (build-time only, for flattening Illustrator SVG → artwork JSON), PyMuPDF (already used by the renderer).

---

## Spec

Design spec: [docs/superpowers/specs/2026-05-29-approach-lighting-design.md](../specs/2026-05-29-approach-lighting-design.md).

## Refinements to the spec (decided during planning)

- **`light_groups` schema** uses `{artwork_key, x, y, dir_x, dir_y, target_width}` (a direction *vector* + target rendered width), **not** `rotate_deg`/`scale`. This matches the codebase's existing `dir_x`/`dir_y` matrix convention in `render_svg_layers.py` and **decouples the layout module from the artwork** — the layout module never needs to read artwork files; the renderer computes `scale = target_width / artwork_native_width`.
- **Artwork library is a checked-in normalized JSON** (`artwork/lights/lights.json`) produced by an offline build script (`build_light_artwork.py`) from the per-artboard SVG exports. Runtime parses no SVG/.ai — it reads JSON. This isolates fragile vector parsing to a rarely-run build step.
- **`target_width = k · (2 · max_half_wid)`** (uniform per diagram); **`gap = g · target_width`**. `k`, `g` are module constants for calibration.

## File Structure

**New files**
- `python/pipeline/approach_lights_data.py` — NASR APT_RWY_END loader + FAA-code→artwork-key mapping (data concerns together).
- `python/pipeline/approach_light_layout.py` — placement math: `compute_approach_light_placements(...)`. Reuses `_runway_extents`, `_normalize_designation`.
- `python/pipeline/light_artwork.py` — runtime loader for `artwork/lights/lights.json` (cached; asserts equal native widths).
- `python/pipeline/build_light_artwork.py` — offline build: per-artboard SVG dir → `lights.json` (uses `svgelements`; normalizes to canonical frame; asserts equal artboard widths).
- `python/tests/conftest.py`, `python/tests/test_*.py`, `python/tests/fixtures/...` — the project's first tests.

**Modified files**
- `python/pyproject.toml` — add a test/build optional-dependency group (`pytest`, `svgelements`).
- `python/classify_pipeline.py` — `--place-lights` flag, `--nasr-rwy-end` arg, step 8b, `payload["light_groups"]`.
- `python/render_svg_layers.py` — read `light_groups`, add "Approach Lights" layer, inject symbols.
- `classify.sh`, `classify-airports.sh` — pass `--place-lights` through (thin).

## Canonical artwork frame (used by build + render)

After normalization (done in `build_light_artwork.py`), every symbol's paths are stored so that:
- **attach point = origin (0, 0)** (the point that sits at the threshold).
- **forward (away from the runway) = +X**; lateral = ±Y.
- **`native_width`** = the runway-width reference the symbol was drawn against (identical across all artboards; asserted equal).

The renderer then emits each symbol under `transform="matrix(s·dir_x, s·dir_y, -s·dir_y, s·dir_x, x, y)"` where `s = target_width / native_width` and `(dir_x, dir_y)` is the runway outward unit vector — identical in form to the existing text matrix at `render_svg_layers.py:308-312`.

---

## Task 0: Test harness bootstrap

**Files:**
- Modify: `python/pyproject.toml`
- Create: `python/tests/__init__.py` (empty)
- Create: `python/tests/conftest.py`
- Create: `python/tests/test_smoke.py`

- [ ] **Step 1: Add a test/build optional-dependency group to `pyproject.toml`**

Append after the `dependencies = [...]` array (before `[project.scripts]`):

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "svgelements>=1.9",
]
```

- [ ] **Step 2: Install the dev deps into the existing venv**

Run: `cd /Users/lukehogan/AOA-Code/faa-pro && .venv/bin/python -m pip install "pytest>=8.0" "svgelements>=1.9"`
Expected: pytest and svgelements install successfully.

- [ ] **Step 3: Create `python/tests/__init__.py`** (empty file).

- [ ] **Step 4: Create `python/tests/conftest.py`**

```python
"""Pytest configuration for the FAA-classifier test suite.

Adds the `python/` directory (this file's parent's parent) to sys.path so
tests can `from pipeline.X import Y` exactly as the production scripts do
(production sets PYTHONPATH=<repo>/python in classify.sh). Keeping the path
fix here means `pytest` works without callers having to export PYTHONPATH.
"""
import sys
from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parent.parent
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

# Fixtures directory, importable by tests via `from conftest import FIXTURES`.
FIXTURES = Path(__file__).resolve().parent / "fixtures"
```

- [ ] **Step 5: Create `python/tests/test_smoke.py`**

```python
def test_pipeline_package_importable():
    import pipeline  # noqa: F401
```

- [ ] **Step 6: Run the smoke test**

Run: `cd /Users/lukehogan/AOA-Code/faa-pro && .venv/bin/python -m pytest python/tests/test_smoke.py -v`
Expected: 1 passed.

- [ ] **Step 7: Commit**

```bash
git add python/pyproject.toml python/tests/
git commit -m "test: bootstrap pytest harness (first tests in repo)"
```

---

## Task 1: FAA code → artwork-key mapping

**Files:**
- Create: `python/pipeline/approach_lights_data.py`
- Test: `python/tests/test_approach_lights_data.py`

- [ ] **Step 1: Write the failing test**

```python
from pipeline.approach_lights_data import als_code_to_artwork_key


def test_mapped_codes_resolve_to_artwork_keys():
    assert als_code_to_artwork_key("ALSF2") == "alsf2"
    assert als_code_to_artwork_key("MALSR") == "malsr"
    assert als_code_to_artwork_key("SSALR") == "malsr"   # shares MALSR artwork
    assert als_code_to_artwork_key("ODALS") == "odals"


def test_code_matching_is_case_and_space_insensitive():
    assert als_code_to_artwork_key(" malsr ") == "malsr"
    assert als_code_to_artwork_key("Alsf1") == "alsf1"


def test_none_blank_and_unmapped_codes_return_none():
    assert als_code_to_artwork_key("NONE") is None
    assert als_code_to_artwork_key("") is None
    assert als_code_to_artwork_key(None) is None
    assert als_code_to_artwork_key("NSTD") is None   # out of scope -> no symbol
    assert als_code_to_artwork_key("RLLS") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/lukehogan/AOA-Code/faa-pro && .venv/bin/python -m pytest python/tests/test_approach_lights_data.py -v`
Expected: FAIL with "No module named 'pipeline.approach_lights_data'".

- [ ] **Step 3: Write the mapping (start the new module)**

Create `python/pipeline/approach_lights_data.py`:

```python
"""approach_lights_data.py — FAA approach-lighting data: load per-runway-end
ALS codes from NASR, and map those codes to artwork keys.

Two concerns, kept together because both are about NASR's approach-light
codes:

  * load_approach_lights(csv_path, airport) -> {end_designator: raw_code}
        Reads the NASR runway-end table (APT_RWY_END.csv / equivalent).
  * als_code_to_artwork_key(code) -> artwork_key | None
        Maps an FAA ALS code to the artwork archetype that depicts it.
        Several codes share one symbol; NONE/blank/out-of-scope -> None.

Authoritative FAA code set (apt.txt field "APPROACH LIGHT SYSTEM"):
  ALSAF, ALSF1, ALSF2, MALS, MALSF, MALSR, SSALS, SSALF, SSALR,
  NEON, ODALS, RLLS, MIL OVRN, NSTD, NONE.
v1 scope is centerline ALS only; anything not in CODE_TO_ARTWORK_KEY
degrades to no symbol (the layout module logs a diagnostic for codes that
were present but unmapped, so new/edge codes surface).
"""
from __future__ import annotations


# FAA code -> artwork key. Codes grouped together are visually equivalent at
# chart-symbol resolution. [VERIFY] groupings against chart symbology before
# shipping; unmapped codes degrade safely to no-symbol.
CODE_TO_ARTWORK_KEY: dict[str, str] = {
    "ALSF1": "alsf1",
    "ALSF2": "alsf2",
    "MALSR": "malsr",
    "SSALR": "malsr",
    "MALSF": "malsf",
    "SSALF": "malsf",
    "MALS": "mals",
    "SSALS": "mals",
    "SALS": "sals",
    "SALSF": "sals",
    "ODALS": "odals",
}


def als_code_to_artwork_key(code: str | None) -> str | None:
    """Return the artwork key for an FAA ALS code, or None when there is no
    symbol to draw (NONE / blank / out-of-scope code)."""
    if not code:
        return None
    return CODE_TO_ARTWORK_KEY.get(code.strip().upper())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/lukehogan/AOA-Code/faa-pro && .venv/bin/python -m pytest python/tests/test_approach_lights_data.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add python/pipeline/approach_lights_data.py python/tests/test_approach_lights_data.py
git commit -m "feat: add FAA ALS code -> artwork-key mapping"
```

---

## Task 2: NASR runway-end ALS loader

**Files:**
- Modify: `python/pipeline/approach_lights_data.py`
- Create: `python/tests/fixtures/nasr_apt_rwy_end_sample.csv`
- Test: `python/tests/test_approach_lights_data.py`

- [ ] **Step 1: Create the fixture CSV** `python/tests/fixtures/nasr_apt_rwy_end_sample.csv`

```csv
ARPT_ID,RWY_ID,RWY_END_ID,APPROACH_LIGHT_SYSTEM
DEN,16/34,16,MALSR
DEN,16/34,34,NONE
DEN,08/26,08,ALSF2
DEN,08/26,26,ODALS
APF,NE/SW,NE,MALSF
ATL,09L/27R,09L,MALSR
ATL,09L/27R,27R,
```

- [ ] **Step 2: Write the failing test (append to `test_approach_lights_data.py`)**

```python
from pathlib import Path
from pipeline.approach_lights_data import load_approach_lights

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "nasr_apt_rwy_end_sample.csv"


def test_load_returns_normalized_end_to_code_for_airport():
    result = load_approach_lights(FIXTURE, "den")
    # Designators are normalized the same way runway-number tokens are:
    # leading zeros stripped, L/R/C kept, compass uppercased.
    assert result == {"16": "MALSR", "34": "NONE", "8": "ALSF2", "26": "ODALS"}


def test_load_is_airport_case_insensitive_and_handles_compass_and_suffix():
    assert load_approach_lights(FIXTURE, "APF") == {"NE": "MALSF"}
    atl = load_approach_lights(FIXTURE, "atl")
    assert atl["9L"] == "MALSR"      # "09L" -> "9L"
    assert "27R" not in atl          # blank code row dropped


def test_load_unknown_airport_returns_empty():
    assert load_approach_lights(FIXTURE, "zzz") == {}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /Users/lukehogan/AOA-Code/faa-pro && .venv/bin/python -m pytest python/tests/test_approach_lights_data.py -v`
Expected: FAIL with "cannot import name 'load_approach_lights'".

- [ ] **Step 4: Implement the loader (append to `approach_lights_data.py`)**

```python
import csv as _csv
from pathlib import Path

from pipeline.extract_pdf_text import _normalize_designation


# Column-header aliases. NASR's modern CSV header names are not 100%
# pinned ([VERIFY] against a freshly downloaded APT_RWY_END.csv and trim
# this list to the real names). First matching header in each list wins.
_AIRPORT_COLS = ("ARPT_ID", "LOCATION_ID", "LOCID", "LOC_ID")
_END_ID_COLS = ("RWY_END_ID", "RWY_END_IDENT", "BASE_END_ID", "RUNWAY_END_IDENT")
_ALS_COLS = (
    "APPROACH_LIGHT_SYSTEM",
    "APCH_LGT_SYSTEM_CODE",
    "RWY_END_LGT_APCH_LGT_SYSTEM_TYPE",
    "APCH_LGTG_SYS_CODE",
)


def _pick_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str:
    """Return the first candidate header present in fieldnames, else raise
    with the actual headers so a schema mismatch is obvious."""
    present = {fn.strip().upper(): fn for fn in (fieldnames or [])}
    for cand in candidates:
        if cand.upper() in present:
            return present[cand.upper()]
    raise ValueError(
        f"None of {candidates} found in APT_RWY_END columns: {fieldnames}"
    )


def load_approach_lights(csv_path: Path, airport_code: str) -> dict[str, str]:
    """Load per-runway-end approach-light codes for one airport.

    Returns {normalized_end_designator: als_code}, e.g. {"16": "MALSR"}.
    End designators are normalized via the same `_normalize_designation`
    the runway-number matcher uses, so they join directly to the tokens in
    step3_diag. Rows with a blank code are dropped. Match on the airport id
    is case-insensitive.
    """
    target = airport_code.strip().upper()
    out: dict[str, str] = {}
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = _csv.DictReader(f)
        if not reader.fieldnames:
            return out
        ap_col = _pick_column(reader.fieldnames, _AIRPORT_COLS)
        end_col = _pick_column(reader.fieldnames, _END_ID_COLS)
        als_col = _pick_column(reader.fieldnames, _ALS_COLS)
        for row in reader:
            if (row.get(ap_col) or "").strip().upper() != target:
                continue
            code = (row.get(als_col) or "").strip()
            if not code:
                continue
            desig = _normalize_designation((row.get(end_col) or "").strip())
            if desig:
                out[desig] = code.upper()
    return out
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /Users/lukehogan/AOA-Code/faa-pro && .venv/bin/python -m pytest python/tests/test_approach_lights_data.py -v`
Expected: PASS (all tests in the file).

- [ ] **Step 6: Commit**

```bash
git add python/pipeline/approach_lights_data.py python/tests/test_approach_lights_data.py python/tests/fixtures/nasr_apt_rwy_end_sample.csv
git commit -m "feat: load per-runway-end ALS codes from NASR APT_RWY_END"
```

---

## Task 3: Reciprocal-designation helper

**Files:**
- Create: `python/pipeline/approach_light_layout.py`
- Test: `python/tests/test_approach_light_layout.py`

- [ ] **Step 1: Write the failing test**

```python
from pipeline.approach_light_layout import _reciprocal_designation


def test_numeric_reciprocals_swap_lr_and_add_18():
    assert _reciprocal_designation("17R") == "35L"
    assert _reciprocal_designation("35L") == "17R"
    assert _reciprocal_designation("9L") == "27R"
    assert _reciprocal_designation("18") == "36"
    assert _reciprocal_designation("36") == "18"
    assert _reciprocal_designation("4C") == "22C"


def test_compass_reciprocals():
    assert _reciprocal_designation("NE") == "SW"
    assert _reciprocal_designation("N") == "S"
    assert _reciprocal_designation("W") == "E"


def test_unrecognized_returns_none():
    assert _reciprocal_designation("XYZ") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/lukehogan/AOA-Code/faa-pro && .venv/bin/python -m pytest python/tests/test_approach_light_layout.py -v`
Expected: FAIL with "No module named 'pipeline.approach_light_layout'".

- [ ] **Step 3: Start the module with the helper**

Create `python/pipeline/approach_light_layout.py`:

```python
"""approach_light_layout.py — approach-lighting symbol placement.

Runs after final classification (step 8b in classify_pipeline.py), parallel
to runway_label_layout but with a key difference: lights MAY overlap runways
and taxiways, so there is NO collision-march. Each symbol sits at a fixed gap
off the pavement end, centered on the runway centerline, oriented to the
runway, and scaled by a single per-document factor (keyed to the widest
runway's depicted width) so all lights on a page share one size relationship.

Per-end ALS codes come from NASR (approach_lights_data.load_approach_lights).
Which geometric end is "16" vs "34" is resolved from step3_diag's existing
runway-number↔end pairing — no lat/long or compass needed. If only one end's
number was detected, the other end's designator is the reciprocal.

Output (light_groups), in PyMuPDF top-left (y-down) coords — same frame as the
per-polygon translate, so the renderer applies it with no flip:
  {artwork_key, x, y, dir_x, dir_y, target_width}
where (x, y) is the attach point at the threshold (gap already applied),
(dir_x, dir_y) is the outward unit vector (forward axis of the symbol), and
target_width = k * (2 * max_half_wid) is the rendered width one runway-width
reference maps to (same for every light on the page).
"""
from __future__ import annotations

import math
import re

from pipeline.taxi_detection import _runway_extents
from pipeline.extract_pdf_text import _normalize_designation
from pipeline.approach_lights_data import als_code_to_artwork_key


# Tuning knobs (calibrate visually on DEN + a small field — see plan Task 10).
DEFAULT_K = 0.5   # overall light size relative to the widest depicted width
DEFAULT_G = 0.3   # gap off the pavement end, as a fraction of target_width

_DESIG_RE = re.compile(r"^(\d{1,2})([LRC]?)$")
_COMPASS_RECIP = {
    "N": "S", "S": "N", "E": "W", "W": "E",
    "NE": "SW", "SW": "NE", "NW": "SE", "SE": "NW",
}
_LR_SWAP = {"L": "R", "R": "L", "C": "C", "": ""}


def _reciprocal_designation(desig: str) -> str | None:
    """Return the opposite-end designation: numeric ends add 18 (mod 36) and
    swap L<->R; compass ends flip to their opposite. None if unrecognized.
    Input/output are in canonical form (no leading zero)."""
    m = _DESIG_RE.match(desig or "")
    if m:
        n = int(m.group(1)) + 18
        if n > 36:
            n -= 36
        return f"{n}{_LR_SWAP[m.group(2)]}"
    return _COMPASS_RECIP.get((desig or "").upper())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/lukehogan/AOA-Code/faa-pro && .venv/bin/python -m pytest python/tests/test_approach_light_layout.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add python/pipeline/approach_light_layout.py python/tests/test_approach_light_layout.py
git commit -m "feat: add reciprocal-designation helper for approach lights"
```

---

## Task 4: Approach-light placement (`compute_approach_light_placements`)

**Files:**
- Modify: `python/pipeline/approach_light_layout.py`
- Test: `python/tests/test_approach_light_layout.py`

Notes on test fixtures: a runway poly needs `subpaths` (a closed ring) and `rect`, because `_runway_extents(subpaths, rect)` runs PCA over the ring's anchors. The helper below builds an axis-aligned rectangle runway centered at (cx, cy) with given length (along x) and width (along y).

- [ ] **Step 1: Write the failing test (append to `test_approach_light_layout.py`)**

```python
from pipeline.approach_light_layout import (
    compute_approach_light_placements,
    DEFAULT_K,
)


def _rect_runway(cx, cy, length, width):
    """Axis-aligned runway polygon: long axis along x, width along y."""
    hl, hw = length / 2.0, width / 2.0
    ring = [
        (cx - hl, cy - hw), (cx + hl, cy - hw),
        (cx + hl, cy + hw), (cx - hl, cy + hw),
    ]
    return {
        "cx": cx, "cy": cy,
        "rect": (cx - hl, cy - hw, cx + hl, cy + hw),
        "subpaths": [ring],
        "filled": True, "stroked": False, "stroke_width": 0.0,
        "is_taxi_surface": False, "is_near_black": False,
    }


def _diag(runway_idx, end_sign, token):
    return {
        "runway_idx": runway_idx, "end_sign": end_sign,
        "matched": token is not None, "token": token, "claimed": [],
    }


def test_places_one_symbol_per_end_with_als_oriented_and_scaled():
    # One horizontal runway, idx 0, length 400 width 40 -> half_wid 20.
    polys = [_rect_runway(500.0, 300.0, 400.0, 40.0)]
    step3 = [_diag(0, +1, "16"), _diag(0, -1, "34")]
    als = {"16": "MALSR", "34": "ALSF2"}
    groups, diags = compute_approach_light_placements(
        polys, step3, runway_indices=[0], als_by_end=als,
    )
    assert len(groups) == 2
    by_key = {g["artwork_key"]: g for g in groups}
    assert set(by_key) == {"malsr", "alsf2"}
    # Uniform target width = k * (2 * max_half_wid) = 0.5 * 40 = 20.
    for g in groups:
        assert g["target_width"] == DEFAULT_K * 40.0
    # Horizontal runway: outward unit vector is +/-(1, 0).
    assert {round(g["dir_x"]) for g in groups} == {-1, 1}
    assert all(round(g["dir_y"]) == 0 for g in groups)
    # Attach points sit beyond each end (|x - 500| > half_len 200) on the
    # centerline (y == 300).
    for g in groups:
        assert abs(g["x"] - 500.0) > 200.0
        assert abs(g["y"] - 300.0) < 1e-6


def test_scale_is_keyed_to_widest_runway_across_the_document():
    # Wide runway idx 0 (width 60) and narrow runway idx 1 (width 20).
    polys = [
        _rect_runway(500.0, 200.0, 400.0, 60.0),   # max half_wid = 30
        _rect_runway(500.0, 600.0, 400.0, 20.0),
    ]
    step3 = [_diag(1, +1, "09"), _diag(1, -1, "27")]   # lights only on narrow
    als = {"9": "MALSR", "27": "MALSR"}
    groups, _ = compute_approach_light_placements(
        polys, step3, runway_indices=[0, 1], als_by_end=als,
    )
    # target width uses the WIDEST runway (half_wid 30), not the narrow one.
    assert all(g["target_width"] == DEFAULT_K * 60.0 for g in groups)


def test_reciprocal_inference_when_one_end_unmatched():
    polys = [_rect_runway(500.0, 300.0, 400.0, 40.0)]
    step3 = [_diag(0, +1, "16"), _diag(0, -1, None)]   # only +1 matched
    als = {"34": "ALSF2"}                              # ALS on the recip end
    groups, _ = compute_approach_light_placements(
        polys, step3, runway_indices=[0], als_by_end=als,
    )
    assert [g["artwork_key"] for g in groups] == ["alsf2"]
    assert round(groups[0]["dir_x"]) == -1   # the -1 (reciprocal) end


def test_none_and_unmapped_codes_emit_no_symbol():
    polys = [_rect_runway(500.0, 300.0, 400.0, 40.0)]
    step3 = [_diag(0, +1, "16"), _diag(0, -1, "34")]
    als = {"16": "NONE", "34": "NSTD"}   # one NONE, one out-of-scope
    groups, _ = compute_approach_light_placements(
        polys, step3, runway_indices=[0], als_by_end=als,
    )
    assert groups == []


def test_runway_with_no_matched_number_is_skipped():
    polys = [_rect_runway(500.0, 300.0, 400.0, 40.0)]
    step3 = [_diag(0, +1, None), _diag(0, -1, None)]
    als = {"16": "MALSR"}
    groups, diags = compute_approach_light_placements(
        polys, step3, runway_indices=[0], als_by_end=als,
    )
    assert groups == []
    assert any(d.get("skipped") == "no_matched_number" for d in diags)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/lukehogan/AOA-Code/faa-pro && .venv/bin/python -m pytest python/tests/test_approach_light_layout.py -v`
Expected: FAIL with "cannot import name 'compute_approach_light_placements'".

- [ ] **Step 3: Implement `compute_approach_light_placements` (append to `approach_light_layout.py`)**

```python
def _document_width_reference(all_polys, runway_indices) -> float:
    """Largest depicted half-width across all runways on the page. Behind one
    function so it can be swapped for a median/percentile during tuning."""
    max_half_wid = 0.0
    for ri in runway_indices:
        poly = all_polys[ri]
        _, _, _, _, _, half_wid = _runway_extents(
            poly.get("subpaths") or [], poly["rect"],
        )
        if half_wid > max_half_wid:
            max_half_wid = half_wid
    return max_half_wid


def compute_approach_light_placements(
    all_polys: list[dict],
    step3_diag: list[dict],
    runway_indices,
    als_by_end: dict[str, str],
    *,
    k: float = DEFAULT_K,
    g: float = DEFAULT_G,
) -> tuple[list[dict], list[dict]]:
    """Compute one light_group per runway end that has an in-scope ALS.

    See module docstring for the output schema. `als_by_end` maps normalized
    end designators to raw FAA codes (from approach_lights_data). `k`/`g` are
    the size/gap tuning knobs.
    """
    light_groups: list[dict] = []
    diagnostics: list[dict] = []

    max_half_wid = _document_width_reference(all_polys, runway_indices)
    if max_half_wid <= 0.0:
        return light_groups, diagnostics
    target_width = k * (2.0 * max_half_wid)
    gap = g * target_width

    # Group step3 entries by runway.
    by_runway: dict[int, list[dict]] = {}
    for d in step3_diag:
        by_runway.setdefault(d["runway_idx"], []).append(d)

    for ri in runway_indices:
        ends = by_runway.get(ri, [])

        # Designator per end_sign, from the matched runway-number token.
        desig_by_sign: dict[int, str] = {}
        for d in ends:
            if d.get("matched") and d.get("token"):
                nd = _normalize_designation(str(d["token"]))
                if nd:
                    desig_by_sign[int(d["end_sign"])] = nd
        if not desig_by_sign:
            diagnostics.append({"runway_idx": ri, "skipped": "no_matched_number"})
            continue
        # Reciprocal inference when only one end carried a number.
        if len(desig_by_sign) == 1:
            (known_sign, known_desig), = desig_by_sign.items()
            recip = _reciprocal_designation(known_desig)
            if recip:
                desig_by_sign.setdefault(-known_sign, recip)

        poly = all_polys[ri]
        cx, cy, ux, uy, half_len, _ = _runway_extents(
            poly.get("subpaths") or [], poly["rect"],
        )
        if not math.isfinite(half_len):
            continue

        for end_sign, desig in desig_by_sign.items():
            code = als_by_end.get(desig)
            key = als_code_to_artwork_key(code)
            if key is None:
                if code:
                    diagnostics.append({
                        "runway_idx": ri, "end_sign": end_sign,
                        "designator": desig, "code": code, "skipped": "unmapped_code",
                    })
                continue
            out_x, out_y = end_sign * ux, end_sign * uy
            ax = cx + (half_len + gap) * out_x
            ay = cy + (half_len + gap) * out_y
            light_groups.append({
                "artwork_key": key,
                "x": round(ax, 4),
                "y": round(ay, 4),
                "dir_x": round(out_x, 6),
                "dir_y": round(out_y, 6),
                "target_width": round(target_width, 4),
            })
            diagnostics.append({
                "runway_idx": ri, "end_sign": end_sign,
                "designator": desig, "code": code, "artwork_key": key,
            })

    return light_groups, diagnostics
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/lukehogan/AOA-Code/faa-pro && .venv/bin/python -m pytest python/tests/test_approach_light_layout.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add python/pipeline/approach_light_layout.py python/tests/test_approach_light_layout.py
git commit -m "feat: compute approach-light placements (no march, doc-uniform scale)"
```

---

## Task 5: Runtime artwork library loader

**Files:**
- Create: `python/pipeline/light_artwork.py`
- Create: `python/tests/fixtures/lights_sample.json`
- Test: `python/tests/test_light_artwork.py`

The runtime loader reads a normalized artwork JSON. Schema (produced by Task 9's build script):

```json
{
  "native_width": 100.0,
  "symbols": {
    "malsr": {"paths": [{"d": "M0 0 L10 0", "fill": "#000000", "stroke": "none", "stroke_width": 0.0}]},
    "odals": {"paths": [{"d": "M0 0 L5 0", "fill": "#000000", "stroke": "none", "stroke_width": 0.0}]}
  }
}
```

`native_width` is shared across symbols (all artboards same width — asserted at build time). Stored once at top level.

- [ ] **Step 1: Create the fixture** `python/tests/fixtures/lights_sample.json`

```json
{
  "native_width": 100.0,
  "symbols": {
    "malsr": {"paths": [{"d": "M0 0 L10 0", "fill": "#000000", "stroke": "none", "stroke_width": 0.0}]},
    "odals": {"paths": [{"d": "M0 0 L5 0", "fill": "#000000", "stroke": "none", "stroke_width": 0.0}]}
  }
}
```

- [ ] **Step 2: Write the failing test**

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /Users/lukehogan/AOA-Code/faa-pro && .venv/bin/python -m pytest python/tests/test_light_artwork.py -v`
Expected: FAIL with "No module named 'pipeline.light_artwork'".

- [ ] **Step 4: Implement the loader**

Create `python/pipeline/light_artwork.py`:

```python
"""light_artwork.py — runtime loader for the normalized approach-light artwork
library (artwork/lights/lights.json), produced offline by build_light_artwork.

The library is already normalized to the canonical frame (attach point at the
origin, forward = +X) and all symbols share one `native_width` (the runway-
width reference). Runtime does NO SVG/.ai parsing — it reads JSON and caches.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class LightArtwork:
    native_width: float
    symbols: dict  # {key: {"paths": [{"d", "fill", "stroke", "stroke_width"}]}}


@lru_cache(maxsize=8)
def load_light_artwork(json_path: Path) -> LightArtwork:
    """Load and cache the artwork library. Raises ValueError if native_width
    is missing or non-positive (an unbuilt/corrupt library)."""
    data = json.loads(Path(json_path).read_text())
    native_width = data.get("native_width")
    if not native_width or float(native_width) <= 0.0:
        raise ValueError(
            f"lights.json missing positive native_width: {json_path}"
        )
    return LightArtwork(
        native_width=float(native_width),
        symbols=dict(data.get("symbols") or {}),
    )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /Users/lukehogan/AOA-Code/faa-pro && .venv/bin/python -m pytest python/tests/test_light_artwork.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add python/pipeline/light_artwork.py python/tests/test_light_artwork.py python/tests/fixtures/lights_sample.json
git commit -m "feat: add runtime approach-light artwork library loader"
```

---

## Task 6: Renderer injection (`emit_approach_lights` + wire into main)

**Files:**
- Modify: `python/render_svg_layers.py`
- Test: `python/tests/test_render_lights.py`

- [ ] **Step 1: Write the failing test**

```python
import xml.etree.ElementTree as ET

from render_svg_layers import emit_approach_lights
from pipeline.light_artwork import LightArtwork


def _artwork():
    return LightArtwork(
        native_width=100.0,
        symbols={"malsr": {"paths": [
            {"d": "M0 0 L10 0", "fill": "#000000", "stroke": "none", "stroke_width": 0.0},
        ]}},
    )


def test_emit_creates_layer_with_scaled_rotated_translated_symbol():
    svg = ET.Element("svg")
    groups = [{
        "artwork_key": "malsr", "x": 700.0, "y": 300.0,
        "dir_x": 1.0, "dir_y": 0.0, "target_width": 50.0,
    }]
    emit_approach_lights(svg, groups, _artwork())

    layer = svg.find("g")
    assert layer is not None
    assert layer.get("id") == "Approach Lights"
    sym = layer.find("g")
    # scale = target_width / native_width = 50/100 = 0.5; dir = (1,0).
    assert sym.get("transform") == "matrix(0.500000,0.000000,-0.000000,0.500000,700.0000,300.0000)"
    path = sym.find("path")
    assert path.get("d") == "M0 0 L10 0"
    assert path.get("fill") == "#000000"


def test_emit_skips_unknown_key_and_emits_nothing_for_empty():
    svg = ET.Element("svg")
    emit_approach_lights(svg, [], _artwork())
    assert svg.find("g") is None  # no layer when there are no lights

    svg2 = ET.Element("svg")
    emit_approach_lights(svg2, [{"artwork_key": "nope", "x": 0, "y": 0,
                                 "dir_x": 1.0, "dir_y": 0.0, "target_width": 50.0}],
                         _artwork())
    layer = svg2.find("g")
    assert layer is not None and layer.find("g") is None  # layer, no symbols
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/lukehogan/AOA-Code/faa-pro && .venv/bin/python -m pytest python/tests/test_render_lights.py -v`
Expected: FAIL with "cannot import name 'emit_approach_lights'".

- [ ] **Step 3: Add `"Approach Lights"` to `TARGET_LAYERS`**

In `python/render_svg_layers.py`, change the list at lines 34-46 so the entry sits just below the topmost text-token layer:

```python
TARGET_LAYERS = [
    "PDF Text Tokens",
    "Approach Lights",
    "Runway Labels (Text)",
    "Taxiway Labels (Text)",
    "Footprints",
    "Runways",
    "Runway Labels",
    "Taxiways",
    "Taxiway Labels",
    "Stars",
    "Other",
    "Metadata",
]
```

- [ ] **Step 4: Add the `emit_approach_lights` function**

Add near the top-level functions of `render_svg_layers.py` (e.g. just above `def main()`):

```python
def emit_approach_lights(svg, light_groups, artwork):
    """Inject approach-light symbols into a (new or existing) "Approach Lights"
    SVG layer <g>. Each light is one <g> with a scale+rotate+translate matrix
    wrapping the symbol's <path>s.

    light_groups entries are {artwork_key, x, y, dir_x, dir_y, target_width} in
    PyMuPDF top-left (y-down) coords — applied directly, no flip (matching the
    per-polygon translate convention, NOT the y-flipped text-label convention).
    Does nothing when light_groups is empty.
    """
    if not light_groups:
        return
    # Find or create the layer (same pattern as the PDF Text Tokens layer).
    layer = None
    for child in svg:
        if child.get("id") == "Approach Lights":
            layer = child
            break
    if layer is None:
        layer = ET.SubElement(svg, "g", {
            "id": "Approach Lights",
            "inkscape:label": "Approach Lights",
            "inkscape:groupmode": "layer",
        })

    native_width = artwork.native_width
    for lg in light_groups:
        sym = artwork.symbols.get(lg["artwork_key"])
        if not sym:
            continue
        s = float(lg["target_width"]) / native_width
        dx = float(lg["dir_x"])
        dy = float(lg["dir_y"])
        x = float(lg["x"])
        y = float(lg["y"])
        transform = (
            f"matrix({s * dx:.6f},{s * dy:.6f},"
            f"{-s * dy:.6f},{s * dx:.6f},"
            f"{x:.4f},{y:.4f})"
        )
        g = ET.SubElement(layer, "g", {"transform": transform})
        for p in sym.get("paths", []):
            attrs = {"d": p["d"], "fill": p.get("fill") or "none"}
            stroke = p.get("stroke")
            if stroke and stroke != "none":
                attrs["stroke"] = stroke
                attrs["stroke-width"] = f"{float(p.get('stroke_width', 0.0)):.3f}"
            ET.SubElement(g, "path", attrs)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /Users/lukehogan/AOA-Code/faa-pro && .venv/bin/python -m pytest python/tests/test_render_lights.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Wire into `main()`**

In `render_svg_layers.py`: (a) after the `label_groups = payload.get("label_groups", [])` line (~152) add:

```python
    light_groups = payload.get("light_groups", [])
```

(b) After the per-layer emission loop ends (after line 334, the `ET.SubElement(g, "path", attrs)` inside the path-bucket loop) and **before** the `# Add the PDF Text Tokens layer last` block (~336), add:

```python
    # Inject approach-light artwork (new geometry, not from the PDF) into the
    # "Approach Lights" layer. Placed before PDF Text Tokens so tokens stay
    # topmost. No-op when there are no lights or no artwork library.
    if light_groups:
        from pipeline.light_artwork import load_light_artwork
        lights_json = Path(__file__).parent.parent / "artwork" / "lights" / "lights.json"
        if lights_json.exists():
            emit_approach_lights(svg, light_groups, load_light_artwork(lights_json))
        else:
            print(f"  WARN: light_groups present but {lights_json} missing")
```

- [ ] **Step 7: Re-run all tests (nothing regressed)**

Run: `cd /Users/lukehogan/AOA-Code/faa-pro && .venv/bin/python -m pytest python/tests/ -v`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add python/render_svg_layers.py python/tests/test_render_lights.py
git commit -m "feat: inject approach-light artwork into Approach Lights SVG layer"
```

---

## Task 7: Pipeline integration — step 8b + flags

**Files:**
- Modify: `python/classify_pipeline.py`

This task has no new unit test (the placement logic is already covered by Task 4, the loader by Task 2). It wires those together and is verified by Task 10's end-to-end run.

- [ ] **Step 1: Add imports** near the other `from pipeline...` imports (~104-114):

```python
from pipeline.approach_lights_data import load_approach_lights
from pipeline.approach_light_layout import compute_approach_light_placements
```

- [ ] **Step 2: Add CLI flags** in the argparse block (after `--footprint-threshold`, ~391):

```python
    ap.add_argument("--place-lights", action="store_true",
                    help="Place approach-lighting symbols (step 8b). Requires "
                         "--nasr-rwy-end and artwork/lights/lights.json.")
    ap.add_argument("--nasr-rwy-end", type=Path,
                    default=Path(__file__).parent.parent / "data" / "nasr_apt_rwy_end.csv",
                    help="NASR runway-end CSV (per-end approach-light codes).")
```

- [ ] **Step 3: Compute light_groups after step 8** — insert immediately after the `step3b.json` write (~644), before the records loop (~661):

```python
    # Step 8b: approach-light placement. Lights may overlap pavement, so
    # unlike runway labels there is NO collision-march — symbols sit at a
    # fixed gap off the pavement end. Gated by --place-lights.
    light_groups: list[dict] = []
    if args.place_lights:
        if not args.nasr_rwy_end or not args.nasr_rwy_end.exists():
            raise SystemExit(
                f"--place-lights requires runway-end data; not found: {args.nasr_rwy_end}"
            )
        als_by_end = load_approach_lights(args.nasr_rwy_end, airport_code)
        light_groups, step8b_diag = compute_approach_light_placements(
            all_polys, step3_diag, runway_indices, als_by_end,
        )
        args.out.with_suffix(".step8b.json").write_text(
            json.dumps(step8b_diag, indent=2, default=str)
        )
        print(f"  Approach lights placed: {len(light_groups)}")
```

- [ ] **Step 4: Add `light_groups` to the payload** — in the payload dict (~780-789) add the key:

```python
        "label_groups": label_groups,
        "light_groups": light_groups,
    }
```

- [ ] **Step 5: Verify the module imports and argparse parse cleanly**

Run: `cd /Users/lukehogan/AOA-Code/faa-pro && PYTHONPATH=./python .venv/bin/python python/classify_pipeline.py --help`
Expected: help text shows `--place-lights` and `--nasr-rwy-end`; exit 0.

- [ ] **Step 6: Commit**

```bash
git add python/classify_pipeline.py
git commit -m "feat: wire approach-light placement into pipeline as step 8b"
```

---

## Task 8: Shell wrapper pass-through

**Files:**
- Modify: `classify.sh`
- Modify: `classify-airports.sh`

- [ ] **Step 1: Inspect how `classify.sh` builds the `classify_pipeline.py` invocation** and how `PIPELINE_EXTRA` is used (the `--skip-hull` precedent).

Run: `cd /Users/lukehogan/AOA-Code/faa-pro && grep -n "PIPELINE_EXTRA\|classify_pipeline.py\|--skip-hull" classify.sh`
Expected: shows the env var and the python invocation line.

- [ ] **Step 2: Make `classify.sh` forward a `PLACE_LIGHTS` env var.** Add near the top (after other config), then append to the pipeline args. Concretely, where `PIPELINE_EXTRA` is assembled, add:

```bash
# Optional: place approach-lighting symbols (step 8b). Enable by exporting
# PLACE_LIGHTS=1 (classify-airports.sh sets this from an interactive prompt).
if [[ "${PLACE_LIGHTS:-0}" == "1" ]]; then
    PIPELINE_EXTRA="${PIPELINE_EXTRA:-} --place-lights"
fi
```

(If `classify.sh` does not use a `PIPELINE_EXTRA` variable, instead append `--place-lights` directly to the `"$PYTHON" "$PROJECT/python/classify_pipeline.py" ...` argument list under the same `if` guard.)

- [ ] **Step 3: Add an interactive prompt to `classify-airports.sh`** after the existing "Proceed? [y/N]" confirmation (~104-109), before the processing loop:

```bash
read -e -p "Place approach lights? [y/N] " place_lights_in
case "$(echo "$place_lights_in" | tr '[:upper:]' '[:lower:]')" in
    y|yes) export PLACE_LIGHTS=1 ;;
    *)     export PLACE_LIGHTS=0 ;;
esac
```

- [ ] **Step 4: Verify scripts still parse** (bash syntax check):

Run: `cd /Users/lukehogan/AOA-Code/faa-pro && bash -n classify.sh && bash -n classify-airports.sh && echo OK`
Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add classify.sh classify-airports.sh
git commit -m "feat: add --place-lights pass-through to classify wrappers"
```

---

## Task 9: Offline artwork build script (Illustrator SVG → lights.json)

**Files:**
- Create: `python/pipeline/build_light_artwork.py`
- Create: `python/tests/fixtures/artboards/malsr.svg`
- Create: `python/tests/fixtures/artboards/odals.svg`
- Test: `python/tests/test_build_light_artwork.py`

The build script flattens each per-artboard SVG (Illustrator "Export for Screens" output) to absolute paths via `svgelements`, normalizes to the canonical frame (attach point → origin; forward, drawn as artboard "up", → +X; lateral → ±Y), asserts all artboards share one width, and writes `lights.json`. The attach point is the **bottom-center** of the artboard (per the authoring convention).

Canonical transform applied to each absolute point `(px, py)` in artboard SVG coords (origin top-left, y-down, artboard `W × H`):
1. Translate to attach point at origin: `(px - W/2, py - H)`.
2. Rotate so artboard "up" `(0, -1)` maps to canonical `+X` via `R = [[0, -1], [1, 0]]`: `(X, Y) = (-(py - H), (px - W/2))`.

So `(px, py) -> (X, Y) = (H - py, px - W/2)`. `native_width = W`.

- [ ] **Step 1: Create fixture artboards.** `python/tests/fixtures/artboards/malsr.svg` (100×120, a vertical centerline from bottom-center upward):

```xml
<svg xmlns="http://www.w3.org/2000/svg" width="100" height="120" viewBox="0 0 100 120">
  <path d="M50 120 L50 20" stroke="#000000" stroke-width="2" fill="none"/>
</svg>
```

`python/tests/fixtures/artboards/odals.svg` (same 100×120 canvas, shorter line):

```xml
<svg xmlns="http://www.w3.org/2000/svg" width="100" height="120" viewBox="0 0 100 120">
  <path d="M50 120 L50 70" stroke="#000000" stroke-width="2" fill="none"/>
</svg>
```

- [ ] **Step 2: Write the failing test**

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /Users/lukehogan/AOA-Code/faa-pro && .venv/bin/python -m pytest python/tests/test_build_light_artwork.py -v`
Expected: FAIL with "No module named 'pipeline.build_light_artwork'".

- [ ] **Step 4: Implement the build script**

Create `python/pipeline/build_light_artwork.py`:

```python
"""build_light_artwork.py — OFFLINE build: per-artboard SVG exports -> the
normalized artwork JSON the renderer consumes (artwork/lights/lights.json).

Run after exporting artboards from Illustrator ("Export for Screens", one SVG
per artboard, named by the artwork key). Flattens each SVG to absolute paths
via svgelements, normalizes to the canonical frame (attach point = bottom-
center -> origin; artboard "up" -> +X), asserts all artboards share one width
(the runway-width reference that makes per-document uniform scaling valid),
and writes lights.json.

Usage:
    python pipeline/build_light_artwork.py artwork/lights/ artwork/lights/lights.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from svgelements import SVG, Path as SvgPath, Shape


def _symbol_from_svg(svg_path: Path) -> tuple[float, list[dict]]:
    """Return (artboard_width, paths) with paths normalized to the canonical
    frame. canonical (X, Y) = (H - py, px - W/2)."""
    svg = SVG.parse(str(svg_path))
    W = float(svg.width)
    H = float(svg.height)
    paths: list[dict] = []
    for el in svg.elements():
        if not isinstance(el, Shape):
            continue
        seg = SvgPath(el)               # absolute, transforms applied
        d_abs = seg.d()
        if not d_abs.strip():
            continue
        # Re-emit with canonical-frame points. svgelements lets us map points;
        # simplest robust path: rebuild from the segment endpoints.
        canon = SvgPath()
        for sub in seg.as_subpaths():
            pts = [(seg_pt.end.x, seg_pt.end.y) for seg_pt in sub]
            first = sub[0].start
            allpts = [(first.x, first.y)] + pts
            canon_pts = [(H - py, px - W / 2.0) for (px, py) in allpts]
            canon += "M {} {}".format(*canon_pts[0])
            for (X, Y) in canon_pts[1:]:
                canon += " L {} {}".format(X, Y)
        stroke = el.stroke
        fill = el.fill
        paths.append({
            "d": canon.d(),
            "fill": str(fill) if fill and fill.value is not None else "none",
            "stroke": str(stroke) if stroke and stroke.value is not None else "none",
            "stroke_width": float(getattr(el, "stroke_width", 0.0) or 0.0),
        })
    return W, paths


def build_light_artwork(artboards_dir: Path, out_json: Path) -> None:
    """Build lights.json from a directory of per-artboard SVGs."""
    artboards_dir = Path(artboards_dir)
    svgs = sorted(p for p in artboards_dir.glob("*.svg"))
    if not svgs:
        raise ValueError(f"No .svg artboards in {artboards_dir}")
    symbols: dict[str, dict] = {}
    widths: list[float] = []
    for svg_path in svgs:
        W, paths = _symbol_from_svg(svg_path)
        widths.append(W)
        symbols[svg_path.stem] = {"paths": paths}
    # Equal-width invariant: this is what makes one doc_scale correct for all.
    if max(widths) - min(widths) > 1e-6:
        raise ValueError(
            f"Artboards must share one width; got {dict(zip([p.stem for p in svgs], widths))}"
        )
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(
        {"native_width": widths[0], "symbols": symbols}, indent=2,
    ))


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: build_light_artwork.py <artboards_dir> <out.json>")
    build_light_artwork(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /Users/lukehogan/AOA-Code/faa-pro && .venv/bin/python -m pytest python/tests/test_build_light_artwork.py -v`
Expected: PASS. If `svgelements`'s `as_subpaths()`/point API differs in the installed version, adjust `_symbol_from_svg` to the installed API (the test pins the required canonical output, so iterate against it). Per CLAUDE.md: if the svgelements API fails twice, web-search the installed svgelements version's API before a third attempt.

- [ ] **Step 6: Commit**

```bash
git add python/pipeline/build_light_artwork.py python/tests/test_build_light_artwork.py python/tests/fixtures/artboards/
git commit -m "feat: offline build of normalized approach-light artwork library"
```

---

## Task 10: End-to-end verification + calibration (manual)

**Files:**
- Create (by the user): `artwork/lights/*.svg` (the real artboards) and the built `artwork/lights/lights.json`.
- Obtain: `data/nasr_apt_rwy_end.csv` (real NASR runway-end file).
- Modify (after calibration): `DEFAULT_K` / `DEFAULT_G` in `approach_light_layout.py`.

- [ ] **Step 1: [VERIFY] real NASR headers.** Download a current `APT_RWY_END.csv` from the FAA 28-day subscription. Confirm the airport-id, runway-end-id, and approach-light column headers; trim `_AIRPORT_COLS` / `_END_ID_COLS` / `_ALS_COLS` in `approach_lights_data.py` to the real names. Place the file at `data/nasr_apt_rwy_end.csv`.

- [ ] **Step 2: Build the artwork library** from the real artboards:

Run: `cd /Users/lukehogan/AOA-Code/faa-pro && PYTHONPATH=./python .venv/bin/python python/pipeline/build_light_artwork.py artwork/lights/ artwork/lights/lights.json`
Expected: `lights.json` written; no width-mismatch error.

- [ ] **Step 3: Run the full pipeline with lights on a known airport** (e.g. one with MALSR/ALSF2). Use the same invocation `classify.sh` uses, adding `--place-lights`:

Run: `cd /Users/lukehogan/AOA-Code/faa-pro && PLACE_LIGHTS=1 bash classify.sh /path/to/<airport>-faa.pdf`
Expected: console prints `Approach lights placed: N` (N > 0); a `*.step8b.json` diagnostic is written.

- [ ] **Step 4: Inspect the rendered SVG.** Open the output SVG; confirm an "Approach Lights" layer exists with symbols at the correct runway ends, oriented along each runway, on the centerline, a visible gap off the pavement.

- [ ] **Step 5: Calibrate.** Render DEN (zoomed-out) and a small field. Adjust `DEFAULT_K` (overall size) and `DEFAULT_G` (gap) in `approach_light_layout.py` until the proportion looks right and is consistent across both. Confirm all lights on one diagram are the same scale.

- [ ] **Step 6: Commit calibration + the data/artwork artifacts** (per the repo's data conventions for committing `data/` and `artwork/`):

```bash
git add python/pipeline/approach_light_layout.py python/pipeline/approach_lights_data.py
git commit -m "chore: calibrate approach-light scale/gap and pin NASR headers"
```

---

## Self-Review

**Spec coverage** — every spec section maps to a task:
- Data (APT_RWY_END / field A49) → Tasks 2, 10·1.
- End→designator bridge (step-3 pairing, reciprocal, skip-if-none) → Tasks 3, 4.
- `approach_light_layout` no-march + doc-uniform scale + gap → Task 4.
- Artwork: master `.ai` → named artboards → per-artboard SVG → normalized library → Tasks 5 (runtime), 9 (build), 10·2.
- FAA code→artwork-key table → Task 1.
- Step 8b + `--place-lights` → Task 7.
- Renderer "Approach Lights" layer + matrix injection (no y-flip) → Task 6.
- Thin shell wrappers → Task 8.
- `k`/`g` calibration, normalizer swappable, `[VERIFY]` items → Tasks 4 (`_document_width_reference`), 10.

**Placeholder scan** — no "TBD/handle edge cases/similar to" steps; every code step shows complete code. The two `[VERIFY]` markers (NASR headers, artwork-key groupings) are explicit confirmation tasks in Task 10, not unfinished code — the loader resolves headers from an alias list and errors loudly if absent, so the code is runnable before verification.

**Type/name consistency** — `light_groups` schema `{artwork_key, x, y, dir_x, dir_y, target_width}` is identical across Tasks 4 (producer), 6 (consumer), 7 (payload). `load_approach_lights(csv_path, airport_code) -> dict[str,str]`, `als_code_to_artwork_key(code) -> str|None`, `compute_approach_light_placements(all_polys, step3_diag, runway_indices, als_by_end, *, k, g)`, `LightArtwork(native_width, symbols)`, `emit_approach_lights(svg, light_groups, artwork)`, `build_light_artwork(artboards_dir, out_json)` — names/signatures match between definition and call sites.
