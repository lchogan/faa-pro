# Approach Lighting Symbols — Design Spec

- **Date:** 2026-05-29
- **Status:** Approved for implementation planning (no code written yet)
- **Scope (v1):** Centerline approach lighting systems only

## Purpose

Render an approach-lighting symbol at each runway end that has an approach
lighting system (ALS), on the classified FAA airport-diagram output. Each
symbol is:

1. **Sourced** from per-runway-end FAA data (which ALS, if any, exists).
2. **Oriented** to align with the runway centerline.
3. **Scaled** proportional to the runway width *as depicted on the page* (not
   the real-world width), so a zoomed-out airport like DEN gets smaller lights
   than a zoomed-in small field — and, critically, **all lights on one diagram
   share a single scale** (see "Scale model").
4. **Positioned** centered on the runway centerline, a visible gap off the end
   of the pavement.

Unlike runway-number labels, approach lights **are allowed to overlap** runways
and taxiways, so there is **no collision-march**. Lights may also overlap
runway-number labels — the two are separate, individually toggleable layers and
only one is shown at a time.

## Out of scope (v1)

- **REIL** (paired threshold flashers — corner placement, not centerline).
- **PAPI / VASI** (side-mounted glideslope indicators — lateral placement).
- **Touchdown-zone lights** (on-runway).
- Automatic 28-day NASR download automation (the runway-end file is dropped in
  manually, mirroring the existing `data/nasr_apt_rwy.csv` workflow).

These each need a different placement model or data handling and are deferred.

---

## Key insight: the "which end is 16 vs 34?" problem is already solved

NASR keys approach lighting to the magnetic end-designator ("16" vs "34"), but
a detected PDF runway is a single polygon with two *geometric* ends, and the
page is not reliably north-up. Naïvely mapping designator → geometric end would
require latitude/longitude plus page-orientation detection.

We avoid all of that. The pipeline's step 3 (`_match_runway_labels` in
`python/classify_pipeline.py`) **already pairs each geometric runway end
(`end_sign ∈ {-1,+1}`) with the painted runway-number token** ("16", "34L", …),
stored in `step3_diag`. That pairing is the bridge:

- matched token at an end → its base designator → look up the ALS code for
  `(this airport, that designator)` in the runway-end data → place that symbol
  at that geometric end.
- Designators are reciprocal (differ by 18), so **one** matched number
  disambiguates **both** ends.
- If **neither** end matched a number → skip that runway, emit a diagnostic.

This reuses machinery the pipeline already trusts and needs no geographic data.

---

## Scale model — one scale per diagram

A single `doc_scale` for the whole page, computed once from the widest detected
runway and applied to **every** symbol:

```
doc_scale = k · (2 · max_half_wid) / artboard_width
```

- `max_half_wid` = the largest `half_wid` (depicted half-width, in PDF points)
  across all detected runways on the page. `half_wid` is the 6th element of
  `_runway_extents(...)` in `python/pipeline/taxi_detection.py`.
- `artboard_width` = the shared native width of the artwork artboards (the
  runway-width reference the symbols were drawn against; see "Artwork").
- `k` = a single tuning constant for overall light size.

All lights hold the same size relationship to one another — their *relative*
differences (e.g. ALSF wider than ODALS) come only from how they were drawn; the
*absolute* size is set once by the widest runway. A large symbol on a small
runway is the intended, accepted result.

**Gap** (off the pavement end) is also document-uniform and proportional to the
lighting scale:

```
gap = g · k · (2 · max_half_wid)
```

`k` and `g` are the two tuning knobs; calibrate on DEN + a small field once
artwork exists.

**Normalizer is swappable.** "Largest runway width" lives behind one named
function (e.g. `_document_width_reference(runway_extents)`), so it can be
changed to median/percentile during tuning if a fragmented or mis-detected
runway ever inflates the max. Largest is the default.

---

## Artwork — one master `.ai`, named artboards → per-artboard SVG

**Authoring (manual, by the user):**
- One master `.ai` with **one named artboard per archetype**: `alsf1`,
  `alsf2`, `malsr`, `malsf`, `mals`, `sals`, `odals`.
- **All artboards share identical dimensions.** The artboard width represents
  **one runway-width reference**. Each symbol is drawn at its true size
  relative to that width (narrower systems like ODALS leave more whitespace).
- **Attach point = bottom-center**; symbol points "up" (away from the runway);
  centerline runs vertically through the artboard center.
- Export via Illustrator **File → Export → Export for Screens**, producing one
  SVG per artboard, *named by the artboard*, written to `artwork/lights/`.

`artwork/lights/<key>.svg` is the checked-in source of truth (human-inspectable,
diffable). The `.ai` is the editable master.

**FAA code → artwork key table** (the only hand-maintained mapping; several
codes share a symbol):

| Artwork key | FAA codes |
|---|---|
| `alsf1` | `ALSF1` |
| `alsf2` | `ALSF2` |
| `malsr` | `MALSR`, `SSALR` |
| `malsf` | `MALSF`, `SSALF` |
| `mals`  | `MALS`, `SSALS` |
| `sals`  | `SALS`, `SALSF` |
| `odals` | `ODALS` |
| *(none)* | `NONE` / blank / unmapped → no symbol |

`SSALR`/`MALSR`, `SSALF`/`MALSF`, `SSALS`/`MALS` are grouped because they are
visually equivalent at chart-symbol resolution. [VERIFY] grouping against the
actual chart symbology and the full code list in a downloaded NASR file before
finalizing; unmapped codes degrade safely to "no symbol" + diagnostic.

---

## Components

### 1. Runway-end data loader (new)
- **What:** read the NASR runway-end table, return
  `{(ARPT_ID, end_designator): als_code}`.
- **How:** drop `data/nasr_apt_rwy_end.csv` (NASR APT_RWY_END, same free 28-day
  subscription as the existing APT_RWY file). Loader extracts the airport id,
  the runway-end designator, and the approach-light-system code (NASR field
  **A49 "APPROACH LIGHT SYSTEM"**).
- **Depends on:** the CSV file existing in `data/`.
- **[VERIFY]:** exact column header for the ALS code in `APT_RWY_END.csv` (the
  workflow confirmed field A49 by fixed position in the legacy `apt.txt` layout;
  confirm the CSV column name against a current downloaded file).

### 2. End → designator bridge (extend existing)
- **What:** for each detected runway end, get the ALS code.
- **How:** read `step3_diag` (already produced). Per matched end: token →
  base designator → loader lookup keyed by `(airport, designator)`. Fill the
  reciprocal end's designator if only one end matched. Skip + diagnostic if
  neither matched.
- **Depends on:** component 1; `step3_diag` from `_match_runway_labels`.

### 3. `python/pipeline/approach_light_layout.py` (new module)
- **What:** compute one placement per (runway end, ALS) →
  `light_groups: [{artwork_key, x, y, rotate_deg, scale, runway_idx, end_sign}]`
  plus diagnostics.
- **How:** mirrors `runway_label_layout.py` but **drops the march loop**.
  - `doc_scale` computed once from `max_half_wid` over all runways.
  - Per end: `cx, cy, ux, uy, half_len, _ = _runway_extents(...)`;
    `out = end_sign · (ux, uy)`;
    attach target `(x, y) = (cx, cy) + (half_len + gap) · out`;
    `rotate` aligns artwork's local "up" axis to `out`;
    `scale = doc_scale`.
  - Output positions are in **PyMuPDF top-left (y-down)** — the same frame as
    the per-polygon `translate`, **not** the AI-y-up frame the text
    label-groups use. No coordinate flip at render time.
- **Depends on:** `_runway_extents` (from `taxi_detection.py`); components 1–2.
- **Public:** `compute_approach_light_placements(all_polys, step3_diag,
  als_by_end, runway_indices, k, g) -> (light_groups, diagnostics)`.

### 4. Artwork loader `python/pipeline/light_artwork.py` (new)
- **What:** load `artwork/lights/*.svg` → `{key: (paths, native_width)}`,
  cached.
- **How:** parse each SVG's `<path d="…">` and viewBox. Normalize so the attach
  point is at the local origin and the symbol axis is the local "up". **Assert
  all artboards share the same width**, failing loudly if not (this is what
  guarantees the uniform-scale relationship).
- **Depends on:** the exported SVG files. No PyMuPDF / `.ai` parsing at runtime.

### 5. Pipeline hook — step 8b (extend `classify_pipeline.py`)
- **What:** after step 8 (runway labels), if enabled, compute light placements
  and attach `light_groups` to the predictions payload.
- **How:** gated by a `--place-lights` flag. Call
  `compute_approach_light_placements(...)`; add `payload["light_groups"]`
  alongside the existing `payload["label_groups"]` (built around
  `classify_pipeline.py:787`). The `compute_runway_label_translations` call at
  `classify_pipeline.py:639` is the structural template.
- **Depends on:** components 1–4.

### 6. Renderer injection (extend `render_svg_layers.py`)
- **What:** emit the light symbols into a new SVG layer.
- **How:**
  - Add `"Approach Lights"` to `TARGET_LAYERS` (currently
    `render_svg_layers.py:34`), at the top of the z-order (overlay above
    pavement). Separate, individually toggleable layer.
  - Read `payload.get("light_groups", [])` (mirrors the `label_groups` read at
    `render_svg_layers.py:152`).
  - For each light group, look up its artwork paths and emit `<g><path …/></g>`
    with a `matrix(...)` transform composing **scale · rotate · translate**.
    Extend the existing text-matrix pattern at `render_svg_layers.py:308–312`
    (`matrix(dx, dy, -dy, dx, e, f)`) with scale:
    `matrix(s·dx, s·dy, -s·dy, s·dx, e, f)`, where `(dx, dy)` is the runway
    direction unit vector and `(e, f)` is the attach target in y-down SVG
    coordinates (no flip, since light_groups are already y-down).
  - **[VERIFY]:** the exact local-axis convention vs the matrix derivation is
    an implementation step covered by a unit test (a known symbol placed at a
    known end with a known angle renders to expected coordinates).
- **Depends on:** components 4–5.

### 7. Shell wrappers (thin pass-through)
- **What:** let the operator turn light placement on.
- **How:** `classify.sh` / `classify-airports.sh` pass `--place-lights` through
  to `classify_pipeline.py` (flag or interactive prompt). **No lighting logic in
  bash** — the wrappers stay thin orchestration. This corrects the original
  framing ("add a function to `classify-airports.sh`"): the function lives in
  the Python pipeline.

---

## Data flow

```
data/nasr_apt_rwy_end.csv ──► (1) loader ──► als_by_end {(airport,desig):code}
                                                    │
classify_pipeline step 3 step3_diag ──► (2) bridge ─┤
                                                    ▼
all_polys + step3_diag ──► (3) approach_light_layout.compute_approach_light_placements
                                                    │  (uses _runway_extents; doc_scale; no march)
                                                    ▼
                                          light_groups [{artwork_key,x,y,rotate_deg,scale,…}]
                                                    │  (step 8b, --place-lights)
                                                    ▼
                              payload["light_groups"]  ──► <airport>_predictions.json
                                                    │
artwork/lights/*.svg ──► (4) light_artwork loader ──┤
                                                    ▼
                       (6) render_svg_layers ──► "Approach Lights" <g> with matrix() transforms ──► SVG
```

## Error handling

- **No `nasr_apt_rwy_end.csv`:** with `--place-lights`, fail loudly with a clear
  message; without the flag, no effect.
- **Airport absent from runway-end data:** no lights, diagnostic logged.
- **ALS code `NONE`/blank/unmapped:** no symbol for that end (unmapped also logs
  a diagnostic so new codes surface).
- **Neither runway end matched a number:** skip that runway, diagnostic.
- **Artboards differ in width:** artwork loader raises at load time (protects the
  uniform-scale guarantee).
- **Missing artwork file for a mapped key:** skip that symbol, diagnostic.

## Testing

- **`approach_light_layout` unit tests:** placement at a known end with synthetic
  runway geometry — correct attach point, rotation, uniform `doc_scale` across
  multiple runways of different widths, correct `gap`, no-march (a symbol that
  overlaps a crossing runway stays put), reciprocal-designator inference,
  skip-when-unmatched.
- **`light_artwork` unit tests:** path/viewBox parse, attach-point
  normalization, the equal-width assertion fires on mismatch.
- **Bridge tests:** token → designator → ALS code, including `L/R/C` suffixes
  and compass-named runways (e.g. `NE/SW`).
- **Render integration:** a fixture airport with a known ALS renders a
  matrix-transformed `<path>` at expected coordinates in the
  "Approach Lights" layer.
- **Visual calibration (manual):** render DEN + a small field, tune `k` and `g`.

## Open tuning items (not blockers)

- `k` (overall light size) and `g` (gap proportion) — visual calibration.
- Whether "largest runway width" needs to become median/percentile (swappable).
- Final artwork-key grouping vs chart symbology ([VERIFY] above).
- Exact NASR `APT_RWY_END.csv` ALS column header ([VERIFY] above).
