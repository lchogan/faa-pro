# faa-pro — FAA Airport Diagram Classifier

Classifies vector objects in FAA airport diagram PDFs into semantic
layers: **Taxiways, Taxiway Labels, Runways, Runway Labels, Footprints,
Lights, Stars, Other**. Output is a layered SVG that opens in Adobe
Illustrator with native AI layers (save-as `.ai` if you want the
extension).

## Quick start

```bash
bash classify.sh /path/to/<airport>-faa.pdf
```

Produces `<airport>-diagram.svg` next to the source PDF. Open in
Illustrator → File > Open. Each top-level group becomes a native
Illustrator layer.

End-to-end on ORD (4331 polygons): **~7 seconds**, ~5 of which is the
ML model.

## The pipeline

`classify.sh` orchestrates three Python steps. There is no Illustrator
scripting in the production path.

```
<airport>-faa.pdf
        │
        │  Step 1   PyMuPDF
        ▼
<airport>_paths.csv             4331 polygons w/ geometric features
        │
        │  Step 2   classify_pipeline.py — the brain (6 substeps)
        ▼
<airport>_predictions.json      one record per polygon, AI y-up bbox
        │
        │  Step 3   render_svg_layers.py
        ▼
<airport>-diagram.svg           ten <g> layers, Inkscape-tagged
```

### `classify_pipeline.py` — the 6 substeps

Polygons claimed in earlier substeps are **removed from the pool** seen
by every later substep. ML never decides Taxiways, Taxiway Labels, or
Runways.

1. **Taxiways (rule-based).** Filled polygons whose RGB is gray
   (~#cfcfcf with leeway: avg 175–235, channel spread ≤ 20). This is
   the *only* source for Taxiways.

2. **Taxiway Labels (rule-based).** PDF text tokens matching
   `^([A-Z][A-Z]?[0-9]{0,2}|[0-9])$` (e.g. `C`, `C1`, `A11`, `3`)
   whose bbox **touches** a Taxiways surface from step 1. For each
   qualifying token, the K = `len(token)` nearest unclaimed near-black
   filled polygons (the actual glyph polygons) are claimed as
   Taxiway Labels.

3. **Runways (rule-based, NASR-driven).** Look up the airport in
   `data/nasr_apt_rwy.csv`, count its non-helipad runways → `N`. From
   the unclaimed pool, take the `N` largest polygons by polygon area
   (shoelace, robust to rotation) that are either filled near-black
   (paved) or stroked-only (grass strip outlines). Two safeguards:
   a bbox-area smell test (≤ 50% of page) rejects the chart frame,
   and a PCA-derived aspect-ratio sanity check (≥ 20% of the
   airport's smallest NASR runway aspect, floor 4:1) rejects label
   boxes whose polygon area might rival a small runway. Nested clip
   groups exposed by `get_drawings(extended=True)` are also
   candidates; when a clip wins, the largest polygon fully contained
   in its scissor is claimed (this is how F45's grass strip — drawn
   only as a clipped hatch pattern — gets picked up).

4. **ML — Footprints, Lights, Stars, Other.** The trained LightGBM
   model (`python/runs/v24/model.lgb`) runs on every polygon *not*
   claimed by step 1, 2, or 3. The Taxiways / Taxiway Labels /
   Runways / Runway Labels classes are masked out of the probability
   matrix so an unclaimed polygon can't fall back into them. The
   existing `pdf_char_override.apply_pdf_char_override` runs on the
   filtered set, preserving the Lights-by-stroke heuristic that
   detects ~1000 light-fixture stripes.

5. **Runway Labels (rule-based, post-ML).** For each rule-claimed
   Runway polygon, compute its principal axis via PCA. From each
   endpoint, search outward through widths `(1, 10, 25, 50, 100)pt`
   for any runway-pattern (`^(0?[1-9]|[12][0-9]|3[0-6])[LRC]?$`)
   token whose centroid sits in that band. The first width that
   yields a candidate wins; ties are broken by closeness to the
   endpoint. The chosen token claims K nearest unclaimed near-black
   filled polygons → **Runway Labels** (whatever ML had assigned them
   gets overridden). Tokens are reserved across runway ends, so a
   `9L` matched at one runway can't be re-used at another.

6. **Concave-hull rejection (post-everything).** Build a concave hull
   (`shapely.concave_hull(ratio=0.0)`, no buffer) over the rule-claimed
   Runways + Taxiways' anchor points. Any polygon whose centroid sits
   outside the hull is demoted to **Other**. Runways, Taxiways, Runway
   Labels, and Taxiway Labels are exempt — they're rule-trusted, and
   labels can legitimately sit at chart edges. The bulk of the work
   here is collapsing ML false-positive Lights and Footprints scattered
   across legend areas, scale bars, and surrounding text.

The PDF Text Tokens debug layer is always emitted: every word in the
PDF text stream as a magenta 4pt text frame at its bbox center.
Useful for spot-checking why a token did or didn't qualify.

## Layout

```
faa-pro/
├── classify.sh                          # entry point
├── README.md                            # this file
├── python/
│   ├── classify_pipeline.py             # 6-step orchestrator (Step 2)
│   ├── extract_paths_fitz.py            # PyMuPDF path extraction (Step 1)
│   ├── render_svg_layers.py             # SVG export (Step 3)
│   ├── chart_scene.py                   # PDF → polygons + clips + tokens (single source of truth)
│   ├── taxi_detection.py                # rule-based taxi (substeps 1+2)
│   ├── runway_detection.py              # rule-based runway (substep 3, NASR-driven)
│   ├── hull_filter.py                   # concave-hull rejection (substep 6)
│   ├── pdf_char_override.py             # Lights heuristic + axis helpers
│   ├── relational.py                    # ML feature engineering
│   ├── load.py                          # CSV schema + LABELS tuple
│   ├── predict_one.py                   # PDF text helper (kept for imports)
│   ├── extract_pdf_text.py              # NASR + text extraction
│   ├── runs/v24/model.lgb               # trained LightGBM
│   ├── render_pdf_layers.py             # PDF/OCG renderer (alternate, not used)
│   ├── render_char_layers_charbox.py    # taxi-only debug SVG renderer
│   └── char_training_legacy/            # old char-classifier code, kept for reuse
├── data/
│   ├── nasr_apt_rwy.csv                 # FAA NASR runway designations
│   └── char_training_legacy/            # old char-corpus training data
├── ImportPredictedLayers.jsx            # legacy JSX renderer, no longer invoked
├── ExportClassifiedPaths.jsx            # used during retraining (manual labeling)
├── PrepareForLabeling.jsx               # used during retraining (scaffold creation)
├── AddTargetLayers.jsx, AddTargetLayersBatch.jsx
└── _deprecated/                         # historical debug scripts + experiment outputs
```

## Retraining

The current LightGBM at `python/runs/v24/model.lgb` is trained on
~20–30 manually-labeled airports. To retrain see `python/README.md`.
Important rule: **don't modify the geometry** of training files (no
Pathfinder, no scale/rotation, no fill changes), or the model's
geometric features won't match what it sees at inference time.

## Architecture notes

- **Why SVG, not PDF.** A PDF-with-OCGs renderer is in
  `python/render_pdf_layers.py` and produces a structurally correct
  PDF (intent=View+Design, OCProperties.D.Order set, etc.) but
  Illustrator dumps everything into Layer 1. AI requires
  `/PieceInfo/Private/AIPrivateData1`–`16` (16 binary blobs of
  proprietary undocumented Illustrator serialization) to map OCGs to
  native layers. The SVG importer doesn't need that, so SVG is the
  practical path.
- **Why ML can't decide Taxiways, Taxi Labels, or Runways.** The
  rule-based detection is more reliable: gray fill is unambiguous,
  the K-nearest token-driven match is essentially perfect on
  diagrams where labels sit on pavement, and NASR tells us exactly
  how many runways an airport has so picking the N largest polygons
  is more robust than ML when the chart's runway depiction varies
  (paved black-fill, grass strip stroked outline, nested clip group
  with hatch pattern only). Letting ML override these would only
  introduce errors.
- **Why nested clip groups are first-class candidates in step 3.**
  Some FAA charts (F45 is the canonical example) draw a grass-strip
  runway as a clipped hatch pattern with no visible outline polygon.
  The simple-rectangle outline you see in Illustrator is the
  clip-group's clipping shape, which `page.get_drawings()` hides by
  default. Switching to `get_drawings(extended=True)` exposes
  `clip`-typed entries; `chart_scene.py` carries them alongside
  regular polygons, and `runway_detection.py` ranks them as
  candidates. When a clip wins, the largest polygon fully contained
  in its scissor is claimed.
- **Why an aspect-ratio sanity check.** A label-box rectangle on a
  small chart can have polygon area comparable to a 1850ft turf
  strip. NASR's per-airport minimum runway aspect (length/width)
  gives us a per-chart threshold: candidates must be at least 20% as
  elongated as the most square-ish real runway, floor 4:1. PCA on
  polygon points is used so rotated rectangles don't get punished
  by their square bboxes.
- **Why centerline-based runway-label matching is a thin band, not a
  bbox-touch test.** Runway designators on FAA charts often sit at
  the threshold *off* the runway pavement. A bbox-touch test against
  the Runway polygon misses them; the principal-axis line extended
  through the polygon reliably passes near the threshold marking.

## Known limitations

- The PDF Text Tokens debug layer adds ~700 text frames per chart.
  Toggle it off in Illustrator if it gets in the way.
- The remaining ML class — `Footprints` vs `Other` — still has the
  same false-positive issues as before (filled circles, arrowheads,
  letter polygons leaking into Footprints). The plan to refactor
  this with momepy morphology features and the combined 30+125-file
  training corpus is captured in memory but not yet implemented.

## Pipeline status (six-step plan)

The user's six-step plan for the rebuilt pipeline:

1. **Taxiways → gray fill.** Done (pre-rebuild rule).
2. **Runways → deterministic.** Done — NASR-driven top-N rule with
   nested clip-group support and aspect-ratio sanity check.
   Validated on ARB, APF, ELM, F45.
3. **Taxiway and runway labels.** Done — taxi labels via
   step-2 K-nearest, runway labels via centerline-token search
   (now consumes step 3's rule-claimed runways).
4. **Concave hull rejection.** Done — `python/hull_filter.py` runs as
   pipeline substep 6. `shapely.concave_hull(ratio=0.0)` over rule-
   claimed Runways + Taxiways' anchor points, no buffer; centroid-in-
   hull test demotes anything outside to Other; Runways, Taxiways,
   Runway Labels, and Taxiway Labels are exempt.
5. **Footprint ML refactor.** Not started. Train footprint binary
   on the union of the 30 clean files plus the 125 NASR-matched
   airports from the legacy 160-file corpus (international airports
   excluded — sourced from OSM, different stylization). Plan: use
   momepy morphology features + relational signals; keep stroked
   items in this pass for context.
6. **Stroked → Other.** Not started. Final sweep moves stroked
   items to Other, with the Runways layer exempt (grass strips
   stay).

## Commit history (rebuild)

The repo history starts at `91ab78c rebuild in progress`. The
rebuild added the rule-based taxi pipeline, the centerline-token
runway-label search, and the SVG renderer. The character-classifier
code from before the rebuild is preserved under
`python/char_training_legacy/` and `data/char_training_legacy/` for
potential future reuse.
