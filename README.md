# faa-pro — FAA Airport Diagram Classifier

Classifies vector objects in FAA airport diagram PDFs into semantic
layers: **Taxiways, Taxiway Labels, Runways, Runway Labels, Footprints,
Stars, Other**. Output is a layered SVG, plus — for single-airport
runs — a layer-organized `.ai` opened in Illustrator ready for review.

## Quick start

Single airport (default — opens in Illustrator when finished):

```bash
bash classify.sh /path/to/<airport>-faa.pdf
```

Produces `<airport>-diagram.svg` next to the source PDF, then opens
it in Illustrator, saves as `<airport>-diagram.ai`, reorganizes the
layer panel via `PrepareForInspection.jsx`, and leaves the document
open for review (see [Layer organization](#layer-organization-step-4)
below).

Batch (multiple airports — Illustrator step is skipped):

```bash
bash classify.sh airports/atl-faa.pdf airports/bna-faa.pdf airports/ord-faa.pdf
```

End-to-end on ORD (4331 polygons): **~7 seconds**, ~5 of which is the
ML model. Step 4 (Illustrator open) adds a few seconds on top when it
runs.

### Auto-open behavior

`classify.sh` decides whether to open the result in Illustrator by
counting positional args:

- **1 PDF arg** → opens in Illustrator after rendering (default).
- **>1 PDF arg** → skips the Illustrator step so you don't end up with
  one window per airport.

Override with the `OPEN_AFTER_CLASSIFY` env var when you need different
behavior:

```bash
# Single airport, but don't open in Illustrator
OPEN_AFTER_CLASSIFY=0 bash classify.sh ord-faa.pdf

# Batch, but force-open every airport (you'll get N windows)
OPEN_AFTER_CLASSIFY=1 bash classify.sh atl-faa.pdf bna-faa.pdf
```

The Illustrator step requires Adobe Illustrator installed locally and
accessible to AppleScript (`osascript`). On a machine without
Illustrator, run with `OPEN_AFTER_CLASSIFY=0` and open the SVG by hand
when convenient.

## The pipeline

`classify.sh` orchestrates three Python steps plus an optional
Illustrator step. There is no Illustrator scripting in the Python path
itself — step 4 just opens the finished SVG and tidies up its layer
panel.

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
        │
        │  Step 4   PrepareForInspection.jsx (single-airport only)
        ▼
<airport>-diagram.ai            opened in Illustrator, layers sorted
```

### Layer organization (step 4)

`PrepareForInspection.jsx` runs inside Illustrator on a single-airport
run. It:

1. Saves the SVG as `<airport>-diagram.ai` next to it.
2. Promotes each SVG-import sublayer out of the "Layer 1" wrapper so
   they sit at the document root.
3. Ensures every target layer exists, including a manual-only **Lights**
   placeholder that the classifier doesn't populate.
4. Sorts the panel top-to-bottom into the standard inspection stack:
   `Runway Labels → Taxiway Labels → Stars → Lights → Footprints →
   Runways → Taxiways → Other → PDF Text Tokens → Metadata`.
5. Locks and hides **Other**, **PDF Text Tokens**, and **Metadata** —
   reference data, not part of the editable diagram.
6. Saves the `.ai` and leaves it open for review.

### `classify_pipeline.py` — the 7 substeps

Polygons claimed in earlier substeps are **removed from the pool** seen
by every later substep. ML only decides Footprints / Stars / Other; the
four rule-based classes (Taxiways, Taxiway Labels, Runways, Runway
Labels) are all claimed in steps 1–4 before ML runs.

1. **Taxi surfaces (rule-based).** Filled polygons whose RGB is gray
   (~#cfcfcf with leeway: avg 175–235, channel spread ≤ 20). This is
   the *only* source for Taxiways — every stroked-unfilled polygon is
   demoted to Other by the final stroked-only sweep (step 7).

2. **Runways (rule-based, NASR-driven).** Look up the airport in
   `data/nasr_apt_rwy.csv`, count its non-helipad runways → `N`. From
   the unclaimed pool, take the `N` largest polygons by polygon area
   (shoelace, robust to rotation) that are either filled near-black
   (paved) or stroked-only (grass strip outlines). Two safeguards: a
   bbox-area smell test (≤ 50% of page) rejects the chart frame, and
   a PCA-derived aspect-ratio sanity check (≥ 20% of the airport's
   smallest NASR runway aspect, floor 4:1) rejects label boxes whose
   polygon area might rival a small runway. Nested clip groups
   exposed by `get_drawings(extended=True)` are also candidates; when
   a clip wins, the largest polygon fully contained in its scissor is
   claimed (this is how F45's grass strip — drawn only as a clipped
   hatch pattern — gets picked up).

3. **Runway Labels (rule-based).** For each rule-claimed Runway,
   compute its principal axis via PCA; from each endpoint, search
   outward through widths `(1, 10, 25, 50, 100)pt` for a token whose
   normalized form is one of NASR's listed designations for this
   airport (`08L → 8L`; compass directions `NE/SW` for turf strips).
   The chosen token claims K=`len(token)` nearest unclaimed near-black
   filled polygons. Tokens are reserved so `9L` can't be re-used at
   another runway end. Runs **before** taxi labels (step 4) so a digit
   glyph belonging to a runway designator (e.g. APF "5" sitting over a
   taxiway) is reserved before taxi-label matching can grab it.

   **3b. Runway-label move along the centerline (layout).** Layout-
   only step — does NOT change which layer any polygon is on.
   Implemented in [`python/pipeline/runway_label_layout.py`](python/pipeline/runway_label_layout.py).
   For every label group claimed in step 3, picks the **first
   filled-Taxi polygon along the extended centerline past this end**
   and aligns the label `2pt` past that polygon's far edge. The
   translation has TWO components combined into one rigid `(dx, dy)`
   applied to every polygon in the group:

   - **Longitudinal**: cast a ray from the runway-end point along the
     outward principal axis. The first filled-Taxi polygon the ray
     enters is the contiguous extension — its smallest ray-vs-boundary
     intersection is `near_t`, its largest is `far_t`. The label's
     nearest-glyph anchor (computed from actual polygon anchors, NOT
     bbox — an angled "22L" has empty bbox corners that would push the
     visible glyph too far) is shifted to land at
     `half_len + far_t + 2pt` along the outward axis. If the runway
     end happens to sit INSIDE the polygon (gray fill extending across
     the threshold, common on FAA charts), `near_t` is treated as 0.
     Sanity gate: `near_t ≤ 1pt` so a polygon 30pt past the threshold
     across a gap doesn't get picked.

   - **Lateral**: also center the group on the centerline. The
     label's anchor-bounds midpoint perpendicular to the runway axis
     is computed from actual polygon anchors
     (`(lat_min + lat_max) / 2`), and the group is shifted so that
     midpoint sits at `lat = 0`. Without this, labels drawn slightly
     off-axis (left or right of the centerline in the source PDF)
     would still be off-axis after the longitudinal move.

   The translation per polygon is carried through the predictions
   JSON as `translate_x` / `translate_y` and applied as an SVG
   `transform="translate(...)"` per `<path>` in stage 3 (renderer) —
   original PDF geometry stays intact.

4. **Taxi Labels (rule-based).** PDF text tokens matching
   `^([A-Z][A-Z]?[0-9]{0,2}|[0-9])$` whose centroid sits inside a
   Taxi surface; same K-nearest claim as runway labels. Tokens
   already consumed by step 3 are excluded so a runway-designator
   digit glyph isn't re-claimed here.

5. **Concave-hull rejection (pre-ML).** Build a concave hull
   (`shapely.concave_hull(ratio=0.0)`, no buffer) over the
   rule-claimed Runways + Taxi surfaces' anchor points. An unclaimed
   polygon is demoted to **Other** only when its bbox doesn't
   intersect the hull at all — anything that touches or overlaps the
   hull is kept and passed to ML. (The earlier centroid-in-hull test
   was too strict for buildings flush with apron edges, where the
   centroid sat just outside.) Runways, Taxi surfaces, Runway Labels,
   and Taxi Labels are exempt — they're rule-trusted, and labels can
   legitimately sit at chart edges. **Production runs with
   `--skip-hull`** (set in `classify.sh`) because too many legitimate
   building footprints sit just outside the hull on real charts; the
   step is preserved for experimentation.

6. **ML — Footprints / Stars / Other.** The v25 LightGBM
   (`python/ml/runs/v25/model.lgb`) runs on the unclaimed pool. It's
   a 3-class classifier trained on the 30-airport clean corpus + 118
   NASR-matched legacy airports (148 airports / 84,746 training rows).
   Stroked items stay in the pool here — they provide neighbour-
   context features the model uses to recognise symbols. No
   mask/postprocessing on the probability matrix; argmax wins.

7. **Stroked-only sweep (final).** Any polygon whose `stroked &&
   !filled` *and* isn't on the Runways layer is demoted to **Other**.
   This is the absolute gate — every stroked-unfilled artifact
   (Lights stripes, arrowheads, decorative line-art, taxiway outline
   polygons drawn over the gray fill, painted hold-position bars,
   centerline marks) ends up on Other. Runways are the only
   exemption because grass-strip runways are drawn as stroked
   rectangles (F45 is the canonical case).

The PDF Text Tokens debug layer is always emitted: every word in the
PDF text stream as a magenta 4pt text frame at its bbox center.
Useful for spot-checking why a token did or didn't qualify.

## Layout

```
faa-pro/
├── classify.sh                          # entry point
├── classify-airports.sh                 # interactive wrapper: prompts, mv, classify
├── classify-wrapper.conf                # default folder paths for the wrapper
├── README.md                            # this file
├── python/
│   ├── classify_pipeline.py             # 7-substep orchestrator (Stage 2)
│   ├── render_svg_layers.py             # SVG export (Stage 3)
│   ├── pipeline/                        # rule-based detection modules
│   │   ├── chart_scene.py               #   PDF → polygons + clips + tokens
│   │   ├── extract_paths_fitz.py        #   PyMuPDF path extraction (Stage 1)
│   │   ├── extract_paths_batch.py       #   batch wrapper around extract_paths
│   │   ├── extract_pdf_text.py          #   NASR + text extraction
│   │   ├── hull_filter.py               #   concave-hull rejection (substep 5)
│   │   ├── runway_detection.py          #   rule-based runway (substep 2)
│   │   ├── runway_label_layout.py       #   centerline label move (substep 3b)
│   │   └── taxi_detection.py            #   gray-fill + taxi-label K-nearest
│   ├── ml/                              # ML training + utilities
│   │   ├── load.py                      #   CSV schema + LABELS constants
│   │   ├── relational.py                #   feature engineering
│   │   ├── extract_labeled_corpus.py    #   batch labeled-AI extraction
│   │   ├── train.py                     #   LightGBM trainer
│   │   └── runs/v25/model.lgb           #   trained LightGBM (production)
│   ├── _deprecated/                     # confirmed dead, kept for git history
│   ├── char_training_legacy/            # pre-rebuild char-classifier code
│   └── README.md                        # Python-side details + retraining
├── data/
│   ├── nasr_apt_rwy.csv                 # FAA NASR runway designations
│   └── char_training_legacy/            # old char-corpus training data
├── ImportPredictedLayers.jsx            # legacy JSX renderer, no longer invoked
├── ExportClassifiedPaths.jsx            # legacy labeled-export (replaced by ml/extract_labeled_corpus.py)
├── PrepareForLabeling.jsx               # used during retraining (scaffold creation)
├── PrepareForInspection.jsx             # post-classify Illustrator layer organizer
└── _deprecated/                         # historical debug scripts + experiment outputs
```

## Retraining

The current model at `python/ml/runs/v25/model.lgb` is a 3-class LightGBM
(Footprints / Stars / Other) trained on **148 airports / 84,746 rows**:
the 30-airport clean corpus + 118 NASR-matched airports from the legacy
160-file Pathfinder-unioned corpus. International legacy airports are
excluded — they were sourced from OSM and have different stylization.

Layer extraction is fully Python now (no Illustrator round-trip):
`extract_labeled_corpus.py` reads `<code>-diagram.ai` files via PyMuPDF,
forces all OCG layers visible (the user hides Other / Uncertain / Lines
/ Text / Arrowheads in the UI config so the file *displays* clean), and
maps each drawing's `layer` field to a canonical training label via
`load.layer_name_to_label`. Substring matching folds legacy variants
("Footprints copy", "Footprints Small", "Footrpints Small" [typo]) into
their canonical class.

To retrain see `python/README.md`. Important rule: **don't modify the
geometry** of training files (no Pathfinder, no scale/rotation, no fill
changes), or the model's geometric features won't match what it sees at
inference time.

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
- Buildings whose bbox doesn't intersect the concave hull are demoted
  to Other when hull rejection is enabled (substep 5 — runs before
  ML). Production sets `--skip-hull` so this isn't active by default,
  but the code path exists. The bbox-intersect test keeps anything
  that touches the hull, more lenient than the earlier centroid-in-
  hull test, but a building whose bbox sits fully outside the hull
  (e.g. detached terminal across a road from the apron) will still
  be rejected.

## Pipeline status

All substeps in the rebuilt pipeline are landed:

1. **Taxi surfaces → gray fill.** `pipeline/taxi_detection.py`'s
   gray-RGB rule via `chart_scene.is_taxi_surface`.
2. **Runways → NASR-driven.** `pipeline/runway_detection.py` —
   top-N rule with nested clip-group support and PCA aspect-ratio
   sanity check. Validated on ARB, APF, ELM, F45.
3. **Runway labels → centerline-token search.**
   `classify_pipeline._match_runway_labels` — for each rule-claimed
   runway, search outward through widening centerline bands for a
   NASR-listed token, claim K=`len(token)` nearest near-black filled
   polygons. Tokens reserved across ends and across step 4.
   3b. **Runway-label centerline move (layout).**
       `pipeline/runway_label_layout.py` — pick the first filled-Taxi
       polygon along the extended centerline, place the label group
       2pt past its far edge, lateral-center on the centerline.
       Translation carried to renderer via `translate_x` /
       `translate_y` in the predictions JSON.
4. **Taxi labels → K-nearest.** `pipeline/taxi_detection.match_taxi_labels`
   — token regex K-nearest gated on centroid-in-Taxi-surface. Runs
   after step 3 so runway-designator digit glyphs are reserved.
5. **Concave-hull rejection (pre-ML).** `pipeline/hull_filter.py` —
   `shapely.concave_hull(ratio=0.0)` over rule-claimed Runways +
   Taxiways' anchor points; bbox-intersects-hull test keeps anything
   touching the hull. **Production sets `--skip-hull`** because real
   charts have legitimate detached buildings outside the hull.
6. **ML — Footprints / Stars / Other.** v25 3-class LightGBM at
   `python/ml/runs/v25/model.lgb`, trained on 148 airports / 84,746
   rows (30 clean + 118 NASR-matched legacy). Morphology features
   (convexity, circularity, rectangularity, shape index, vertex
   density, hull_area_rel) in `ml/relational.py`. OOF macro-F1 =
   0.887 (Footprints F1 = 0.894).
7. **Stroked-only sweep.** `classify_pipeline.py` final pass demotes
   any polygon with `stroked && !filled` to Other, except those on
   the Runways layer (grass-strip runways are stroked rectangles).
   This is the absolute gate — no stroked-unfilled artwork survives
   onto a target layer.

## Commit history (rebuild)

The repo history starts at `91ab78c rebuild in progress`. The
rebuild added the rule-based taxi pipeline, the centerline-token
runway-label search, and the SVG renderer. The character-classifier
code from before the rebuild is preserved under
`python/char_training_legacy/` and `data/char_training_legacy/` for
potential future reuse.

## Notes (v1, 2026-05-07)

Captured for the next person (or next session) picking this up.

### Production defaults locked in
- **Model**: `python/ml/runs/v25/model.lgb` (3-class:
  Footprints / Stars / Other, OOF macro-F1 = 0.887).
- **Hull rejection is OFF** in `classify.sh`
  (`PIPELINE_EXTRA="--skip-hull"`). Validated end-to-end on
  ARB / APF / COS / ELM / F45 / MCO / OGG / ORD. Many legitimate
  building footprints (e.g. ARB FIRE STATION at 277,138–285,152)
  sit fully outside the hull. Without `--skip-hull`, step 4
  demoted them to Other before ML ever saw them.
- **Argmax decision rule** (no `--footprint-threshold`). On
  ORD-class charts, lowering the threshold past argmax pulls in
  arrow symbols. The flag is plumbed through and available via
  `PIPELINE_EXTRA="--skip-hull --footprint-threshold 0.10"` if
  you want to experiment per-airport.
- **Taxi-label gate**: token *centroid* must be inside the taxi
  surface polygon (`taxi_detection._bbox_touches`). Earlier
  bbox-corner test let runway-slope annotations like OGG's "UP"
  qualify; the centroid test is stricter and aligned with the
  user's intent ("center, not just touching").

### Known limits to plan around when retraining
- **Symbol negatives are underrepresented.** The 78K Other rows
  in v25's training corpus are mostly text + lines + arrowheads.
  Chart symbols (arrows, fuel circles, hot-spot markers, compass
  rose tick marks) show up rarely as labeled negatives, so the
  model can confuse a wide-bodied arrowhead with a footprint.
  When labeling new airports, **explicitly drag chart symbols
  into Other** rather than leaving them in Layer 1.
- **The persistent ORD arrow** that survives even the strict
  argmax rule is exactly the kind of symbol-negative the model
  hasn't been trained on. Worth identifying its `object_id` in
  `ord_predictions.json` and labeling it as a high-value example
  in the next training pass.
- **Stars is noisy** — only 25 training samples, OOF F1 = 0.776.
  Most airports have ≤ 1 Star. If accuracy matters, label more
  Stars in the new corpus.

### Workflow for the next training pass
1. Drag-label new diagrams via `PrepareForLabeling.jsx` →
   `<code>-diagram.ai` files. Hard rule: **no Pathfinder, no
   transforms, no fill changes.** See
   [python/README.md](python/README.md) for full rules.
2. Place new files anywhere. Run scripts from project root with
   `PYTHONPATH=python` so `from pipeline.* import …` resolves.
   To rebuild the labeled corpus:
   ```
   PYTHONPATH=python python python/ml/extract_labeled_corpus.py \
       --root /path/to/new/labeling/folder \
       --root /Users/lukehogan/AOA-Code/faa-downloader/airports-class \
       --root /Users/lukehogan/Documents/startups/aoa/products/artwork/airports \
       --out  python/labeled_corpus.csv \
       --us-only
   ```
3. `PYTHONPATH=python python python/ml/relational.py --in python/labeled_corpus.csv --out python/v26_features.parquet`
4. `PYTHONPATH=python python python/ml/train.py --features python/v26_features.parquet --out-dir python/ml/runs/v26`
5. Update the `--model` and `--feature-list` defaults in
   `classify_pipeline.py` (or just rename `python/ml/runs/v26` →
   `python/ml/runs/v25`).

### Validation discipline
Hold out the v1-validated set
(ARB / APF / COS / ELM / F45 / MCO / OGG / ORD) and compare
Footprint counts before vs. after retrain. If v26 regresses on
those airports, the new corpus has a labeling drift to find
before shipping.

### Useful flags during experimentation
- `PIPELINE_EXTRA="--skip-hull --footprint-threshold 0.10"
  bash classify.sh ord-faa.pdf` — gentle promotion threshold
- `PIPELINE_EXTRA="" bash classify.sh arb-faa.pdf` — restore the
  pre-v1 hull-on behaviour (useful for debugging hull-vs-no-hull
  diffs)
