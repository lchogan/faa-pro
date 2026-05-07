# faa-pro — FAA Airport Diagram Classifier

Classifies vector objects in FAA airport diagram PDFs into semantic
layers: **Taxiways, Taxiway Labels, Runways, Runway Labels, Footprints,
Stars, Other**. Output is a layered SVG that opens in Adobe Illustrator
with native AI layers (save-as `.ai` if you want the extension).

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
by every later substep. ML only decides Footprints / Stars / Other; the
four rule-based classes (Taxiways, Taxiway Labels, Runways, Runway
Labels) are claimed in steps 1–3 before ML runs.

1. **Taxi surfaces (rule-based).** Filled polygons whose RGB is gray
   (~#cfcfcf with leeway: avg 175–235, channel spread ≤ 20). This is
   the *only* source for Taxiways.

2. **Runways (rule-based, NASR-driven).** Look up the airport in
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

3. **Runway + Taxiway Labels (rule-based).** Two passes, runway-first.
   *Runway Labels:* for each rule-claimed Runway, compute its principal
   axis via PCA; from each endpoint, search outward through widths
   `(1, 10, 25, 50, 100)pt` for a token whose normalized form is one of
   NASR's listed designations for this airport (`08L → 8L`; compass
   directions `NE/SW` for turf strips). The chosen token claims
   K=`len(token)` nearest unclaimed near-black filled polygons. Tokens
   are reserved so `9L` can't be re-used at another runway end. *Taxi
   Labels:* PDF text tokens matching `^([A-Z][A-Z]?[0-9]{0,2}|[0-9])$`
   whose bbox touches a Taxi surface; same K-nearest claim. Runway
   Labels run first so a digit glyph belonging to a runway designator
   (e.g. APF "5" sitting over a taxiway) is reserved before taxi-label
   matching can grab it.

4. **Concave-hull rejection (pre-ML).** Build a concave hull
   (`shapely.concave_hull(ratio=0.0)`, no buffer) over the rule-claimed
   Runways + Taxi surfaces' anchor points. An unclaimed polygon is
   demoted to **Other** only when its bbox doesn't intersect the hull
   at all — anything that touches or overlaps the hull is kept and
   passed to ML. (The earlier centroid-in-hull test was too strict for
   buildings flush with apron edges, where the centroid sat just
   outside.) Runways, Taxi surfaces, Runway Labels, and Taxi Labels
   are exempt — they're rule-trusted, and labels can legitimately sit
   at chart edges. Doing this *before* ML stops the model from spending
   capacity on legend/scale-bar/note polygons.

5. **ML — Footprints / Stars / Other.** The v25 LightGBM
   (`python/runs/v25/model.lgb`) runs on the unclaimed in-hull pool.
   It's a 3-class classifier trained on the 30-airport clean corpus +
   118 NASR-matched legacy airports (148 airports / 84,746 training
   rows). Stroked items stay in the pool here — they provide
   neighbour-context features the model uses to recognise symbols.
   No mask/postprocessing on the probability matrix; argmax wins.

6. **Stroked-only sweep (final).** Any polygon whose `stroked && !filled`
   *and* isn't on the Runways layer is demoted to **Other**. This
   catches Lights stripes, arrowheads, and decorative line-art that ML
   classed as Other-or-Footprint. Runways are exempt because grass-
   strip runways are drawn as stroked rectangles (F45 is the canonical
   case).

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
│   ├── extract_paths_fitz.py            # PyMuPDF path extraction (inference + labeled .ai)
│   ├── extract_labeled_corpus.py        # batch labeled extraction → training CSV
│   ├── render_svg_layers.py             # SVG export (Step 3)
│   ├── chart_scene.py                   # PDF → polygons + clips + tokens (single source of truth)
│   ├── taxi_detection.py                # rule-based taxi surfaces + labels (substeps 1, 3)
│   ├── runway_detection.py              # rule-based runway (substep 2, NASR-driven)
│   ├── hull_filter.py                   # concave-hull rejection (substep 4)
│   ├── relational.py                    # ML feature engineering (incl. morphology)
│   ├── load.py                          # CSV schema + LABELS / ML_LABELS / RULE_CLAIMED_LABELS
│   ├── train.py                         # v25 3-class trainer
│   ├── extract_pdf_text.py              # NASR + text extraction (used by runway-label search)
│   ├── runs/v25/model.lgb               # trained LightGBM (Footprints / Stars / Other)
│   ├── pdf_char_override.py             # legacy override (no longer in production path)
│   ├── render_pdf_layers.py             # PDF/OCG renderer (alternate, not used)
│   └── char_training_legacy/            # pre-rebuild char-classifier code, kept for reuse
├── data/
│   ├── nasr_apt_rwy.csv                 # FAA NASR runway designations
│   └── char_training_legacy/            # old char-corpus training data
├── ImportPredictedLayers.jsx            # legacy JSX renderer, no longer invoked
├── ExportClassifiedPaths.jsx            # legacy labeled-export (replaced by extract_labeled_corpus.py)
├── PrepareForLabeling.jsx               # used during retraining (scaffold creation)
└── _deprecated/                         # historical debug scripts + experiment outputs
```

## Retraining

The current model at `python/runs/v25/model.lgb` is a 3-class LightGBM
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
  to Other (step 4 runs before ML). The bbox-intersect test keeps
  anything that touches the hull, which is more lenient than the
  earlier centroid-in-hull test, but a building whose bbox sits fully
  outside the hull (e.g. detached terminal across a road from the
  apron) will still be rejected.

## Pipeline status (six-step plan)

All six steps in the rebuilt pipeline are now landed:

1. **Taxiways → gray fill.** Done (pre-rebuild rule).
2. **Runways → deterministic.** Done — NASR-driven top-N rule with
   nested clip-group support and aspect-ratio sanity check.
   Validated on ARB, APF, ELM, F45.
3. **Taxiway and runway labels.** Done — runway labels via
   centerline-token search, then taxi labels via K-nearest. Token
   reservation prevents double-claiming.
4. **Concave hull rejection (pre-ML).** Done —
   `python/hull_filter.py` runs as substep 4, before ML.
   `shapely.concave_hull(ratio=0.0)` over rule-claimed Runways +
   Taxiways' anchor points; centroid-in-hull test demotes anything
   outside to Other and removes from the ML pool. Runways,
   Taxiways, Runway Labels, and Taxiway Labels are exempt.
5. **Footprint ML refactor.** Done — v25 3-class LightGBM
   (Footprints / Stars / Other) at `python/runs/v25/model.lgb`,
   trained on 148 airports / 84,746 rows (30 clean + 118
   NASR-matched legacy). Morphology features (convexity,
   circularity, rectangularity, shape index, vertex density,
   hull_area_rel) added to `relational.py`; convexity is the
   model's #1 feature by gain. OOF macro-F1 = 0.887 (Footprints
   F1 = 0.894). Stroked items stay in the ML input pool so the
   model can use neighbour context.
6. **Stroked → Other (final sweep).** Done — `classify_pipeline.py`
   step 7 demotes any polygon with `stroked && !filled` to Other,
   except those on the Runways layer (grass-strip runways are
   stroked rectangles).

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
- **Model**: `python/runs/v25/model.lgb` (3-class:
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
2. Place new files anywhere. To rebuild the labeled corpus:
   ```
   python python/extract_labeled_corpus.py \
       --root /path/to/new/labeling/folder \
       --root /Users/lukehogan/AOA-Code/faa-downloader/airports-class \
       --root /Users/lukehogan/Documents/startups/aoa/products/artwork/airports \
       --out  python/labeled_corpus.csv \
       --us-only
   ```
3. `python python/relational.py --in python/labeled_corpus.csv --out python/v26_features.parquet`
4. `python python/train.py --features python/v26_features.parquet --out-dir python/runs/v26`
5. Update the `--model` and `--feature-list` defaults in
   `classify_pipeline.py` (or just rename `runs/v26` → `runs/v25`).

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
