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
        │  Step 2   classify_pipeline.py — the brain (5 substeps)
        ▼
<airport>_predictions.json      one record per polygon, AI y-up bbox
        │
        │  Step 3   render_svg_layers.py
        ▼
<airport>-diagram.svg           ten <g> layers, Inkscape-tagged
```

### `classify_pipeline.py` — the 5 substeps

Polygons claimed in earlier substeps are **removed from the pool** seen
by every later substep. ML never decides Taxiways or Taxiway Labels.

1. **Taxiways (rule-based).** Filled polygons whose RGB is gray
   (~#cfcfcf with leeway: avg 175–235, channel spread ≤ 20). This is
   the *only* source for Taxiways.

2. **Taxiway Labels (rule-based).** PDF text tokens matching
   `^([A-Z][A-Z]?[0-9]{0,2}|[0-9])$` (e.g. `C`, `C1`, `A11`, `3`)
   whose bbox **touches** a Taxiways surface from step 1. For each
   qualifying token, the K = `len(token)` nearest unclaimed near-black
   filled polygons (the actual glyph polygons) are claimed as
   Taxiway Labels.

3. **(no step 3 — placeholder for the deferred runway-label
   pre-claim that we removed; see step 5).**

4. **ML — Footprints, Runways, Lights, Stars, Other.** The trained
   LightGBM model (`python/runs/v24/model.lgb`) runs on every polygon
   *not* claimed by step 1 or 2. The Taxiways / Taxiway Labels /
   Runway Labels classes are masked out of the probability matrix so
   an unclaimed polygon can't fall back into them. The existing
   `pdf_char_override.apply_pdf_char_override` runs on the filtered
   set, preserving the Lights-by-stroke heuristic that detects ~1000
   light-fixture stripes.

5. **Runway Labels (rule-based, post-ML).** For each ML-predicted
   Runway polygon, compute its principal axis via PCA. From each
   endpoint, search outward through widths `(1, 10, 25, 50, 100)pt`
   for any runway-pattern (`^(0?[1-9]|[12][0-9]|3[0-6])[LRC]?$`)
   token whose centroid sits in that band. The first width that
   yields a candidate wins; ties are broken by closeness to the
   endpoint. The chosen token claims K nearest unclaimed near-black
   filled polygons → **Runway Labels** (whatever ML had assigned them
   gets overridden). Tokens are reserved across runway ends, so a
   `9L` matched at one runway can't be re-used at another.

The PDF Text Tokens debug layer is always emitted: every word in the
PDF text stream as a magenta 4pt text frame at its bbox center.
Useful for spot-checking why a token did or didn't qualify.

## Layout

```
faa-pro/
├── classify.sh                          # entry point
├── README.md                            # this file
├── python/
│   ├── classify_pipeline.py             # 5-step orchestrator (Step 2)
│   ├── extract_paths_fitz.py            # PyMuPDF path extraction (Step 1)
│   ├── render_svg_layers.py             # SVG export (Step 3)
│   ├── taxi_detection.py                # rule-based taxi (steps 1+2 of pipeline)
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
- **Why no Step 3 pre-claim.** An earlier version of step 3 had each
  runway-pattern token greedily claim K nearest polygons before ML
  ran, validated against centerlines after. False positives from
  bare-digit tokens (`1`, `2`, `3`) grabbed unrelated glyphs that
  happened to lie near a centerline. The current centerline-token
  search anchors on the runways themselves and picks at most one
  token per runway end, eliminating that whole class of error.
- **Why ML can't decide Taxiways or Taxi Labels.** The rule-based
  detection is more reliable: gray fill is unambiguous, and the
  K-nearest token-driven match is essentially perfect on diagrams
  where labels sit on pavement. Letting ML override that would only
  introduce errors.
- **Why centerline-based runway-label matching is a thin band, not a
  bbox-touch test.** Runway designators on FAA charts often sit at
  the threshold *off* the runway pavement. A bbox-touch test against
  the Runway polygon misses them; the principal-axis line extended
  through the polygon reliably passes near the threshold marking.

## Known limitations

- ML sometimes misses parallel runways (ORD detected 6 of 8). When a
  runway pavement isn't predicted, its labels can't be matched in
  step 5 and stay as whatever ML assigned them (usually Other).
  Improving runway recall in the model would close this.
- The PDF Text Tokens debug layer adds ~700 text frames per chart.
  Toggle it off in Illustrator if it gets in the way.

## Commit history (rebuild)

The repo history starts at `91ab78c rebuild in progress`. The
rebuild added the rule-based taxi pipeline, the centerline-token
runway-label search, and the SVG renderer. The character-classifier
code from before the rebuild is preserved under
`python/char_training_legacy/` and `data/char_training_legacy/` for
potential future reuse.
