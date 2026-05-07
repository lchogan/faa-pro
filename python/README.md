# faa-classifier — Python pipeline

The Python side of the airport-diagram classifier. Production
inference is invoked from the parent `classify.sh`; see the parent
[README](../README.md) for an end-to-end overview.

This README is the **training** workflow. Inference details are in
the parent README and in the docstrings of `classify_pipeline.py`.

## Inference (production)

`classify_pipeline.py` is the orchestrator. From the parent dir:

```bash
bash classify.sh /path/to/<airport>-faa.pdf
```

That runs three Python steps:

| Step | Script | Purpose |
|------|--------|---------|
| 1    | `extract_paths_fitz.py` | PyMuPDF path extraction → `<airport>_paths.csv` |
| 2    | `classify_pipeline.py`  | 6-step classification → `<airport>_predictions.json` |
| 3    | `render_svg_layers.py`  | SVG render with native AI layers → `<airport>-diagram.svg` |

`classify_pipeline.py` itself imports `taxi_detection.detect_taxi`
for the rule-based steps, the LightGBM `Booster` from
`runs/v24/model.lgb` for ML, and `pdf_char_override.apply_pdf_char_override`
for the Lights-by-stroke heuristic.

## Training (LightGBM model retraining)

The model in `runs/v24/` is trained on per-PathItem geometric +
relational features extracted from manually-labeled `*-diagram.ai`
files. To retrain on a refreshed corpus:

```
Source FAA PDF
       │
       ▼  Stage 0: PrepareForLabeling.jsx       (one-time per training airport)
<code>-diagram.ai (empty layered scaffold)
       │
       ▼  Manual labeling in Illustrator        (drag paths into target layers)
<code>-diagram.ai (labeled, geometry preserved)
       │
       ▼  Stage 1: ExportClassifiedPaths.jsx
classified_paths.csv
       │
       ▼  Stage 2: relational.py (adds neighbor / spatial features)
features.parquet
       │
       ▼  Stage 3: train.py
runs/v<N>/model.lgb + metrics.json + feature_list.json
```

After training, point `classify_pipeline.py` at the new run dir via
its `--model` and `--feature-list` args (or rename to `runs/v<N>` and
update the defaults in the file).

### Producing clean training data

Geometry must match what the model sees at inference time. The
existing pre-rebuild training corpus has been post-edited (Pathfinder
unite, color/scale changes) and is unreliable for retraining. For
new training files:

1. In Illustrator: `File → Scripts → Other Script → PrepareForLabeling.jsx`
   in the parent dir. Pick the FAA source PDF and enter the airport
   code (e.g. `cha`). The script saves `<code>-diagram.ai` next to the
   PDF with the seven empty target layers and a holding layer
   `Unclassified` containing all raw paths.
2. Open the Layers panel. Drag each PathItem from `Unclassified` into
   the correct target layer. Leftover items go to `Other`. The
   `Unclassified` layer must be empty when you save.
3. **Hard rules** (or the file becomes unusable as training data):
   - No Pathfinder (Unite, Merge, Minus Front, etc).
   - No fill or stroke color changes.
   - No rotate / scale / move / transform.
   - No artboard modifications.
   - No Live Paint, Image Trace, or any operation that reshapes paths.
4. Save. Aim for ~20–30 diverse airports across class B/C/D.

### Train

```bash
# 1. Add neighbor / spatial features to the exported CSV
python relational.py \
    --in /path/to/classified_paths.csv \
    --out features.parquet

# 2. Train (writes model.lgb + metrics.json + confusion.csv)
python train.py --features features.parquet --out-dir runs/v25
```

## Files

```
python/
├── classify_pipeline.py         # production: 6-step orchestrator
├── hull_filter.py               # concave-hull rejection (substep 6)
├── extract_paths_fitz.py        # production: PyMuPDF path/feature extraction
├── render_svg_layers.py         # production: layered SVG export
├── taxi_detection.py            # rule-based gray-fill + taxi-label K-nearest
├── pdf_char_override.py         # Lights stroke heuristic + runway-axis helpers
├── relational.py                # neighbor / spatial features for training+predict
├── load.py                      # CSV schema + LABELS = (Taxiways, Footprints, ...)
├── predict_one.py               # PDF text extraction helper (still imported)
├── extract_pdf_text.py          # NASR runway designations + text extraction
├── train.py                     # LightGBM training
├── predict.py                   # legacy predictor; classify_pipeline.py replaced it
├── runs/v24/                    # trained model, feature list, metrics
│
├── render_pdf_layers.py         # PDF/OCG renderer (alternate, not in classify.sh)
├── render_char_layers_charbox.py # taxi-only debug SVG; experimental
├── pdf_page_to_svg.py
├── extract_paths_batch.py
├── build_demo_predictions.py
├── dump_pdf_text.py
└── char_training_legacy/        # pre-rebuild char-classifier code, kept for reuse
```

## Setup

```bash
cd python
uv sync
# or: python -m venv ../.venv && source ../.venv/bin/activate && pip install -e .
```

The parent `classify.sh` expects the venv at `../.venv/bin/python3`.

## Data contract

`load.py` is the single source of truth for the CSV schema and label
mapping. Any change to the JSX exporter (`ExportClassifiedPaths.jsx`)
or the PyMuPDF extractor (`extract_paths_fitz.py`) must be mirrored
there.
