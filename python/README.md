# faa-classifier — Python pipeline

The Python side of the airport-diagram classifier. Production
inference is invoked from the parent `classify.sh`; see the parent
[README](../README.md) for an end-to-end overview.

This README covers the **directory layout** and **training** workflow.
Inference details are in the parent README and in the docstring at the
top of `classify_pipeline.py`.

## Directory layout

```
python/
├── classify_pipeline.py    # orchestrator (entry point — called by classify.sh)
├── render_svg_layers.py    # SVG output (entry point — called by classify.sh)
├── pipeline/               # rule-based detection modules
│   ├── chart_scene.py      #   PDF → polygons + clips + text tokens
│   ├── extract_paths_fitz.py  # PyMuPDF path/feature extraction
│   ├── extract_paths_batch.py # batch wrapper around extract_paths
│   ├── extract_pdf_text.py    # NASR designations + PDF text extraction
│   ├── hull_filter.py         # concave-hull rejection (step 5)
│   ├── runway_detection.py    # rule-based runway claim (step 2)
│   ├── runway_label_layout.py # runway-label centerline move (step 3b)
│   └── taxi_detection.py      # rule-based gray-fill + taxi-label K-nearest
├── ml/                     # ML training + utilities
│   ├── load.py             #   CSV schema + LABELS / ML_LABELS / RULE_CLAIMED_LABELS
│   ├── relational.py       #   neighbor / spatial feature engineering
│   ├── extract_labeled_corpus.py  # batch labeled-AI extraction → training CSV
│   ├── train.py            #   LightGBM trainer
│   └── runs/               #   trained models (v1 … v25); production = runs/v25/
├── _deprecated/            # confirmed dead — kept for git history
│   ├── predict.py          #   legacy predictor (replaced by classify_pipeline)
│   ├── predict_one.py      #   legacy single-airport wrapper
│   ├── pdf_char_override.py
│   ├── build_demo_predictions.py
│   ├── render_pdf_layers.py        # alternate PDF/OCG renderer
│   ├── render_char_layers_charbox.py
│   ├── pdf_page_to_svg.py
│   └── dump_pdf_text.py
├── char_training_legacy/   # pre-rebuild char-classifier code
└── (data files: features.parquet, labeled_corpus.csv, …)
```

`classify.sh` exports `PYTHONPATH="$PROJECT/python"` so any script
under `python/` can resolve `from pipeline.X import …` and
`from ml.X import …` regardless of where it lives.

## Inference (production)

`classify_pipeline.py` is the orchestrator. From the parent dir:

```bash
bash classify.sh /path/to/<airport>-faa.pdf
```

That runs three Python steps:

| Stage | Script | Purpose |
|-------|--------|---------|
| 1     | `pipeline/extract_paths_fitz.py` | PyMuPDF path extraction → `<airport>_paths.csv` |
| 2     | `classify_pipeline.py`           | 7-substep classification → `<airport>_predictions.json` |
| 3     | `render_svg_layers.py`           | SVG render with native AI layers → `<airport>-diagram.svg` |

Substeps inside `classify_pipeline.py`: 1 taxi surfaces · 2 runways
(NASR-driven) · 3 runway labels · 3b runway-label centerline move ·
4 taxi labels · 5 hull rejection (skipped in production via
`--skip-hull`) · 6 ML on remaining unclaimed · 7 stroked-only sweep
to Other. See the docstring at the top of `classify_pipeline.py` for
the gory details.

The production model is `ml/runs/v25/model.lgb` (3-class:
`Footprints` / `Stars` / `Other`). Taxiways, Taxiway Labels, Runways,
Runway Labels are all rule-claimed before ML runs and never go
through the model.

## Training (LightGBM model retraining)

The model in `ml/runs/v25/` is trained on per-PathItem geometric +
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
       ▼  Stage 1: ml/extract_labeled_corpus.py (batch reads <code>-diagram.ai files)
labeled_corpus.csv
       │
       ▼  Stage 2: ml/relational.py             (adds neighbor / spatial features)
features.parquet
       │
       ▼  Stage 3: ml/train.py
ml/runs/v<N>/model.lgb + metrics.json + feature_list.json
```

After training, point `classify_pipeline.py` at the new run dir via
its `--model` and `--feature-list` args (or rename to
`ml/runs/v<N>` and update the defaults in the file).

### Producing clean training data

Geometry must match what the model sees at inference time. The
pre-rebuild legacy training corpus has been post-edited (Pathfinder
unite, color/scale changes) and is unreliable for retraining. For
new training files:

1. In Illustrator: `File → Scripts → Other Script → PrepareForLabeling.jsx`
   in the parent dir. Pick the FAA source PDF and enter the airport
   code (e.g. `cha`). The script saves `<code>-diagram.ai` next to
   the PDF with the empty target layers and a holding layer
   `Unclassified` containing all raw paths.
2. Open the Layers panel. Drag each PathItem from `Unclassified`
   into the correct target layer. Leftover items go to `Other`. The
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
python ml/relational.py \
    --in /path/to/labeled_corpus.csv \
    --out features.parquet

# 2. Train (writes model.lgb + metrics.json + confusion.csv)
python ml/train.py --features features.parquet --out-dir ml/runs/v26
```

## Setup

```bash
cd python
uv sync
# or: python -m venv ../.venv && source ../.venv/bin/activate && pip install -e .
```

The parent `classify.sh` expects the venv at `../.venv/bin/python3`.

## Data contract

`ml/load.py` is the single source of truth for the CSV schema and
label mapping. Any change to the JSX exporter
(`ExportClassifiedPaths.jsx`) or the PyMuPDF extractor
(`pipeline/extract_paths_fitz.py`) must be mirrored there.
