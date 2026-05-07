# faa-classifier

ML pipeline for the FAA-PRO airport diagram classifier. Trains on per-PathItem
features extracted from classified `*-diagram.ai` files and predicts one of
seven classes: **Taxiways, Footprints, Runways, Lights, Taxiway Labels,
Runway Labels, Other**.

## Pipeline

```
Source FAA PDF
        │
        ▼  Stage 0: PrepareForLabeling.jsx          (one-time, per new training airport)
<code>-diagram.ai (empty layered scaffold)
        │
        ▼  Manual labeling in Illustrator           (drag paths into target layers — DO NOT modify geometry)
<code>-diagram.ai (labeled, geometry preserved)
        │
        ▼  Stage 1: ExportClassifiedPaths.jsx
classified_paths.csv  ──── per-path geometric features + label
        │
        ▼  Stage 2: relational.py
features.parquet      ──── adds neighbor / spatial features
        │
        ▼  Stage 3: train.py
model.lgb + metrics
        │
        ▼  Stage 4: predict.py
predictions.json      ──── consumed by ImportPredictedLayers.jsx
```

## Producing clean training data

The classifier needs training files whose **geometry matches what it will see at
inference time**. The existing `*-diagram.ai` corpus has been post-edited
(pathfinder unite, color/scale changes, artboard rotation) and is unreliable for
geometry-shaped training of Taxiways / Runways / merged Footprints.

For each new airport you label:

1. In Illustrator: `File → Scripts → Other Script → PrepareForLabeling.jsx`.
   Pick the FAA source PDF and enter the airport code (e.g. `cha`). The script
   saves `<code>-diagram.ai` next to the PDF with seven empty target layers and
   a holding layer called `Unclassified` containing all the raw paths.
2. Open the Layers panel (`Window → Layers`). Drag each PathItem from
   `Unclassified` into the correct labeled layer. Leftover items can be moved
   to `Other`. The `Unclassified` layer should be empty when you save.
3. **Hard rules** (or the file becomes unusable as training data):
   - Do **not** use Pathfinder (Unite, Merge, Minus Front, etc).
   - Do **not** change fill or stroke colors.
   - Do **not** rotate, scale, move, or transform any artwork.
   - Do **not** modify the artboard.
   - Do **not** use Live Paint, Image Trace, or any operation that reshapes paths.
4. Save. Repeat for ~20–30 diverse airports (mix of small regional, medium,
   large hub, international).

## Setup

Recommended (`uv`):

```bash
cd python
uv sync                # creates .venv and installs deps
```

Or with plain pip:

```bash
cd python
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Run

```bash
# 1. Add neighbor features
python relational.py --in /path/to/classified_paths.csv --out features.parquet

# 2. Train (writes model.lgb + metrics.json + confusion.csv)
python train.py --features features.parquet --out-dir runs/v1

# 3. Predict on an unlabeled diagram's exported CSV
python predict.py --model runs/v1/model.lgb \
                  --in unlabeled_paths.csv \
                  --out predictions.json
```

## Data contract

`load.py` is the single source of truth for the CSV schema and label mapping.
Any change to the JSX exporter must be mirrored there.
