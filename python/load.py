"""
Loader and schema validator for the CSV produced by ExportClassifiedPaths.jsx.

Single source of truth for:
  - expected column names and dtypes
  - the canonical training-label set
  - how to derive a stable per-row id (airport + object_id)

`load_features` returns a clean DataFrame with no UNLABELED rows. Use
`load_features_all` if you also want unlabeled rows for inference.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

LABELS = (
    "Taxiways",
    "Footprints",
    "Runways",
    "Taxiway Labels",
    "Runway Labels",
    "Stars",
    "Other",
)
# ML classes — the 3-class subset the v25 LightGBM is trained to predict.
# Everything outside this set is rule-claimed earlier in the pipeline (steps
# 1-4) or swept by the stroked-only step 6.
ML_LABELS = (
    "Footprints",
    "Stars",
    "Other",
)
# Labels that are rule-claimed by classify_pipeline.py steps 1-4. Training
# rows with these labels are excluded from the ML training pool because at
# inference time the ML model never sees those polygons.
RULE_CLAIMED_LABELS = (
    "Taxiways", "Taxiway Labels", "Runways", "Runway Labels",
)
UNLABELED = "UNLABELED"

# Columns the JSX exporter emits, in order. Keep in sync with COLUMNS in
# ExportClassifiedPaths.jsx.
EXPECTED_COLUMNS = [
    "airport", "object_id", "kind", "source_layer", "label",
    "left", "top", "right", "bottom", "width", "height",
    "bbox_area", "poly_area", "perimeter",
    "centroid_x", "centroid_y", "aspect",
    "num_anchors", "subpath_count", "closed",
    "filled", "fill_kind", "fill_r", "fill_g", "fill_b",
    "stroked", "stroke_kind", "stroke_r", "stroke_g", "stroke_b", "stroke_width",
    "principal_angle", "principal_ratio",
    "longest_segment_angle", "longest_segment_length",
    "artboard_left", "artboard_top", "artboard_right", "artboard_bottom",
]

NUMERIC_COLS = [
    "object_id",
    "left", "top", "right", "bottom", "width", "height",
    "bbox_area", "poly_area", "perimeter",
    "centroid_x", "centroid_y", "aspect",
    "num_anchors", "subpath_count", "closed",
    "filled", "fill_r", "fill_g", "fill_b",
    "stroked", "stroke_r", "stroke_g", "stroke_b", "stroke_width",
    "principal_angle", "principal_ratio",
    "longest_segment_angle", "longest_segment_length",
    "artboard_left", "artboard_top", "artboard_right", "artboard_bottom",
]

# Optional numeric columns produced by the PyMuPDF extractor (extract_paths_fitz)
# but absent in older JSX-exported CSVs. _coerce skips silently when missing.
OPTIONAL_NUMERIC_COLS = ["hull_area"]


def _validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing expected columns: {missing}")
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if extra:
        # Extras are non-fatal; log via print so callers see it.
        print(f"[load] note: ignoring extra columns: {extra}")


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in OPTIONAL_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["object_id"] = df["object_id"].astype("Int64")
    df["airport"] = df["airport"].astype("string")
    df["kind"] = df["kind"].astype("category")
    df["source_layer"] = df["source_layer"].astype("string")
    df["label"] = df["label"].astype("string")
    df["fill_kind"] = df["fill_kind"].fillna("none").astype("category")
    df["stroke_kind"] = df["stroke_kind"].fillna("none").astype("category")
    return df


def _add_row_id(df: pd.DataFrame) -> pd.DataFrame:
    df["row_id"] = df["airport"].astype(str) + "#" + df["object_id"].astype(str)
    return df


# Source-layer name (case-insensitive, whitespace-collapsed) → canonical label.
# Mirrors mapLayerToLabel() in ExportClassifiedPaths.jsx and is also used by
# extract_paths_fitz.py when reading labeled .ai files via PyMuPDF (every
# drawing dict carries a 'layer' field naming the OCG layer it belongs to).
# Legacy 160-corpus files use a sparser layer set (Footprints/Lights/Runways/
# Taxiways only) — anything else falls through to "Other".
LAYER_TO_LABEL = {
    "taxiways": "Taxiways", "taxiway": "Taxiways",
    "footprints": "Footprints", "footprint": "Footprints",
    "runways": "Runways", "runway": "Runways",
    # Lights aren't a model class anymore (step 6 sweeps stroked items to
    # Other); folded into Other so legacy "Lights" layer rows still serve
    # as negatives during training.
    "lights": "Other", "light": "Other",
    "taxiway labels": "Taxiway Labels", "taxiwaylabels": "Taxiway Labels", "taxiway label": "Taxiway Labels",
    "runway labels": "Runway Labels", "runwaylabels": "Runway Labels", "runway label": "Runway Labels",
    "stars": "Stars", "star": "Stars",
    "layer 1": UNLABELED, "unclassified": UNLABELED,
    # Legacy variants seen in the 160-corpus that need explicit dispositions:
    "runway-marking": "Other",       # osh: 7 stray instances
    "labels": UNLABELED,             # ATL-style merged label layer — not training-usable
    "emas": "Other", "windsock": "Other", "text": "Other",
}


def _normalize_layer(name: str) -> str:
    if not isinstance(name, str):
        return ""
    return " ".join(name.lower().split())


def layer_name_to_label(name: str | None) -> str:
    """Map a raw OCG/Illustrator layer name to its canonical training label.

    Returns UNLABELED for unknown / Layer 1 / merged-label layers (callers
    drop those rows from training). The 125-airport legacy corpus has a
    chaotic spread of layer names (Footprints / Footprints Small /
    Footrpints Small [typo] / New Footprints / Taxiways Copy / Apron copy /
    OSM Taxiways / etc.). We pull these into the canonical training labels
    via substring matching:
      - any layer name containing "footprint" / "footrpint"  → Footprints
      - "taxiway" / "apron" / "parking"                     → Taxiways
      - "runway" (but not "runway label*")                  → Runways
      - "label"                                             → catchall Labels
        (legacy ATL-style merged label layers — UNLABELED, drop in training)
    Anything unrecognized that *isn't* the UNLABELED holding tank falls
    through to "Other" so legacy negatives (Lines, Text, Arrowheads, EMAS,
    Windsock, etc.) contribute as Other rather than being silently dropped.
    """
    norm = _normalize_layer(name or "")
    if not norm:
        return UNLABELED
    if norm in LAYER_TO_LABEL:
        return LAYER_TO_LABEL[norm]
    # Substring fallbacks for legacy layer-name variants. Order matters:
    # "runway labels" / "taxiway labels" must be matched before the bare
    # "runway" / "taxiway" rules, but since both are already in the exact
    # dict above we never reach this code with those names.
    if "footprint" in norm or "footrpint" in norm:
        return "Footprints"
    if "label" in norm:
        return UNLABELED  # merged-label legacy layers, not training-usable
    if "runway" in norm:
        return "Runways"
    if "taxiway" in norm or "apron" in norm or "parking" in norm:
        return "Taxiways"
    return "Other"


def relabel_from_source_layer(df: pd.DataFrame) -> pd.DataFrame:
    """Re-derive the `label` column from `source_layer` using the current
    layer-name mapping. Use this to fix older CSVs that were exported with
    a stale mapping (e.g. before "Star" → "Stars" was added) and to fold
    legacy variants ("Footprints copy", "Footprints Small") into their
    canonical label.

    Goes through layer_name_to_label so substring matching matches the
    extractor's behavior; without this, exact-only dict lookup would silently
    demote ~1k legacy footprint variants into Other.
    """
    df = df.copy()
    df["label"] = df["source_layer"].astype(str).map(layer_name_to_label)
    return df


def load_features_all(csv_path: str | Path, relabel: bool = True) -> pd.DataFrame:
    """Load every row, including unlabeled. Use for inference.

    `relabel=True` (default) re-derives the `label` column from `source_layer`
    using the current `_LAYER_TO_LABEL` mapping. This keeps older CSVs in sync
    with mapping changes (e.g. when a new layer name like "Star" is added)
    without re-running the JSX export.
    """
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, na_values=[""])
    _validate_schema(df)
    df = _coerce(df)
    df = _add_row_id(df)
    if relabel:
        df = relabel_from_source_layer(df)
    return df


def load_features(csv_path: str | Path) -> pd.DataFrame:
    """Load only labeled rows, ready for training."""
    df = load_features_all(csv_path)
    df = df[df["label"].isin(LABELS)].copy()
    df["label"] = df["label"].astype(pd.CategoricalDtype(categories=list(LABELS), ordered=False))
    return df.reset_index(drop=True)


def label_distribution(df: pd.DataFrame) -> pd.Series:
    return df["label"].value_counts().reindex(LABELS).fillna(0).astype(int)


if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser(description="Inspect a classified_paths.csv export.")
    ap.add_argument("csv", type=Path)
    args = ap.parse_args()

    df = load_features_all(args.csv)
    print(f"Total rows: {len(df):,}")
    print(f"Airports:   {df['airport'].nunique()}")
    print()
    print("Label distribution:")
    counts = df["label"].value_counts()
    for lab, n in counts.items():
        print(f"  {lab:<12} {n:>8,}")
    print()
    if df["label"].eq(UNLABELED).any():
        print(f"Unlabeled rows (Layer 1): {(df['label'] == UNLABELED).sum():,}")
    print()
    print("Source layers (raw, top 20):")
    for lab, n in df["source_layer"].value_counts().head(20).items():
        print(f"  {lab:<25} {n:>8,}")
