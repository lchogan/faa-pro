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
    "Lights",
    "Taxiway Labels",
    "Runway Labels",
    "Stars",
    "Other",
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
# Mirrors mapLayerToLabel() in ExportClassifiedPaths.jsx and is used to *re-derive*
# labels from the source_layer column when an older export emitted the wrong label.
_LAYER_TO_LABEL = {
    "taxiways": "Taxiways", "taxiway": "Taxiways",
    "footprints": "Footprints", "footprint": "Footprints",
    "runways": "Runways", "runway": "Runways",
    "lights": "Lights", "light": "Lights",
    "taxiway labels": "Taxiway Labels", "taxiwaylabels": "Taxiway Labels", "taxiway label": "Taxiway Labels",
    "runway labels": "Runway Labels", "runwaylabels": "Runway Labels", "runway label": "Runway Labels",
    "stars": "Stars", "star": "Stars",
    "layer 1": UNLABELED, "unclassified": UNLABELED,
}


def _normalize_layer(name: str) -> str:
    if not isinstance(name, str):
        return ""
    return " ".join(name.lower().split())


def relabel_from_source_layer(df: pd.DataFrame) -> pd.DataFrame:
    """Re-derive the `label` column from `source_layer` using the current
    layer-name mapping. Use this to fix older CSVs that were exported with
    a stale mapping (e.g. before "Star" → "Stars" was added)."""
    df = df.copy()
    norm = df["source_layer"].astype(str).map(_normalize_layer)
    df["label"] = norm.map(_LAYER_TO_LABEL).fillna("Other")
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
