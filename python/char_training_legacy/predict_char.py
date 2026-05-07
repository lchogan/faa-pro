"""
predict_char.py — run the trained character classifier (Phase 3 model) on
every polygon in an airport's paths/edges CSVs and return a DataFrame with
predicted-char + confidence columns.

This is the inference half of the character-recognition pipeline. Phase 4
calls it from inside predict.py to drive the new label-override logic that
replaces the brittle PDF-text-bbox-overlap rule.

Public API:
    add_char_predictions(paths_df, edges_df, model_dir) -> DataFrame
        Adds three columns to paths_df: char_pred, char_prob, char_top3.

CLI for ad-hoc inspection:
    python predict_char.py \\
        --paths data/char_corpus/atl/atl_paths.csv \\
        --edges data/char_corpus/atl/atl_paths_edges.csv \\
        --model-dir python/runs/char/v1 \\
        --out /tmp/atl_chars.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from build_char_training import polygon_invariant_features


def add_char_predictions(
    paths_df: pd.DataFrame,
    edges_df: pd.DataFrame | None,
    model_dir: Path,
) -> pd.DataFrame:
    """Return a copy of paths_df with char_pred / char_prob / char_top3 columns.

    char_pred   one of "0".."9", "A".."Z" — argmax over the 36 classes
    char_prob   model probability of the argmax class
    char_top3   "X|Y|Z" string of the top-3 chars (descending prob)
    """
    feature_meta = json.loads((model_dir / "feature_list.json").read_text())
    feature_cols: list[str] = feature_meta["feature_cols"]
    label_classes: list[str] = feature_meta["label_classes"]
    is_binary = bool(feature_meta.get("binary", False))
    booster = lgb.Booster(model_file=str(model_dir / "model.lgb"))

    # Build the same per-polygon invariant features used at training time.
    edges_by_obj: dict[int, pd.DataFrame] = {}
    if edges_df is not None and len(edges_df) > 0:
        edges_by_obj = {int(oid): grp for oid, grp in edges_df.groupby("object_id")}

    # The training-time helper takes top_k from the column names. Recover it
    # from the saved feature list so the inference matrix matches exactly.
    seg_cols = [c for c in feature_cols if c.startswith("seg_ratio_")]
    top_k = max(int(c.split("_")[-1]) for c in seg_cols) + 1 if seg_cols else 16

    feat_rows: list[dict] = []
    for _, row in paths_df.iterrows():
        edges_for_obj = edges_by_obj.get(int(row["object_id"]))
        feat_rows.append(polygon_invariant_features(row, edges_for_obj, top_k))
    X = pd.DataFrame(feat_rows)[feature_cols].astype(np.float32)

    raw = booster.predict(X)  # (N,) for binary, (N, K) for multiclass

    out = paths_df.copy()
    if is_binary:
        # Binary detector returns P(CHAR). label_classes is ["NOT_CHAR","CHAR"].
        char_prob = raw.astype(float)
        out["char_pred"] = np.where(char_prob >= 0.5, label_classes[1], label_classes[0])
        out["char_prob"] = char_prob
        # No char identity here — top3 is just the binary call repeated for
        # schema parity with multiclass output.
        out["char_top3"] = out["char_pred"]
    else:
        top1 = raw.argmax(axis=1)
        top1_p = raw[np.arange(len(raw)), top1]
        top3_idx = np.argsort(raw, axis=1)[:, -3:][:, ::-1]
        top3_chars = [
            "|".join(label_classes[j] for j in row_idx) for row_idx in top3_idx
        ]
        out["char_pred"] = [label_classes[i] for i in top1]
        out["char_prob"] = top1_p.astype(float)
        out["char_top3"] = top3_chars
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=Path, required=True)
    ap.add_argument("--edges", type=Path, default=None,
                    help="Auto-detected from --paths if omitted")
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None,
                    help="If set, write predictions CSV here. Otherwise, stdout summary only.")
    args = ap.parse_args()

    if args.edges is None:
        guess = args.paths.with_name(args.paths.stem + "_edges" + args.paths.suffix)
        args.edges = guess if guess.exists() else None

    paths_df = pd.read_csv(args.paths)
    edges_df = pd.read_csv(args.edges) if args.edges else None
    print(f"[predict_char] {len(paths_df):,} polygons, "
          f"{0 if edges_df is None else len(edges_df):,} edges")

    out = add_char_predictions(paths_df, edges_df, args.model_dir)

    summary = out["char_pred"].value_counts().sort_index()
    print("[predict_char] predicted-char distribution:")
    for ch, n in summary.items():
        print(f"   {ch}  {int(n):>5}")

    if args.out is not None:
        out[["airport", "object_id", "char_pred", "char_prob", "char_top3"]].to_csv(
            args.out, index=False
        )
        print(f"[predict_char] wrote -> {args.out}")


if __name__ == "__main__":
    main()
