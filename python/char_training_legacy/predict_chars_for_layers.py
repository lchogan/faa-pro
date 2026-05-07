"""
predict_chars_for_layers.py — single-airport char-classifier inference.

Reads the paths CSV produced by ClassifyAirport.jsx (and the edges sidecar),
runs the v1 character classifier on every polygon, and writes a JSON file
keyed by JSX object_id that the matching JSX (CharLayerImport.jsx) consumes
to assign each path to a per-character layer.

This is the standalone "show me what the char classifier thinks each polygon
is" visualization tool — it does NOT touch the 7-class production pipeline.

Output JSON shape:
    {
      "airport": "atl",
      "label_classes": ["0", "1", ..., "9", "A", ..., "Z", "Not a Character"],
      "predictions": [
        {"object_id": 0, "char_pred": "A", "char_prob": 0.92,
         "char_top3": "A|R|N"},
        ...
      ]
    }

Usage:
    python predict_chars_for_layers.py \\
        --paths /tmp/atl_paths.csv \\
        --edges /tmp/atl_paths_edges.csv \\
        --out   /tmp/atl_char_predictions.json \\
        --reject-prob 0.30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from predict_char import add_char_predictions

# Paths whose top-1 probability is below this threshold are routed to a
# "Not a Character" layer in the visualization. The classifier was only
# trained on letter-shaped polygons, so confidence on non-letters tends to
# be diffuse — a soft reject keeps the per-letter layers cleaner for visual
# inspection.
DEFAULT_REJECT_PROB = 0.30
NOT_A_CHARACTER_LABEL = "Not a Character"
NEGATIVE_CLASS = "_NOT_"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=Path, required=True)
    ap.add_argument("--edges", type=Path, default=None,
                    help="Auto-detected from --paths if omitted")
    ap.add_argument("--model-dir", type=Path,
                    default=Path(__file__).parent / "runs" / "char" / "v1")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reject-prob", type=float, default=DEFAULT_REJECT_PROB,
                    help=f"Top-1 probability below this routes to '{NOT_A_CHARACTER_LABEL}'")
    args = ap.parse_args()

    if args.edges is None:
        guess = args.paths.with_name(args.paths.stem + "_edges" + args.paths.suffix)
        args.edges = guess if guess.exists() else None

    paths_df = pd.read_csv(args.paths)
    edges_df = pd.read_csv(args.edges) if args.edges else None
    print(f"[predict_chars_for_layers] {len(paths_df):,} polygons, "
          f"{0 if edges_df is None else len(edges_df):,} edges")

    if not (args.model_dir / "model.lgb").exists():
        raise SystemExit(f"char model not found: {args.model_dir}/model.lgb")

    annotated = add_char_predictions(paths_df, edges_df, args.model_dir)

    # Apply soft-reject threshold for visualization sanity.
    rejected = (annotated["char_prob"] < args.reject_prob).sum()
    print(f"[predict_chars_for_layers] reject-prob={args.reject_prob}; "
          f"{rejected:,}/{len(annotated):,} polygons routed to "
          f"'{NOT_A_CHARACTER_LABEL}'")

    label_classes = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [NOT_A_CHARACTER_LABEL]
    records = []
    for _, row in annotated.iterrows():
        char = str(row["char_pred"])
        prob = float(row["char_prob"])
        if char == NEGATIVE_CLASS or prob < args.reject_prob:
            label = NOT_A_CHARACTER_LABEL
        else:
            label = char
        records.append({
            "object_id": int(row["object_id"]),
            "char_pred": str(char),
            "char_prob": round(prob, 4),
            "char_top3": str(row["char_top3"]),
            "layer": label,
        })

    payload = {
        "airport": str(paths_df["airport"].iloc[0]) if len(paths_df) else None,
        "reject_prob": args.reject_prob,
        "label_classes": label_classes,
        "predictions": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"[predict_chars_for_layers] wrote {len(records):,} predictions -> {args.out}")

    counts = pd.Series([r["layer"] for r in records]).value_counts()
    print("[predict_chars_for_layers] layer distribution:")
    for layer in label_classes:
        n = int(counts.get(layer, 0))
        if n == 0:
            continue
        bar = "#" * min(40, n // max(1, len(records) // 200))
        print(f"   {layer:<18} {n:>5}  {bar}")


if __name__ == "__main__":
    main()
