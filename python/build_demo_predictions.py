"""
Build per-airport prediction JSON files from saved OOF probabilities.

Each airport's predictions here are genuinely held-out from its training
fold (GroupKFold by airport), so this demonstrates how the model would
perform on a brand-new airport — no train-test leakage.

Usage:
    python build_demo_predictions.py \
      --oof runs/v24/oof_predictions.parquet \
      --airports cha,21d,sjc \
      --out-dir demo_predictions/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from load import LABELS
from predict import apply_overrides, MAYBE_OF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", type=Path, required=True,
                    help="oof_predictions.parquet from train.py")
    ap.add_argument("--airports", type=str, required=True,
                    help="Comma-separated list of airport codes")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--confidence", type=float, default=0.60)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df_all = pd.read_parquet(args.oof)
    targets = [a.strip().lower() for a in args.airports.split(",")]

    for airport in targets:
        sub = df_all[df_all["airport"] == airport].reset_index(drop=True)
        if len(sub) == 0:
            print(f"[demo] {airport}: NOT FOUND in OOF parquet")
            continue
        prob_cols = [f"prob_{l}" for l in LABELS]
        probs = sub[prob_cols].to_numpy(dtype=float)

        # Build a frame with just the columns apply_overrides needs.
        ov_in = pd.DataFrame({
            "bbox_area_rel": sub["bbox_area_rel"],
            "filled": sub["filled"],
            "inside_taxiway_bbox": sub["inside_taxiway_bbox"],
            "pdf_text_match": sub["pdf_text_match"],
        })
        final_labels, overrode = apply_overrides(ov_in, probs, args.confidence)

        records = []
        pred_cls = probs.argmax(axis=1)
        top_p = probs.max(axis=1)
        for i in range(len(sub)):
            records.append({
                "airport": str(sub.iloc[i]["airport"]),
                "object_id": int(sub.iloc[i]["object_id"]),
                "kind": str(sub.iloc[i]["kind"]),
                "label": final_labels[i],
                "true_label": str(sub.iloc[i]["label"]),
                "model_top": LABELS[pred_cls[i]],
                "model_top_prob": float(top_p[i]),
                "override_applied": bool(overrode[i]),
                "probs": {LABELS[k]: round(float(probs[i, k]), 4) for k in range(len(LABELS))},
            })

        payload = {
            "airport": airport,
            "model_labels": list(LABELS),
            "maybe_labels": list(MAYBE_OF.values()),
            "confidence_threshold": args.confidence,
            "predictions": records,
        }
        out = args.out_dir / f"{airport}_predictions.json"
        out.write_text(json.dumps(payload, indent=2))

        # Summary
        counts = pd.Series([r["label"] for r in records]).value_counts()
        true_counts = pd.Series([r["true_label"] for r in records]).value_counts()
        print(f"\n[demo] {airport}: wrote {len(records):,} predictions -> {out}")
        print(f"   {'class':<22} {'pred':>5}  {'true':>5}")
        for lab in list(LABELS) + list(MAYBE_OF.values()):
            p = int(counts.get(lab, 0))
            t = int(true_counts.get(lab, 0))
            if p == 0 and t == 0: continue
            marker = "  ?" if lab.startswith("Maybe") else "   "
            print(f"   {marker} {lab:<22} {p:>5}  {t:>5}")


if __name__ == "__main__":
    main()
