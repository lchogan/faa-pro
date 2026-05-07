"""
Predict layer assignments for an unlabeled diagram, with confidence-banded
"Maybe X" buckets so reviewers can quickly spot ambiguous calls.

Usage:
    python predict.py --model runs/v24/model.lgb \
                      --feature-list runs/v24/feature_list.json \
                      --in unlabeled_paths.csv \
                      --pdf-text pdf_text.csv \
                      --out predictions.json

Decision logic (in order):
  1) NASR override fires (PDF text matches a confirmed FAA runway designation,
     path is letter-sized, and the model didn't already settle on a polygon
     class). If the model also picked "Runway Labels" → final "Runway Labels".
     If the model disagreed → "Maybe Runway Label" (the conflict case).
  2) Same for taxiway-pattern + on-pavement → Taxiway Labels / Maybe Taxiway Label.
  3) For everything else: argmax of model probabilities. If max prob ≥
     `--confidence` (default 0.60), route to that class. Otherwise route to
     "Maybe <class>".

Output JSON:
    {
      "airport": "xxx",
      "model_labels": ["Taxiways", ...],
      "maybe_labels": ["Maybe Taxiway", ...],
      "predictions": [
        {"airport": "xxx", "object_id": 0, "kind": "path",
         "label": "Maybe Runway Label",
         "model_top": "Other", "model_top_prob": 0.45,
         "probs": {"Taxiways": 0.01, ...}},
        ...
      ]
    }

The companion JSX (`ImportPredictedLayers.jsx`) reads this and assigns paths
to layers (creating "Maybe X" sub-layers as needed).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from load import load_features_all, LABELS
from relational import add_relational_features, load_pdf_text

# Class → its "Maybe" sibling. Other has no Maybe (Other is already the
# catch-all uncertain class).
MAYBE_OF = {
    "Taxiways":       "Maybe Taxiway",
    "Footprints":     "Maybe Footprint",
    "Runways":        "Maybe Runway",
    "Lights":         "Maybe Light",
    "Stars":          "Maybe Star",
    "Taxiway Labels": "Maybe Taxiway Label",
    "Runway Labels":  "Maybe Runway Label",
}


def apply_overrides(df: pd.DataFrame, probs: np.ndarray, confidence: float) -> tuple[list[str], list[bool]]:
    """Return (final_label_per_row, was_override_per_row).

    Logic mirrors the post-processing in train.py so the predictions you
    get here match the OOF metrics produced during training.
    """
    n = len(df)
    pred_cls = probs.argmax(axis=1)
    top_p = probs.max(axis=1)
    rl_idx = LABELS.index("Runway Labels")
    tl_idx = LABELS.index("Taxiway Labels")
    other_idx = LABELS.index("Other")

    bbox_rel = df["bbox_area_rel"].fillna(1.0).to_numpy()
    is_letter = (bbox_rel < 5e-4) & (bbox_rel < 0.5)
    pdf_match = df["pdf_text_match"].astype(str).to_numpy() if "pdf_text_match" in df.columns else np.array(["no_text"] * n)
    in_taxi_bbox = df["inside_taxiway_bbox"].fillna(0).to_numpy().astype(bool) if "inside_taxiway_bbox" in df.columns else np.zeros(n, dtype=bool)
    prob_tl = probs[:, tl_idx]
    OVERRIDE_PROB_THRESH = 0.02

    # Eligibility for each override.
    elig_runway_override = np.isin(pred_cls, [other_idx, tl_idx])
    runway_override_fires = (
        elig_runway_override & is_letter & (pdf_match == "runway_known")
    )
    elig_taxiway_override = (pred_cls == other_idx)
    taxiway_override_fires = (
        elig_taxiway_override & is_letter & (pdf_match == "taxiway")
        & in_taxi_bbox & (prob_tl > OVERRIDE_PROB_THRESH)
    )

    final_labels: list[str] = []
    overrode = []
    for i in range(n):
        if runway_override_fires[i]:
            # Override fires. If the model also picked Runway Labels (or was
            # already routing to Runway Labels via taxi-confusion), trust as
            # confident. Otherwise mark as "Maybe Runway Label" so the
            # reviewer can confirm — these are exactly the conflict cases.
            if pred_cls[i] == rl_idx:
                final_labels.append("Runway Labels")
            else:
                final_labels.append("Maybe Runway Label")
            overrode.append(True)
            continue
        if taxiway_override_fires[i]:
            if pred_cls[i] == tl_idx:
                final_labels.append("Taxiway Labels")
            else:
                final_labels.append("Maybe Taxiway Label")
            overrode.append(True)
            continue

        # No override — confidence-band the model prediction.
        cls_name = LABELS[pred_cls[i]]
        if cls_name == "Other":
            # Other is already the "uncertain" bucket; never demote.
            final_labels.append("Other")
        elif top_p[i] >= confidence:
            final_labels.append(cls_name)
        else:
            final_labels.append(MAYBE_OF.get(cls_name, "Other"))
        overrode.append(False)

    return final_labels, overrode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--feature-list", type=Path, required=True)
    ap.add_argument("--in", dest="in_path", type=Path, required=True)
    ap.add_argument("--pdf-text", dest="pdf_text_path", type=Path, default=None,
                    help="CSV from extract_pdf_text.py (auto-detect: pdf_text.csv next to --in)")
    ap.add_argument("--out", dest="out_path", type=Path, required=True)
    ap.add_argument("--confidence", type=float, default=0.60,
                    help="Top-class probability threshold for confident routing "
                         "(below this → 'Maybe X')")
    ap.add_argument("--pdf", type=Path, default=None,
                    help="Path to the source <airport>-faa.pdf. Required for "
                         "the new pdf_char_override pipeline (per-character-"
                         "bbox token detection + group spatial gates). Auto-"
                         "detect: looks for <stem-without-_paths>-faa.pdf "
                         "next to --in.")
    ap.add_argument("--nasr-rwy", type=Path,
                    default=Path(__file__).parent.parent / "data" / "nasr_apt_rwy.csv",
                    help="FAA NASR APT_RWY.csv for the runway-known check.")
    ap.add_argument("--char-model-dir", type=Path, default=None,
                    help="DEPRECATED: previously selected the char-classifier "
                         "model. The new pdf_char_override doesn't need a "
                         "trained model — left here so old shell scripts "
                         "passing --char-model-dir don't error out.")
    args = ap.parse_args()

    feature_meta = json.loads(args.feature_list.read_text())
    feature_cols = feature_meta["feature_cols"]
    cat_cols = feature_meta["categorical_cols"]
    labels = feature_meta["labels"]
    if labels != list(LABELS):
        print(f"[predict] warning: label set in model differs: {labels} vs {LABELS}")

    print(f"[predict] loading {args.in_path}")
    df = load_features_all(args.in_path)
    print(f"[predict] {len(df):,} rows across {df['airport'].nunique()} airport(s)")

    text_df = None
    pdf_text_path = args.pdf_text_path
    if pdf_text_path is None:
        guess = args.in_path.with_name("pdf_text.csv")
        if guess.exists():
            pdf_text_path = guess
            print(f"[predict] auto-detected pdf_text: {pdf_text_path}")
    if pdf_text_path is not None:
        text_df = load_pdf_text(pdf_text_path)
        print(f"[predict] {len(text_df):,} text words")

    # Auto-detect edges sidecar
    edges_path = args.in_path.with_name(args.in_path.stem + "_edges" + args.in_path.suffix)
    edges_df = None
    if edges_path.exists():
        from relational import load_edges
        edges_df = load_edges(edges_path)
        print(f"[predict] auto-detected edges: {edges_path} ({len(edges_df):,} rows)")

    df = add_relational_features(df, edges_df=edges_df, text_df=text_df)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input missing feature columns required by model: {missing}")
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype("category")

    booster = lgb.Booster(model_file=str(args.model))
    probs = booster.predict(X)
    pred_cls = probs.argmax(axis=1)
    top_p = probs.max(axis=1)

    # Locate the source PDF — required by the new override pipeline.
    pdf_path = args.pdf
    if pdf_path is None:
        # Best-effort auto-detect: <airport>-faa.pdf next to --in if --in is
        # named <airport>_paths.csv.
        airport_guess = args.in_path.stem.replace("_paths", "").lower()
        guess = args.in_path.with_name(f"{airport_guess}-faa.pdf")
        if guess.exists():
            pdf_path = guess
            print(f"[predict] auto-detected pdf: {pdf_path}")

    if pdf_path is not None and pdf_path.exists():
        from pdf_char_override import apply_pdf_char_override
        from extract_pdf_text import load_nasr_runways as _load_nasr
        nasr_full = _load_nasr(args.nasr_rwy) if args.nasr_rwy and args.nasr_rwy.exists() else {}
        airport_code = pdf_path.stem.replace("-faa", "").lower()
        nasr_for_airport = nasr_full.get(airport_code, set())
        final_labels, overrode = apply_pdf_char_override(
            df.reset_index(drop=True),
            probs,
            pdf_path,
            nasr_runways=nasr_for_airport,
            confidence=args.confidence,
            verbose=True,
        )
    else:
        print(f"[predict] no --pdf available; falling back to legacy apply_overrides")
        final_labels, overrode = apply_overrides(
            df.reset_index(drop=True), probs, args.confidence
        )

    out_records = []
    df_r = df.reset_index(drop=True)
    for i, row in df_r.iterrows():
        out_records.append({
            "airport": row["airport"],
            "object_id": int(row["object_id"]),
            "kind": str(row["kind"]),
            "label": final_labels[i],
            "model_top": labels[pred_cls[i]],
            "model_top_prob": float(top_p[i]),
            "override_applied": bool(overrode[i]),
            "probs": {labels[k]: round(float(probs[i, k]), 4) for k in range(len(labels))},
            "left": round(float(row["left"]), 4),
            "top": round(float(row["top"]), 4),
            "right": round(float(row["right"]), 4),
            "bottom": round(float(row["bottom"]), 4),
        })

    out_payload = {
        "airport": df["airport"].iloc[0] if len(df) else None,
        "model_labels": labels,
        "maybe_labels": list(MAYBE_OF.values()),
        "confidence_threshold": args.confidence,
        "predictions": out_records,
    }
    args.out_path.write_text(json.dumps(out_payload, indent=2))
    print(f"[predict] wrote {len(out_records):,} predictions -> {args.out_path}")
    counts = pd.Series([r["label"] for r in out_records]).value_counts()
    print(f"\nLabel distribution:")
    for lab, n in counts.items():
        marker = "  ?" if lab.startswith("Maybe") else "   "
        print(f"          {marker} {lab:<22} {int(n):>8,}")


if __name__ == "__main__":
    main()
