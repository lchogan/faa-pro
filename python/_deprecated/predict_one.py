"""
End-to-end inference on a single airport you didn't train on.

After you've exported the paths CSV from Illustrator (ExportClassifiedPaths.jsx
on a folder containing just that airport's <code>-diagram.ai), this wrapper
chains the rest: PDF-text extraction → relational features → predict.

Usage:
    python predict_one.py \
      --paths /path/to/cmi_paths.csv \
      --airport-folder /path/to/airports/cmi \
      --out /path/to/cmi_predictions.json

Auto-detects paths_edges.csv next to --paths and the FAA-faa.pdf in
--airport-folder. Uses the bundled NASR runway file by default.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from _deprecated.predict import apply_overrides, MAYBE_OF
from ml.load import load_features_all, LABELS
from ml.relational import add_relational_features, load_edges, load_pdf_text
from pipeline.extract_pdf_text import _normalize_designation, find_known_runway_designations, load_nasr_runways


def extract_pdf_text_for_one(pdf_path: Path, nasr_csv: Path, out_csv: Path) -> int:
    """Single-airport version of extract_pdf_text.py."""
    import fitz
    nasr_runways = load_nasr_runways(nasr_csv) if nasr_csv and nasr_csv.exists() else {}
    airport = pdf_path.stem.replace("-faa", "").lower()
    doc = fitz.open(pdf_path)
    page = doc[0]
    ph = float(page.rect.height)
    pw = float(page.rect.width)
    words = page.get_text("words")
    known_runways = nasr_runways.get(airport)
    source = "NASR"
    if not known_runways:
        known_runways = find_known_runway_designations(words)
        source = "chart-pair fallback"

    n = 0
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["airport", "text", "x_min", "y_min", "x_max", "y_max",
                    "page_height", "page_width", "is_known_runway"])
        for word in words:
            x0, y0, x1, y1, text, *_ = word
            if not text or not text.strip():
                continue
            norm = _normalize_designation(text)
            is_known = bool(norm and norm in known_runways)
            w.writerow([
                airport, text,
                f"{float(x0):.4f}", f"{ph - float(y1):.4f}",
                f"{float(x1):.4f}", f"{ph - float(y0):.4f}",
                f"{ph:.4f}", f"{pw:.4f}", int(is_known),
            ])
            n += 1
    doc.close()
    print(f"[predict_one] extracted {n} PDF words ({len(known_runways)} known runways via {source}: {sorted(known_runways)})")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=Path, required=True,
                    help="CSV from ExportClassifiedPaths.jsx for the single airport")
    ap.add_argument("--airport-folder", type=Path, required=True,
                    help="Folder with <code>-faa.pdf for PDF-text extraction")
    ap.add_argument("--model", type=Path,
                    default=Path(__file__).parent / "runs" / "v24" / "model.lgb")
    ap.add_argument("--feature-list", type=Path,
                    default=Path(__file__).parent / "runs" / "v24" / "feature_list.json")
    ap.add_argument("--nasr-rwy", type=Path,
                    default=Path(__file__).parent.parent / "data" / "nasr_apt_rwy.csv")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--confidence", type=float, default=0.60)
    ap.add_argument("--char-model-dir", type=Path,
                    default=Path(__file__).parent / "runs" / "char" / "v1",
                    help="Trained character classifier (Phase 3 / runs/char/v1). "
                         "Pass empty string to fall back to legacy bbox override.")
    args = ap.parse_args()

    # Locate the FAA PDF inside the airport folder
    pdfs = list(args.airport_folder.glob("*-faa.pdf"))
    if not pdfs:
        raise SystemExit(f"No *-faa.pdf in {args.airport_folder}")
    pdf_path = pdfs[0]

    # 1) PDF text
    pdf_text_csv = args.out.with_suffix(".pdf_text.csv")
    print(f"[predict_one] step 1/3 — extracting PDF text from {pdf_path}")
    extract_pdf_text_for_one(pdf_path, args.nasr_rwy, pdf_text_csv)

    # 2) Relational features
    print(f"[predict_one] step 2/3 — building features from {args.paths}")
    df = load_features_all(args.paths)
    edges_csv = args.paths.with_name(args.paths.stem + "_edges" + args.paths.suffix)
    edges_df = load_edges(edges_csv) if edges_csv.exists() else None
    if edges_df is not None:
        print(f"[predict_one]   edges: {len(edges_df):,} rows")
    text_df = load_pdf_text(pdf_text_csv)
    df = add_relational_features(df, edges_df=edges_df, text_df=text_df)

    # 3) Predict + apply overrides + confidence-band into Maybe buckets
    print(f"[predict_one] step 3/3 — predicting with model {args.model}")
    feature_meta = json.loads(args.feature_list.read_text())
    feature_cols = feature_meta["feature_cols"]
    cat_cols = feature_meta["categorical_cols"]
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype("category")
    booster = lgb.Booster(model_file=str(args.model))
    probs = booster.predict(X)

    # New override pipeline (replaces the broken pdf_text_match bbox check
    # AND the char-classifier detour): pdf_char_override uses PyMuPDF's
    # per-character bboxes to map polygons to PDF text tokens, color
    # pre-classifies taxiway pavement, and gates runway/taxiway-label
    # overrides on group centroids.
    from pdf_char_override import apply_pdf_char_override
    nasr_runways_full = (
        load_nasr_runways(args.nasr_rwy)
        if args.nasr_rwy and args.nasr_rwy.exists() else {}
    )
    airport_code = pdf_path.stem.replace("-faa", "").lower()
    nasr_for_airport = nasr_runways_full.get(airport_code, set())
    final_labels, overrode = apply_pdf_char_override(
        df.reset_index(drop=True),
        probs,
        pdf_path,
        nasr_runways=nasr_for_airport,
        confidence=args.confidence,
        verbose=True,
    )
    df_r = df.reset_index(drop=True)
    pred_cls = probs.argmax(axis=1)
    top_p = probs.max(axis=1)
    records = [{
        "airport": str(df_r.iloc[i]["airport"]),
        "object_id": int(df_r.iloc[i]["object_id"]),
        "kind": str(df_r.iloc[i]["kind"]),
        "label": final_labels[i],
        "model_top": LABELS[pred_cls[i]],
        "model_top_prob": float(top_p[i]),
        "override_applied": bool(overrode[i]),
        # Bbox in AI y-up coords. ImportPredictedLayers.jsx uses these to
        # match each Illustrator path/compound to its prediction by spatial
        # proximity — needed because PyMuPDF's drawing order differs from
        # Illustrator's pathItems/compoundPathItems enumeration order.
        "left": round(float(df_r.iloc[i]["left"]), 4),
        "top": round(float(df_r.iloc[i]["top"]), 4),
        "right": round(float(df_r.iloc[i]["right"]), 4),
        "bottom": round(float(df_r.iloc[i]["bottom"]), 4),
    } for i in range(len(df_r))]

    payload = {
        "airport": df_r["airport"].iloc[0] if len(df_r) else None,
        "model_labels": list(LABELS),
        "maybe_labels": list(MAYBE_OF.values()),
        "confidence_threshold": args.confidence,
        "predictions": records,
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\n[predict_one] wrote {len(records):,} predictions -> {args.out}")
    counts = pd.Series([r["label"] for r in records]).value_counts()
    print("\nLabel distribution:")
    for lab in list(LABELS) + list(MAYBE_OF.values()):
        n = int(counts.get(lab, 0))
        if n == 0: continue
        marker = "  ?" if lab.startswith("Maybe") else "   "
        print(f"   {marker} {lab:<22} {n:>6,}")


if __name__ == "__main__":
    main()
