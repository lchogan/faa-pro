"""
Train a LightGBM multiclass classifier on extracted+relational features.

Usage:
    python train.py --features features.parquet --out-dir runs/v1

The features file should be the output of `relational.py`. Cross-validation
holds out *airports*, not paths, so the model can't memorize per-diagram quirks.
The final saved model is trained on all labeled rows.

Outputs in --out-dir:
    model.lgb               LightGBM booster, loadable via lightgbm.Booster
    metrics.json            per-class precision/recall/F1, macro/weighted, OOF
    confusion.csv           OOF confusion matrix
    feature_importance.csv  gain-based importance, sorted desc
    feature_list.json       exact feature columns + categorical mask (used at predict time)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from load import LABELS, UNLABELED  # noqa: F401

# Columns that are inputs to the model. Exclude ids, label, raw layer name,
# and absolute coordinates (we have normalized + relative versions instead).
ID_OR_LABEL_COLS = {
    "airport", "object_id", "source_layer", "label", "row_id",
    # absolute coords are airport-specific; we keep relative ones added by relational.py
    "left", "top", "right", "bottom",
    "centroid_x", "centroid_y",
    "artboard_left", "artboard_top", "artboard_right", "artboard_bottom",
    # All pdf_text-derived features are excluded from model training and applied
    # only as a deterministic post-processing override below. Empirically, mixing
    # them into the multiclass model hurts the polygon classes (Taxiways P drops
    # from 0.88 to 0.65) because filled-gray polygons whose centroid lands in a
    # taxiway label word get pulled into the Labels boundary.
    "pdf_text_match",
    "pdf_word_length",
    "pdf_word_dist",
    "pdf_inside_word_bbox",
    "runway_label_signature",
    "taxiway_label_signature",
}
CATEGORICAL_COLS = ("kind", "fill_kind", "stroke_kind")


def build_xy(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in ID_OR_LABEL_COLS]
    X = df[feature_cols].copy()
    for c in CATEGORICAL_COLS:
        if c in X.columns:
            X[c] = X[c].astype("category")
    y = df["label"].astype(pd.CategoricalDtype(categories=list(LABELS)))
    y_int = y.cat.codes.to_numpy()
    groups = df["airport"].to_numpy()
    cat_idx = [feature_cols.index(c) for c in CATEGORICAL_COLS if c in feature_cols]
    return X, y_int, groups, feature_cols, cat_idx


def lgb_params(num_classes: int) -> dict:
    return dict(
        objective="multiclass",
        num_class=num_classes,
        metric="multi_logloss",
        learning_rate=0.05,
        num_leaves=63,
        min_data_in_leaf=50,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        verbose=-1,
    )


def balanced_sample_weights(y: np.ndarray, num_classes: int, mode: str = "sqrt") -> np.ndarray:
    """Per-row weight schemes for handling class imbalance.

    mode="inverse"  classic balanced (weight_c = N / (K * count_c)). Most
                    aggressive — rare classes get weight ∝ 1/count. Tends to
                    over-predict minorities when other features are already
                    informative.
    mode="sqrt"     compressed: max-count class gets 1.0, others get
                    sqrt(max_count / count_c). Gentler boost, keeps the model
                    from going crazy on highly informative features.
    mode="none"     equivalent to no weighting (all 1s).
    """
    counts = np.bincount(y, minlength=num_classes)
    counts = np.maximum(counts, 1)
    if mode == "inverse":
        class_w = len(y) / (num_classes * counts)
    elif mode == "sqrt":
        class_w = np.sqrt(counts.max() / counts)
    elif mode == "none":
        class_w = np.ones(num_classes, dtype=float)
    else:
        raise ValueError(f"Unknown weighting mode: {mode}")
    return class_w[y].astype(np.float32)


def cross_val(X, y, groups, cat_idx, n_splits=5, num_boost_round=600, early_stopping=50,
              weight_mode: str = "sqrt"):
    gkf = GroupKFold(n_splits=n_splits)
    oof_pred = np.full((len(y), len(LABELS)), np.nan)
    fold_metrics = []
    n_unique_groups = len(np.unique(groups))
    n_splits = min(n_splits, n_unique_groups)
    print(f"[train] CV with {n_splits} airport-grouped folds across {n_unique_groups} airports"
          + (f"  (weight_mode={weight_mode})" if weight_mode != "none" else ""))

    for fold, (tr, va) in enumerate(gkf.split(X, y, groups), start=1):
        tr_w = balanced_sample_weights(y[tr], len(LABELS), mode=weight_mode) if weight_mode != "none" else None
        tr_set = lgb.Dataset(X.iloc[tr], y[tr], weight=tr_w,
                             categorical_feature=cat_idx, free_raw_data=False)
        va_set = lgb.Dataset(X.iloc[va], y[va],
                             categorical_feature=cat_idx, reference=tr_set, free_raw_data=False)
        model = lgb.train(
            lgb_params(len(LABELS)),
            tr_set,
            num_boost_round=num_boost_round,
            valid_sets=[va_set],
            callbacks=[lgb.early_stopping(early_stopping), lgb.log_evaluation(0)],
        )
        pred = model.predict(X.iloc[va], num_iteration=model.best_iteration)
        oof_pred[va] = pred
        pred_cls = pred.argmax(axis=1)
        f1 = f1_score(y[va], pred_cls, average="macro")
        print(f"[train] fold {fold}: best_iter={model.best_iteration}  macro-F1={f1:.4f}")
        fold_metrics.append({"fold": fold, "best_iter": int(model.best_iteration), "macro_f1": float(f1)})
    return oof_pred, fold_metrics


def train_final(X, y, cat_idx, num_boost_round: int, weight_mode: str = "sqrt") -> lgb.Booster:
    w = balanced_sample_weights(y, len(LABELS), mode=weight_mode) if weight_mode != "none" else None
    full_set = lgb.Dataset(X, y, weight=w, categorical_feature=cat_idx, free_raw_data=False)
    model = lgb.train(
        lgb_params(len(LABELS)),
        full_set,
        num_boost_round=num_boost_round,
    )
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True, help="Input parquet/csv from relational.py")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--early-stopping", type=int, default=50)
    ap.add_argument("--no-class-weights", action="store_true",
                    help="Disable balanced sample weighting (alias for --weight-mode none)")
    ap.add_argument("--weight-mode", choices=("inverse", "sqrt", "none"), default="sqrt",
                    help="Sample-weight scheme for class imbalance (default: sqrt)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.features.suffix.lower() == ".parquet":
        df = pd.read_parquet(args.features)
    else:
        df = pd.read_csv(args.features)
    df = df[df["label"].isin(LABELS)].reset_index(drop=True)
    print(f"[train] labeled rows: {len(df):,}  airports: {df['airport'].nunique()}")
    print("[train] label distribution:")
    for lab, n in df["label"].value_counts().reindex(LABELS).items():
        print(f"          {lab:<12} {int(n):>8,}")

    X, y, groups, feature_cols, cat_idx = build_xy(df)

    weight_mode = "none" if args.no_class_weights else args.weight_mode
    oof, fold_metrics = cross_val(
        X, y, groups, cat_idx,
        n_splits=args.folds,
        num_boost_round=args.rounds,
        early_stopping=args.early_stopping,
        weight_mode=weight_mode,
    )

    # OOF metrics
    oof_pred_cls = np.where(np.isnan(oof[:, 0]), -1, oof.argmax(axis=1))
    mask = oof_pred_cls >= 0
    rep = classification_report(y[mask], oof_pred_cls[mask], target_names=list(LABELS), output_dict=True, zero_division=0)
    cm = confusion_matrix(y[mask], oof_pred_cls[mask], labels=list(range(len(LABELS))))
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in LABELS], columns=[f"pred_{l}" for l in LABELS])
    cm_df.to_csv(args.out_dir / "confusion.csv")

    # ----- post-processing: pdf_text override -----
    # Override "model said Other" → matching Label class when:
    #   1) the path's centroid is inside a PDF word matching the runway /
    #      taxiway designation pattern,
    #   2) the path is letter-sized and filled,
    #   3) it's geometrically near the relevant pavement: for runway labels
    #      that means within ~1.5x of the runway's half-length along its
    #      axis (i.e. within or just past a runway end); for taxiway labels
    #      it means the centroid is inside a taxiway-candidate bbox.
    # The geometric gate is what stops random digits in notes from being
    # promoted to runway labels.
    rep_pp = None
    if "pdf_text_match" in df.columns and "bbox_area_rel" in df.columns:
        OVERRIDE_PROB_THRESH = 0.02
        oof_pp = oof_pred_cls.copy()
        rl_idx = LABELS.index("Runway Labels")
        tl_idx = LABELS.index("Taxiway Labels")
        other_idx = LABELS.index("Other")
        # is_letter: small relative to artboard. Loosened from 1e-4 to 5e-4
        # because legitimate runway-label glyphs at large airports can sit
        # just over the tighter threshold. Excludes the rare pathological
        # full-artboard outliers via the rel < 0.5 cap.
        bbox_rel = df["bbox_area_rel"].fillna(1.0).to_numpy()
        is_letter = (bbox_rel < 5e-4) & (bbox_rel < 0.5)
        # NB: do NOT require is_filled. Letters with holes ("0", "8", "B",
        # "9", "6", etc.) export as CompoundPathItems whose .filled flag
        # is false due to Illustrator's even-odd fill rule. ~36% of true
        # Runway Labels in the corpus are exactly these compound letters.
        pdf_match = df["pdf_text_match"].astype(str).to_numpy()
        was_other = (oof_pp == other_idx)
        was_taxiway_label = (oof_pp == tl_idx)
        prob_rl = oof[:, rl_idx]
        prob_tl = oof[:, tl_idx]
        in_taxi_bbox = df["inside_taxiway_bbox"].fillna(0).to_numpy().astype(bool)
        # Runway-label override: must look like a runway label both to the
        # text (pdf word matches runway pattern) and to the model (non-trivial
        # prob given to Runway Labels). Combined probability gate filters out
        # incidental digits in notes far from any runway.
        # Strict path: word matches a confirmed FAA runway designation for
        # this airport (NASR APT_RWY.csv). NASR text is authoritative, so we
        # trust it over the model regardless of what class the model picked
        # — including Taxiway Labels (the model frequently confuses runway
        # vs taxiway letter glyphs since they look identical per-character).
        # We also drop the probability gate here: when the FAA-confirmed
        # text says "27L", the glyph IS part of a runway designation, full
        # stop. Restrictions: must be letter-sized, and the model wasn't
        # already calling it Runway Labels (no-op) or one of the polygon
        # classes (model got the polygon class right by other means).
        # No geometric gate: we tried `(rax < 2.0) | inside_runway_bbox` but
        # it lost 32 true labels (recall 0.972 → 0.843) without proportional
        # precision gain. The data shows true labels sit *just outside* the
        # runway bbox at the threshold, so geometric proximity isn't a clean
        # discriminator with the features we have.
        rax = df["runway_axis_position"].fillna(np.inf).to_numpy()
        eligible_for_runway_override = was_other | was_taxiway_label
        runway_strict = (
            eligible_for_runway_override & is_letter
            & (pdf_match == "runway_known")
        )
        # Fallback path: airport not in NASR (rare — NASR covers ~19K) or
        # word matches runway pattern but not airport's NASR list. Tighter
        # geometric gate (near a runway end) since text isn't authoritative.
        near_runway_end = (rax > 0.7) & (rax < 1.5)
        runway_fallback = (
            was_other & is_letter
            & (pdf_match == "runway_other")
            & (prob_rl > OVERRIDE_PROB_THRESH)
            & near_runway_end
        )
        oof_pp = np.where(runway_strict | runway_fallback, rl_idx, oof_pp)
        # Taxiway-label override: combined gate — pdf-text + on-taxiway + the
        # model gave Taxiway Labels at least some probability. Each gate alone
        # is too permissive; the conjunction is what isolates real Taxiway
        # Labels from incidental letters in non-pavement contexts.
        oof_pp = np.where(
            was_other & is_letter & (pdf_match == "taxiway")
            & in_taxi_bbox & (prob_tl > OVERRIDE_PROB_THRESH),
            tl_idx, oof_pp,
        )
        rep_pp = classification_report(y[mask], oof_pp[mask], target_names=list(LABELS), output_dict=True, zero_division=0)
        cm_pp = confusion_matrix(y[mask], oof_pp[mask], labels=list(range(len(LABELS))))
        pd.DataFrame(cm_pp, index=[f"true_{l}" for l in LABELS],
                     columns=[f"pred_{l}" for l in LABELS]).to_csv(args.out_dir / "confusion_postprocessed.csv")

    metrics = {
        "fold_metrics": fold_metrics,
        "oof_classification_report": rep,
        "oof_classification_report_postprocessed": rep_pp,
        "labels": list(LABELS),
        "n_rows_labeled": int(len(df)),
        "n_airports": int(df["airport"].nunique()),
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Final model on all data — average best_iter across folds, rounded up
    avg_best_iter = int(np.ceil(np.mean([m["best_iter"] for m in fold_metrics]))) if fold_metrics else args.rounds
    print(f"[train] training final model on all {len(df):,} rows for {avg_best_iter} rounds")
    model = train_final(X, y, cat_idx, num_boost_round=avg_best_iter, weight_mode=weight_mode)
    model.save_model(str(args.out_dir / "model.lgb"))

    # Feature importance
    imp = pd.DataFrame({
        "feature": feature_cols,
        "gain": model.feature_importance(importance_type="gain"),
        "split": model.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)
    imp.to_csv(args.out_dir / "feature_importance.csv", index=False)

    # Feature list with categorical mask, for predict.py
    (args.out_dir / "feature_list.json").write_text(json.dumps({
        "feature_cols": feature_cols,
        "categorical_cols": [c for c in CATEGORICAL_COLS if c in feature_cols],
        "labels": list(LABELS),
    }, indent=2))

    # Save OOF probabilities + per-row metadata so demo predictions can be
    # built per airport without retraining (each airport's predictions are
    # genuinely held-out from its OOF fold).
    oof_df = df[["airport", "object_id", "kind", "label",
                 "bbox_area_rel", "filled", "inside_taxiway_bbox"]].copy()
    if "pdf_text_match" in df.columns:
        oof_df["pdf_text_match"] = df["pdf_text_match"].astype(str)
    for k, lab in enumerate(LABELS):
        oof_df[f"prob_{lab}"] = oof[:, k]
    oof_df.to_parquet(args.out_dir / "oof_predictions.parquet", index=False)

    # Print headline numbers
    def _print_report(title, r):
        print(f"\n=== {title} ===")
        for lab in LABELS:
            m = r[lab]
            print(f"  {lab:<14}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1-score']:.3f}  N={int(m['support'])}")
        print(f"  macro-F1     {r['macro avg']['f1-score']:.3f}")
        print(f"  weighted-F1  {r['weighted avg']['f1-score']:.3f}")

    _print_report("OOF results (model only)", rep)
    if rep_pp is not None:
        _print_report("OOF results (with pdf_text override)", rep_pp)
    print(f"\nArtifacts written to {args.out_dir}/")


if __name__ == "__main__":
    main()
