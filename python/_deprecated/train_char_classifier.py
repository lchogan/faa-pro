"""
train_char_classifier.py — Phase 3 of the character-recognition build plan.

Trains a 36-class character classifier (A-Z, 0-9) on the distant-supervision
parquet built by build_char_training.py.

The training data is *noisy by construction*: each row's label is the PDF-text
character at the position of the polygon nearest to that character's slice
center. Some rows are mislabeled — but with thousands of examples per common
character, the dominant signal is the actual letter shape, and a tree-based
classifier is robust to a steady noise floor.

Held-out evaluation is **airport-grouped**: all polygons from the same airport
go to the same fold. A random row-level split would leak per-airport rendering
quirks (line thickness, anti-aliasing, exact glyph variant) into the test set
and overstate accuracy.

Usage:
  python train_char_classifier.py \\
    --in /path/to/char_training.parquet \\
    --out-dir python/runs/char/v1 \\
    --val-frac 0.15

Outputs (under --out-dir):
  model.lgb            LightGBM Booster, multiclass softmax over 36 chars
  feature_list.json    {feature_cols, label_classes} — for inference parity
  metrics.json         {top1, top3, per_class_acc, val_airports}
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

NEGATIVE_CLASS = "_NOT_"
# Multiclass mode: 36 character classes + the explicit "not a character"
# reject class. When build_char_training.py is run with --neg-ratio 0 the
# parquet contains no negatives, in which case _NOT_ is harmlessly absent
# at inference time.
ALL_CHARS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [NEGATIVE_CLASS]
# Binary-mode label set — the only thing the detector decides.
BINARY_CHARS = ["NOT_CHAR", "CHAR"]
META_COLS = {"airport", "object_id", "token", "char_pos", "char"}


def split_by_airport(df: pd.DataFrame, val_frac: float, seed: int) -> tuple[np.ndarray, list[str]]:
    """Return (is_val mask, val_airport_codes). Holds out whole airports."""
    rng = random.Random(seed)
    airports = sorted(df["airport"].unique().tolist())
    rng.shuffle(airports)
    n_val = max(1, int(round(len(airports) * val_frac)))
    val_set = set(airports[:n_val])
    is_val = df["airport"].isin(val_set).to_numpy()
    return is_val, sorted(val_set)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=Path, required=True,
                    help="Parquet from build_char_training.py")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="Fraction of airports to hold out for evaluation")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--num-leaves", type=int, default=63)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--num-rounds", type=int, default=600)
    ap.add_argument("--early-stopping", type=int, default=40)
    ap.add_argument("--binary", action="store_true",
                    help="Train a CHAR-vs-NOT_CHAR binary detector instead "
                         "of the 36-way recognizer. The parquet must contain "
                         "negative examples (label == _NOT_).")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.in_path)
    print(f"[train] loaded {len(df):,} rows from {args.in_path}")
    n_airports = df["airport"].nunique()
    print(f"[train] {n_airports} airports, "
          f"{df['char'].nunique()} distinct char labels")

    # --- features and labels ---
    feature_cols = [c for c in df.columns if c not in META_COLS]

    if args.binary:
        # Binary detector: every original letter label collapses to CHAR (1),
        # NEGATIVE_CLASS collapses to NOT_CHAR (0). All non-canonical labels
        # are dropped. This is the model the visualization tool uses to
        # answer the only question the user wants right now: "is this
        # polygon a character glyph at all?"
        valid = df["char"].isin(set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ") | {NEGATIVE_CLASS})
        if (~valid).any():
            print(f"[train] dropping {(~valid).sum()} rows with unrecognized labels")
        df = df[valid].reset_index(drop=True)
        y = (df["char"] != NEGATIVE_CLASS).to_numpy(dtype=np.int32)
        label_classes = BINARY_CHARS
        print(f"[train] binary mode: {(y == 1).sum():,} CHAR / {(y == 0).sum():,} NOT_CHAR")
    else:
        label_to_idx = {c: i for i, c in enumerate(ALL_CHARS)}
        in_set = df["char"].isin(label_to_idx)
        if (~in_set).any():
            print(f"[train] dropping {(~in_set).sum()} rows with off-set chars")
        df = df[in_set].reset_index(drop=True)
        y = df["char"].map(label_to_idx).to_numpy(dtype=np.int32)
        label_classes = ALL_CHARS

    X = df[feature_cols].astype(np.float32)

    # --- airport-grouped split ---
    is_val, val_airports = split_by_airport(df, args.val_frac, args.seed)
    n_train = (~is_val).sum()
    n_val = is_val.sum()
    print(f"[train] split: {n_train:,} train rows / {n_val:,} val rows "
          f"(val airports = {len(val_airports)} of {n_airports})")

    X_train, y_train = X[~is_val], y[~is_val]
    X_val, y_val = X[is_val], y[is_val]

    # --- train ---
    train_set = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    val_set = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=train_set)

    if args.binary:
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": args.learning_rate,
            "num_leaves": args.num_leaves,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 5,
            "verbose": -1,
            "is_unbalance": True,
        }
    else:
        params = {
            "objective": "multiclass",
            "num_class": len(ALL_CHARS),
            "metric": "multi_logloss",
            "learning_rate": args.learning_rate,
            "num_leaves": args.num_leaves,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 5,
            "verbose": -1,
            "is_unbalance": True,
        }

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=args.num_rounds,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(args.early_stopping),
            lgb.log_evaluation(period=20),
        ],
    )

    # --- evaluate ---
    val_probs = booster.predict(X_val, num_iteration=booster.best_iteration)
    if args.binary:
        # LightGBM returns shape (N,) for binary objective — probability of
        # the positive class (label==1, i.e. CHAR).
        pred_idx = (val_probs >= 0.5).astype(np.int32)
        top1 = (pred_idx == y_val).mean()
        # For binary we report precision/recall on the CHAR class — that
        # tells the user "of polygons we said are characters, how many
        # actually were" (precision) and "of true characters, how many
        # we caught" (recall).
        tp = int(((pred_idx == 1) & (y_val == 1)).sum())
        fp = int(((pred_idx == 1) & (y_val == 0)).sum())
        fn = int(((pred_idx == 0) & (y_val == 1)).sum())
        tn = int(((pred_idx == 0) & (y_val == 0)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        print("")
        print(f"[train] best iter = {booster.best_iteration}")
        print(f"[train] CHAR detector — accuracy={top1:.3f}")
        print(f"[train]   CHAR precision={precision:.3f}  recall={recall:.3f}  F1={f1:.3f}")
        print(f"[train]   confusion: TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}")
        metrics_payload = {
            "binary": True,
            "top1_val_acc": float(top1),
            "char_precision": float(precision),
            "char_recall": float(recall),
            "char_f1": float(f1),
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "val_airports": val_airports,
            "n_train": int(n_train),
            "n_val": int(n_val),
            "best_iteration": int(booster.best_iteration or 0),
        }
    else:
        pred_idx = val_probs.argmax(axis=1)
        top1 = (pred_idx == y_val).mean()
        top3 = np.mean([
            y_val[i] in val_probs[i].argsort()[-3:]
            for i in range(len(y_val))
        ])
        label_to_idx = {c: i for i, c in enumerate(ALL_CHARS)}
        per_class = {}
        for ch, idx in label_to_idx.items():
            mask = y_val == idx
            if mask.sum() == 0:
                per_class[ch] = {"n": 0, "acc": None}
                continue
            correct = (pred_idx[mask] == idx).mean()
            per_class[ch] = {"n": int(mask.sum()), "acc": float(correct)}
        print("")
        print(f"[train] best iter = {booster.best_iteration}")
        print(f"[train] top-1 val accuracy: {top1:.3f}")
        print(f"[train] top-3 val accuracy: {top3:.3f}")
        print("[train] per-class accuracy (sorted by support desc):")
        for ch, m in sorted(per_class.items(), key=lambda kv: -(kv[1]["n"] or 0)):
            if m["n"] == 0:
                continue
            bar = "#" * int((m["acc"] or 0) * 30)
            print(f"   {ch}  n={m['n']:>5}  acc={m['acc']:.3f}  {bar}")
        metrics_payload = {
            "binary": False,
            "top1_val_acc": float(top1),
            "top3_val_acc": float(top3),
            "per_class_acc": per_class,
            "val_airports": val_airports,
            "n_train": int(n_train),
            "n_val": int(n_val),
            "best_iteration": int(booster.best_iteration or 0),
        }

    # --- save ---
    model_path = args.out_dir / "model.lgb"
    booster.save_model(str(model_path), num_iteration=booster.best_iteration)
    (args.out_dir / "feature_list.json").write_text(json.dumps({
        "feature_cols": feature_cols,
        "label_classes": label_classes,
        "binary": args.binary,
    }, indent=2))
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))
    print(f"\n[train] saved -> {args.out_dir}")


if __name__ == "__main__":
    main()
