"""
char_validated_match.py — replacement for the brittle PDF-text-bbox-overlap
override used in predict.py.

The original override fired when a polygon's centroid landed inside any PDF
text token's bbox. That breaks because PDF-rendered text and Illustrator
vector geometry are offset by ~15-20pt in the Y direction on FAA charts —
chevrons get matched to runway-pair labels and footprints get matched to
taxiway labels, producing false-positive overrides.

This module replaces the spatial bbox-containment test with a *shape-based*
test: for each PDF text token T, we slice its bbox into len(T) horizontal
character slots, find the nearest polygon to each slot's center (within
token bbox + margin), and only fire the override on that polygon when the
trained character classifier confirms the polygon's shape matches the
expected character.

Public entry point:
    compute_char_validated_match(df_paths, text_df, char_model_dir,
                                  edges_df=None, margin=15.0)
        -> Series[str] indexed like df_paths, values in
           {'runway_known', 'runway_other', 'taxiway', 'no_text'}.

The returned series should overwrite the existing `pdf_text_match` column
on df_paths before apply_overrides runs. The semantic is the same — only
the *means of arriving at it* changes from "centroid-in-bbox" to
"shape-confirms-token-position".
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from predict_char import add_char_predictions

# Patterns mirror the ones in relational.py / extract_pdf_text.py so the
# token classification stays consistent across the codebase.
_RUNWAY_TEXT_RE = re.compile(r"^\d{1,2}[LRC]?$")
_TAXIWAY_TEXT_RE = re.compile(r"^[A-Z][A-Z0-9]?$")

# Token chars must be in the 36-class label space the char classifier knows.
_ALLOWED_CHARS = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _classify_token(text: str, is_known_runway: bool) -> str | None:
    """Return the override category for a token, or None if not relevant."""
    if not text:
        return None
    if _RUNWAY_TEXT_RE.match(text):
        return "runway_known" if is_known_runway else "runway_other"
    if _TAXIWAY_TEXT_RE.match(text):
        return "taxiway"
    return None


# Precedence when a polygon validates against multiple tokens — runway_known
# beats taxiway beats runway_other beats no_text. (runway_other is rarely
# acted on by apply_overrides but kept here for symmetry with the old field.)
_PRECEDENCE = {"no_text": 0, "runway_other": 1, "taxiway": 2, "runway_known": 3}


def compute_char_validated_match(
    df_paths: pd.DataFrame,
    text_df: pd.DataFrame | None,
    char_model_dir: Path,
    edges_df: pd.DataFrame | None = None,
    margin: float = 15.0,
) -> pd.Series:
    """Return char-validated pdf_text_match values for each row in df_paths.

    Args:
        df_paths: rows from load_features_all (need at least centroid_x,
            centroid_y, object_id, and the columns build_char_training/
            polygon_invariant_features expects: bbox_area, perimeter,
            num_anchors, subpath_count, closed, principal_ratio).
        text_df: rows from load_pdf_text (need text, x_min/x_max/y_min/y_max
            in Illustrator y-up frame, optional is_known_runway).
        char_model_dir: directory containing model.lgb + feature_list.json
            from train_char_classifier.py.
        edges_df: paths_edges sidecar — needed by polygon_invariant_features
            for the seg_ratio_* features. Highly recommended.
        margin: PDF-units margin around each token's bbox when collecting
            candidate polygons. Default 15.0 covers the worst-case PDF-vs-AI
            offset observed on FAA charts.
    """
    n = len(df_paths)
    out = np.full(n, "no_text", dtype=object)
    if n == 0:
        return pd.Series(out, index=df_paths.index, name="pdf_text_match")
    if text_df is None or len(text_df) == 0:
        return pd.Series(out, index=df_paths.index, name="pdf_text_match")

    # Run char classifier once for the whole dataframe — much faster than
    # per-token batches (LightGBM does its own vectorization internally).
    annotated = add_char_predictions(
        df_paths.reset_index(drop=True), edges_df, Path(char_model_dir)
    )
    char_top3 = annotated["char_top3"].to_numpy()  # "X|Y|Z" strings
    centroids = annotated[["centroid_x", "centroid_y"]].to_numpy(dtype=float)

    # Match-precedence tracker so the strongest validated category wins
    # if a polygon ends up matching multiple tokens.
    rank = np.zeros(n, dtype=int)

    text_df = text_df.reset_index(drop=True)
    text_arr = text_df["text"].astype(str).to_numpy()
    xmin = text_df["x_min"].to_numpy(dtype=float)
    xmax = text_df["x_max"].to_numpy(dtype=float)
    ymin = text_df["y_min"].to_numpy(dtype=float)
    ymax = text_df["y_max"].to_numpy(dtype=float)
    known = (
        text_df["is_known_runway"].to_numpy(dtype=int)
        if "is_known_runway" in text_df.columns
        else np.zeros(len(text_df), dtype=int)
    )

    n_validated = 0
    n_tokens_relevant = 0
    n_tokens_with_match = 0

    for ti in range(len(text_df)):
        text = text_arr[ti].strip()
        category = _classify_token(text, bool(known[ti]))
        if category is None:
            continue
        n_tokens_relevant += 1

        # Restrict the token's chars to the classifier's label space; if
        # any chars fall outside, we still attempt the alphanum subset.
        if not all(c in _ALLOWED_CHARS for c in text):
            continue

        x0, x1 = xmin[ti], xmax[ti]
        y0, y1 = ymin[ti], ymax[ti]
        if x1 <= x0 or y1 <= y0:
            continue
        n_chars = len(text)
        char_w = (x1 - x0) / n_chars
        slice_cy = (y0 + y1) / 2.0

        # Candidate polygons: centroid in token bbox + margin
        in_box = (
            (centroids[:, 0] >= x0 - margin)
            & (centroids[:, 0] <= x1 + margin)
            & (centroids[:, 1] >= y0 - margin)
            & (centroids[:, 1] <= y1 + margin)
        )
        cand = np.where(in_box)[0]
        if len(cand) == 0:
            continue
        cand_pts = centroids[cand]

        # Greedy 1-1 assignment: each char position picks the nearest
        # unused candidate polygon (matching build_char_training.py).
        used: set[int] = set()
        token_validated_any = False
        for i, expected_char in enumerate(text):
            slice_cx = x0 + (i + 0.5) * char_w
            d2 = (cand_pts[:, 0] - slice_cx) ** 2 + (cand_pts[:, 1] - slice_cy) ** 2
            order = np.argsort(d2)
            chosen = None
            for j in order:
                if int(j) not in used:
                    chosen = int(j)
                    used.add(chosen)
                    break
            if chosen is None:
                continue
            poly_idx = int(cand[chosen])
            top3_str = char_top3[poly_idx]
            if not isinstance(top3_str, str):
                continue
            top3 = top3_str.split("|")
            # Validate: char classifier's top-3 contains the expected char.
            # Top-3 (vs top-1) handles confusable pairs like O/0 and U/V
            # where the classifier may not pick the right one but still
            # ranks it highly — the PDF text already tells us which one
            # is correct, so the top-3 gate is enough to confirm the
            # polygon really is a glyph.
            if expected_char in top3:
                if _PRECEDENCE[category] > rank[poly_idx]:
                    out[poly_idx] = category
                    rank[poly_idx] = _PRECEDENCE[category]
                n_validated += 1
                token_validated_any = True
        if token_validated_any:
            n_tokens_with_match += 1

    print(
        f"[char_validated_match] tokens_relevant={n_tokens_relevant:,}  "
        f"tokens_validated={n_tokens_with_match:,}  "
        f"polygons_validated={n_validated:,}  "
        f"polygons={n}"
    )

    return pd.Series(out, index=df_paths.index, name="pdf_text_match")
