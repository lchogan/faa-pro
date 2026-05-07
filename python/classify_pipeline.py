"""
classify_pipeline.py — single-pass orchestrator for the new pipeline.

Steps in strict order. Polygons claimed in earlier steps are removed
from the pool seen by every later step.

  1+2. Taxiways / Taxi Labels    rule-based (gray-fill detection +
                                 token regex K-nearest gated on
                                 bbox-touches-taxi-surface).
  3.   Runways                    rule-based (NASR runway count +
                                  top-N polygons by polygon area among
                                  filled-near-black or stroked
                                  candidates, after taxi removal).
                                  Replaces ML-based runway prediction.
  4.   ML                         on the remaining unclaimed polygons.
                                  Classes Taxiways / Taxiway Labels /
                                  Runway Labels / Runways are masked
                                  out of ML's probability matrix so an
                                  unclaimed polygon can't fall back
                                  into them.
  5.   Centerline-token search    For each rule-claimed Runway, extend
                                  a thin centerline past each end. The
                                  closest runway-pattern text token
                                  whose centroid sits within a
                                  progressively-widened band along the
                                  centerline picks the label. Claim K =
                                  len(token) nearest near-black filled
                                  unclaimed polygons -> Runway Labels.

Output: predictions JSON in the same format as predict_one.py — one
record per polygon, with bbox in AI y-up coordinates so
ImportPredictedLayers.jsx can match each Illustrator path.

Usage:
    python classify_pipeline.py \\
        --paths /path/to/<airport>_paths.csv \\
        --pdf /path/to/<airport>-faa.pdf \\
        --out /path/to/<airport>_predictions.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import fitz
import lightgbm as lgb
import numpy as np
import pandas as pd

from extract_pdf_text import (
    _normalize_designation,
    find_known_runway_designations,
    load_nasr_runways,
)
from load import LABELS, load_features_all
from pdf_char_override import apply_pdf_char_override
from predict_one import extract_pdf_text_for_one
from relational import add_relational_features, load_edges, load_pdf_text
from runway_detection import detect_runways
from taxi_detection import detect_taxi


RUNWAY_RE = re.compile(r"^(0?[1-9]|[12][0-9]|3[0-6])[LRC]?$")
# Step 5: centerline-band widths used to search for runway-name tokens
# past each runway end. We walk the widths in order and stop at the
# first width that yields a candidate. 1pt is effectively "directly on
# the centerline"; we expand to 100pt before giving up on this end.
CENTERLINE_WIDTHS_PT = (1.0, 10.0, 25.0, 50.0, 100.0)
# Runways now come from the rule-based detector (step 3), so ML must
# not assign them either.
ML_EXCLUDE_CLASSES = ("Taxiways", "Taxiway Labels", "Runway Labels", "Runways")


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _runway_axis(subpaths, fallback_rect):
    """PCA principal axis of a runway polygon's anchor points. Returns
    (cx, cy, ux, uy, half_len) where ux,uy is a unit vector along the
    axis and half_len is the longitudinal half-extent."""
    pts: list[tuple[float, float]] = []
    for ring in subpaths or []:
        pts.extend(ring)
    x0, y0, x1, y1 = fallback_rect
    if len(pts) < 3:
        if (x1 - x0) >= (y1 - y0):
            return ((x0 + x1) / 2, (y0 + y1) / 2, 1.0, 0.0, (x1 - x0) / 2)
        return ((x0 + x1) / 2, (y0 + y1) / 2, 0.0, 1.0, (y1 - y0) / 2)
    arr = np.asarray(pts, dtype=float)
    centroid = arr.mean(axis=0)
    centered = arr - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    norm = float(np.linalg.norm(direction)) or 1.0
    ux, uy = float(direction[0]) / norm, float(direction[1]) / norm
    proj = centered @ np.array([ux, uy])
    return (float(centroid[0]), float(centroid[1]), ux, uy,
            float((proj.max() - proj.min()) / 2.0))


def _segment_intersects_bbox(p1, p2, x0, y0, x1, y1) -> bool:
    """Liang-Barsky line-clipping segment-vs-axis-aligned-rect.
    Currently unused; kept around in case the centerline gate needs to
    be reintroduced later."""
    ax, ay = p1
    bx, by = p2
    dx, dy = bx - ax, by - ay
    p_arr = (-dx, dx, -dy, dy)
    q_arr = (ax - x0, x1 - ax, ay - y0, y1 - ay)
    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p_arr, q_arr):
        if abs(pi) < 1e-12:
            if qi < 0:
                return False
        else:
            u = qi / pi
            if pi < 0:
                if u > u2:
                    return False
                if u > u1:
                    u1 = u
            else:
                if u < u1:
                    return False
                if u < u2:
                    u2 = u
    return u1 <= u2


# ---------------------------------------------------------------------------
# Step 5 — centerline-token search and K-nearest claim
# ---------------------------------------------------------------------------

def _match_runway_labels(runway_indices, all_polys, text_tokens,
                         claimed_polys: set[int]):
    """For each ML-predicted Runway polygon, search past each end of
    the polygon's principal axis for a runway-pattern text token whose
    centroid sits within a progressively-widened band along the
    centerline (CENTERLINE_WIDTHS_PT). The first token found at the
    smallest-width band that contains any candidate wins that end. Ties
    inside a band are broken by closeness-to-end (longitudinal
    distance past the endpoint).

    For each matched token, claim K = len(token.text) nearest unclaimed
    near-black filled polygons as Runway Labels. Polygons claimed for
    one runway are not available to a later runway. Tokens used by one
    end aren't reused at any other end.

    Mutates `claimed_polys`.

    Returns:
      (label_indices, diagnostics)
        label_indices: set[int] of polygon indices to label as Runway Labels
        diagnostics:   list[dict] one per (runway, end), records what
                       happened — token text, width that succeeded, lat
                       offset, longitudinal distance past end, claimed
                       polygon indices. Used by step-5 print-out.
    """
    # Pre-filter tokens to those matching the runway regex.
    pattern_tokens = [t for t in text_tokens if RUNWAY_RE.match(t["text"])]
    used_token_ids: set[int] = set()    # id() of token dicts already used
    label_indices: set[int] = set()
    diagnostics: list[dict] = []

    for ri in runway_indices:
        rp = all_polys[ri]
        cx, cy, ux, uy, half_len = _runway_axis(rp["subpaths"], rp["rect"])
        # Pre-compute per-token longitudinal/lateral coordinates for
        # this runway.
        tok_proj = []
        for tok in pattern_tokens:
            if id(tok) in used_token_ids:
                tok_proj.append(None)
                continue
            dxv = tok["cx"] - cx
            dyv = tok["cy"] - cy
            long_pos = dxv * ux + dyv * uy
            lat = abs(-dxv * uy + dyv * ux)
            tok_proj.append((long_pos, lat))

        for end_sign in (-1, +1):
            chosen = None       # (width, distance_past_end, lat, tok)
            for width in CENTERLINE_WIDTHS_PT:
                cands = []
                for tok, proj in zip(pattern_tokens, tok_proj):
                    if proj is None:
                        continue
                    if id(tok) in used_token_ids:
                        continue
                    long_pos, lat = proj
                    # Past this endpoint along the axis.
                    if end_sign * long_pos < half_len:
                        continue
                    if lat >= width / 2.0:
                        continue
                    dist_past = end_sign * long_pos - half_len
                    cands.append((dist_past, lat, tok))
                if cands:
                    cands.sort(key=lambda x: x[0])
                    dist_past, lat, tok = cands[0]
                    chosen = (width, dist_past, lat, tok)
                    break
            diag = {
                "runway_idx": ri,
                "end_sign": end_sign,
                "matched": False,
                "token": None,
                "width": None,
                "lat": None,
                "dist_past_end": None,
                "claimed": [],
            }
            if chosen is None:
                diagnostics.append(diag)
                continue
            width, dist_past, lat, tok = chosen
            used_token_ids.add(id(tok))
            k = len(tok["text"])
            # Claim K nearest unclaimed near-black filled polygons.
            scored = []
            for i, p in enumerate(all_polys):
                if i in claimed_polys:
                    continue
                if not (p["filled"] and p["is_near_black"]
                        and not p["is_taxi_surface"]):
                    continue
                ddx = tok["cx"] - p["cx"]
                ddy = tok["cy"] - p["cy"]
                scored.append((ddx * ddx + ddy * ddy, i))
            scored.sort()
            nearest = [i for _, i in scored[:k]]
            if len(nearest) < k:
                # Couldn't find enough polygons to satisfy K. Free the
                # token so it can be considered for the next end (rare
                # but possible at airports with very crowded glyph
                # pools).
                used_token_ids.discard(id(tok))
                diag["matched"] = False
                diag["token"] = tok["text"]
                diag["width"] = width
                diag["lat"] = lat
                diag["dist_past_end"] = dist_past
                diag["claimed"] = []
                diag["note"] = f"insufficient_polys k={k} found={len(nearest)}"
                diagnostics.append(diag)
                continue
            claimed_polys.update(nearest)
            label_indices.update(nearest)
            diag["matched"] = True
            diag["token"] = tok["text"]
            diag["width"] = width
            diag["lat"] = lat
            diag["dist_past_end"] = dist_past
            diag["claimed"] = list(nearest)
            diagnostics.append(diag)

    return label_indices, diagnostics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=Path, required=True,
                    help="Full path-features CSV from extract_paths_fitz.py")
    ap.add_argument("--pdf", type=Path, required=True,
                    help="Source <airport>-faa.pdf")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", type=Path,
                    default=Path(__file__).parent / "runs" / "v24" / "model.lgb")
    ap.add_argument("--feature-list", type=Path,
                    default=Path(__file__).parent / "runs" / "v24" / "feature_list.json")
    ap.add_argument("--nasr-rwy", type=Path,
                    default=Path(__file__).parent.parent / "data" / "nasr_apt_rwy.csv")
    args = ap.parse_args()

    # ---- Steps 1+2: rule-based taxi detection ------------------------
    print(f"[pipeline] step 1+2: rule-based taxi detection")
    det = detect_taxi(args.pdf)
    all_polys = det["all_polys"]
    text_tokens = det["text_tokens"]
    page_w, page_h = det["page_w"], det["page_h"]
    surf_set = set(det["taxi_surface_indices"])
    label_set = set(det["taxi_label_indices"])
    print(f"  taxiways: {len(surf_set)}, taxi labels: {len(label_set)}")

    # Claimed = removed-from-pool. Each later step adds to this set.
    claimed: set[int] = surf_set | label_set

    # ---- Step 3: rule-based runway detection -------------------------
    airport_code = args.pdf.stem.replace("-faa", "").lower()
    print(f"[pipeline] step 3: rule-based runway detection ({airport_code})")
    rwy_set = detect_runways(
        all_polys, det["clips"], airport_code, args.nasr_rwy,
        page_w=page_w, page_h=page_h, claimed_indices=claimed,
    )
    print(f"  runways: {len(rwy_set)}")
    claimed |= rwy_set

    # ---- Step 4: ML on the remaining unclaimed polygons --------------
    print(f"[pipeline] step 4: ML on {len(all_polys) - len(claimed)} "
          f"unclaimed polygons")
    df = load_features_all(args.paths)
    if len(df) != len(all_polys):
        raise ValueError(
            f"Polygon count mismatch — CSV has {len(df)}, "
            f"PyMuPDF detected {len(all_polys)}. Both should iterate "
            f"page.get_drawings() in the same order."
        )

    edges_csv = args.paths.with_name(args.paths.stem + "_edges" + args.paths.suffix)
    edges_df = load_edges(edges_csv) if edges_csv.exists() else None

    pdf_text_csv = args.out.with_suffix(".pdf_text.csv")
    print(f"  extracting PDF text -> {pdf_text_csv.name}")
    extract_pdf_text_for_one(args.pdf, args.nasr_rwy, pdf_text_csv)
    text_df = load_pdf_text(pdf_text_csv)

    df = add_relational_features(df, edges_df=edges_df, text_df=text_df)
    df = df.reset_index(drop=True)

    unclaimed_mask = np.array([i not in claimed for i in range(len(df))])
    df_unclaimed = df[unclaimed_mask]

    feature_meta = json.loads(args.feature_list.read_text())
    feature_cols = feature_meta["feature_cols"]
    cat_cols = feature_meta["categorical_cols"]
    X = df_unclaimed[feature_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype("category")
    booster = lgb.Booster(model_file=str(args.model))
    probs = booster.predict(X)

    # Mask classes ML must not assign. Taxiways and Taxi Labels are
    # rule-based-only (handled in steps 1+2). Runway Labels come from
    # step 5. Setting these to 0 forces argmax onto the remaining valid
    # classes (Footprints / Runways / Lights / Stars / Other).
    for cls in ML_EXCLUDE_CLASSES:
        if cls in LABELS:
            probs[:, LABELS.index(cls)] = 0.0

    # Apply pdf_char_override on the filtered (unclaimed-only) df. This
    # preserves the existing Lights-by-stroke detection and runway
    # axis/text logic. Its taxi/runway-label overrides can't fire here
    # because we already removed those polygons from the input pool.
    nasr_runways_full = (
        load_nasr_runways(args.nasr_rwy)
        if args.nasr_rwy and args.nasr_rwy.exists() else {}
    )
    airport_code = args.pdf.stem.replace("-faa", "").lower()
    nasr_for_airport = nasr_runways_full.get(airport_code, set())
    df_unclaimed_r = df_unclaimed.reset_index(drop=True)
    final_labels_unclaimed, _override_mask = apply_pdf_char_override(
        df_unclaimed_r,
        probs,
        args.pdf,
        nasr_runways=nasr_for_airport,
        confidence=0.60,
        verbose=False,
    )
    # Sanity-strip — the override shouldn't assign these classes on the
    # filtered set, but enforce it.
    ml_labels = [
        ("Other" if lab in ML_EXCLUDE_CLASSES else lab)
        for lab in final_labels_unclaimed
    ]
    ml_top_prob = probs.max(axis=1)

    # Map: polygon index -> final label.
    final_label = [None] * len(all_polys)
    final_top = [None] * len(all_polys)
    final_prob = [0.0] * len(all_polys)
    final_override = [False] * len(all_polys)
    final_source = [None] * len(all_polys)

    for i in surf_set:
        final_label[i] = "Taxiways"
        final_top[i] = "Taxiways"
        final_override[i] = True
        final_source[i] = "rule_taxi_surface"

    for i in label_set:
        final_label[i] = "Taxiway Labels"
        final_top[i] = "Taxiway Labels"
        final_override[i] = True
        final_source[i] = "rule_taxi_label"

    for i in rwy_set:
        final_label[i] = "Runways"
        final_top[i] = "Runways"
        final_override[i] = True
        final_source[i] = "rule_runway_nasr_topn"

    unclaimed_indices = list(df_unclaimed.index)
    for j, i in enumerate(unclaimed_indices):
        final_label[i] = ml_labels[j]
        final_top[i] = ml_labels[j]
        final_prob[i] = float(ml_top_prob[j])
        final_source[i] = "ml"

    print(f"  ML label distribution (unclaimed only):")
    counts = pd.Series(ml_labels).value_counts()
    for lab, n in counts.items():
        print(f"    {lab:<22} {n}")

    # ---- Step 5: centerline-token search -----------------------------
    # Runway polygons now come from step 3 (rule-based), not ML.
    runway_indices = sorted(rwy_set)
    print(f"[pipeline] step 5: centerline-token search over "
          f"{len(runway_indices)} rule-claimed runways")

    # `claimed` already includes taxi surfaces + taxi labels. The
    # K-nearest pool inside _match_runway_labels also uses this set so
    # we never reuse a polygon that's already labeled.
    rwy_label_idx, step5_diag = _match_runway_labels(
        runway_indices, all_polys, text_tokens, claimed,
    )
    n_matched_ends = sum(1 for d in step5_diag if d.get("matched"))
    print(f"  runway ends processed: {len(step5_diag)}, "
          f"token matches: {n_matched_ends}, "
          f"polygons reassigned: {len(rwy_label_idx)}")

    for i in rwy_label_idx:
        final_label[i] = "Runway Labels"
        final_top[i] = "Runway Labels"
        final_override[i] = True
        final_source[i] = "rule_runway_label_centerline"

    # Stash diagnostics for the one-shot diag script and inline view.
    args.out.with_suffix(".step5.json").write_text(
        json.dumps(step5_diag, indent=2, default=str)
    )

    # ---- Build records JSON in predict_one.py format -----------------
    records = []
    for i in range(len(all_polys)):
        records.append({
            "airport": str(df.iloc[i]["airport"]),
            "object_id": int(df.iloc[i]["object_id"]),
            "kind": str(df.iloc[i]["kind"]),
            "label": final_label[i],
            "model_top": final_top[i],
            "model_top_prob": float(final_prob[i]),
            "override_applied": bool(final_override[i]),
            "override_source": final_source[i],
            "left":   round(float(df.iloc[i]["left"]),   4),
            "top":    round(float(df.iloc[i]["top"]),    4),
            "right":  round(float(df.iloc[i]["right"]),  4),
            "bottom": round(float(df.iloc[i]["bottom"]), 4),
        })

    # Debug: PDF Text Tokens. Each word from the PDF text stream with
    # its bbox center in AI y-up coords. Consumed by
    # ImportPredictedLayers.jsx to build a "PDF Text Tokens" layer in
    # the final AI file.
    token_records = [{
        "text": t["text"],
        "x":    round(t["cx"], 4),
        "y":    round(page_h - t["cy"], 4),  # PDF y-down -> AI y-up
    } for t in text_tokens]

    payload = {
        "airport": records[0]["airport"] if records else None,
        "model_labels": list(LABELS),
        "maybe_labels": [],
        "confidence_threshold": None,
        "predictions": records,
        "text_tokens": token_records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\n[pipeline] wrote {len(records)} predictions -> {args.out}")

    final_counts = pd.Series([r["label"] for r in records]).value_counts()
    print("\nFinal label distribution:")
    for lab in list(LABELS):
        n = int(final_counts.get(lab, 0))
        if n:
            print(f"  {lab:<22} {n}")


if __name__ == "__main__":
    main()
