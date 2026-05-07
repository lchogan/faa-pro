"""
classify_pipeline.py — single-pass orchestrator for the new pipeline.

Steps in strict order. Polygons claimed in earlier steps are removed
from the pool seen by every later step.

  1.   Taxi surfaces      rule-based (gray-fill detection from
                          chart_scene.is_taxi_surface).
  2.   Runways            rule-based (NASR runway count + top-N
                          polygons by polygon area among filled-
                          near-black or stroked candidates, after
                          taxi-surface removal). Replaces ML-based
                          runway prediction.
  3.   Runway Labels      For each rule-claimed Runway, extend a
                          thin centerline past each end. The closest
                          NASR-listed runway-name token whose
                          centroid sits within a progressively-
                          widened band along the centerline picks
                          the label. Claim K = len(token) nearest
                          near-black filled unclaimed polygons. NASR
                          designations are pulled per-airport so a
                          chart-only token like APF "6" is rejected,
                          and compass-direction names ("NE/SW")
                          qualify for turf strips.
  4.   Taxi Labels        Token regex K-nearest gated on bbox-
                          touches-taxi-surface, restricted to the
                          unclaimed pool. Runs AFTER runway-label
                          matching so a digit glyph that belongs to
                          a runway designator is no longer available
                          and a taxi token with the same numeric
                          string (e.g. APF runway "5" vs a hypothetical
                          taxi "5") still finds its own glyph
                          polygons elsewhere on the chart.
  5.   ML                 On the remaining unclaimed polygons.
                          Classes Taxiways / Taxiway Labels / Runway
                          Labels / Runways are masked out of ML's
                          probability matrix so an unclaimed polygon
                          can't fall back into them.
  6.   Hull rejection     Build a concave hull (no buffer) over the
                          rule-claimed Runways + Taxi surfaces'
                          anchor points. Any polygon whose centroid
                          sits outside the hull is demoted to Other.
                          Runways, Taxi surfaces, Runway Labels, and
                          Taxi Labels are exempt — rule-trusted, and
                          labels can legitimately sit at chart edges.

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

import lightgbm as lgb
import numpy as np
import pandas as pd

from chart_scene import read_chart
from extract_pdf_text import (
    _normalize_designation,
    load_nasr_runways,
)
from hull_filter import hull_reject
from load import LABELS, ML_LABELS, load_features_all
from relational import add_relational_features, load_edges
from runway_detection import detect_runways
from taxi_detection import match_taxi_labels


RUNWAY_RE = re.compile(r"^(0?[1-9]|[12][0-9]|3[0-6])[LRC]?$")
# Centerline-band widths used to search for runway-name tokens past each
# runway end. We walk the widths in order and stop at the first width that
# yields a candidate. 1pt is effectively "directly on the centerline"; we
# expand to 100pt before giving up on this end.
CENTERLINE_WIDTHS_PT = (1.0, 10.0, 25.0, 50.0, 100.0)
# Hull rejection (step 4): classes whose polygons are exempt from the
# centroid-in-hull test. Rule-claimed and never re-tested. Labels in
# particular can sit at chart edges (runway numbers past the threshold,
# taxiway letters at the far end of a stub) so they must not be culled.
HULL_EXEMPT_CLASSES = frozenset({
    "Runways", "Taxiways", "Runway Labels", "Taxiway Labels",
})


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _runway_axis(subpaths, fallback_rect):
    """PCA principal axis of a runway polygon's anchor points. Returns
    (cx, cy, ux, uy, half_len) where ux,uy is a unit vector along the
    axis and half_len is the longitudinal half-extent.

    The center is the midpoint of the (min, max) projections along the
    axis, NOT the naive arr.mean(axis=0). Subpath rings often repeat
    their first point as a closing anchor, and PathItem geometry can
    distribute anchors non-uniformly along the rectangle (more on one
    side than the other). Either skews the naive mean and produces
    asymmetric half-extents, so a token painted just past the
    "shorter" end appears to fall short of the threshold. Using the
    projection midpoint guarantees symmetric per-end half-lengths.
    """
    pts: list[tuple[float, float]] = []
    for ring in subpaths or []:
        pts.extend(ring)
    x0, y0, x1, y1 = fallback_rect
    if len(pts) < 3:
        if (x1 - x0) >= (y1 - y0):
            return ((x0 + x1) / 2, (y0 + y1) / 2, 1.0, 0.0, (x1 - x0) / 2)
        return ((x0 + x1) / 2, (y0 + y1) / 2, 0.0, 1.0, (y1 - y0) / 2)
    arr = np.asarray(pts, dtype=float)
    naive_centroid = arr.mean(axis=0)
    centered = arr - naive_centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    norm = float(np.linalg.norm(direction)) or 1.0
    ux, uy = float(direction[0]) / norm, float(direction[1]) / norm
    axis = np.array([ux, uy])
    proj = centered @ axis
    long_min = float(proj.min())
    long_max = float(proj.max())
    long_offset = (long_min + long_max) / 2.0
    cx = float(naive_centroid[0]) + long_offset * ux
    cy = float(naive_centroid[1]) + long_offset * uy
    half_len = (long_max - long_min) / 2.0
    return (cx, cy, ux, uy, half_len)


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
                         claimed_polys: set[int],
                         nasr_designations: set[str] | None = None):
    """For each rule-claimed Runway polygon, search past each end of
    the polygon's principal axis for a runway-name text token whose
    centroid sits within a progressively-widened band along the
    centerline (CENTERLINE_WIDTHS_PT). The first token found at the
    smallest-width band that contains any candidate wins that end. Ties
    inside a band are broken by closeness-to-end (longitudinal
    distance past the endpoint).

    Token eligibility: when `nasr_designations` is provided, only
    tokens whose normalized form is one of the NASR-listed runway
    designations for this airport qualify. Without NASR, the function
    falls back to a generic runway-name regex (matches any 1-36
    designator with optional L/R/C suffix). NASR is far more reliable
    — APF's chart contains a "6" token that matches the regex but is
    not a runway here, and the regex-only path was incorrectly
    claiming polygons near it.

    For each matched token, claim K = len(token.text) nearest unclaimed
    near-black filled polygons as Runway Labels. Polygons claimed for
    one runway are not available to a later runway. Tokens used by one
    end aren't reused at any other end.

    Mutates `claimed_polys`.

    Returns:
      (label_indices, diagnostics, used_token_ids)
        label_indices:   set[int] of polygon indices labeled Runway Labels
        diagnostics:     list[dict] one per (runway, end), records what
                         happened — token text, width that succeeded, lat
                         offset, longitudinal distance past end, claimed
                         polygon indices. Used by step-3 print-out.
        used_token_ids:  set[int] of id() values for token dicts that
                         were consumed by this matcher. The pipeline
                         passes these into the taxi-label matcher so
                         the same token can't be claimed twice (which
                         would otherwise pull arrowheads / nearby
                         symbols into Taxiway Labels at airports where
                         a runway designator sits over a taxiway).
    """
    rwy_set = nasr_designations or set()
    if rwy_set:
        pattern_tokens = [
            t for t in text_tokens
            if _normalize_designation(t["text"]) in rwy_set
        ]
    else:
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

    return label_indices, diagnostics, used_token_ids


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
                    default=Path(__file__).parent / "runs" / "v25" / "model.lgb")
    ap.add_argument("--feature-list", type=Path,
                    default=Path(__file__).parent / "runs" / "v25" / "feature_list.json")
    ap.add_argument("--nasr-rwy", type=Path,
                    default=Path(__file__).parent.parent / "data" / "nasr_apt_rwy.csv")
    ap.add_argument("--skip-hull", action="store_true",
                    help="Experiment toggle: skip hull rejection (step 4) entirely.")
    ap.add_argument("--footprint-threshold", type=float, default=None,
                    help="If set, label as Footprints whenever P(Footprints) >= threshold "
                         "(overriding argmax). Lower = more sensitive. 0.5 ≈ argmax, "
                         "try 0.3 or 0.2 to catch borderline buildings.")
    args = ap.parse_args()

    # NASR runway designations for THIS airport. Used by the runway-label
    # centerline-token search to identify which PDF text tokens are
    # actually runway names — rejects APF's stray "6" token (regex
    # would accept it; NASR confirms APF runways are 5/23, 14/32, NE/SW).
    airport_code = args.pdf.stem.replace("-faa", "").lower()
    nasr_runways_full = (
        load_nasr_runways(args.nasr_rwy)
        if args.nasr_rwy and args.nasr_rwy.exists() else {}
    )
    nasr_for_airport = nasr_runways_full.get(airport_code, set())

    # ---- Step 1: taxi-surface detection (gray fill) ------------------
    print(f"[pipeline] step 1: taxi-surface detection")
    scene = read_chart(args.pdf)
    all_polys = scene["all_polys"]
    text_tokens = scene["text_tokens"]
    clips = scene["clips"]
    page_w, page_h = scene["page_w"], scene["page_h"]
    surf_set = {i for i, p in enumerate(all_polys) if p["is_taxi_surface"]}
    taxi_surfaces = [all_polys[i] for i in sorted(surf_set)]
    print(f"  taxi surfaces: {len(surf_set)}")

    # Claimed = removed-from-pool. Each later step adds to this set.
    claimed: set[int] = set(surf_set)

    # ---- Step 2: rule-based runway detection -------------------------
    print(f"[pipeline] step 2: rule-based runway detection ({airport_code})")
    rwy_set = detect_runways(
        all_polys, clips, airport_code, args.nasr_rwy,
        page_w=page_w, page_h=page_h, claimed_indices=claimed,
    )
    print(f"  runways: {len(rwy_set)}")
    claimed |= rwy_set

    # ---- Step 3: runway-label centerline-token search ----------------
    # Done BEFORE taxi-label matching so digit-glyph polygons that
    # belong to a runway designator are reserved first. This matters
    # at airports where a runway (e.g. APF "5") sits over a taxiway
    # and would otherwise be claimed as a taxi label.
    runway_indices = sorted(rwy_set)
    print(f"[pipeline] step 3: centerline-token search over "
          f"{len(runway_indices)} rule-claimed runways")
    rwy_label_idx, step3_diag, step3_used_token_ids = _match_runway_labels(
        runway_indices, all_polys, text_tokens, claimed,
        nasr_designations=nasr_for_airport,
    )
    n_matched_ends = sum(1 for d in step3_diag if d.get("matched"))
    print(f"  runway ends processed: {len(step3_diag)}, "
          f"token matches: {n_matched_ends}, "
          f"polygons reassigned: {len(rwy_label_idx)}")
    claimed |= rwy_label_idx

    # ---- Step 4: taxi-label K-nearest claim --------------------------
    # Tokens consumed by step 3 are excluded — each PDF text token
    # identifies polygons at most once across the pipeline.
    print(f"[pipeline] step 4: taxi-label K-nearest claim")
    taxi_label_list = match_taxi_labels(
        all_polys, taxi_surfaces, text_tokens,
        claimed_polys=claimed,
        excluded_token_ids=step3_used_token_ids,
    )
    label_set = set(taxi_label_list)
    claimed |= label_set
    print(f"  taxi labels: {len(label_set)}")

    # ---- Initialize per-polygon final labels from rule-claimed steps -
    final_label: list[str | None] = [None] * len(all_polys)
    final_top: list[str | None] = [None] * len(all_polys)
    final_prob: list[float] = [0.0] * len(all_polys)
    final_override: list[bool] = [False] * len(all_polys)
    final_source: list[str | None] = [None] * len(all_polys)

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

    for i in rwy_label_idx:
        final_label[i] = "Runway Labels"
        final_top[i] = "Runway Labels"
        final_override[i] = True
        final_source[i] = "rule_runway_label_centerline"

    # Stash centerline-token diagnostics from step 3 for inline view.
    args.out.with_suffix(".step3.json").write_text(
        json.dumps(step3_diag, indent=2, default=str)
    )

    # ---- Step 5 (was 6): concave-hull rejection BEFORE ML ------------
    # Build a concave hull (no buffer) over the rule-claimed Runways +
    # Taxiways' anchor points. A non-exempt polygon is demoted only
    # when its bbox doesn't intersect the hull at all — anything that
    # touches or overlaps the hull is kept (footprints flush with the
    # apron edge belong here even when their centroid sits just
    # outside). Exempt classes: Runways, Taxiways, Runway Labels, Taxi
    # Labels — rule-trusted, and labels can legitimately sit at chart
    # edges.
    if args.skip_hull:
        print(f"[pipeline] step 5: concave-hull rejection SKIPPED (--skip-hull)")
    else:
        print(f"[pipeline] step 5: concave-hull rejection (pre-ML)")
        hull_candidates = [
            i for i in range(len(all_polys))
            if final_label[i] not in HULL_EXEMPT_CLASSES
        ]
        demote_idx, hull_diag = hull_reject(
            all_polys,
            anchor_indices=surf_set | rwy_set,
            candidate_indices=hull_candidates,
        )
        print(f"  hull anchor pts: {hull_diag['n_anchor_points']}, "
              f"area: {hull_diag['hull_area']:.0f}")
        print(f"  candidates tested: {hull_diag['n_candidates_tested']}, "
              f"demoted to Other: {hull_diag['n_demoted']}")
        for i in demote_idx:
            final_label[i] = "Other"
            final_top[i] = "Other"
            final_override[i] = True
            final_source[i] = "hull_reject"
        claimed |= demote_idx

    # ---- Step 6: ML on the remaining unclaimed polygons --------------
    print(f"[pipeline] step 6: ML on {len(all_polys) - len(claimed)} "
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

    df = add_relational_features(df, edges_df=edges_df, text_df=None)
    df = df.reset_index(drop=True)

    unclaimed_mask = np.array([i not in claimed for i in range(len(df))])
    df_unclaimed = df[unclaimed_mask]

    feature_meta = json.loads(args.feature_list.read_text())
    feature_cols = feature_meta["feature_cols"]
    cat_cols = feature_meta["categorical_cols"]
    model_labels = feature_meta.get("labels", list(ML_LABELS))
    X = df_unclaimed[feature_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype("category")
    booster = lgb.Booster(model_file=str(args.model))
    probs = booster.predict(X)

    # v25 emits only ML_LABELS (Footprints / Stars / Other). Default
    # decision rule is argmax over the 3 columns; --footprint-threshold
    # promotes any polygon with P(Footprints) >= threshold even when
    # Other narrowly wins argmax — useful for catching borderline
    # buildings the model isn't quite confident on.
    ml_pred_idx = probs.argmax(axis=1)
    ml_top_prob = probs.max(axis=1)
    ml_labels = [model_labels[k] for k in ml_pred_idx]
    if args.footprint_threshold is not None:
        fp_idx = model_labels.index("Footprints")
        promoted = 0
        for j in range(len(ml_labels)):
            if ml_labels[j] != "Footprints" and probs[j, fp_idx] >= args.footprint_threshold:
                ml_labels[j] = "Footprints"
                ml_top_prob[j] = float(probs[j, fp_idx])
                promoted += 1
        print(f"  --footprint-threshold {args.footprint_threshold}: "
              f"promoted {promoted} polygons to Footprints")

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

    # ---- Step 7 (user-facing step 6): stroked-only sweep -------------
    # Final pass: any polygon that's stroked-only (stroked && not filled)
    # AND not on the Runways layer is demoted to Other. Catches Lights,
    # arrowheads, line art, decorative symbols that survive ML. Runways
    # are exempt — grass-strip runways are drawn as stroked rectangles
    # and must stay on the Runways layer (rule-claimed in step 2).
    print(f"[pipeline] step 7: stroked-only sweep → Other (Runways exempt)")
    stroked_demoted = 0
    for i, p in enumerate(all_polys):
        if final_label[i] == "Runways":
            continue
        if final_label[i] == "Other":
            continue  # already there
        if p.get("stroked") and not p.get("filled"):
            prev = final_label[i]
            final_label[i] = "Other"
            final_top[i] = "Other"
            final_override[i] = True
            final_source[i] = f"stroked_sweep_from_{prev}"
            stroked_demoted += 1
    print(f"  stroked-only polygons demoted: {stroked_demoted}")

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
