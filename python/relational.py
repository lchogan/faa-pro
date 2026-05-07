"""
Relational / spatial-context features.

Per-object features (size, color, stroke) are not enough to distinguish Lights
from random small lines, because Lights are defined by their *relationship* to
each other and to runways. This module adds neighbor-based features that the
LightGBM model can learn from.

Features added per row, computed within each airport's coordinate frame:

  knn_dist_{1,3,5,10}        Euclidean distance to k-th nearest centroid (pts)
  local_density              # objects within a fixed-fraction-of-artboard radius
  similar_neighbor_count     # neighbors with similar size + stroke (any angle)
  aligned_neighbor_count     # similar neighbors whose principal angle ≈ this one's
  dist_to_runway             centroid distance to nearest runway-shaped object (pts)
  angle_to_runway            angular diff (deg) between this principal angle
                             and the nearest runway candidate's
  parallelism_to_runway      cos²(angle_diff) — 1 if parallel, 0 if perpendicular
  perp_to_runway             sin²(angle_diff) — 1 if perpendicular, 0 if parallel
  inside_runway_bbox         1 if centroid lies inside any runway candidate's bbox
  longitudinal_offset_runway signed distance along runway long-axis from runway centroid (pts)
  lateral_offset_runway      signed perpendicular distance from runway long-axis (pts)
  runway_axis_position       |longitudinal_offset| / runway_half_length
                             (0 = at runway center, 1 = at end, >1 = beyond)
  dist_to_taxiway            centroid distance to nearest gray-fill candidate (pts)
  inside_taxiway_bbox        1 if centroid lies inside any taxiway candidate's bbox
  angle_to_taxiway           angular diff (deg) vs nearest taxiway candidate's principal angle
  parallelism_to_taxiway     cos²(angle_diff) — 1 if parallel to local taxiway axis
  size_rel_to_artboard       max(width, height) / artboard diagonal
  bbox_area_rel              bbox_area / artboard area
  centroid_norm_x/y          centroid normalized to [0,1] in artboard frame

When a sidecar `*_edges.csv` file is supplied (one row per anchor-to-anchor
edge of every polygon), three more features are added per object using the
*nearest local edge* of the relevant candidate set — this is what handles
curved/L-shaped taxiway polygons where the polygon's PCA principal_angle
points along the diagonal but the local tangent at the label position is
clearly horizontal or vertical:

  dist_to_taxiway_edge       distance to nearest edge of any taxiway candidate
  angle_to_taxiway_edge      min angle diff (deg) between this object's
                             principal/longest-segment angle and that edge
  parallelism_to_taxiway_edge cos²(angle_to_taxiway_edge)
  dist_to_runway_edge        same, vs runway-candidate edges
  angle_to_runway_edge
  parallelism_to_runway_edge

"Runway candidates" are detected without labels: the most elongated
high-area objects in the airport. Even if not literally runways they give a
stable orientation reference, which is what the Lights signal needs.
"""

from __future__ import annotations

import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from load import load_features_all

# Patterns used to classify PDF-extracted word text. Runway designations like
# "9", "27L", "9R" vs taxiway designations like "A", "B1", "K2", "AA".
_RUNWAY_TEXT_RE = re.compile(r"^\d{1,2}[LRC]?$")
_TAXIWAY_TEXT_RE = re.compile(r"^[A-Z][A-Z0-9]?$")

KS = (1, 3, 5, 10)
LOCAL_DENSITY_FRAC = 0.05       # radius = 5% of artboard diagonal
SIMILAR_TOL_STROKE = 0.15       # ±15% stroke width
SIMILAR_TOL_SIZE = 0.5          # ±50% size
ALIGNED_ANGLE_DEG = 10          # orientation match within ±10°
RUNWAY_RATIO_MIN = 5.0          # principal ratio threshold for "elongated"
RUNWAY_AREA_QUANTILE = 0.9      # top decile by bbox area
RUNWAY_FALLBACK_TOP_N = 5       # if no candidate matches both criteria

# Taxiway candidate detection (filled gray polygons, larger than median,
# not extremely elongated — runways can be gray-ish in some diagrams).
TAXIWAY_GRAY_MIN = 150          # min RGB value for "grayish"
TAXIWAY_GRAY_MAX = 235          # max RGB value
TAXIWAY_GRAY_TOL = 20           # max |R-G|, |R-B|, |G-B|
TAXIWAY_AREA_QUANTILE = 0.5     # candidates must be above the airport's median area
TAXIWAY_RATIO_MAX = 5.0         # exclude runway-like elongated shapes
TAXIWAY_FALLBACK_TOP_N = 10     # if no gray candidate found, fall back to largest filled


def _angle_diff_rad(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Diff between two angles in [0, π); both treated as undirected lines."""
    d = np.abs(a - b)
    return np.minimum(d, np.pi - d)


def _centroid_inside_any_bbox(centroids: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
    """Return a boolean mask of which centroids fall inside any of the given bboxes.

    bboxes are stored Illustrator-style: [left, top, right, bottom] with top > bottom.
    """
    if bboxes.size == 0:
        return np.zeros(len(centroids), dtype=bool)
    cx = centroids[:, 0:1]  # (N, 1)
    cy = centroids[:, 1:2]
    left = bboxes[:, 0]
    top = bboxes[:, 1]
    right = bboxes[:, 2]
    bottom = bboxes[:, 3]
    return ((cx >= left) & (cx <= right) & (cy >= bottom) & (cy <= top)).any(axis=1)


def _per_airport(
    df: pd.DataFrame,
    edges_df: pd.DataFrame | None = None,
    text_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    n = len(df)
    if n == 0:
        return df

    centroids = df[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    angles_rad = df["principal_angle"].to_numpy(dtype=float) * np.pi / 180.0
    sizes = np.maximum(df["width"].to_numpy(dtype=float),
                       df["height"].to_numpy(dtype=float))
    stroke_w = df["stroke_width"].fillna(0).to_numpy(dtype=float)
    bbox_area = df["bbox_area"].to_numpy(dtype=float)
    pca_ratio = df["principal_ratio"].fillna(1).to_numpy(dtype=float)

    artboard_w = float(df["artboard_right"].iloc[0] - df["artboard_left"].iloc[0])
    artboard_h = float(df["artboard_top"].iloc[0] - df["artboard_bottom"].iloc[0])
    artboard_diag = np.hypot(artboard_w, artboard_h) or 1.0
    artboard_area = abs(artboard_w * artboard_h) or 1.0

    df["size_rel_to_artboard"] = sizes / artboard_diag
    df["bbox_area_rel"] = bbox_area / artboard_area
    df["centroid_norm_x"] = (df["centroid_x"] - df["artboard_left"]) / (artboard_w or 1.0)
    df["centroid_norm_y"] = (df["centroid_y"] - df["artboard_bottom"]) / (artboard_h or 1.0)

    tree = cKDTree(centroids)

    # ----- kNN distances -----
    if n > 1:
        max_k = min(max(KS), n - 1)
        d, _ = tree.query(centroids, k=max_k + 1)  # column 0 is self (dist=0)
        for k in KS:
            col = min(k, max_k)
            df[f"knn_dist_{k}"] = d[:, col]
    else:
        for k in KS:
            df[f"knn_dist_{k}"] = np.nan

    # ----- pair-wise predicates within local-density radius -----
    R = LOCAL_DENSITY_FRAC * artboard_diag
    pairs = tree.query_pairs(r=R, output_type="ndarray") if n > 1 else np.empty((0, 2), dtype=int)
    if pairs.size:
        i_idx, j_idx = pairs[:, 0], pairs[:, 1]
        sw_i, sw_j = stroke_w[i_idx], stroke_w[j_idx]
        size_i, size_j = sizes[i_idx], sizes[j_idx]
        ang_i, ang_j = angles_rad[i_idx], angles_rad[j_idx]

        denom_sw = np.maximum(np.maximum(sw_i, sw_j), 1e-6)
        denom_sz = np.maximum(np.maximum(size_i, size_j), 1e-6)
        sim_stroke = np.abs(sw_i - sw_j) / denom_sw < SIMILAR_TOL_STROKE
        sim_size = np.abs(size_i - size_j) / denom_sz < SIMILAR_TOL_SIZE
        similar = sim_stroke & sim_size

        ang_diff = _angle_diff_rad(ang_i, ang_j)
        aligned = similar & (ang_diff < np.deg2rad(ALIGNED_ANGLE_DEG))

        sim_counts = (np.bincount(i_idx[similar], minlength=n) +
                      np.bincount(j_idx[similar], minlength=n))
        ali_counts = (np.bincount(i_idx[aligned], minlength=n) +
                      np.bincount(j_idx[aligned], minlength=n))
        # local density = pairs touching i (every pair within R)
        loc_counts = (np.bincount(i_idx, minlength=n) +
                      np.bincount(j_idx, minlength=n))
    else:
        sim_counts = np.zeros(n, dtype=int)
        ali_counts = np.zeros(n, dtype=int)
        loc_counts = np.zeros(n, dtype=int)

    df["local_density"] = loc_counts
    df["similar_neighbor_count"] = sim_counts
    df["aligned_neighbor_count"] = ali_counts

    bboxes = df[["left", "top", "right", "bottom"]].to_numpy(dtype=float)

    # ----- runway-candidate features -----
    if n > 0:
        area_thresh = np.quantile(bbox_area, RUNWAY_AREA_QUANTILE)
        cand_mask = (pca_ratio >= RUNWAY_RATIO_MIN) & (bbox_area >= area_thresh)
        cand_idx = np.where(cand_mask)[0]
        if cand_idx.size == 0:
            cand_idx = np.argsort(-bbox_area)[: min(RUNWAY_FALLBACK_TOP_N, n)]

        cand_centroids = centroids[cand_idx]
        cand_angles = angles_rad[cand_idx]
        cand_bboxes = bboxes[cand_idx]
        # Runway half-length along its principal axis: a thin elongated path is
        # well-approximated by half of max(width, height) of its bbox.
        cand_widths = cand_bboxes[:, 2] - cand_bboxes[:, 0]
        cand_heights = cand_bboxes[:, 1] - cand_bboxes[:, 3]
        cand_half_length = np.maximum(np.abs(cand_widths), np.abs(cand_heights)) / 2.0

        cand_tree = cKDTree(cand_centroids)
        d_runway, nn = cand_tree.query(centroids, k=1)
        nn_angle = cand_angles[nn]
        nn_centroid = cand_centroids[nn]
        nn_half_length = np.maximum(cand_half_length[nn], 1e-6)

        diff = _angle_diff_rad(angles_rad, nn_angle)
        df["dist_to_runway"] = d_runway
        df["angle_to_runway"] = np.rad2deg(diff)
        df["parallelism_to_runway"] = np.cos(diff) ** 2
        df["perp_to_runway"] = np.sin(diff) ** 2
        df["inside_runway_bbox"] = _centroid_inside_any_bbox(centroids, cand_bboxes).astype(int)

        # Project (object_centroid - nearest_runway_centroid) onto the
        # runway's long-axis direction (cos θ, sin θ).
        dx = centroids[:, 0] - nn_centroid[:, 0]
        dy = centroids[:, 1] - nn_centroid[:, 1]
        cos_t = np.cos(nn_angle)
        sin_t = np.sin(nn_angle)
        long_off = dx * cos_t + dy * sin_t
        lat_off = -dx * sin_t + dy * cos_t
        df["longitudinal_offset_runway"] = long_off
        df["lateral_offset_runway"] = lat_off
        df["runway_axis_position"] = np.abs(long_off) / nn_half_length
    else:
        df["dist_to_runway"] = np.nan
        df["angle_to_runway"] = np.nan
        df["parallelism_to_runway"] = np.nan
        df["perp_to_runway"] = np.nan
        df["inside_runway_bbox"] = 0
        df["longitudinal_offset_runway"] = np.nan
        df["lateral_offset_runway"] = np.nan
        df["runway_axis_position"] = np.nan

    # ----- taxiway-candidate features -----
    # Taxiways are large filled gray polygons. We approximate "gray" by
    # checking R≈G≈B in a mid-range, then keep only above-median area to
    # avoid picking up small gray text.
    filled_mask = df["filled"].fillna(0).to_numpy(dtype=int).astype(bool)
    fr = df["fill_r"].to_numpy(dtype=float)
    fg = df["fill_g"].to_numpy(dtype=float)
    fb = df["fill_b"].to_numpy(dtype=float)
    rgb_present = ~(np.isnan(fr) | np.isnan(fg) | np.isnan(fb))
    in_range = (
        (fr >= TAXIWAY_GRAY_MIN) & (fr <= TAXIWAY_GRAY_MAX)
        & (fg >= TAXIWAY_GRAY_MIN) & (fg <= TAXIWAY_GRAY_MAX)
        & (fb >= TAXIWAY_GRAY_MIN) & (fb <= TAXIWAY_GRAY_MAX)
    )
    grayish = (
        (np.abs(fr - fg) <= TAXIWAY_GRAY_TOL)
        & (np.abs(fr - fb) <= TAXIWAY_GRAY_TOL)
        & (np.abs(fg - fb) <= TAXIWAY_GRAY_TOL)
    )
    if n > 0:
        tx_area_thresh = np.quantile(bbox_area, TAXIWAY_AREA_QUANTILE)
        not_elongated = pca_ratio <= TAXIWAY_RATIO_MAX
        tx_mask = (filled_mask & rgb_present & in_range & grayish
                   & (bbox_area >= tx_area_thresh) & not_elongated)
        tx_idx = np.where(tx_mask)[0]
        if tx_idx.size == 0:
            # fallback: largest filled objects, regardless of color
            filled_idx = np.where(filled_mask)[0]
            if filled_idx.size:
                top_n = min(TAXIWAY_FALLBACK_TOP_N, filled_idx.size)
                tx_idx = filled_idx[np.argsort(-bbox_area[filled_idx])[:top_n]]

        if tx_idx.size > 0:
            tx_centroids = centroids[tx_idx]
            tx_angles = angles_rad[tx_idx]
            tx_tree = cKDTree(tx_centroids)
            d_tx, tx_nn = tx_tree.query(centroids, k=1)
            df["dist_to_taxiway"] = d_tx
            df["inside_taxiway_bbox"] = _centroid_inside_any_bbox(centroids, bboxes[tx_idx]).astype(int)
            tx_diff = _angle_diff_rad(angles_rad, tx_angles[tx_nn])
            df["angle_to_taxiway"] = np.rad2deg(tx_diff)
            df["parallelism_to_taxiway"] = np.cos(tx_diff) ** 2
        else:
            df["dist_to_taxiway"] = np.nan
            df["inside_taxiway_bbox"] = 0
            df["angle_to_taxiway"] = np.nan
            df["parallelism_to_taxiway"] = np.nan
    else:
        tx_idx = np.empty(0, dtype=int)
        df["dist_to_taxiway"] = np.nan
        df["inside_taxiway_bbox"] = 0
        df["angle_to_taxiway"] = np.nan
        df["parallelism_to_taxiway"] = np.nan

    # ----- edge-based features -----
    # When polygon-edge data is available (one row per anchor-to-anchor edge
    # of every polygon), compute parallelism against the *nearest local edge*
    # of any candidate. This is what catches taxiway labels on curved or
    # L-shaped taxiway polygons, where the polygon's PCA axis is meaningless
    # but the edge nearest the label still has the right local tangent.
    obj_ids = df["object_id"].to_numpy()
    ls_rad = df["longest_segment_angle"].fillna(0).to_numpy(dtype=float) * np.pi / 180.0

    def _edge_features(cand_obj_ids: np.ndarray, prefix: str) -> None:
        if edges_df is None or len(edges_df) == 0 or cand_obj_ids.size == 0:
            df[f"dist_to_{prefix}_edge"] = np.nan
            df[f"angle_to_{prefix}_edge"] = np.nan
            df[f"parallelism_to_{prefix}_edge"] = np.nan
            return
        sub = edges_df[edges_df["object_id"].isin(cand_obj_ids)]
        if len(sub) == 0:
            df[f"dist_to_{prefix}_edge"] = np.nan
            df[f"angle_to_{prefix}_edge"] = np.nan
            df[f"parallelism_to_{prefix}_edge"] = np.nan
            return
        edge_xy = sub[["mid_x", "mid_y"]].to_numpy(dtype=float)
        edge_ang = sub["angle"].to_numpy(dtype=float) * np.pi / 180.0
        edge_tree = cKDTree(edge_xy)
        d_e, e_nn = edge_tree.query(centroids, k=1)
        nn_ang = edge_ang[e_nn]
        # min angle diff using object's principal_angle OR longest_segment_angle
        d1 = _angle_diff_rad(angles_rad, nn_ang)
        d2 = _angle_diff_rad(ls_rad, nn_ang)
        best = np.minimum(d1, d2)
        df[f"dist_to_{prefix}_edge"] = d_e
        df[f"angle_to_{prefix}_edge"] = np.rad2deg(best)
        df[f"parallelism_to_{prefix}_edge"] = np.cos(best) ** 2

    _edge_features(obj_ids[cand_idx] if cand_idx.size else np.empty(0, dtype=int), "runway")
    _edge_features(obj_ids[tx_idx] if tx_idx.size else np.empty(0, dtype=int), "taxiway")

    df = _add_pdf_text_features(df, text_df)

    return df


def add_relational_features(
    df: pd.DataFrame,
    edges_df: pd.DataFrame | None = None,
    text_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add per-airport relational features.

    If `edges_df` is provided (the sidecar `*_edges.csv` produced by
    ExportClassifiedPaths.jsx), compute the edge-based parallelism features as
    well — these handle curved/L-shaped taxiways correctly, where the polygon's
    PCA axis is misleading but the local tangent at each edge is right.

    If `text_df` is provided (output of extract_pdf_text.py), match each path's
    centroid against the PDF text layer and expose pattern features (runway
    designation / taxiway designation / other text / no text).
    """
    pieces = []
    for airport, sub in df.groupby("airport", sort=False):
        e_sub = None if edges_df is None else edges_df[edges_df["airport"] == airport]
        t_sub = None if text_df is None else text_df[text_df["airport"] == airport]
        pieces.append(_per_airport(sub, e_sub, t_sub))
    return pd.concat(pieces, ignore_index=True)


def load_edges(path: str | Path) -> pd.DataFrame:
    """Load the *_edges.csv sidecar produced by ExportClassifiedPaths.jsx."""
    df = pd.read_csv(path)
    expected = {"airport", "object_id", "subpath_index", "edge_index", "mid_x", "mid_y", "angle", "length"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Edges file missing expected columns: {missing}")
    df["airport"] = df["airport"].astype(str)
    for c in ("object_id", "subpath_index", "edge_index"):
        df[c] = df[c].astype(int)
    for c in ("mid_x", "mid_y", "angle", "length"):
        df[c] = df[c].astype(float)
    return df


def load_pdf_text(path: str | Path) -> pd.DataFrame:
    """Load PDF-extracted text from extract_pdf_text.py.

    Bounding boxes are already in Illustrator coordinates (y-flipped during
    extraction), so they directly compare to PathItem bounds. The optional
    `is_known_runway` column flags words that match a runway designation
    actually present in the airport (parsed from "27L-9R"-style pair tokens),
    which is much cleaner than trusting the runway-pattern regex alone.
    """
    df = pd.read_csv(path, dtype={"text": str})
    expected = {"airport", "text", "x_min", "y_min", "x_max", "y_max"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"pdf_text file missing expected columns: {missing}")
    df["airport"] = df["airport"].astype(str)
    df["text"] = df["text"].astype(str)
    for c in ("x_min", "y_min", "x_max", "y_max"):
        df[c] = df[c].astype(float)
    if "is_known_runway" in df.columns:
        df["is_known_runway"] = df["is_known_runway"].fillna(0).astype(int)
    else:
        df["is_known_runway"] = 0
    return df


def _classify_text(t: str, is_known_runway: bool = False) -> str:
    """Return the kind of word.

    'runway_known' means the word matches the runway pattern *and* is one of
    the airport's actual runway designations (parsed from runway-pair text in
    the PDF). 'runway_other' means it matches the regex but isn't a known
    designation (compass-rose labels, magnetic-variation degrees, etc.).
    """
    if not t:
        return "other_text"
    if _RUNWAY_TEXT_RE.match(t):
        return "runway_known" if is_known_runway else "runway_other"
    if _TAXIWAY_TEXT_RE.match(t):
        return "taxiway"
    return "other_text"


def _add_pdf_text_features(df: pd.DataFrame, text_df: pd.DataFrame | None) -> pd.DataFrame:
    """For each path, find which PDF-extracted word (if any) contains its
    centroid, and expose features the model can learn from.

    Features added:
      pdf_text_match         categorical: 'runway', 'taxiway', 'other_text', 'no_text'
      pdf_word_length        character length of the matched word (0 if none)
      pdf_word_dist          distance from path centroid to nearest word center
      pdf_inside_word_bbox   1 if path centroid falls inside any word bbox
    """
    n = len(df)
    if n == 0 or text_df is None or len(text_df) == 0:
        df["pdf_text_match"] = "no_text"
        df["pdf_word_length"] = 0
        df["pdf_word_dist"] = np.nan
        df["pdf_inside_word_bbox"] = 0
        return df

    cx = df["centroid_x"].to_numpy(dtype=float)
    cy = df["centroid_y"].to_numpy(dtype=float)

    words = text_df.reset_index(drop=True)
    wxmin = words["x_min"].to_numpy(dtype=float)
    wxmax = words["x_max"].to_numpy(dtype=float)
    wymin = words["y_min"].to_numpy(dtype=float)
    wymax = words["y_max"].to_numpy(dtype=float)
    word_texts = words["text"].to_numpy()
    word_known = (
        words["is_known_runway"].to_numpy(dtype=int)
        if "is_known_runway" in words.columns
        else np.zeros(len(words), dtype=int)
    )

    word_areas = (wxmax - wxmin) * (wymax - wymin)
    best_word = np.full(n, -1, dtype=int)
    best_area = np.full(n, np.inf, dtype=float)

    # Avoid an n_paths x n_words boolean matrix. Large airport diagrams can
    # have many thousands of Illustrator objects, and that matrix was enough
    # to push the one-shot classifier into "application out of memory".
    path_tree = cKDTree(np.column_stack([cx, cy]))
    for wi in range(len(words)):
        area = word_areas[wi]
        if not np.isfinite(area) or area <= 0:
            continue
        center = ((wxmin[wi] + wxmax[wi]) / 2.0, (wymin[wi] + wymax[wi]) / 2.0)
        radius = float(np.hypot(wxmax[wi] - wxmin[wi], wymax[wi] - wymin[wi]) / 2.0)
        candidates = path_tree.query_ball_point(center, r=radius)
        if not candidates:
            continue
        idx = np.asarray(candidates, dtype=int)
        contained = (
            (cx[idx] >= wxmin[wi])
            & (cx[idx] <= wxmax[wi])
            & (cy[idx] >= wymin[wi])
            & (cy[idx] <= wymax[wi])
            & (area < best_area[idx])
        )
        if contained.any():
            hit_idx = idx[contained]
            best_word[hit_idx] = wi
            best_area[hit_idx] = area

    has_match = best_word >= 0

    text_match = np.full(n, "no_text", dtype=object)
    word_length = np.zeros(n, dtype=int)
    for i in range(n):
        if has_match[i]:
            raw = word_texts[best_word[i]]
            t = "" if raw is None or (isinstance(raw, float) and np.isnan(raw)) else str(raw).strip()
            word_length[i] = len(t)
            text_match[i] = _classify_text(t, bool(word_known[best_word[i]]))

    # Distance to nearest word center regardless of containment
    word_centers = np.column_stack([(wxmin + wxmax) / 2.0, (wymin + wymax) / 2.0])
    tree = cKDTree(word_centers)
    d, _ = tree.query(np.column_stack([cx, cy]), k=1)

    df = df.copy()
    df["pdf_text_match"] = pd.Categorical(
        text_match,
        categories=["runway_known", "runway_other", "taxiway", "other_text", "no_text"],
    )
    df["pdf_word_length"] = word_length
    df["pdf_word_dist"] = d
    df["pdf_inside_word_bbox"] = has_match.astype(int)

    # Interaction features: pdf_text alone is too noisy because many big
    # polygons (taxiway pavement) sit under taxiway-pattern label words. The
    # signal we *actually* want is "this is a small letter-shaped polygon
    # inside a label word." Combining the text pattern with size makes the
    # signal class-specific.
    bbox_rel = df["bbox_area_rel"].fillna(1.0).to_numpy(dtype=float)
    is_letter_sized = bbox_rel < 1e-4  # ~ tiny relative to artboard
    is_runway_known_word = (text_match == "runway_known")
    is_taxiway_word = (text_match == "taxiway")
    df["runway_label_signature"] = (is_letter_sized & is_runway_known_word).astype(int)
    df["taxiway_label_signature"] = (is_letter_sized & is_taxiway_word).astype(int)
    return df


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Add relational features to a paths CSV.")
    ap.add_argument("--in", dest="in_path", type=Path, required=True,
                    help="Input CSV from ExportClassifiedPaths.jsx")
    ap.add_argument("--edges", dest="edges_path", type=Path, default=None,
                    help="Sidecar *_edges.csv from ExportClassifiedPaths.jsx (auto-detected if next to --in)")
    ap.add_argument("--pdf-text", dest="pdf_text_path", type=Path, default=None,
                    help="CSV from extract_pdf_text.py")
    ap.add_argument("--out", dest="out_path", type=Path, required=True,
                    help="Output Parquet (preferred) or CSV")
    args = ap.parse_args()

    print(f"[relational] loading {args.in_path}")
    df = load_features_all(args.in_path)
    print(f"[relational] {len(df):,} rows across {df['airport'].nunique()} airports")

    edges_path = args.edges_path
    if edges_path is None:
        guess = args.in_path.with_name(args.in_path.stem + "_edges" + args.in_path.suffix)
        if guess.exists():
            edges_path = guess
            print(f"[relational] auto-detected edges sidecar: {edges_path}")

    edges_df = None
    if edges_path is not None:
        print(f"[relational] loading edges from {edges_path}")
        edges_df = load_edges(edges_path)
        print(f"[relational] {len(edges_df):,} edges")

    text_df = None
    if args.pdf_text_path is not None:
        print(f"[relational] loading PDF text from {args.pdf_text_path}")
        text_df = load_pdf_text(args.pdf_text_path)
        print(f"[relational] {len(text_df):,} text words across {text_df['airport'].nunique()} airports")

    print("[relational] computing neighbor features per airport ...")
    df_out = add_relational_features(df, edges_df, text_df)

    out = args.out_path
    if out.suffix.lower() == ".parquet":
        df_out.to_parquet(out, index=False)
    else:
        df_out.to_csv(out, index=False)
    print(f"[relational] wrote {len(df_out):,} rows -> {out}")
