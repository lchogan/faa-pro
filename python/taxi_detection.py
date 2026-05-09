"""
taxi_detection.py — rule-based taxiway-surface + taxiway-label detection.

This is the authoritative source for Taxiways and Taxiway Labels in the
new pipeline. The ML model is *not* used for these two classes.

Detection logic:
  - Taxiway surface (gray fill): filled polygon with gray fill
    (~#cfcfcf, with leeway for chart variation). Detection of the gray
    fill itself happens in chart_scene.read_chart and is exposed via
    the `is_taxi_surface` flag on each polygon dict.
  - Taxiway surface (runway end pad): unfilled stroked polygon sitting
    flush with a rule-claimed runway's longitudinal end. These are
    run-up / hold pads — drawn as stroked-only rectangles by FAA chart
    convention rather than filled gray. Without an explicit claim, the
    final stroked-only sweep in classify_pipeline demotes them to
    Other; ML can't rescue them either (the v25 head emits only
    Footprints/Stars/Other). See `detect_runway_end_pads`.
  - Taxiway label: a word token matching `^([A-Z][A-Z]?[0-9]{0,2}|[0-9])$`
    (e.g. "C", "C1", "A11", "3") whose bbox touches a taxi surface
    polygon. The K = len(token) nearest unclaimed near-black filled
    glyph polygons are claimed for that token.

The pipeline calls `match_taxi_labels` AFTER runway detection +
runway-label matching, so digit-glyph polygons that belong to a
runway designator are already claimed and unavailable to the
taxi-label K-nearest. This avoids the failure mode where APF runway
"5" sits over a taxiway and gets claimed as a taxi label, leaving
the digit-5 polygon unavailable when step 5 looks for runway labels.

Coordinates are PDF y-down (matching PyMuPDF). Convert to AI y-up at
the consumer boundary if needed.

Public:
    detect_taxi(pdf_path) -> dict with all_polys, clips,
        taxi_surface_indices, taxi_label_indices, text_tokens,
        page_w, page_h. Convenience wrapper that runs both steps.
    match_taxi_labels(all_polys, taxi_surfaces, text_tokens,
                      claimed_polys=None) -> list[int]
        Run only the K-nearest claim, given an externally-loaded
        scene and a set of polygon indices already claimed by other
        steps. Production pipeline calls this directly.
    detect_runway_end_pads(all_polys, runway_indices,
                           claimed_polys=None, max_distance=5.0)
                           -> set[int]
        Find stroked-unfilled run-up/hold pads sitting flush with a
        runway's longitudinal end. Production pipeline calls this
        between rule-based runway detection and runway-label matching.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from chart_scene import read_chart


TAXIWAY_RE = re.compile(r"^([A-Z][A-Z]?[0-9]{0,2}|[0-9])$")


def _point_in_subpaths(x: float, y: float,
                       subpaths: list[list[tuple[float, float]]]) -> bool:
    """Even-odd ray-cast test against a compound polygon's anchor rings."""
    inside = False
    for ring in subpaths:
        n = len(ring)
        if n < 3:
            continue
        j = n - 1
        for i in range(n):
            xi, yi = ring[i]
            xj, yj = ring[j]
            if (yi > y) != (yj > y):
                xt = xi + (xj - xi) * (y - yi) / ((yj - yi) or 1e-12)
                if x < xt:
                    inside = not inside
            j = i
    return inside


def _bbox_touches(t: dict, surfaces: list[dict]) -> bool:
    """True if the token's center sits inside any surface polygon.

    Earlier this was a 5-sample bbox-corners test; OGG showed it was
    too lenient — a runway-slope annotation like "UP" painted just off
    pavement still passed because its bbox corners reached into the
    adjacent taxiway. Real taxi-letter labels are painted *on* the
    pavement, so requiring the centroid to be inside is both more
    conservative and more semantically correct.
    """
    cx, cy = t["cx"], t["cy"]
    for s in surfaces:
        sx0, sy0, sx1, sy1 = s["rect"]
        if not (sx0 <= cx <= sx1 and sy0 <= cy <= sy1):
            continue
        if _point_in_subpaths(cx, cy, s["subpaths"] or []):
            return True
    return False


def match_taxi_labels(all_polys: list[dict],
                      taxi_surfaces: list[dict],
                      text_tokens: list[dict],
                      claimed_polys: set[int] | None = None,
                      excluded_token_ids: set[int] | None = None) -> list[int]:
    """For each token matching the taxi-label regex whose bbox touches
    a taxi-surface polygon, claim the K = len(token.text) nearest
    near-black filled polygons (excluding ones already in
    `claimed_polys`) as taxiway-label glyphs.

    Production pipeline passes a `claimed_polys` set that already
    contains taxi surfaces, runways, and runway labels — so a digit
    glyph that belongs to a runway designator is unavailable here.

    `excluded_token_ids` is a set of id(token) values for tokens
    already consumed by an earlier label step (e.g. runway-label
    matching). Each token can identify polygons at most once across
    the pipeline; without this, a runway designator that overlaps a
    taxi surface would re-qualify here and pull arrowheads or other
    near-black symbols into Taxiway Labels via the K-nearest claim.

    Returns the list of claimed indices in claim order.
    """
    claimed: set[int] = set(claimed_polys or [])
    excluded_ids: set[int] = set(excluded_token_ids or [])
    pattern_tokens = [
        t for t in text_tokens
        if TAXIWAY_RE.match(t["text"]) and id(t) not in excluded_ids
    ]
    qualifying = [t for t in pattern_tokens if _bbox_touches(t, taxi_surfaces)]

    def _is_taxi_candidate(p):
        return p["filled"] and p["is_near_black"] and not p["is_taxi_surface"]

    label_indices: list[int] = []
    for tok in qualifying:
        k = len(tok["text"])
        scored = []
        for i, p in enumerate(all_polys):
            if i in claimed or not _is_taxi_candidate(p):
                continue
            dx = tok["cx"] - p["cx"]
            dy = tok["cy"] - p["cy"]
            scored.append((dx * dx + dy * dy, i))
        scored.sort()
        for _, i in scored[:k]:
            claimed.add(i)
            label_indices.append(i)
    return label_indices


def _runway_extents(subpaths: list[list[tuple[float, float]]],
                    fallback_rect: tuple[float, float, float, float]
                    ) -> tuple[float, float, float, float, float, float]:
    """Principal axis center, unit direction, and half-extents both
    along and across the axis.

    Mirrors classify_pipeline._runway_axis but additionally returns the
    lateral half-extent. End-pad detection needs the lateral half-width
    so it can reject stroked polygons that sit alongside the runway
    (those are not run-up pads — they're either curb stripes or chart
    annotation).

    Returns (cx, cy, ux, uy, half_len, half_wid) where:
      (cx, cy)   — projection-midpoint center (symmetric per-end)
      (ux, uy)   — unit vector along the principal axis
      half_len   — longitudinal half-extent
      half_wid   — lateral half-extent
    """
    pts: list[tuple[float, float]] = []
    for ring in subpaths or []:
        pts.extend(ring)
    x0, y0, x1, y1 = fallback_rect
    if len(pts) < 3:
        if (x1 - x0) >= (y1 - y0):
            return ((x0 + x1) / 2, (y0 + y1) / 2,
                    1.0, 0.0, (x1 - x0) / 2, (y1 - y0) / 2)
        return ((x0 + x1) / 2, (y0 + y1) / 2,
                0.0, 1.0, (y1 - y0) / 2, (x1 - x0) / 2)
    arr = np.asarray(pts, dtype=float)
    naive_centroid = arr.mean(axis=0)
    centered = arr - naive_centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    norm = float(np.linalg.norm(direction)) or 1.0
    ux, uy = float(direction[0]) / norm, float(direction[1]) / norm
    long_axis = np.array([ux, uy])
    lat_axis = np.array([-uy, ux])
    long_proj = centered @ long_axis
    lat_proj = centered @ lat_axis
    long_min, long_max = float(long_proj.min()), float(long_proj.max())
    lat_min, lat_max = float(lat_proj.min()), float(lat_proj.max())
    long_offset = (long_min + long_max) / 2.0
    cx = float(naive_centroid[0]) + long_offset * ux
    cy = float(naive_centroid[1]) + long_offset * uy
    half_len = (long_max - long_min) / 2.0
    half_wid = (lat_max - lat_min) / 2.0
    return (cx, cy, ux, uy, half_len, half_wid)


# Tight tolerance for runway-end-pad detection. Run-up / hold pads on
# FAA charts are drawn flush with the runway threshold — the stroke of
# the pad and the stroke/edge of the runway are practically coincident.
# 5pt is "almost touching"; loosening this risks pulling in arrowheads
# or annotation rectangles that happen to live near a runway end.
RUNWAY_END_PAD_MAX_DISTANCE_PT = 5.0


def _point_to_segment_distance(px: float, py: float,
                               ax: float, ay: float,
                               bx: float, by: float) -> float:
    """Min distance from point (px, py) to the line segment a→b.
    Standard projection-with-clamp; degenerate (a == b) collapses to
    point-to-point distance.
    """
    abx = bx - ax
    aby = by - ay
    denom = abx * abx + aby * aby
    if denom < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * abx + (py - ay) * aby) / denom
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    qx = ax + t * abx
    qy = ay + t * aby
    return math.hypot(px - qx, py - qy)


def _min_polygon_boundary_distance(
    sp_a: list[list[tuple[float, float]]],
    sp_b: list[list[tuple[float, float]]],
) -> float:
    """Approximate min boundary-to-boundary distance between two
    compound polygons defined by their subpath rings.

    For each anchor in A, measure the minimum distance to any segment
    of B; then symmetrically. The smaller of the two is the boundary
    distance — this is exact when the polygons don't intersect (the
    closest pair is always anchor-of-one to segment-of-other in 2D),
    and returns 0 (or near-zero) when they touch or overlap.

    Subpath anchors are taken from `subpaths` directly. Bbox is
    deliberately NOT used — runways are arbitrarily rotated on FAA
    charts and a bbox-based projection would introduce large errors.
    """
    def _segments(sp):
        for ring in sp or []:
            n = len(ring)
            if n < 2:
                continue
            for i in range(n):
                yield ring[i], ring[(i + 1) % n]

    def _anchors(sp):
        for ring in sp or []:
            for pt in ring:
                yield pt

    min_d = math.inf
    segs_b = list(_segments(sp_b))
    if segs_b:
        for px, py in _anchors(sp_a):
            for (ax, ay), (bx, by) in segs_b:
                d = _point_to_segment_distance(px, py, ax, ay, bx, by)
                if d < min_d:
                    min_d = d
                    if min_d == 0.0:
                        return 0.0
    segs_a = list(_segments(sp_a))
    if segs_a:
        for px, py in _anchors(sp_b):
            for (ax, ay), (bx, by) in segs_a:
                d = _point_to_segment_distance(px, py, ax, ay, bx, by)
                if d < min_d:
                    min_d = d
                    if min_d == 0.0:
                        return 0.0
    return min_d


def detect_runway_end_pads(all_polys: list[dict],
                           runway_indices,
                           claimed_polys: set[int] | None = None,
                           max_distance: float = RUNWAY_END_PAD_MAX_DISTANCE_PT
                           ) -> set[int]:
    """Find stroked-unfilled polygons sitting flush with a rule-claimed
    runway's longitudinal end. These are taxiway run-up / hold pads —
    drawn by FAA chart convention as stroked-only rectangles instead of
    filled gray, so the gray-fill rule (step 1) misses them.

    Why this is a *narrow* rule and not an ML class:
      - The pads sit practically touching the runway when they exist;
        a 5pt tolerance is sufficient and avoids false positives.
      - Step 7 of classify_pipeline (the stroked-only sweep) demotes
        any surviving stroked polygon to Other, so without an explicit
        rule claim these pads end up on the Other layer regardless of
        what ML thinks.
      - The current ML head (v25) emits only Footprints / Stars / Other
        and so cannot route these polygons to Taxiways even if step 7
        were relaxed.

    Detection geometry — works in the runway's principal-axis frame
    (NOT bboxes, which are misleading on rotated runways):

      1. Candidate is stroked AND NOT filled AND not already claimed.
      2. Candidate has at least one anchor point that is BOTH past one
         of the runway's longitudinal ends AND within the centerline
         band laterally (|lat| ≤ half_wid + max_distance). This is the
         "polygon sits on the extended centerline past the threshold"
         test, computed from anchor projections so chart rotation is
         handled exactly.
      3. Polygon-boundary-to-polygon-boundary minimum distance between
         the candidate and the runway is ≤ max_distance. This is the
         "practically touching" test — measured anchor-of-one against
         segments-of-the-other (exact for non-intersecting polygons in
         2D), so it doesn't depend on bbox alignment.

    Args:
        all_polys: full polygon list from chart_scene.read_chart.
        runway_indices: indices in all_polys claimed as Runways by
            step 2.
        claimed_polys: indices already claimed by earlier steps; these
            are skipped.
        max_distance: longitudinal/lateral/boundary tolerance in
            points; default 5pt. See module-level
            RUNWAY_END_PAD_MAX_DISTANCE_PT.

    Returns set of polygon indices to add to the Taxiways layer.
    """
    claimed = set(claimed_polys or [])
    pad_indices: set[int] = set()
    for ri in runway_indices:
        rp = all_polys[ri]
        rwy_subpaths = rp.get("subpaths") or []
        cx, cy, ux, uy, half_len, half_wid = _runway_extents(
            rwy_subpaths, rp["rect"]
        )
        # Lateral band for "on the extended centerline" — runway lateral
        # half-width plus the same touching tolerance. Hold-pad shoulders
        # sometimes flare a few points wider than the runway proper, so
        # we accept anchors slightly outside the runway's lateral
        # footprint. Anchors farther out laterally are rejected because
        # they're not on the centerline at all (e.g. stroked annotation
        # off to the side).
        lat_band = half_wid + max_distance
        for i, p in enumerate(all_polys):
            if i in claimed or i in pad_indices:
                continue
            if not p.get("stroked") or p.get("filled"):
                continue
            cand_subpaths = p.get("subpaths") or []
            if not cand_subpaths:
                continue
            # Step 2: any anchor on the extended centerline past an end?
            on_centerline_past_end = False
            for ring in cand_subpaths:
                for ax_, ay_ in ring:
                    dxv = ax_ - cx
                    dyv = ay_ - cy
                    long_pos = dxv * ux + dyv * uy
                    lat = -dxv * uy + dyv * ux
                    if abs(long_pos) > half_len and abs(lat) <= lat_band:
                        on_centerline_past_end = True
                        break
                if on_centerline_past_end:
                    break
            if not on_centerline_past_end:
                continue
            # Step 3: polygon boundaries practically touching.
            d = _min_polygon_boundary_distance(rwy_subpaths, cand_subpaths)
            if d > max_distance:
                continue
            pad_indices.add(i)
    return pad_indices


def detect_taxi(pdf_path: Path) -> dict:
    """Run rule-based taxiway-surface and taxiway-label detection on a
    single-page FAA airport diagram PDF. Convenience wrapper that
    runs surface detection + label matching back-to-back. The
    production pipeline calls `read_chart` and `match_taxi_labels`
    directly so it can interleave runway and runway-label claims
    between the two steps.
    """
    scene = read_chart(pdf_path)
    all_polys = scene["all_polys"]
    text_tokens = scene["text_tokens"]
    clips = scene["clips"]
    page_w = scene["page_w"]
    page_h = scene["page_h"]

    taxi_surface_indices = [i for i, p in enumerate(all_polys)
                            if p["is_taxi_surface"]]
    taxi_surfaces = [all_polys[i] for i in taxi_surface_indices]

    label_indices = match_taxi_labels(
        all_polys, taxi_surfaces, text_tokens,
        claimed_polys=set(taxi_surface_indices),
    )

    return {
        "all_polys": all_polys,
        "clips": clips,
        "taxi_surface_indices": taxi_surface_indices,
        "taxi_label_indices": label_indices,
        "text_tokens": text_tokens,
        "page_w": page_w,
        "page_h": page_h,
    }
