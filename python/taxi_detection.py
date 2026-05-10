"""
taxi_detection.py — rule-based taxiway-surface + taxiway-label detection.

This is the authoritative source for Taxiways and Taxiway Labels in the
new pipeline. The ML model is *not* used for these two classes.

Detection logic:
  - Taxiway surface (gray fill): filled polygon with gray fill
    (~#cfcfcf, with leeway for chart variation). Detection of the gray
    fill itself happens in chart_scene.read_chart and is exposed via
    the `is_taxi_surface` flag on each polygon dict.
  - Taxiway surface (paired stroked outline): a stroked-unfilled polygon
    whose centroid lies inside one of the gray-fill Taxiways. FAA charts
    draw each gray taxi surface (including hold pads) as a filled gray
    polygon AND a separate stroked outline polygon stacked on top.
    Step 1 catches the gray fill; without this rule the matching outline
    would be demoted to Other by the final stroked-only sweep, breaking
    the visual when the Taxiways layer is rendered. Centroid-in-filled-
    taxi uniquely identifies the outline relationship — arbitrary lines
    or markings near pavement do not have their centroid inside a
    gray-fill polygon. See `detect_paired_stroked_outlines`.
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
    detect_paired_stroked_outlines(all_polys, taxi_surface_indices,
                                   claimed_polys=None) -> set[int]
        Find stroked-unfilled polygons whose centroid sits inside any
        already-claimed gray-fill Taxiway. These are the outline-on-fill
        companion polygons that FAA charts draw over each taxi surface.
    _runway_extents(subpaths, fallback_rect) -> 6-tuple
        (cx, cy, ux, uy, half_len, half_wid) — runway principal-axis
        center, unit direction, and half-extents in long/lat. Re-used
        by runway_label_layout.py.
    _min_polygon_boundary_distance(sp_a, sp_b) -> float
        Approximate min boundary-to-boundary distance between two
        compound polygons. Used by runway_label_layout.py for the
        contiguous-extension touching test.
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


def detect_paired_stroked_outlines(
    all_polys: list[dict],
    taxi_surface_indices,
    claimed_polys: set[int] | None = None,
) -> set[int]:
    """Find stroked-unfilled polygons that are visual outlines of an
    already-claimed gray-fill taxi surface — i.e. their centroid lies
    inside one of the filled-Taxi polygons.

    Why this rule exists
    --------------------
    FAA airport diagrams draw each gray taxiway (apron, run-up area,
    hold pad, taxi segment) as TWO stacked polygons:
      1. A filled gray polygon — the pavement fill (caught by step 1).
      2. A separate stroked-unfilled polygon over it — the pavement
         outline.
    Both are part of the visible taxiway. Without this rule the
    outline polygon falls through to step 7's stroked-only sweep and
    ends up on Other, leaving the rendered Taxiways layer with the
    fill but no outline.

    The detection test
    ------------------
    A stroked-unfilled polygon's centroid is inside one of the filled-
    Taxi polygons. Centroid-in-fill is a robust proxy for the
    fill/outline pairing because:
      - A polygon's stroked outline shares its filled twin's centroid
        exactly (same shape, same anchor extent).
      - Arbitrary stroked artwork near pavement (centerline marks,
        threshold stripes, arrowheads, line-art annotations) does NOT
        have its centroid sitting inside a gray-fill region — those
        live at the boundary or outside.
      - 1-D line segments fail the test naturally: a line's centroid
        is on the line itself, which is not inside any 2-D fill region
        unless the line genuinely overlaps the pavement, in which case
        it IS visually part of the taxiway and belongs on Taxiways
        anyway.

    Args:
        all_polys: full polygon list from chart_scene.read_chart.
        taxi_surface_indices: indices already claimed as Taxiways by
            step 1 (gray-fill detection).
        claimed_polys: indices already claimed by any step; skipped.

    Returns:
        set of polygon indices to add to the Taxiways layer.
    """
    claimed = set(claimed_polys or [])
    surfaces = [all_polys[i] for i in taxi_surface_indices]
    outline_indices: set[int] = set()
    for i, p in enumerate(all_polys):
        if i in claimed:
            continue
        if not p.get("stroked") or p.get("filled"):
            continue
        cx, cy = p["cx"], p["cy"]
        for s in surfaces:
            sx0, sy0, sx1, sy1 = s["rect"]
            # Cheap bbox prefilter before the ray-cast.
            if not (sx0 <= cx <= sx1 and sy0 <= cy <= sy1):
                continue
            if _point_in_subpaths(cx, cy, s.get("subpaths") or []):
                outline_indices.add(i)
                break
    return outline_indices


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
