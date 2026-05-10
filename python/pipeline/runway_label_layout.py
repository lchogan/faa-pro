"""
runway_label_layout.py — runway-label repositioning along the runway
centerline.

Runs after final classification in classify_pipeline.py. For every
(runway, end, label-polygon-group) triple step 3 identified, this
module computes ONE translation vector per group so the label sits
centered on the runway centerline AND is shifted outward along that
centerline far enough that no glyph polygon overlaps any visible
polygon (taxiways, runways, footprints, taxiway labels, other runway
labels).

Why a separate module
---------------------
classify_pipeline.py orchestrates classification (which polygon goes
on which layer). Repositioning is a layout / presentation concern,
not a classification concern, and bundling them muddies both. The
geometry here (polygon-polygon overlap, principal-axis projection,
march-and-test) also benefits from being self-contained and unit-
testable.

Why not mutate polygon dicts in place
-------------------------------------
The production renderer (render_svg_layers.py) reads source geometry
directly from the PDF, not from polygon dicts or the paths CSV.
Mutating `all_polys` therefore would not change the rendered SVG.
Instead this module returns a `dict[polygon_index] -> (dx, dy)` and
the pipeline writes those translations into the predictions JSON;
the renderer applies them as SVG `transform="translate(dx, dy)"`
attributes per `<path>` element. Original geometry stays intact.

Why march-and-test instead of "find the hold-pad polygon"
---------------------------------------------------------
The previous algorithm picked the FIRST gray-fill polygon along the
extended centerline and aligned the label past that polygon's far
edge. It was brittle in two ways: (a) when the centerline crosses a
second runway past the runway end, the rule "FIRST gray-fill polygon"
ignored the crossing runway and the label ended up sitting on it;
(b) any classification gap (hold pad missing from gray fill, runway
end blunt, fragmented pavement) silently produced a wrong placement.
March-and-test is geometry-only: it asks "is the label glyph hitting
anything visible?" and steps outward until the answer is no. It
naturally handles crossing runways, fragmented pavement, missing
hold pads, and any future class added to the collision set.

Algorithm — per (runway, end, label_polygon_indices) triple
-----------------------------------------------------------
  1. Principal axis of the runway via PCA: (cx, cy, ux, uy, half_len).
     Outward unit vector for this end is end_sign * (ux, uy).
     Runway-end point on this side: (cx, cy) + half_len * outward.
  2. Lateral component. Center the label group's anchor-bounds
     midpoint on the centerline (lat = 0). Handles labels drawn
     slightly off-axis from the runway. Anchor-bounds (NOT bbox)
     so an asymmetric token like "9R" doesn't get pulled off-center
     by the wider R glyph.
  3. Longitudinal march.
     Start with the label inner edge AT the runway end:
       delta_long_start = half_len - current_inner
     Then in MARCH_STEP_PT (0.5pt) increments outward, translate
     every label polygon by (delta_long * out_unit + lat_delta) and
     test polygon-polygon overlap of each translated glyph against
     every collision polygon (taxiways, runways, footprints, taxi
     labels, other runway labels — the group's own polys excluded).
     The first step at which no glyph overlaps anything is the
     resting position.
  4. Add LABEL_CLEARANCE_PT (2.0pt) outward as a minimum-clearance
     buffer. Net result: the label is between 2.0pt and
     (2.0 + march_step) pt past the nearest pavement edge — matching
     the visual spacing of the older hold-pad-based algorithm.
  5. Final translation vector:
       (dx, dy) = delta_long_final * out_unit + delta_lat * perp_unit
     applied uniformly to every polygon in the label group — rigid
     translation, no rotation, no glyph-relative deformation.

Geometry coordinates
--------------------
Inputs use PyMuPDF top-left page coordinates (the same frame used by
chart_scene.read_chart and render_svg_layers). Output translations
are in the same frame, so they can be applied directly as SVG
transforms with no flip.

Public
------
  compute_runway_label_translations(
    all_polys, step3_diag, collision_indices,
    label_clearance=2.0, march_step=0.5, max_march_pt=200.0,
  ) -> (translations, diagnostics)
"""

from __future__ import annotations

import math

from pipeline.taxi_detection import _point_in_subpaths, _runway_extents


LABEL_CLEARANCE_PT = 2.0
MARCH_STEP_PT = 0.5
MAX_MARCH_PT = 200.0


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _iter_anchors(poly: dict):
    """Yield every anchor point (x, y) across all subpaths of a poly."""
    for ring in poly.get("subpaths") or []:
        for pt in ring:
            yield pt


def _polygon_segments(subpaths):
    """Yield (a, b) tuples for every closed-ring segment in subpaths."""
    for ring in subpaths or []:
        n = len(ring)
        if n < 2:
            continue
        for i in range(n):
            yield ring[i], ring[(i + 1) % n]


def _segments_intersect(p1, p2, q1, q2) -> bool:
    """True iff segment p1-p2 crosses segment q1-q2.

    Standard CCW orientation test plus a collinear-overlap fallback.
    Endpoint-on-segment counts as intersecting — for label clearance
    "edge touching edge" is still overlap that the march should push
    past.
    """
    def ccw(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    d1 = ccw(q1, q2, p1)
    d2 = ccw(q1, q2, p2)
    d3 = ccw(p1, p2, q1)
    d4 = ccw(p1, p2, q2)
    if (((d1 > 0) and (d2 < 0)) or ((d1 < 0) and (d2 > 0))) and \
       (((d3 > 0) and (d4 < 0)) or ((d3 < 0) and (d4 > 0))):
        return True

    def on_segment(a, b, c):
        return (min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and
                min(a[1], b[1]) <= c[1] <= max(a[1], b[1]))

    if d1 == 0 and on_segment(q1, q2, p1):
        return True
    if d2 == 0 and on_segment(q1, q2, p2):
        return True
    if d3 == 0 and on_segment(p1, p2, q1):
        return True
    if d4 == 0 and on_segment(p1, p2, q2):
        return True
    return False


def _rects_overlap(a, b) -> bool:
    """True iff axis-aligned rects (x0, y0, x1, y1) overlap (touching counts)."""
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _polygons_overlap(a_subpaths, b_subpaths) -> bool:
    """True iff two compound polygons (subpaths) overlap.

    Tests in order: any vertex of A inside B, any vertex of B inside
    A, any edge of A crosses any edge of B. Together these catch
    containment in either direction and partial intersection. Caller
    should bbox-prefilter to avoid the O(N*M) edge crossings on
    obviously-disjoint pairs.
    """
    for ring in a_subpaths or []:
        for x, y in ring:
            if _point_in_subpaths(x, y, b_subpaths):
                return True
    for ring in b_subpaths or []:
        for x, y in ring:
            if _point_in_subpaths(x, y, a_subpaths):
                return True
    for a, b in _polygon_segments(a_subpaths):
        for c, d in _polygon_segments(b_subpaths):
            if _segments_intersect(a, b, c, d):
                return True
    return False


def _translate_subpaths(subpaths, dx: float, dy: float):
    """Return a new subpaths list with every anchor shifted by (dx, dy)."""
    return [[(x + dx, y + dy) for (x, y) in ring] for ring in subpaths or []]


def _translate_rect(rect, dx: float, dy: float):
    """Return (x0+dx, y0+dy, x1+dx, y1+dy)."""
    return (rect[0] + dx, rect[1] + dy, rect[2] + dx, rect[3] + dy)


# ---------------------------------------------------------------------------
# Label-group collision check (translated)
# ---------------------------------------------------------------------------

def _label_group_collides(
    label_polys: list[dict],
    dx: float, dy: float,
    collision_polys: list[dict],
) -> bool:
    """True iff any polygon in label_polys, after translation by
    (dx, dy), overlaps any polygon in collision_polys.

    Bbox-prefilters every candidate pair before the deep polygon test.
    Builds the translated subpaths for each label glyph at most once
    per call (lazily, only when at least one collision polygon's bbox
    intersects).
    """
    for lp in label_polys:
        l_rect = _translate_rect(lp["rect"], dx, dy)
        translated = None  # built lazily on the first deep test
        for cp in collision_polys:
            if not _rects_overlap(l_rect, cp["rect"]):
                continue
            if translated is None:
                translated = _translate_subpaths(lp["subpaths"], dx, dy)
            if _polygons_overlap(translated, cp["subpaths"]):
                return True
    return False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_runway_label_translations(
    all_polys: list[dict],
    step3_diag: list[dict],
    collision_indices,
    label_clearance: float = LABEL_CLEARANCE_PT,
    march_step: float = MARCH_STEP_PT,
    max_march_pt: float = MAX_MARCH_PT,
) -> tuple[dict[int, tuple[float, float]], list[dict]]:
    """Compute translation vectors for every label group identified
    by step 3.

    Returns:
      (translations, diagnostics)
        translations: dict[polygon_index] = (dx, dy). Polygons not in
            the dict get no translation at render time.
        diagnostics:  list[dict] — one entry per matched (runway, end)
            with the values fed into the calculation. Useful for
            debugging unexpected layout shifts and for the inline
            print-out from classify_pipeline.

    Inputs:
      all_polys:           full polygon list from chart_scene.read_chart.
      step3_diag:          diagnostics list emitted by classify_pipeline's
                           `_match_runway_labels`. Only entries with
                           `matched=True` and a non-empty `claimed` list
                           drive a move.
      collision_indices:   iterable of polygon indices to treat as
                           visible obstacles. Caller picks classes —
                           typically Taxiways, Runways, Footprints,
                           Taxiway Labels, Runway Labels. Each label
                           group's own polygons are removed from this
                           set automatically before its march.
      label_clearance:     extra outward distance added once the march
                           finds a no-overlap step (default 2.0 pt —
                           matches the spacing the older hold-pad-
                           based algorithm produced).
      march_step:          increment along the centerline per step
                           (default 0.5 pt). Smaller = tighter stop,
                           more checks.
      max_march_pt:        safety cap on march distance (default 200
                           pt). If the label hasn't cleared by this
                           distance, it stops here and the diagnostic
                           records `collision_at_rest=True`.
    """
    translations: dict[int, tuple[float, float]] = {}
    diagnostics: list[dict] = []
    collision_index_set = set(collision_indices)

    for diag in step3_diag:
        if not diag.get("matched"):
            continue
        claimed_indices: list[int] = list(diag.get("claimed") or [])
        if not claimed_indices:
            continue
        ri = diag["runway_idx"]
        end_sign = diag["end_sign"]
        rwy_poly = all_polys[ri]
        cx, cy, ux, uy, half_len, _ = _runway_extents(
            rwy_poly.get("subpaths") or [], rwy_poly["rect"],
        )
        out_ux = end_sign * ux
        out_uy = end_sign * uy
        # Lateral basis: principal-axis perpendicular. Sign is
        # independent of end_sign — perpendicular is symmetric.
        perp_x = -uy
        perp_y = ux

        # Scan every anchor of every polygon in the label group ONCE,
        # collecting:
        #   - current_inner: smallest outward-projection (glyph anchor
        #     closest to the runway). Drives the longitudinal start.
        #   - lat_min / lat_max: range of lateral positions across the
        #     whole group's anchors. Drives the lateral centering move.
        # Bbox is deliberately NOT used — an angled label has empty
        # bbox corners that misrepresent the actual visible glyph
        # extent.
        current_inner = math.inf
        lat_min = math.inf
        lat_max = -math.inf
        for li in claimed_indices:
            for px, py in _iter_anchors(all_polys[li]):
                long_pos_signed = (px - cx) * ux + (py - cy) * uy
                outward_pos = end_sign * long_pos_signed
                if outward_pos < current_inner:
                    current_inner = outward_pos
                lat = (px - cx) * perp_x + (py - cy) * perp_y
                if lat < lat_min:
                    lat_min = lat
                if lat > lat_max:
                    lat_max = lat
        if not math.isfinite(current_inner) or not math.isfinite(lat_min):
            continue

        # Lateral component: center anchor-bounds midpoint on centerline.
        lat_center = (lat_min + lat_max) / 2.0
        delta_lat = -lat_center

        # Build collision polygon list for this group (exclude self).
        own = set(claimed_indices)
        collision_polys = [
            all_polys[i] for i in collision_index_set
            if i not in own
        ]
        label_polys = [all_polys[i] for i in claimed_indices]

        # Longitudinal march. Canonical starting position is the
        # label inner edge AT the runway end, regardless of where
        # the chart artist drew it (we'll either pull it back to the
        # threshold and march outward, or push it forward — whichever
        # finds the first clear position).
        delta_long_start = half_len - current_inner
        n_steps = int(max_march_pt / march_step) + 1
        cleared_offset_pt: float | None = None
        for k in range(n_steps):
            offset_pt = k * march_step
            delta_long = delta_long_start + offset_pt
            dx = delta_long * out_ux + delta_lat * perp_x
            dy = delta_long * out_uy + delta_lat * perp_y
            if not _label_group_collides(
                label_polys, dx, dy, collision_polys,
            ):
                cleared_offset_pt = offset_pt
                break

        if cleared_offset_pt is None:
            # Hit the cap. Park the label at max_march_pt past the
            # runway end and flag for inspection. Sane charts should
            # never reach this branch.
            cleared_offset_pt = max_march_pt
            collision_at_rest = True
        else:
            collision_at_rest = False

        # Add the minimum-clearance buffer so the label is at least
        # `label_clearance` pt past the nearest pavement edge.
        delta_long_final = (
            delta_long_start + cleared_offset_pt + label_clearance
        )
        dx = delta_long_final * out_ux + delta_lat * perp_x
        dy = delta_long_final * out_uy + delta_lat * perp_y

        for li in claimed_indices:
            translations[li] = (dx, dy)

        diagnostics.append({
            "runway_idx": ri,
            "end_sign": end_sign,
            "token": diag.get("token"),
            "current_inner": current_inner,
            "delta_long_start": delta_long_start,
            "march_offset_pt": cleared_offset_pt,
            "delta_long_final": delta_long_final,
            "collision_at_rest": collision_at_rest,
            "lat_center": lat_center,
            "delta_lat": delta_lat,
            "dx": dx,
            "dy": dy,
            "claimed": claimed_indices,
        })

    return translations, diagnostics
