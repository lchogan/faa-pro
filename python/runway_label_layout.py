"""
runway_label_layout.py — runway-label repositioning along the runway
centerline.

Runs after step 3 (runway-label matching) in classify_pipeline.py.
For each (runway, end, label-polygon-group) triple step 3 identified,
this module computes ONE translation vector per group so the label's
nearest-glyph anchor lands exactly LABEL_CLEARANCE_PT points past the
last contiguous pavement edge along the runway centerline.

Why a separate module
---------------------
classify_pipeline.py orchestrates classification (which polygon goes
on which layer). Repositioning is a layout / presentation concern,
not a classification concern, and bundling them muddies both. The
geometry here (ray-vs-segment intersection, principal-axis projection,
rigid translation along an arbitrarily-rotated centerline) also
benefits from being self-contained and unit-testable.

Why not mutate polygon dicts in place
-------------------------------------
The production renderer (render_svg_layers.py) reads source geometry
directly from the PDF, not from polygon dicts or the paths CSV.
Mutating `all_polys` therefore would not change the rendered SVG.
Instead this module returns a `dict[polygon_index] -> (dx, dy)` and
the pipeline writes those translations into the predictions JSON;
the renderer applies them as SVG `transform="translate(dx, dy)"`
attributes per `<path>` element. Original geometry stays intact.

Algorithm — per (runway, end, label_polygon_indices) triple
-----------------------------------------------------------
  1. Principal axis of the runway via PCA: (cx, cy, ux, uy, half_len).
     Outward unit vector for this end is end_sign * (ux, uy).
     Runway-end point on this side: (cx, cy) + half_len * outward.
  2. Cast a ray from the runway-end point along the outward unit.
     For each filled-Taxi polygon, compute every ray-vs-boundary
     intersection. Pick the polygon whose smallest intersection
     ("near_t") is closest to the runway end (i.e. the FIRST polygon
     the centerline encounters past the threshold). The polygon's
     largest intersection ("far_t") is the centerline exit — where
     the ray walks out of that polygon's pavement, and where the
     label needs to clear by `label_clearance` points.
       - Sanity gate: near_t must be ≤ CONTIGUITY_TOLERANCE_PT (1pt).
         Any larger and the polygon doesn't actually start at the
         threshold — there's a gap, and we don't want to align across
         it.
       - This finder takes only FILLED-Taxi indices as candidates, NOT
         the stroked outlines from step 2b. Outlines duplicate fill
         geometry; including them would make "first polygon along
         centerline" ambiguous between the fill and its outline twin.
  3. If no polygon passes, t_exit = 0 and the label lands 2pt past the
     runway end itself.
  4. Lateral centering. The translation also has a perpendicular-to-
     centerline component so the label group's lateral bounds midpoint
     sits exactly on the extended centerline (lat = 0). This handles
     the case where a label group was originally drawn slightly off-
     axis from the runway — without this, the label would still be
     pushed 2pt past the pavement but offset to one side. The midpoint
     is computed from the actual anchor-point min/max lat (NOT bbox)
     so an asymmetric token like "9R" doesn't get pulled off-center
     by the wider R glyph.
  4. Compute current label inner edge:
       current_inner = min over all anchors of all label polygons of
                       end_sign * ((px - cx) * ux + (py - cy) * uy)
     This is the outward-projection of the label's nearest-to-runway
     glyph anchor. Crucially NOT a bbox: an axis-aligned bbox of an
     angled multi-glyph label has empty corners that would push the
     visible glyph too far from the pavement.
  5. Desired inner edge:
       target_inner = half_len + t_exit + LABEL_CLEARANCE_PT
  6. Single translation vector for the whole group:
       delta = target_inner - current_inner
       (dx, dy) = delta * outward_unit
     Same vector applied to every polygon in the group, so the label
     translates rigidly along the centerline. No rotation, no
     glyph-relative deformation.

Geometry coordinates
--------------------
Inputs use PyMuPDF top-left page coordinates (the same frame used by
chart_scene.read_chart and render_svg_layers). Output translations
are in the same frame, so they can be applied directly as SVG
transforms with no flip.

Public
------
  compute_runway_label_translations(
    all_polys, step3_diag, taxi_indices,
    contiguity_tolerance=1.0, label_clearance=2.0,
  ) -> (translations, diagnostics)
"""

from __future__ import annotations

import math

from taxi_detection import _point_in_subpaths, _runway_extents


CONTIGUITY_TOLERANCE_PT = 1.0
LABEL_CLEARANCE_PT = 2.0


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


def _ray_segment_intersection_t(rx: float, ry: float,
                                ux: float, uy: float,
                                ax: float, ay: float,
                                bx: float, by: float) -> float | None:
    """Return the parameter t >= 0 along the ray
    P(t) = (rx, ry) + t * (ux, uy) at which the ray crosses the
    segment a→b within the segment's bounds, else None.

    Solve r + t*u = a + s*(b - a) for (t, s):
        t*ux - s*sx = ax - rx
        t*uy - s*sy = ay - ry
    where (sx, sy) = (bx - ax, by - ay).
    """
    sx = bx - ax
    sy = by - ay
    denom = ux * (-sy) - uy * (-sx)
    if abs(denom) < 1e-12:
        return None
    t = ((ax - rx) * (-sy) - (ay - ry) * (-sx)) / denom
    s = (ux * (ay - ry) - uy * (ax - rx)) / denom
    # Allow a tiny tolerance on s so a corner crossing isn't missed.
    if t < 0.0 or s < -1e-9 or s > 1.0 + 1e-9:
        return None
    return t


# ---------------------------------------------------------------------------
# First-extension-along-centerline search
# ---------------------------------------------------------------------------

def _first_extension_along_centerline(
    candidate_polys_by_idx: dict[int, dict],
    rwy_end_x: float, rwy_end_y: float,
    out_ux: float, out_uy: float,
    near_tolerance: float,
) -> tuple[int, float, float] | None:
    """Cast a ray outward from the runway-end point along
    (out_ux, out_uy) and return the FIRST candidate polygon the ray
    enters — the one whose entry intersection (smallest t > 0) is
    closest to the runway end. Returns (idx, near_t, far_t) or None.

    Why "first along centerline" instead of "closest by polygon-
    boundary distance":
      - Two adjacent gray-fill Taxiways (a hold pad at the threshold
        and the apron immediately behind it) can BOTH touch the runway
        boundary, so a "min boundary distance" tiebreaker becomes
        arbitrary iteration order. The user observed this as the
        algorithm picking the LATER polygon and aligning the label
        past its far end instead of the first polygon's far end.
      - The label sits on the centerline. What matters for label
        positioning is the FIRST polygon the centerline encounters
        past the threshold, and that polygon's FAR edge (where the
        label clears the pavement). Geometric ray-vs-boundary directly
        captures that: a polygon's "near_t" is exactly its centerline
        entry point.

    near_tolerance: maximum allowed near_t. The polygon must START at
        or near the runway end (gap ≤ near_tolerance), not be a
        polygon 30pt past it across a gap.
    """
    best_near_t = math.inf
    best_far_t = 0.0
    best_idx: int | None = None
    for idx, poly in candidate_polys_by_idx.items():
        sp = poly.get("subpaths") or []
        if not sp:
            continue
        # If the runway-end point is INSIDE the polygon, near_t = 0 —
        # the polygon already contains the threshold and the ray
        # starts within it. This is the common case on FAA charts
        # where the gray fill extends across the runway threshold,
        # so a strict "ray must enter from outside" check would
        # otherwise miss the actual contiguous extension.
        point_inside = _point_in_subpaths(rwy_end_x, rwy_end_y, sp)
        ts: list[float] = []
        for (ax, ay), (bx, by) in _polygon_segments(sp):
            t = _ray_segment_intersection_t(
                rwy_end_x, rwy_end_y, out_ux, out_uy,
                ax, ay, bx, by,
            )
            if t is not None and t > 0.0:
                ts.append(t)
        if point_inside:
            near_t = 0.0
            far_t = max(ts) if ts else 0.0
        else:
            if not ts:
                continue
            near_t = min(ts)
            if near_t > near_tolerance:
                continue
            far_t = max(ts)
        if near_t < best_near_t:
            best_near_t = near_t
            best_far_t = far_t
            best_idx = idx
    if best_idx is None:
        return None
    return best_idx, best_near_t, best_far_t


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_runway_label_translations(
    all_polys: list[dict],
    step3_diag: list[dict],
    taxi_indices,
    contiguity_tolerance: float = CONTIGUITY_TOLERANCE_PT,
    label_clearance: float = LABEL_CLEARANCE_PT,
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
      all_polys:     full polygon list from chart_scene.read_chart.
      step3_diag:    diagnostics list emitted by classify_pipeline's
                     `_match_runway_labels`. Only entries with
                     `matched=True` and a non-empty `claimed` list
                     drive a move.
      taxi_indices:  iterable of FILLED-Taxi polygon indices (step 1
                     gray fills only). Stroked outlines from step 2b
                     should NOT be included — they duplicate the fill
                     geometry and would create ambiguous "first along
                     centerline" matches.
    """
    translations: dict[int, tuple[float, float]] = {}
    diagnostics: list[dict] = []
    candidate_polys_by_idx = {i: all_polys[i] for i in taxi_indices}

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
        rwy_end_x = cx + half_len * out_ux
        rwy_end_y = cy + half_len * out_uy

        # Find the FIRST filled-Taxi polygon along the extended
        # centerline past this end. Use that polygon's FAR edge for
        # alignment — the label clears its visible pavement.
        first = _first_extension_along_centerline(
            candidate_polys_by_idx,
            rwy_end_x, rwy_end_y, out_ux, out_uy,
            near_tolerance=contiguity_tolerance,
        )
        if first is not None:
            ext_idx, near_t, t_exit = first
        else:
            ext_idx = None
            near_t = None
            t_exit = 0.0

        # Scan every anchor of every polygon in the label group ONCE,
        # collecting both:
        #   - current_inner: smallest outward-projection (glyph anchor
        #     closest to the runway). Drives the longitudinal move.
        #   - lat_min / lat_max: range of lateral positions across the
        #     whole group's anchors. Drives the lateral centering move.
        # Bbox is deliberately NOT used — an angled label has empty
        # bbox corners that misrepresent the actual visible glyph
        # extent (see module docstring).
        current_inner = math.inf
        lat_min = math.inf
        lat_max = -math.inf
        for li in claimed_indices:
            for px, py in _iter_anchors(all_polys[li]):
                long_pos_signed = (px - cx) * ux + (py - cy) * uy
                outward_pos = end_sign * long_pos_signed
                if outward_pos < current_inner:
                    current_inner = outward_pos
                lat = -(px - cx) * uy + (py - cy) * ux
                if lat < lat_min:
                    lat_min = lat
                if lat > lat_max:
                    lat_max = lat
        if not math.isfinite(current_inner) or not math.isfinite(lat_min):
            continue

        # Longitudinal component: place the inner edge `label_clearance`
        # past the centerline exit of the chosen extension (or past
        # the runway end if no extension).
        target_inner = half_len + t_exit + label_clearance
        delta_long = target_inner - current_inner

        # Lateral component: center the label group's anchor-bounds
        # midpoint on the centerline (lat = 0). Using the midpoint of
        # min/max lateral anchor positions matches "the actual bounds
        # of the art" regardless of glyph asymmetry — for an
        # asymmetric token like "9R" the wide R doesn't pull the
        # lateral center off where it should be.
        lat_center = (lat_min + lat_max) / 2.0
        delta_lat = -lat_center

        # Combine into one (dx, dy). Longitudinal moves along the
        # outward direction (out_ux, out_uy); lateral moves along the
        # principal-axis perpendicular (-uy, ux). The lateral basis
        # is independent of end_sign — perpendicular to the runway is
        # the same regardless of which end you are facing.
        dx = delta_long * out_ux + delta_lat * (-uy)
        dy = delta_long * out_uy + delta_lat * ux

        for li in claimed_indices:
            translations[li] = (dx, dy)

        diagnostics.append({
            "runway_idx": ri,
            "end_sign": end_sign,
            "token": diag.get("token"),
            "extension_idx": ext_idx,
            "near_t": near_t,
            "t_exit": t_exit,
            "current_inner": current_inner,
            "target_inner": target_inner,
            "delta_along_centerline": delta_long,
            "lat_center": lat_center,
            "delta_lat": delta_lat,
            "dx": dx,
            "dy": dy,
            "claimed": claimed_indices,
        })

    return translations, diagnostics
