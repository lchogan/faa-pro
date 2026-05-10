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
  2. Find the contiguous extension polygon at this end (at most one
     by chart convention):
       - It must have at least one anchor projecting past THIS end on
         the centerline (end_sign * long_pos > half_len). This rejects
         a perpendicular taxiway that crosses the runway in the
         middle and would otherwise be ambiguous between the two ends.
       - Its polygon-boundary-to-runway-boundary minimum distance
         must be <= CONTIGUITY_TOLERANCE_PT (1pt). True extensions
         drawn in FAA charts share the runway edge to within drawing
         precision; 1pt rejects unrelated taxiways the centerline
         might happen to clip far away.
  3. Cast a ray from the runway-end point along the outward unit.
     If an extension is found, find its centerline exit — the largest
     t > 0 at which the ray crosses one of its boundary segments.
     If no extension, t_exit = 0.
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

from taxi_detection import _min_polygon_boundary_distance, _runway_extents


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
# Contiguous-extension search
# ---------------------------------------------------------------------------

def _has_anchor_past_end(poly: dict,
                         cx: float, cy: float,
                         ux: float, uy: float,
                         half_len: float,
                         end_sign: int) -> bool:
    """True if at least one anchor of `poly` projects past THIS end on
    the runway's principal axis. This is the gate that says "the
    polygon extends in the outward direction from this specific end."
    """
    for px, py in _iter_anchors(poly):
        long_pos = (px - cx) * ux + (py - cy) * uy
        if end_sign * long_pos > half_len:
            return True
    return False


def _find_contiguous_extension(
    runway_poly: dict,
    candidate_polys_by_idx: dict[int, dict],
    cx: float, cy: float, ux: float, uy: float,
    half_len: float, end_sign: int,
    contiguity_tolerance: float,
) -> tuple[int, dict] | None:
    """Locate the single Taxiway polygon contiguous with the runway at
    this specific end. Returns (idx, poly) or None.

    Two gates:
      1. At least one anchor past THIS end on the centerline
         (rejects perpendicular crossings in the middle of the runway
         and extensions belonging to the OTHER end).
      2. Polygon-boundary-to-runway-boundary distance <=
         contiguity_tolerance (rejects far-off taxiways the
         centerline ray might clip).

    Defensive: if more than one candidate passes both gates (not
    expected in practice), pick the one whose boundary is closest to
    the runway.
    """
    rwy_subpaths = runway_poly.get("subpaths") or []
    matches: list[tuple[int, dict, float]] = []
    for idx, poly in candidate_polys_by_idx.items():
        if not _has_anchor_past_end(poly, cx, cy, ux, uy, half_len,
                                     end_sign):
            continue
        cand_subpaths = poly.get("subpaths") or []
        if not cand_subpaths:
            continue
        d = _min_polygon_boundary_distance(rwy_subpaths, cand_subpaths)
        if d <= contiguity_tolerance:
            matches.append((idx, poly, d))
    if not matches:
        return None
    matches.sort(key=lambda m: m[2])
    idx, poly, _ = matches[0]
    return idx, poly


def _centerline_exit_t(extension_poly: dict,
                       rwy_end_x: float, rwy_end_y: float,
                       out_ux: float, out_uy: float) -> float:
    """Cast a ray from (rwy_end_x, rwy_end_y) along (out_ux, out_uy)
    and return the largest t > 0 at which the ray crosses any
    boundary segment of `extension_poly`. This is the centerline
    exit — where the ray walks OUT of the extension polygon.

    Returns 0.0 if no intersection found, which means the centerline
    ray doesn't actually cross the extension (rare; possible if the
    extension lies entirely off the centerline, in which case it
    shouldn't have qualified as contiguous in the first place).
    """
    max_t = 0.0
    for (ax, ay), (bx, by) in _polygon_segments(
            extension_poly.get("subpaths") or []):
        t = _ray_segment_intersection_t(
            rwy_end_x, rwy_end_y, out_ux, out_uy, ax, ay, bx, by,
        )
        if t is not None and t > max_t:
            max_t = t
    return max_t


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
      taxi_indices:  iterable of polygon indices currently classified
                     as Taxiways (gray-fill surfaces + end-pads from
                     step 2b). The contiguous-extension search only
                     considers these.
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

        extension = _find_contiguous_extension(
            rwy_poly, candidate_polys_by_idx,
            cx, cy, ux, uy, half_len, end_sign,
            contiguity_tolerance,
        )
        if extension is not None:
            ext_idx, ext_poly = extension
            t_exit = _centerline_exit_t(
                ext_poly, rwy_end_x, rwy_end_y, out_ux, out_uy,
            )
            # A polygon can satisfy "boundary touches runway" and "has
            # an anchor past this end" while still being just an apron
            # adjacent to the runway's SIDE rather than an extension
            # at this end. The decisive geometric test is whether the
            # outward centerline ray actually crosses the polygon: if
            # t_exit is 0 the ray never enters it, so it isn't a
            # true centerline extension. Treat as "no extension" and
            # let the label fall back to "2pt past runway end".
            if t_exit <= 0.0:
                ext_idx = None
                t_exit = 0.0
        else:
            ext_idx = None
            t_exit = 0.0

        # Current label inner edge — the smallest outward-projection
        # of any anchor in the group. This is the glyph anchor closest
        # to the runway (along the centerline). Bbox is deliberately
        # NOT used — see module docstring for why.
        current_inner = math.inf
        for li in claimed_indices:
            for px, py in _iter_anchors(all_polys[li]):
                outward_pos = end_sign * (
                    (px - cx) * ux + (py - cy) * uy
                )
                if outward_pos < current_inner:
                    current_inner = outward_pos
        if not math.isfinite(current_inner):
            continue

        target_inner = half_len + t_exit + label_clearance
        delta = target_inner - current_inner
        dx = delta * out_ux
        dy = delta * out_uy

        for li in claimed_indices:
            translations[li] = (dx, dy)

        diagnostics.append({
            "runway_idx": ri,
            "end_sign": end_sign,
            "token": diag.get("token"),
            "extension_idx": ext_idx,
            "t_exit": t_exit,
            "current_inner": current_inner,
            "target_inner": target_inner,
            "delta_along_centerline": delta,
            "dx": dx,
            "dy": dy,
            "claimed": claimed_indices,
        })

    return translations, diagnostics
