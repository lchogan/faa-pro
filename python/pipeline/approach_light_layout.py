"""approach_light_layout.py — approach-lighting symbol placement.

Runs after final classification (step 8b in classify_pipeline.py), parallel
to runway_label_layout but with a key difference: lights MAY overlap runways
and taxiways, so there is NO collision-march. Each symbol sits at a fixed gap
off the pavement end, centered on the runway centerline, oriented to the
runway, and scaled by a single per-document factor (keyed to the widest
runway's depicted width) so all lights on a page share one size relationship.

Per-end ALS codes come from NASR (approach_lights_data.load_approach_lights).
Which geometric end is "16" vs "34" is resolved from step3_diag's existing
runway-number<->end pairing — no lat/long or compass needed. If only one end's
number was detected, the other end's designator is the reciprocal.

Output (light_groups), in PyMuPDF top-left (y-down) coords — same frame as the
per-polygon translate, so the renderer applies it with no flip:
  {artwork_key, x, y, dir_x, dir_y, target_width}
where (x, y) is the attach point at the threshold (gap already applied),
(dir_x, dir_y) is the outward unit vector (forward axis of the symbol), and
target_width = k * (2 * max_half_wid) is the rendered width one runway-width
reference maps to (same for every light on the page).
"""
from __future__ import annotations

import math
import re

from pipeline.taxi_detection import _runway_extents
from pipeline.extract_pdf_text import _normalize_designation
from pipeline.approach_lights_data import als_code_to_artwork_key


# Tuning knobs (calibrate visually on DEN + a small field — see plan Task 10).
DEFAULT_K = 0.5   # overall light size relative to the widest depicted width
DEFAULT_G = 0.3   # gap off the pavement end, as a fraction of target_width

_DESIG_RE = re.compile(r"^(\d{1,2})([LRC]?)$")
_COMPASS_RECIP = {
    "N": "S", "S": "N", "E": "W", "W": "E",
    "NE": "SW", "SW": "NE", "NW": "SE", "SE": "NW",
}
_LR_SWAP = {"L": "R", "R": "L", "C": "C", "": ""}


def _reciprocal_designation(desig: str) -> str | None:
    """Return the opposite-end designation: numeric ends add 18 (mod 36) and
    swap L<->R; compass ends flip to their opposite. None if unrecognized.
    Input/output are in canonical form (no leading zero)."""
    m = _DESIG_RE.match(desig or "")
    if m:
        n = int(m.group(1)) + 18
        if n > 36:
            n -= 36
        return f"{n}{_LR_SWAP[m.group(2)]}"
    return _COMPASS_RECIP.get((desig or "").upper())


def _document_width_reference(all_polys, runway_indices) -> float:
    """Largest depicted half-width across all runways on the page. Behind one
    function so it can be swapped for a median/percentile during tuning."""
    max_half_wid = 0.0
    for ri in runway_indices:
        poly = all_polys[ri]
        _, _, _, _, _, half_wid = _runway_extents(
            poly.get("subpaths") or [], poly["rect"],
        )
        if half_wid > max_half_wid:
            max_half_wid = half_wid
    return max_half_wid


def compute_approach_light_placements(
    all_polys: list[dict],
    step3_diag: list[dict],
    runway_indices,
    als_by_end: dict[str, str],
    *,
    k: float = DEFAULT_K,
    g: float = DEFAULT_G,
) -> tuple[list[dict], list[dict]]:
    """Compute one light_group per runway end that has an in-scope ALS.

    See module docstring for the output schema. `als_by_end` maps normalized
    end designators to raw FAA codes (from approach_lights_data). `k`/`g` are
    the size/gap tuning knobs.
    """
    light_groups: list[dict] = []
    diagnostics: list[dict] = []

    max_half_wid = _document_width_reference(all_polys, runway_indices)
    if max_half_wid <= 0.0:
        return light_groups, diagnostics
    target_width = k * (2.0 * max_half_wid)
    gap = g * target_width

    # Group step3 entries by runway.
    by_runway: dict[int, list[dict]] = {}
    for d in step3_diag:
        by_runway.setdefault(d["runway_idx"], []).append(d)

    for ri in runway_indices:
        ends = by_runway.get(ri, [])

        # Designator per end_sign, from the matched runway-number token.
        desig_by_sign: dict[int, str] = {}
        for d in ends:
            if d.get("matched") and d.get("token"):
                nd = _normalize_designation(str(d["token"]))
                if nd:
                    desig_by_sign[int(d["end_sign"])] = nd
        if not desig_by_sign:
            diagnostics.append({"runway_idx": ri, "skipped": "no_matched_number"})
            continue
        # Reciprocal inference when only one end carried a number.
        if len(desig_by_sign) == 1:
            (known_sign, known_desig), = desig_by_sign.items()
            recip = _reciprocal_designation(known_desig)
            if recip:
                desig_by_sign.setdefault(-known_sign, recip)

        poly = all_polys[ri]
        cx, cy, ux, uy, half_len, _ = _runway_extents(
            poly.get("subpaths") or [], poly["rect"],
        )
        if not math.isfinite(half_len):
            continue

        for end_sign, desig in desig_by_sign.items():
            code = als_by_end.get(desig)
            key = als_code_to_artwork_key(code)
            if key is None:
                # Log only genuinely-unrecognized codes so new/edge FAA codes
                # surface. "NONE" is the explicit no-approach-lights sentinel
                # (most runway ends) — not an error, so don't flag it.
                if code and code.strip().upper() != "NONE":
                    diagnostics.append({
                        "runway_idx": ri, "end_sign": end_sign,
                        "designator": desig, "code": code, "skipped": "unmapped_code",
                    })
                continue
            out_x, out_y = end_sign * ux, end_sign * uy
            ax = cx + (half_len + gap) * out_x
            ay = cy + (half_len + gap) * out_y
            light_groups.append({
                "artwork_key": key,
                "x": round(ax, 4),
                "y": round(ay, 4),
                "dir_x": round(out_x, 6),
                "dir_y": round(out_y, 6),
                "target_width": round(target_width, 4),
            })
            diagnostics.append({
                "runway_idx": ri, "end_sign": end_sign,
                "designator": desig, "code": code, "artwork_key": key,
            })

    return light_groups, diagnostics
