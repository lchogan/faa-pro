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
