"""approach_lights_data.py — FAA approach-lighting data: load per-runway-end
ALS codes from NASR, and map those codes to artwork keys.

Two concerns, kept together because both are about NASR's approach-light
codes:

  * load_approach_lights(csv_path, airport) -> {end_designator: raw_code}
        Reads the NASR runway-end table (APT_RWY_END.csv / equivalent).
  * als_code_to_artwork_key(code) -> artwork_key | None
        Maps an FAA ALS code to the artwork archetype that depicts it.
        Several codes share one symbol; NONE/blank/out-of-scope -> None.

Authoritative FAA code set (apt.txt field "APPROACH LIGHT SYSTEM"):
  ALSAF, ALSF1, ALSF2, MALS, MALSF, MALSR, SSALS, SSALF, SSALR,
  NEON, ODALS, RLLS, MIL OVRN, NSTD, NONE.
v1 scope is centerline ALS only; anything not in CODE_TO_ARTWORK_KEY
degrades to no symbol (the layout module logs a diagnostic for codes that
were present but unmapped, so new/edge codes surface).
"""
from __future__ import annotations


# FAA code -> artwork key. Codes grouped together are visually equivalent at
# chart-symbol resolution. [VERIFY] groupings against chart symbology before
# shipping; unmapped codes degrade safely to no-symbol.
CODE_TO_ARTWORK_KEY: dict[str, str] = {
    "ALSF1": "alsf1",
    "ALSF2": "alsf2",
    "MALSR": "malsr",
    "SSALR": "malsr",
    "MALSF": "malsf",
    "SSALF": "malsf",
    "MALS": "mals",
    "SSALS": "mals",
    "SALS": "sals",
    "SALSF": "sals",
    "ODALS": "odals",
}


def als_code_to_artwork_key(code: str | None) -> str | None:
    """Return the artwork key for an FAA ALS code, or None when there is no
    symbol to draw (NONE / blank / out-of-scope code)."""
    if not code:
        return None
    return CODE_TO_ARTWORK_KEY.get(code.strip().upper())
