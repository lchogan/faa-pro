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


import csv as _csv
from pathlib import Path

from pipeline.extract_pdf_text import _normalize_designation


# Column-header aliases. NASR's modern CSV header names are not 100%
# pinned ([VERIFY] against a freshly downloaded APT_RWY_END.csv and trim
# this list to the real names). First matching header in each list wins.
_AIRPORT_COLS = ("ARPT_ID", "LOCATION_ID", "LOCID", "LOC_ID")
_END_ID_COLS = ("RWY_END_ID", "RWY_END_IDENT", "BASE_END_ID", "RUNWAY_END_IDENT")
_ALS_COLS = (
    "APPROACH_LIGHT_SYSTEM",
    "APCH_LGT_SYSTEM_CODE",
    "RWY_END_LGT_APCH_LGT_SYSTEM_TYPE",
    "APCH_LGTG_SYS_CODE",
)


def _pick_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str:
    """Return the first candidate header present in fieldnames, else raise
    with the actual headers so a schema mismatch is obvious."""
    present = {fn.strip().upper(): fn for fn in (fieldnames or [])}
    for cand in candidates:
        if cand.upper() in present:
            return present[cand.upper()]
    raise ValueError(
        f"None of {candidates} found in APT_RWY_END columns: {fieldnames}"
    )


def load_approach_lights(csv_path: Path, airport_code: str) -> dict[str, str]:
    """Load per-runway-end approach-light codes for one airport.

    Returns {normalized_end_designator: als_code}, e.g. {"16": "MALSR"}.
    End designators are normalized via the same `_normalize_designation`
    the runway-number matcher uses, so they join directly to the tokens in
    step3_diag. Rows with a blank code are dropped. Match on the airport id
    is case-insensitive.
    """
    target = airport_code.strip().upper()
    out: dict[str, str] = {}
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = _csv.DictReader(f)
        if not reader.fieldnames:
            return out
        ap_col = _pick_column(reader.fieldnames, _AIRPORT_COLS)
        end_col = _pick_column(reader.fieldnames, _END_ID_COLS)
        als_col = _pick_column(reader.fieldnames, _ALS_COLS)
        for row in reader:
            if (row.get(ap_col) or "").strip().upper() != target:
                continue
            code = (row.get(als_col) or "").strip()
            if not code:
                continue
            desig = _normalize_designation((row.get(end_col) or "").strip())
            if desig:
                out[desig] = code.upper()
    return out
