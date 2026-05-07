"""
taxi_detection.py — rule-based taxiway-surface + taxiway-label detection.

This is the authoritative source for Taxiways and Taxiway Labels in the
new pipeline. The ML model is *not* used for these two classes.

Detection logic (matches the working render_char_layers_charbox.py):
  - Taxiway surface: filled polygon with gray fill (~#cfcfcf, with
    leeway for chart variation).
  - Taxiway label: a word token matching `^([A-Z][A-Z]?[0-9]{0,2}|[0-9])$`
    (e.g. "C", "C1", "A11", "3") whose bbox touches a taxi surface
    polygon. The K = len(token) nearest unclaimed near-black filled
    glyph polygons are claimed for that token.

Coordinates are PDF y-down (matching PyMuPDF). Convert to AI y-up at
the consumer boundary if needed.

Public:
    detect_taxi(pdf_path) -> dict with
        all_polys, taxi_surface_indices, taxi_label_indices,
        text_tokens, page_w, page_h
"""

from __future__ import annotations

import re
from pathlib import Path

import fitz

from extract_paths_fitz import _color_to_rgb, items_to_subpaths


TAXIWAY_RE = re.compile(r"^([A-Z][A-Z]?[0-9]{0,2}|[0-9])$")
# Gray-fill detection bracket around #cfcfcf = (207, 207, 207).
GRAY_MIN = 175
GRAY_MAX = 235
GRAY_CHANNEL_TOL = 20  # max R/G/B spread to still call it "gray"
# Near-black fill bracket — glyph polygons are essentially pure black.
BLACK_MAX = 60


def _is_near_black(fill_rgb_kind) -> bool:
    r, g, b, kind = fill_rgb_kind
    if kind == "none" or kind == "other" or r == "":
        return False
    if not (isinstance(r, int) and isinstance(g, int) and isinstance(b, int)):
        return False
    return max(r, g, b) <= BLACK_MAX


def _is_taxiway_gray(fill_rgb_kind) -> bool:
    r, g, b, kind = fill_rgb_kind
    if kind == "none" or kind == "other" or r == "":
        return False
    if not (isinstance(r, int) and isinstance(g, int) and isinstance(b, int)):
        return False
    if max(abs(r - g), abs(g - b), abs(r - b)) > GRAY_CHANNEL_TOL:
        return False
    avg = (r + g + b) / 3.0
    return GRAY_MIN <= avg <= GRAY_MAX


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
    """True if the token bbox has any spatial overlap with any surface
    polygon. 5-sample point-in-polygon plus a check for surface
    anchors inside the token bbox catches partial overlap without a
    full edge-intersection test."""
    sample = [
        (t["cx"], t["cy"]),
        (t["x0"], t["y0"]), (t["x1"], t["y0"]),
        (t["x1"], t["y1"]), (t["x0"], t["y1"]),
    ]
    for s in surfaces:
        sx0, sy0, sx1, sy1 = s["rect"]
        if sx1 < t["x0"] or sx0 > t["x1"] or sy1 < t["y0"] or sy0 > t["y1"]:
            continue
        subs = s["subpaths"] or []
        for px, py in sample:
            if _point_in_subpaths(px, py, subs):
                return True
        for ring in subs:
            for ax, ay in ring:
                if t["x0"] <= ax <= t["x1"] and t["y0"] <= ay <= t["y1"]:
                    return True
    return False


def detect_taxi(pdf_path: Path) -> dict:
    """Run rule-based taxiway-surface and taxiway-label detection on a
    single-page FAA airport diagram PDF.

    Returns:
        all_polys: list of polygon dicts. Each has cx, cy, rect (PDF
            y-down x0,y0,x1,y1), filled, is_taxi_surface, is_near_black,
            subpaths (only for surfaces).
        taxi_surface_indices: list[int] of indices into all_polys.
        taxi_label_indices: list[int] of indices.
        text_tokens: list of word-token dicts.
        page_w, page_h: float, page dimensions in PDF units.
    """
    doc = fitz.open(pdf_path)
    page = doc[0]
    page_w = float(page.rect.width)
    page_h = float(page.rect.height)

    # --- 1) All drawings → polygon records.
    drawings = page.get_drawings()
    all_polys: list[dict] = []
    for d in drawings:
        items = d.get("items") or []
        rect = d.get("rect")
        if not items or rect is None:
            continue
        cx = (float(rect.x0) + float(rect.x1)) / 2.0
        cy = (float(rect.y0) + float(rect.y1)) / 2.0
        dtype = d.get("type", "")
        filled = "f" in dtype
        fill_rgb = _color_to_rgb(d.get("fill")) if filled else ("", "", "", "none")
        is_taxi_surface = filled and _is_taxiway_gray(fill_rgb)
        is_near_black = filled and _is_near_black(fill_rgb)
        all_polys.append({
            "cx": cx,
            "cy": cy,
            "rect": (float(rect.x0), float(rect.y0),
                     float(rect.x1), float(rect.y1)),
            "filled": filled,
            "is_taxi_surface": is_taxi_surface,
            "is_near_black": is_near_black,
            # Always store subpaths — Step 5 needs them for ML-runway PCA.
            "subpaths": items_to_subpaths(items),
        })

    taxi_surface_indices = [i for i, p in enumerate(all_polys)
                            if p["is_taxi_surface"]]
    taxi_surfaces = [all_polys[i] for i in taxi_surface_indices]

    # --- 2) PDF word tokens.
    words = page.get_text("words")
    text_tokens: list[dict] = []
    for w in words:
        x0, y0, x1, y1, txt, *_ = w
        if not txt or not txt.strip():
            continue
        text_tokens.append({
            "text": txt,
            "x0": float(x0), "y0": float(y0),
            "x1": float(x1), "y1": float(y1),
            "cx": (float(x0) + float(x1)) / 2.0,
            "cy": (float(y0) + float(y1)) / 2.0,
        })
    doc.close()

    # --- 3) Token gating + K-nearest glyph claim.
    pattern_tokens = [t for t in text_tokens if TAXIWAY_RE.match(t["text"])]
    qualifying = [t for t in pattern_tokens if _bbox_touches(t, taxi_surfaces)]

    def _is_taxi_candidate(p):
        return p["filled"] and p["is_near_black"] and not p["is_taxi_surface"]

    claimed: set[int] = set()
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

    return {
        "all_polys": all_polys,
        "taxi_surface_indices": taxi_surface_indices,
        "taxi_label_indices": label_indices,
        "text_tokens": text_tokens,
        "page_w": page_w,
        "page_h": page_h,
    }
