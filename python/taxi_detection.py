"""
taxi_detection.py — rule-based taxiway-surface + taxiway-label detection.

This is the authoritative source for Taxiways and Taxiway Labels in the
new pipeline. The ML model is *not* used for these two classes.

Detection logic (matches the working render_char_layers_charbox.py):
  - Taxiway surface: filled polygon with gray fill (~#cfcfcf, with
    leeway for chart variation). Detection of the gray fill itself
    happens in chart_scene.read_chart and is exposed via the
    `is_taxi_surface` flag on each polygon dict.
  - Taxiway label: a word token matching `^([A-Z][A-Z]?[0-9]{0,2}|[0-9])$`
    (e.g. "C", "C1", "A11", "3") whose bbox touches a taxi surface
    polygon. The K = len(token) nearest unclaimed near-black filled
    glyph polygons are claimed for that token.

Coordinates are PDF y-down (matching PyMuPDF). Convert to AI y-up at
the consumer boundary if needed.

Public:
    detect_taxi(pdf_path) -> dict with
        all_polys, clips, taxi_surface_indices, taxi_label_indices,
        text_tokens, page_w, page_h
"""

from __future__ import annotations

import re
from pathlib import Path

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
    single-page FAA airport diagram PDF. Return shape matches the
    pre-refactor signature plus a `clips` field exposing nested clip
    groups for runway_detection's use.
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
        "clips": clips,
        "taxi_surface_indices": taxi_surface_indices,
        "taxi_label_indices": label_indices,
        "text_tokens": text_tokens,
        "page_w": page_w,
        "page_h": page_h,
    }
