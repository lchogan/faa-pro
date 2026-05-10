"""
chart_scene.py — single source of truth for reading an FAA airport
diagram PDF into the data structures every classification step uses.

Replaces the polygon/text extraction that used to live inside
`taxi_detection.detect_taxi`. taxi_detection and runway_detection now
both consume `read_chart()`.

Returns a dict with:
    all_polys      — regular drawn polygons (filled and/or stroked).
                     Indices align to extract_paths_fitz CSV row order.
                     Each dict has cx, cy, rect, filled, stroked,
                     stroke_width, is_taxi_surface, is_near_black,
                     subpaths.
    clips          — clip-group rects exposed by get_drawings(extended=True).
                     Page-sized clips (chart frame artifacts) are dropped.
                     Each dict has rect (the scissor) and subpaths
                     (the clip path's items). Used by runway_detection
                     so grass-strip outlines stored as nested clip
                     groups (e.g. F45) become visible candidates.
    text_tokens    — page word tokens with bbox + center.
    page_w, page_h — page dimensions in PDF points.
"""

from __future__ import annotations

from pathlib import Path

import fitz

from pipeline.extract_paths_fitz import (
    _Bounds,
    _color_to_rgb,
    _is_off_artboard,
    items_to_subpaths,
)


# Gray-fill detection bracket around #cfcfcf = (207, 207, 207).
# Mirrors the constants in taxi_detection.py — single source here so
# the rule lives next to the polygon construction that consumes it.
GRAY_MIN = 175
GRAY_MAX = 235
GRAY_CHANNEL_TOL = 20
BLACK_MAX = 60

# Drop clips whose scissor covers more than this fraction of the page.
# Standard FAA charts have several full-page clips at level 0 (the
# entire artboard). Those carry no detection value and would otherwise
# pollute every candidate pool.
PAGE_SIZED_CLIP_FRACTION = 0.50


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


def _is_near_black(fill_rgb_kind) -> bool:
    r, g, b, kind = fill_rgb_kind
    if kind == "none" or kind == "other" or r == "":
        return False
    if not (isinstance(r, int) and isinstance(g, int) and isinstance(b, int)):
        return False
    return max(r, g, b) <= BLACK_MAX


def read_chart(pdf_path: Path) -> dict:
    """Read a single-page FAA airport diagram into the structures every
    rule-based step consumes. See module docstring for return shape."""
    doc = fitz.open(pdf_path)
    page = doc[0]
    page_w = float(page.rect.width)
    page_h = float(page.rect.height)
    page_area = page_w * page_h
    artboard = _Bounds(left=0.0, top=page_h, right=page_w, bottom=0.0)

    drawings = page.get_drawings(extended=True)

    all_polys: list[dict] = []
    clips: list[dict] = []

    for d in drawings:
        dtype = d.get("type", "")
        items = d.get("items") or []

        if dtype == "clip":
            # Clip entry. Use the scissor as the rect, the items as
            # subpaths. Drop page-sized clips (the chart frame and any
            # other artboard-spanning clip group).
            scissor = d.get("scissor")
            if scissor is None:
                continue
            sx0, sy0, sx1, sy1 = (
                float(scissor.x0), float(scissor.y0),
                float(scissor.x1), float(scissor.y1),
            )
            scissor_area = max((sx1 - sx0), 0.0) * max((sy1 - sy0), 0.0)
            if scissor_area <= 0 or scissor_area >= page_area * PAGE_SIZED_CLIP_FRACTION:
                continue
            subpaths = items_to_subpaths(items) if items else []
            clips.append({
                "rect": (sx0, sy0, sx1, sy1),
                "cx": (sx0 + sx1) / 2.0,
                "cy": (sy0 + sy1) / 2.0,
                "subpaths": subpaths,
                "level": int(d.get("level", 0)),
            })
            continue

        # Regular drawn path (type 'f', 's', or 'fs').
        rect = d.get("rect")
        if not items or rect is None:
            continue
        subpaths = items_to_subpaths(items)
        if not subpaths:
            continue
        bbox_ai = _Bounds(
            left=float(rect.x0),
            right=float(rect.x1),
            top=page_h - float(rect.y0),
            bottom=page_h - float(rect.y1),
        )
        if _is_off_artboard(bbox_ai, artboard):
            continue
        cx = (float(rect.x0) + float(rect.x1)) / 2.0
        cy = (float(rect.y0) + float(rect.y1)) / 2.0
        filled = "f" in dtype
        stroked = "s" in dtype
        fill_rgb = _color_to_rgb(d.get("fill")) if filled else ("", "", "", "none")
        stroke_width = float(d.get("width") or 0.0) if stroked else 0.0
        is_taxi_surface = filled and _is_taxiway_gray(fill_rgb)
        is_near_black_v = filled and _is_near_black(fill_rgb)
        all_polys.append({
            "cx": cx,
            "cy": cy,
            "rect": (float(rect.x0), float(rect.y0),
                     float(rect.x1), float(rect.y1)),
            "filled": filled,
            "stroked": stroked,
            "stroke_width": stroke_width,
            "is_taxi_surface": is_taxi_surface,
            "is_near_black": is_near_black_v,
            "subpaths": subpaths,
        })

    # Page word tokens.
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

    return {
        "all_polys": all_polys,
        "clips": clips,
        "text_tokens": text_tokens,
        "page_w": page_w,
        "page_h": page_h,
    }
