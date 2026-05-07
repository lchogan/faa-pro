"""
render_char_layers_simple.py — the brute-simple baseline we should have
tried first.

PyMuPDF's get_text("words") gives every word's bbox in the *same PDF
coordinate system* as get_drawings(). No coordinate offset, no Illustrator
in the loop. So "is this polygon a character?" reduces to:

    is the polygon's bbox center inside any PDF-text word's bbox?

That's it. No training, no classifier, no features. The 15-20pt offset
that broke the original `pdf_text_match` was *Illustrator-vs-PDF*; with
PyMuPDF on both sides of the comparison the offset doesn't exist.

Outputs an SVG with two layers:
    Character        — polygon center sits inside some PDF text word bbox
    Not a Character  — polygon center is outside every text bbox

Usage:
    python render_char_layers_simple.py \\
        --pdf /path/to/<airport>-faa.pdf \\
        --out /path/to/<airport>-chars-simple.svg

Optional:
    --margin 2.0     PDF-units padding added to text bboxes before testing
                     containment (helps catch glyph extents that slightly
                     exceed the reported word bbox).
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import fitz

from extract_paths_fitz import items_to_subpaths, _color_to_rgb


def _items_to_svg_d(items) -> str:
    """Convert PyMuPDF drawing items to an SVG path 'd' attribute (PDF coords)."""
    parts: list[str] = []
    last_end: tuple[float, float] | None = None

    def feq(a, b):
        return abs(a[0] - b[0]) < 1e-3 and abs(a[1] - b[1]) < 1e-3

    for item in items:
        op = item[0]
        if op == "l":
            p1 = (item[1].x, item[1].y)
            p2 = (item[2].x, item[2].y)
            if last_end is None or not feq(p1, last_end):
                parts.append(f"M{p1[0]:.3f},{p1[1]:.3f}")
            parts.append(f"L{p2[0]:.3f},{p2[1]:.3f}")
            last_end = p2
        elif op == "c":
            p1 = (item[1].x, item[1].y)
            c1 = (item[2].x, item[2].y)
            c2 = (item[3].x, item[3].y)
            p4 = (item[4].x, item[4].y)
            if last_end is None or not feq(p1, last_end):
                parts.append(f"M{p1[0]:.3f},{p1[1]:.3f}")
            parts.append(f"C{c1[0]:.3f},{c1[1]:.3f} {c2[0]:.3f},{c2[1]:.3f} {p4[0]:.3f},{p4[1]:.3f}")
            last_end = p4
        elif op == "re":
            r = item[1]
            parts.append(
                f"M{r.x0:.3f},{r.y0:.3f} L{r.x1:.3f},{r.y0:.3f} "
                f"L{r.x1:.3f},{r.y1:.3f} L{r.x0:.3f},{r.y1:.3f} Z"
            )
            last_end = None
        elif op == "qu":
            q = item[1]
            parts.append(
                f"M{q.ul.x:.3f},{q.ul.y:.3f} L{q.ur.x:.3f},{q.ur.y:.3f} "
                f"L{q.lr.x:.3f},{q.lr.y:.3f} L{q.ll.x:.3f},{q.ll.y:.3f} Z"
            )
            last_end = None
    return " ".join(parts)


def _color_to_hex(c) -> str | None:
    r, g, b, kind = _color_to_rgb(c)
    if kind == "none" or kind == "other" or r == "":
        return None
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--margin", type=float, default=2.0,
                    help="PDF-units padding added to each text bbox")
    ap.add_argument("--require-filled", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Require the polygon to be a filled path. PDF letter "
                         "glyphs are always filled; stroked-only paths "
                         "(circled-letter markers, arrowhead outlines, "
                         "dashed lines) are not. Pass --no-require-filled to "
                         "disable.")
    ap.add_argument("--max-fill-rgb", type=int, default=110,
                    help="Maximum fill-color channel value (0-255) for a "
                         "polygon to count as a character. Black=0, dark "
                         "gray~80, brown HS-box ~150, light gray pavement "
                         "~200. Default 110 keeps text and dark gray, "
                         "rejects browns/light grays/colored markers. Set "
                         "to 255 to disable the color filter.")
    ap.add_argument("--cap-per-word", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="For each word T, keep at most len(T) polygons (the "
                         "ones nearest the per-character slice centers). "
                         "Greedy — can drop real letters when a contaminant "
                         "is closer to a slice center. OFF by default.")
    ap.add_argument("--min-height-ratio", type=float, default=0.0,
                    help="Polygon bbox-height as fraction of word bbox-height "
                         "(0 disables). Doesn't work for rotated words "
                         "because PyMuPDF reports axis-aligned word bboxes.")
    ap.add_argument("--min-items", type=int, default=5,
                    help="Minimum number of path items (line/curve segments) "
                         "for a polygon to count as a character. PDF letter "
                         "glyphs typically have 8-30+ items (curves + "
                         "corners). Triangle arrowheads have 3, small "
                         "rectangles 4, dashes 1-2 — all get rejected. The "
                         "thinnest legitimate letter (an unserifed 'I' as a "
                         "rectangle) has 4 items, so 5 is at the edge. Try "
                         "4 if you see 'I' or '1' missing.")
    ap.add_argument("--min-abs-height", type=float, default=2.0,
                    help="Minimum polygon height in PDF points. Belt-and-"
                         "suspenders for tiny dots that might still pass "
                         "other filters (small periods, accent marks).")
    args = ap.parse_args()

    doc = fitz.open(args.pdf)
    page = doc[0]
    page_w = float(page.rect.width)
    page_h = float(page.rect.height)

    # --- 1) text words. Keep the actual text so we can do per-character
    # slice matching downstream, not just bbox containment.
    words_raw = page.get_text("words")
    words: list[dict] = []
    for w in words_raw:
        x0, y0, x1, y1, text, *_ = w
        text = (text or "").strip()
        if not text:
            continue
        wx0 = float(x0) - args.margin
        wy0 = float(y0) - args.margin
        wx1 = float(x1) + args.margin
        wy1 = float(y1) + args.margin
        words.append({
            "text": text,
            "x0": wx0, "y0": wy0, "x1": wx1, "y1": wy1,
            "area": max((wx1 - wx0) * (wy1 - wy0), 1e-6),
        })
    print(f"[simple] {len(words_raw)} words ({len(words)} non-empty)")

    # --- 2) drawings + classification by simple bbox containment
    drawings = page.get_drawings()

    # --- 2) collect per-polygon properties (one pass through drawings)
    # We need to know everything before per-word slice matching.
    polys: list[dict] = []
    rejected: dict[str, int] = {
        "no_word_overlap": 0, "not_filled": 0, "color_too_light": 0,
        "too_few_items": 0, "too_short_abs": 0,
        "too_short_rel": 0, "not_chosen_by_word": 0,
    }

    for d in drawings:
        items = d.get("items") or []
        rect = d.get("rect")
        if not items or rect is None:
            continue
        x0, y0, x1, y1 = float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        poly_area = max((x1 - x0) * (y1 - y0), 0.0)
        poly_h = max(y1 - y0, 0.0)
        dtype = d.get("type", "")
        filled = "f" in dtype
        stroked = "s" in dtype
        fill_rgb = _color_to_rgb(d.get("fill")) if filled else ("", "", "", "none")

        # Per-polygon filters that don't depend on word context.
        passes_filled = (not args.require_filled) or filled
        passes_color = True
        if filled and isinstance(fill_rgb[0], int):
            if max(fill_rgb[0], fill_rgb[1], fill_rgb[2]) > args.max_fill_rgb:
                passes_color = False

        n_items = len(items)
        passes_items = n_items >= args.min_items
        passes_abs_h = poly_h >= args.min_abs_height

        early_reject = None
        if not passes_filled:
            early_reject = "not_filled"
        elif not passes_color:
            early_reject = "color_too_light"
        elif not passes_items:
            early_reject = "too_few_items"
        elif not passes_abs_h:
            early_reject = "too_short_abs"

        polys.append({
            "items": items,
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "cx": cx, "cy": cy, "area": poly_area, "h": poly_h,
            "n_items": n_items,
            "filled": filled, "stroked": stroked,
            "fill": _color_to_hex(d.get("fill")) if filled else None,
            "stroke": _color_to_hex(d.get("color")) if stroked else None,
            "stroke_width": float(d.get("width") or 0.0) if stroked else 0.0,
            "passes_pre_filters": early_reject is None,
            "early_reject": early_reject,
        })

    # --- 3) word-stage matching: for each word, pick polygons.
    # is_char[i] becomes True if polygon i is chosen by *any* word.
    is_char = [False] * len(polys)
    too_short_for_some_word: set[int] = set()

    for w in words:
        bx0, by0, bx1, by1 = w["x0"], w["y0"], w["x1"], w["y1"]
        word_area = w["area"]
        word_h = max(by1 - by0, 1e-6)
        min_h = args.min_height_ratio * word_h
        text = w["text"]
        n_chars = max(len(text), 1)
        char_w = (bx1 - bx0) / n_chars
        word_cy = (by0 + by1) / 2.0

        # Candidates: passes pre-filters, fits in word bbox by centroid AND
        # area, AND polygon height >= min fraction of word height (drops
        # arrowheads / dots / dashes / horizontal bars).
        cand_idx: list[int] = []
        for i, p in enumerate(polys):
            if not p["passes_pre_filters"]:
                continue
            if not (bx0 <= p["cx"] <= bx1 and by0 <= p["cy"] <= by1
                    and p["area"] <= word_area):
                continue
            if p["h"] < min_h:
                too_short_for_some_word.add(i)
                continue
            cand_idx.append(i)
        if not cand_idx:
            continue

        if args.cap_per_word:
            # Greedy 1-1 slice match: each char position picks the nearest
            # unused candidate. Caps total kept polygons at len(text).
            used: set[int] = set()
            for k in range(n_chars):
                slice_cx = bx0 + (k + 0.5) * char_w
                best = None
                best_d = float("inf")
                for ci in cand_idx:
                    if ci in used:
                        continue
                    p = polys[ci]
                    d2 = (p["cx"] - slice_cx) ** 2 + (p["cy"] - word_cy) ** 2
                    if d2 < best_d:
                        best_d = d2
                        best = ci
                if best is not None:
                    used.add(best)
                    is_char[best] = True
        else:
            # No cap: every candidate becomes a char (legacy behavior).
            for ci in cand_idx:
                is_char[ci] = True

    # --- 4) tally + emit
    char_paths: list[dict] = []
    not_char_paths: list[dict] = []
    for i, p in enumerate(polys):
        record = {
            "d": _items_to_svg_d(p["items"]),
            "fill": p["fill"],
            "stroke": p["stroke"],
            "stroke_width": p["stroke_width"],
        }
        if is_char[i]:
            char_paths.append(record)
            continue
        # Not a Character: figure out why for the stats line.
        if p["early_reject"] == "not_filled":
            rejected["not_filled"] += 1
        elif p["early_reject"] == "color_too_light":
            rejected["color_too_light"] += 1
        elif p["early_reject"] == "too_few_items":
            rejected["too_few_items"] += 1
        elif p["early_reject"] == "too_short_abs":
            rejected["too_short_abs"] += 1
        elif i in too_short_for_some_word:
            rejected["too_short_rel"] += 1
        else:
            overlapped = False
            for w in words:
                if (w["x0"] <= p["cx"] <= w["x1"]
                        and w["y0"] <= p["cy"] <= w["y1"]
                        and p["area"] <= w["area"]):
                    overlapped = True
                    break
            if overlapped:
                rejected["not_chosen_by_word"] += 1
            else:
                rejected["no_word_overlap"] += 1
        not_char_paths.append(record)

    doc.close()
    print(f"[simple] Character: {len(char_paths)}  Not a Character: {len(not_char_paths)}")
    print(f"[simple] reject reasons: "
          f"no_word_overlap={rejected['no_word_overlap']}  "
          f"not_filled={rejected['not_filled']}  "
          f"color_too_light={rejected['color_too_light']}  "
          f"too_few_items={rejected['too_few_items']}  "
          f"too_short_abs={rejected['too_short_abs']}  "
          f"too_short_rel={rejected['too_short_rel']}  "
          f"not_chosen_by_word={rejected['not_chosen_by_word']}")

    # --- 3) emit SVG
    svg_attrs = {
        "xmlns": "http://www.w3.org/2000/svg",
        "xmlns:inkscape": "http://www.inkscape.org/namespaces/inkscape",
        "width": f"{page_w}",
        "height": f"{page_h}",
        "viewBox": f"0 0 {page_w} {page_h}",
    }
    svg = ET.Element("svg", svg_attrs)
    ET.SubElement(svg, "rect", {
        "x": "0", "y": "0",
        "width": str(page_w), "height": str(page_h),
        "fill": "#ffffff",
    })

    for layer_name, paths in [("Character", char_paths), ("Not a Character", not_char_paths)]:
        if not paths:
            continue
        g = ET.SubElement(svg, "g", {
            "id": layer_name,
            "inkscape:label": layer_name,
            "inkscape:groupmode": "layer",
        })
        for p in paths:
            attrs = {"d": p["d"], "fill": p["fill"] or "none"}
            if p["stroke"]:
                attrs["stroke"] = p["stroke"]
                attrs["stroke-width"] = f"{p['stroke_width']:.3f}"
            ET.SubElement(g, "path", attrs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(svg).write(args.out, encoding="utf-8", xml_declaration=True)
    print(f"[simple] wrote -> {args.out}")


if __name__ == "__main__":
    main()
