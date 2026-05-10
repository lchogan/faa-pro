"""
render_svg_layers.py — Python SVG renderer for the layered diagram.

Writes an SVG with one top-level <g> per TARGET_LAYERS entry, marked
with both `id="LayerName"` and Inkscape's `inkscape:groupmode="layer"`
attribute. Adobe Illustrator's SVG importer creates a native layer
for each such group (no manual "Release to Layers" needed in modern
AI versions). Polygon geometry is re-emitted from PyMuPDF's drawing
items so every fill/stroke is preserved exactly.

Usage:
    python render_svg_layers.py \\
        --pdf /path/to/<airport>-faa.pdf \\
        --predictions /path/to/<airport>_predictions.json \\
        --out /path/to/<airport>-diagram.svg
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import fitz

from extract_paths_fitz import (
    _Bounds,
    _is_off_artboard,
    items_to_subpaths,
)


TARGET_LAYERS = [
    "PDF Text Tokens",
    "Footprints",
    "Runways",
    "Runway Labels",
    "Taxiways",
    "Taxiway Labels",
    "Stars",
    "Other",
    "Metadata",
]

TOKEN_FONT_SIZE = 4.0
TOKEN_COLOR_HEX = "#cc00cc"


def _color_hex(raw):
    """Convert a PyMuPDF color value (1, 3, or 4 components in [0,1])
    to a hex string. Returns None for absent."""
    if raw is None:
        return None
    if not isinstance(raw, (tuple, list)):
        return None
    if len(raw) == 1:
        g = int(round(float(raw[0]) * 255))
        return f"#{g:02x}{g:02x}{g:02x}"
    if len(raw) == 3:
        r, g, b = (int(round(float(c) * 255)) for c in raw)
        return f"#{r:02x}{g:02x}{b:02x}"
    if len(raw) == 4:
        c, m, y, k = (float(x) for x in raw)
        rr = int(round((1 - c) * (1 - k) * 255))
        gg = int(round((1 - m) * (1 - k) * 255))
        bb = int(round((1 - y) * (1 - k) * 255))
        return f"#{rr:02x}{gg:02x}{bb:02x}"
    return None


def _items_to_svg_d(items) -> str:
    """Re-emit PyMuPDF drawing items as an SVG path 'd' string. Mirrors
    the items-walking logic in extract_paths_fitz.items_to_subpaths."""
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
            parts.append(
                f"C{c1[0]:.3f},{c1[1]:.3f} {c2[0]:.3f},{c2[1]:.3f} "
                f"{p4[0]:.3f},{p4[1]:.3f}"
            )
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    payload = json.loads(args.predictions.read_text())
    preds = payload.get("predictions", [])
    text_tokens = payload.get("text_tokens", [])

    src = fitz.open(args.pdf)
    src_page = src[0]
    page_w = float(src_page.rect.width)
    page_h = float(src_page.rect.height)
    artboard = _Bounds(left=0.0, top=page_h, right=page_w, bottom=0.0)

    # Walk source drawings in extract_paths_fitz order so prediction[i]
    # aligns with drawing[i].
    valid_drawings = []
    for d in src_page.get_drawings():
        items = d.get("items") or []
        rect = d.get("rect")
        if not items or rect is None:
            continue
        if not items_to_subpaths(items):
            continue
        bbox_ai = _Bounds(
            left=float(rect.x0),
            right=float(rect.x1),
            top=page_h - float(rect.y0),
            bottom=page_h - float(rect.y1),
        )
        if _is_off_artboard(bbox_ai, artboard):
            continue
        valid_drawings.append(d)
    src.close()

    if len(valid_drawings) != len(preds):
        raise SystemExit(
            f"Drawing count mismatch — PDF has {len(valid_drawings)}, "
            f"predictions has {len(preds)}."
        )

    # Bucket polygons by target layer.
    # translate_x / translate_y come from classify_pipeline step 3b
    # (runway-label move along centerline). Coordinates are in
    # PyMuPDF top-left frame, same as the path data, so they apply
    # directly as an SVG transform with no flip.
    layer_buckets: dict[str, list[dict]] = {name: [] for name in TARGET_LAYERS}
    for i, d in enumerate(valid_drawings):
        rec = preds[i]
        label = rec.get("label", "Other")
        if label not in layer_buckets:
            label = "Other"
        dtype = d.get("type", "")
        is_filled = "f" in dtype
        is_stroked = "s" in dtype
        tx = float(rec.get("translate_x", 0.0) or 0.0)
        ty = float(rec.get("translate_y", 0.0) or 0.0)
        layer_buckets[label].append({
            "d": _items_to_svg_d(d.get("items")),
            "fill": _color_hex(d.get("fill")) if is_filled else None,
            "stroke": _color_hex(d.get("color")) if is_stroked else None,
            "stroke_width": float(d.get("width") or 0.0) if is_stroked else 0.0,
            "translate": (tx, ty),
        })

    # Build SVG. Inkscape namespace markers tell Illustrator's SVG
    # importer to convert each top-level <g> into a native layer.
    svg_attrs = {
        "xmlns": "http://www.w3.org/2000/svg",
        "xmlns:inkscape": "http://www.inkscape.org/namespaces/inkscape",
        "width": f"{page_w}",
        "height": f"{page_h}",
        "viewBox": f"0 0 {page_w} {page_h}",
    }
    svg = ET.Element("svg", svg_attrs)

    # Layer order in SVG: first child = bottom of stack visually. To
    # match TARGET_LAYERS' top-to-bottom intent, emit in reverse order.
    for name in reversed(TARGET_LAYERS):
        bucket = layer_buckets.get(name, [])
        if not bucket and name not in ("Metadata", "PDF Text Tokens"):
            continue
        g = ET.SubElement(svg, "g", {
            "id": name,
            "inkscape:label": name,
            "inkscape:groupmode": "layer",
        })
        if name == "PDF Text Tokens":
            continue
        for p in bucket:
            attrs = {"d": p["d"], "fill": p["fill"] or "none"}
            if p["stroke"]:
                attrs["stroke"] = p["stroke"]
                attrs["stroke-width"] = f"{p['stroke_width']:.3f}"
            tx, ty = p.get("translate", (0.0, 0.0))
            if abs(tx) > 1e-6 or abs(ty) > 1e-6:
                attrs["transform"] = f"translate({tx:.4f},{ty:.4f})"
            ET.SubElement(g, "path", attrs)

    # Add the PDF Text Tokens layer last so it ends up topmost.
    if text_tokens:
        # Find or create the PDF Text Tokens layer's <g>.
        token_layer = None
        for child in svg:
            if child.get("id") == "PDF Text Tokens":
                token_layer = child
                break
        if token_layer is None:
            token_layer = ET.SubElement(svg, "g", {
                "id": "PDF Text Tokens",
                "inkscape:label": "PDF Text Tokens",
                "inkscape:groupmode": "layer",
            })
        token_layer.set("font-family", "Helvetica, Arial, sans-serif")
        token_layer.set("font-size", str(TOKEN_FONT_SIZE))
        token_layer.set("fill", TOKEN_COLOR_HEX)
        token_layer.set("text-anchor", "middle")
        token_layer.set("dominant-baseline", "middle")
        for tok in text_tokens:
            text = str(tok.get("text", ""))
            if not text:
                continue
            x_ai = float(tok.get("x", 0))
            y_ai = float(tok.get("y", 0))
            # SVG is y-down; convert from AI y-up frame.
            te = ET.SubElement(token_layer, "text", {
                "x": f"{x_ai:.3f}",
                "y": f"{page_h - y_ai:.3f}",
            })
            te.text = text

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(svg).write(args.out, encoding="utf-8", xml_declaration=True)

    print(f"[render_svg] wrote -> {args.out}")
    print("Layer distribution:")
    for name in TARGET_LAYERS:
        n = len(layer_buckets.get(name, []))
        if n:
            print(f"  {name:<22} {n}")
    if text_tokens:
        print(f"  PDF Text Tokens (text)  {len(text_tokens)}")


if __name__ == "__main__":
    main()
