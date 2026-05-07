"""
render_char_layers_charbox.py — render PDF polygons + word-level text
tokens to SVG, with rule-based taxiway-surface and taxiway-label
detection.

Output layers:
  1. Taxiways         — gray-filled polygons (~#cfcfcf, the pavement
                        surfaces).
  2. Polygons         — every other drawing not claimed as a label.
  3. Taxiway Labels   — polygons claimed as glyphs of taxiway-label
                        tokens (red fill for visibility).
  4. PDF Text Tokens  — debug layer: every word string from the PDF
                        text stream, one <text> per word, anchored at
                        bbox center.

Taxiway-label matching:
  A token qualifies as a candidate if:
    (a) its text matches `^([A-Z][A-Z]?[0-9]{0,2}|[0-9])$`
        (e.g. "C", "C1", "A11", "3"), and
    (b) its bbox touches (any overlap with) a Taxiway-surface polygon.
  For each qualifying token, the K = len(token) nearest unclaimed
  candidate polygon centroids are reassigned to the Taxiway-Labels
  layer. A candidate is filled, near-black, and not itself a surface.

Runway-label matching is deferred until the ML model assigns Runways
in Step 4 — the centerline anchors come from the model's output, not
from a heuristic here.

Usage:
    python render_char_layers_charbox.py \\
        --pdf /path/to/<airport>-faa.pdf \\
        --out /path/to/<airport>-chars.svg
"""

from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import fitz

from extract_paths_fitz import _color_to_rgb, items_to_subpaths


TAXIWAY_RE = re.compile(r"^([A-Z][A-Z]?[0-9]{0,2}|[0-9])$")
TAXI_LABEL_FILL = "#ff0000"
# Gray-fill detection bracket around #cfcfcf = (207, 207, 207).
GRAY_MIN = 175
GRAY_MAX = 235
GRAY_CHANNEL_TOL = 20  # max R/G/B spread to still call it "gray"
# Near-black fill bracket — glyph polygons are essentially pure black.
BLACK_MAX = 60


def _items_to_svg_d(items) -> str:
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


def _color_to_hex(c) -> str | None:
    r, g, b, kind = _color_to_rgb(c)
    if kind == "none" or kind == "other" or r == "":
        return None
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


def _is_near_black(fill_rgb_kind) -> bool:
    """True if the fill is essentially black — every channel <= BLACK_MAX.
    Glyphs are filled near-black on FAA diagrams; this excludes gray
    taxiway pavement, brown markers, and pale gridlines."""
    r, g, b, kind = fill_rgb_kind
    if kind == "none" or kind == "other" or r == "":
        return False
    if not (isinstance(r, int) and isinstance(g, int) and isinstance(b, int)):
        return False
    return max(r, g, b) <= BLACK_MAX


def _is_taxiway_gray(fill_rgb_kind) -> bool:
    """True if the fill color is a light gray near #cfcfcf — the
    taxiway-pavement fill on FAA diagrams. Black, white, color, and
    'no fill' all return False."""
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
    """Even-odd ray-cast test against a compound polygon's anchor rings.
    Curved edges are approximated by straight lines between anchors,
    which is close enough for taxiway-surface containment."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    doc = fitz.open(args.pdf)
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
        stroked = "s" in dtype
        fill_rgb = _color_to_rgb(d.get("fill")) if filled else ("", "", "", "none")
        is_taxi_surface = filled and _is_taxiway_gray(fill_rgb)
        is_near_black = filled and _is_near_black(fill_rgb)
        record = {
            "d": _items_to_svg_d(items),
            "fill": _color_to_hex(d.get("fill")) if filled else None,
            "stroke": _color_to_hex(d.get("color")) if stroked else None,
            "stroke_width": float(d.get("width") or 0.0) if stroked else 0.0,
            "cx": cx,
            "cy": cy,
            "rect": (float(rect.x0), float(rect.y0),
                     float(rect.x1), float(rect.y1)),
            "filled": filled,
            "is_taxi_surface": is_taxi_surface,
            "is_near_black": is_near_black,
            # Subpaths kept only for taxi surfaces (point-in-polygon test).
            "subpaths": items_to_subpaths(items) if is_taxi_surface else None,
        }
        all_polys.append(record)

    taxi_surfaces = [p for p in all_polys if p["is_taxi_surface"]]
    print(f"[charbox] {len(all_polys)} polygons "
          f"({len(taxi_surfaces)} taxi-surface)")

    # --- 2) PDF text tokens (word-level: "RWY", "22L", etc.)
    words = page.get_text("words")  # (x0,y0,x1,y1,word,block,line,wno)
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
    print(f"[charbox] {len(text_tokens)} text tokens")

    # --- 3) Taxiway-label matching.
    # A token qualifies if its bbox *touches* (has any overlap with) a
    # taxiway surface polygon. K-nearest is computed against candidate
    # polygons only (filled, near-black, non-surface).

    def _bbox_touches(t, surfaces):
        # 5-point sample of token bbox + check if any surface anchor
        # falls inside the bbox. Together this catches almost any
        # overlap between a small token bbox and a surface polygon
        # without a full edge-intersection test.
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

    def _taxi_candidate(p):
        return p["filled"] and p["is_near_black"] and not p["is_taxi_surface"]

    pattern_tokens = [t for t in text_tokens if TAXIWAY_RE.match(t["text"])]
    qualifying_tokens = [t for t in pattern_tokens if _bbox_touches(t, taxi_surfaces)]

    claimed_polys: set[int] = set()
    label_indices: list[int] = []
    for tok in qualifying_tokens:
        k = len(tok["text"])
        scored = []
        for i, p in enumerate(all_polys):
            if i in claimed_polys or not _taxi_candidate(p):
                continue
            dx = tok["cx"] - p["cx"]
            dy = tok["cy"] - p["cy"]
            scored.append((dx * dx + dy * dy, i))
        scored.sort()
        for _, i in scored[:k]:
            claimed_polys.add(i)
            label_indices.append(i)

    print(f"[charbox] taxi labels: {len(qualifying_tokens)} tokens, "
          f"{len(label_indices)} polygons claimed")

    taxi_label_layer = [all_polys[i] for i in label_indices]
    polygons_layer = [
        p for i, p in enumerate(all_polys)
        if i not in claimed_polys and not p["is_taxi_surface"]
    ]
    print(f"[charbox] Layers — Taxiways: {len(taxi_surfaces)}  "
          f"Polygons: {len(polygons_layer)}  "
          f"Taxi Labels: {len(taxi_label_layer)}")

    # --- 4) Emit SVG
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

    layers = [
        ("Taxiways", taxi_surfaces, None),
        ("Polygons", polygons_layer, None),
        ("Taxiway Labels", taxi_label_layer, TAXI_LABEL_FILL),
    ]
    for layer_name, paths, override_fill in layers:
        if not paths:
            continue
        g = ET.SubElement(svg, "g", {
            "id": layer_name,
            "inkscape:label": layer_name,
            "inkscape:groupmode": "layer",
        })
        for p in paths:
            attrs = {"d": p["d"], "fill": override_fill or p["fill"] or "none"}
            if p["stroke"]:
                attrs["stroke"] = override_fill or p["stroke"]
                attrs["stroke-width"] = f"{p['stroke_width']:.3f}"
            ET.SubElement(g, "path", attrs)

    if text_tokens:
        # Fixed font-size; each <text> anchored at bbox center so (x, y)
        # IS the analyzable center of the original PDF token.
        g = ET.SubElement(svg, "g", {
            "id": "PDF Text Tokens",
            "inkscape:label": "PDF Text Tokens",
            "inkscape:groupmode": "layer",
            "fill": "#cc00cc",
            "font-family": "Helvetica, Arial, sans-serif",
            "font-size": "4",
            "text-anchor": "middle",
            "dominant-baseline": "middle",
        })
        for t in text_tokens:
            te = ET.SubElement(g, "text", {
                "x": f"{t['cx']:.3f}",
                "y": f"{t['cy']:.3f}",
            })
            te.text = t["text"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(svg).write(args.out, encoding="utf-8", xml_declaration=True)
    print(f"[charbox] wrote -> {args.out}")


if __name__ == "__main__":
    main()
