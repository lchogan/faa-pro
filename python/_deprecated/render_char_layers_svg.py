"""
render_char_layers_svg.py — visualize char-classifier predictions as a
layered SVG, fully command-line / no Illustrator required.

Walks an FAA airport PDF with PyMuPDF, extracts both the rotation+scale-
invariant features AND the raw drawing commands for each path. Runs the
trained char classifier on the features. Emits an SVG with one <g
inkscape:groupmode="layer"> per output class.

Two model modes:

  Binary detector (recommended for "is this a character?" inspection):
    Uses runs/char/v3-binary by default. Outputs two layers:
        Character / Not a Character
    The detector scores every polygon char-vs-not directly, without
    forcing the model to also pick *which* character.

  Multiclass recognizer (when --multiclass):
    Outputs 36 per-character layers + Not a Character. Uses the prob
    threshold + the v2/v3 _NOT_ class to populate Not a Character.
    Less reliable for the detection question because the model is
    splitting its attention between detection and recognition.

Usage:
    python render_char_layers_svg.py \\
        --pdf /path/to/<airport>-faa.pdf \\
        --out /path/to/<airport>-chars.svg

Flags:
    --multiclass           use the 37-class recognizer instead of the binary detector
    --threshold 0.5        binary-mode CHAR-probability threshold (default 0.5)
    --model-dir PATH       override the model directory
    --reject-prob 0.30     multiclass-mode soft-reject threshold
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import fitz
import pandas as pd

# Reuse the exact filtering + feature semantics from extract_paths_fitz so
# the per-row indices we feed the classifier line up with the SVG paths
# we emit. (We do an independent walk here so we can collect both the
# features AND the raw items; importing extract_paths and re-walking
# would risk drift.)
from extract_paths_fitz import (
    items_to_subpaths,
    _pca_angle_ratio,
    _perimeter,
    _longest_segment,
    _color_to_rgb,
    _shoelace_signed,
    _Bounds,
    _is_off_artboard,
    CSV_COLUMNS,
    EDGE_COLUMNS,
)
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial import QhullError
from predict_char import add_char_predictions

LAYER_CHARS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
NOT_A_CHAR = "Not a Character"
CHAR_LAYER = "Character"
# Char-classifier sentinel for the trained reject class (multiclass models v2/v3).
# When the model predicts this class we route directly to NOT_A_CHAR
# regardless of probability.
NEGATIVE_CLASS = "_NOT_"


def _items_to_svg_d(items) -> str:
    """Convert PyMuPDF drawing items to an SVG path 'd' attribute.

    Works in PDF native coordinates (y-down). Caller flips the SVG via
    a viewBox transform so the rendering matches what's in the PDF.
    """
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


def extract_features_and_svg(pdf_path: Path, airport: str):
    """Return (features_df, edges_df, svg_data) where svg_data[i] is the
    SVG dict (d, fill, stroke, stroke_width) for the i-th feature row.

    Mirrors extract_paths_fitz.extract_paths exactly except we also
    accumulate per-drawing SVG render metadata in lockstep.
    """
    doc = fitz.open(pdf_path)
    page = doc[0]
    page_h = float(page.rect.height)
    page_w = float(page.rect.width)
    artboard = _Bounds(left=0.0, top=page_h, right=page_w, bottom=0.0)

    rows: list[dict] = []
    edges: list[dict] = []
    svg_data: list[dict] = []
    object_id = 0

    for d in page.get_drawings():
        items = d.get("items", [])
        if not items:
            continue
        sps_pdf = items_to_subpaths(items)
        if not sps_pdf:
            continue
        rect = d.get("rect")
        if rect is None:
            continue
        bbox = _Bounds(
            left=float(rect.x0),
            right=float(rect.x1),
            top=page_h - float(rect.y0),
            bottom=page_h - float(rect.y1),
        )
        if _is_off_artboard(bbox, artboard):
            continue

        # AI-frame subpaths used for features
        sps_ai = [[(x, page_h - y) for (x, y) in sp] for sp in sps_pdf]
        subpath_count = len(sps_ai)
        kind = "compound" if subpath_count > 1 else "path"

        all_anchors: list[tuple[float, float]] = []
        perim = 0.0
        longest_a = 0.0
        longest_l = 0.0
        signed_area_sum = 0.0
        for sp in sps_ai:
            all_anchors.extend(sp)
            perim += _perimeter(sp, closed=False)
            la, ll = _longest_segment(sp, closed=False)
            if ll > longest_l:
                longest_a = la
                longest_l = ll
            signed_area_sum += _shoelace_signed(sp)
        num_anchors = len(all_anchors)
        pca_a, pca_r = _pca_angle_ratio(all_anchors)
        poly_area = abs(signed_area_sum)
        hull_area = 0.0
        if num_anchors >= 3:
            try:
                hull = ConvexHull(np.asarray(all_anchors, dtype=float))
                hull_area = float(hull.volume)
            except (QhullError, ValueError):
                hull_area = 0.0

        width = bbox.width
        height = bbox.height
        bbox_area = abs(width * height)
        aspect = (width / height) if height > 1e-9 else 0.0
        centroid_x = (bbox.left + bbox.right) / 2.0
        centroid_y = (bbox.top + bbox.bottom) / 2.0

        dtype = d.get("type", "")
        filled = "f" in dtype
        stroked = "s" in dtype
        fill_hex = _color_to_hex(d.get("fill")) if filled else None
        stroke_hex = _color_to_hex(d.get("color")) if stroked else None
        stroke_width = float(d.get("width") or 0.0) if stroked else 0.0

        rows.append({
            "airport": airport,
            "object_id": object_id,
            "kind": kind,
            "source_layer": "",
            "label": "UNLABELED",
            "left": round(bbox.left, 4), "top": round(bbox.top, 4),
            "right": round(bbox.right, 4), "bottom": round(bbox.bottom, 4),
            "width": round(width, 4), "height": round(height, 4),
            "bbox_area": round(bbox_area, 4),
            "poly_area": round(poly_area, 4),
            "perimeter": round(perim, 4),
            "centroid_x": round(centroid_x, 4),
            "centroid_y": round(centroid_y, 4),
            "aspect": round(aspect, 4),
            "num_anchors": num_anchors,
            "subpath_count": subpath_count,
            "closed": 0,
            "filled": 1 if filled else 0,
            "fill_kind": "rgb" if fill_hex else "none",
            "fill_r": "", "fill_g": "", "fill_b": "",
            "stroked": 1 if stroked else 0,
            "stroke_kind": "rgb" if stroke_hex else "none",
            "stroke_r": "", "stroke_g": "", "stroke_b": "",
            "stroke_width": round(stroke_width, 4),
            "principal_angle": round(pca_a, 4),
            "principal_ratio": round(pca_r, 4),
            "longest_segment_angle": round(longest_a, 4),
            "longest_segment_length": round(longest_l, 4),
            "artboard_left": round(artboard.left, 4),
            "artboard_top": round(artboard.top, 4),
            "artboard_right": round(artboard.right, 4),
            "artboard_bottom": round(artboard.bottom, 4),
            "hull_area": round(hull_area, 4),
        })

        # Edges per subpath (treated as open, matching the JSX exporter)
        import math
        for sp_idx, sp in enumerate(sps_ai):
            n = len(sp)
            if n < 2:
                continue
            edge_idx = 0
            for i in range(n - 1):
                x1, y1 = sp[i]
                x2, y2 = sp[i + 1]
                dx = x2 - x1
                dy = y2 - y1
                length = math.hypot(dx, dy)
                if length < 0.001:
                    continue
                deg = math.degrees(math.atan2(dy, dx))
                deg = ((deg % 180.0) + 180.0) % 180.0
                edges.append({
                    "airport": airport,
                    "object_id": object_id,
                    "subpath_index": sp_idx,
                    "edge_index": edge_idx,
                    "mid_x": round((x1 + x2) / 2.0, 4),
                    "mid_y": round((y1 + y2) / 2.0, 4),
                    "angle": round(deg, 4),
                    "length": round(length, 4),
                })
                edge_idx += 1

        # SVG render data — kept in PDF y-down coordinates so the SVG
        # renders the same orientation as the source PDF.
        svg_data.append({
            "d": _items_to_svg_d(items),
            "fill": fill_hex,
            "stroke": stroke_hex,
            "stroke_width": stroke_width,
            "filled": filled,
            "stroked": stroked,
        })

        object_id += 1

    doc.close()
    paths_df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    edges_df = pd.DataFrame(edges, columns=EDGE_COLUMNS) if edges else pd.DataFrame(columns=EDGE_COLUMNS)
    return paths_df, edges_df, svg_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True,
                    help="Output SVG path (e.g. bna-chars.svg)")
    ap.add_argument("--model-dir", type=Path, default=None,
                    help="Defaults to v3-binary in --binary mode (default), "
                         "v2 in --multiclass mode")
    ap.add_argument("--multiclass", action="store_true",
                    help="Use the 37-class recognizer for per-character layers "
                         "instead of the binary detector")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Binary-mode CHAR-probability cutoff")
    ap.add_argument("--reject-prob", type=float, default=0.30,
                    help="Multiclass-mode top-1 prob below this routes to 'Not a Character'")
    args = ap.parse_args()

    if args.model_dir is None:
        default_name = "v2" if args.multiclass else "v3-binary"
        args.model_dir = Path(__file__).parent / "runs" / "char" / default_name

    airport = args.pdf.stem.replace("-faa", "").lower()
    print(f"[render] {airport}: {args.pdf}")

    paths_df, edges_df, svg_data = extract_features_and_svg(args.pdf, airport)
    print(f"[render] extracted {len(paths_df)} paths")

    annotated = add_char_predictions(paths_df, edges_df, args.model_dir)

    # Detect binary vs multiclass from the model's saved feature list.
    feature_meta = json.loads((args.model_dir / "feature_list.json").read_text())
    is_binary = bool(feature_meta.get("binary", False))

    layer_indices: dict[str, list[int]] = {}
    if is_binary:
        # Binary detector: char_prob is P(CHAR); compare to threshold
        # directly. Output exactly two layers so the visualization
        # answers the only question: "is this polygon a character at all?"
        for i, row in annotated.iterrows():
            char_p = float(row["char_prob"])
            layer = CHAR_LAYER if char_p >= args.threshold else NOT_A_CHAR
            layer_indices.setdefault(layer, []).append(i)
    else:
        # Multiclass: route the v2/v3 _NOT_ class directly to NOT_A_CHAR
        # and apply the soft prob threshold to anything else.
        for i, row in annotated.iterrows():
            prob = float(row["char_prob"])
            char = str(row["char_pred"])
            if char == NEGATIVE_CLASS or prob < args.reject_prob:
                layer = NOT_A_CHAR
            else:
                layer = char
            layer_indices.setdefault(layer, []).append(i)

    # Build SVG. PDF native is y-down; SVG default is y-down too — so
    # paths emitted in PDF coords render correctly with no flip.
    doc = fitz.open(args.pdf)
    page_h = float(doc[0].rect.height)
    page_w = float(doc[0].rect.width)
    doc.close()

    # Use minimal namespaces. Inkscape's groupmode="layer" attribute
    # makes Illustrator/Inkscape render each <g> as a named layer.
    NS = {
        "xmlns": "http://www.w3.org/2000/svg",
        "xmlns:inkscape": "http://www.inkscape.org/namespaces/inkscape",
        "xmlns:sodipodi": "http://sodipodi.sourceforge.net/DTD/sodipodi-0.0.dtd",
    }
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ET.register_namespace("inkscape", "http://www.inkscape.org/namespaces/inkscape")
    ET.register_namespace("sodipodi", "http://sodipodi.sourceforge.net/DTD/sodipodi-0.0.dtd")

    svg_attrs = {
        "xmlns": "http://www.w3.org/2000/svg",
        "xmlns:inkscape": "http://www.inkscape.org/namespaces/inkscape",
        "width": f"{page_w}",
        "height": f"{page_h}",
        "viewBox": f"0 0 {page_w} {page_h}",
    }
    svg = ET.Element("svg", svg_attrs)

    # Background rect so the chart isn't transparent on a dark theme
    ET.SubElement(svg, "rect", {
        "x": "0", "y": "0",
        "width": str(page_w), "height": str(page_h),
        "fill": "#ffffff",
    })

    # Emit layers in canonical order. Binary mode → just two layers;
    # multiclass → digits, letters, then NOT_A_CHAR last.
    if is_binary:
        layer_order = [CHAR_LAYER, NOT_A_CHAR]
    else:
        layer_order = LAYER_CHARS + [NOT_A_CHAR]
    for layer in layer_order:
        idxs = layer_indices.get(layer, [])
        if not idxs:
            continue
        g = ET.SubElement(svg, "g", {
            "id": layer,
            "inkscape:label": layer,
            "inkscape:groupmode": "layer",
        })
        for i in idxs:
            meta = svg_data[i]
            row = annotated.iloc[i]
            attrs = {
                "d": meta["d"],
                "fill": meta["fill"] or "none",
            }
            if meta["stroke"]:
                attrs["stroke"] = meta["stroke"]
                attrs["stroke-width"] = f"{meta['stroke_width']:.3f}"
            # Embed prediction info as data attributes for hover-debug
            attrs["data-object-id"] = str(int(row["object_id"]))
            attrs["data-pred"] = str(row["char_pred"])
            attrs["data-prob"] = f"{float(row['char_prob']):.3f}"
            attrs["data-top3"] = str(row["char_top3"])
            ET.SubElement(g, "path", attrs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(svg).write(args.out, encoding="utf-8", xml_declaration=True)
    print(f"[render] wrote -> {args.out}")

    print("[render] layer distribution:")
    total = len(annotated)
    for layer in layer_order:
        n = len(layer_indices.get(layer, []))
        if n == 0:
            continue
        bar = "#" * min(40, n // max(1, total // 200))
        print(f"   {layer:<18} {n:>5}  {bar}")


if __name__ == "__main__":
    main()
