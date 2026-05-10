"""
extract_paths_fitz.py — PyMuPDF-based path extraction matching the
ClassifyAirport.jsx CSV format.

Designed as a drop-in replacement for the Illustrator path-export step. The
output schema is identical to what the JSX exporter writes, so build_char_training,
load.py, and v25-style model retraining can all consume it without changes.

Key parity points with the JSX implementation:
  - All coordinates are y-flipped to Illustrator (y-up) frame: y_ai = page_h - y_pdf
  - perimeter uses straight-line distance between anchors (Bezier curvature ignored)
  - poly_area uses the shoelace formula on anchor points (closed subpaths only)
  - principal_angle / principal_ratio is PCA over all anchors across subpaths
  - num_anchors, subpath_count, closed track JSX semantics (closed = subpath 0)
  - "kind" is "compound" if subpath_count > 1, else "path"
  - Off-artboard drawings are dropped using page rect as the artboard

Mismatches that are intentional (and acceptable):
  - object_id ordering differs (PyMuPDF emits in PDF content-stream order;
    JSX emits compounds first then paths). For training data this is irrelevant —
    we match polygons to PDF tokens by spatial position, not object_id.
  - Color extraction: PyMuPDF gives normalized RGB tuples directly; we map
    them to 0-255 ints. Gray/CMYK paths are detected from tuple length.

Usage (single PDF):
    python extract_paths_fitz.py \\
        --pdf /path/to/abi-faa.pdf \\
        --out-dir /path/to/data/char_corpus/abi

Outputs:
    <out-dir>/<airport>_paths.csv
    <out-dir>/<airport>_paths_edges.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import fitz
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial import QhullError

from ml.load import UNLABELED, layer_name_to_label


CSV_COLUMNS = [
    "airport", "object_id", "kind", "source_layer", "label",
    "left", "top", "right", "bottom", "width", "height",
    "bbox_area", "poly_area", "perimeter",
    "centroid_x", "centroid_y", "aspect",
    "num_anchors", "subpath_count", "closed",
    "filled", "fill_kind", "fill_r", "fill_g", "fill_b",
    "stroked", "stroke_kind", "stroke_r", "stroke_g", "stroke_b", "stroke_width",
    "principal_angle", "principal_ratio",
    "longest_segment_angle", "longest_segment_length",
    "artboard_left", "artboard_top", "artboard_right", "artboard_bottom",
    # Extension columns used by the v2 character classifier. JSX-extracted
    # CSVs won't have these — load.py / consumers should treat them as
    # optional with sensible defaults (0.0 implies "unknown").
    "hull_area",
]
EDGE_COLUMNS = [
    "airport", "object_id", "subpath_index", "edge_index",
    "mid_x", "mid_y", "angle", "length",
]


# Tolerance for "this item's start equals previous item's end" — separates
# subpaths from continuations. PDF coordinates are in points (1/72 inch);
# 1e-3pt is far below visible precision.
_PT_EPS = 1e-3


def _approx_eq(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return abs(a[0] - b[0]) < _PT_EPS and abs(a[1] - b[1]) < _PT_EPS


def items_to_subpaths(items) -> list[list[tuple[float, float]]]:
    """Walk a PyMuPDF drawing's items and split them into subpaths.

    Each subpath is a list of anchor points (Bezier control points are
    discarded — only on-curve anchors are kept, matching JSX behavior).

    Subpaths are split where one item's start point != the previous item's
    end point. Rectangles ("re") and quads ("qu") are emitted as their own
    closed subpaths.
    """
    subpaths: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    last_end: tuple[float, float] | None = None

    def flush():
        nonlocal current
        if current:
            subpaths.append(current)
            current = []

    for item in items:
        op = item[0]
        if op == "l":
            p1, p2 = (item[1].x, item[1].y), (item[2].x, item[2].y)
            if last_end is None or not _approx_eq(p1, last_end):
                flush()
                current = [p1]
            current.append(p2)
            last_end = p2
        elif op == "c":
            # cubic: ("c", p1, control1, control2, p4)
            p1 = (item[1].x, item[1].y)
            p4 = (item[4].x, item[4].y)
            if last_end is None or not _approx_eq(p1, last_end):
                flush()
                current = [p1]
            current.append(p4)
            last_end = p4
        elif op == "re":
            r = item[1]
            flush()
            # Rectangle: 4 corners + first repeated to mark closure
            pts = [
                (r.x0, r.y0),
                (r.x1, r.y0),
                (r.x1, r.y1),
                (r.x0, r.y1),
                (r.x0, r.y0),
            ]
            subpaths.append(pts)
            last_end = None
        elif op == "qu":
            q = item[1]
            flush()
            pts = [
                (q.ul.x, q.ul.y), (q.ur.x, q.ur.y),
                (q.lr.x, q.lr.y), (q.ll.x, q.ll.y),
                (q.ul.x, q.ul.y),
            ]
            subpaths.append(pts)
            last_end = None
        # Other ops (h = close path, m = moveto) shouldn't appear in
        # get_drawings() output — it pre-splits into items. Ignore.

    flush()
    return subpaths


# Note on closure semantics: Illustrator's `closed` flag is set only when the
# path has an *explicit* close-path operation in its source data (rare in PDF
# imports — typically only for rect-stroke shapes). Geometric closure (start
# point == end point) does NOT set the flag. The JSX exporter records this
# Illustrator flag, so to match its output we always emit closed=0 unless the
# PDF source data explicitly closes — which PyMuPDF doesn't expose, so we just
# emit 0 universally. The 4-of-932 cases where JSX reports closed=1 don't
# affect downstream feature signal.
#
# Anchor counts: JSX reports `pathPoints.length`, which includes any duplicate
# close-anchor present in the source. We don't normalize subpaths so this
# matches.


def _shoelace_signed(pts: list[tuple[float, float]]) -> float:
    """Signed shoelace area. Positive = counterclockwise winding,
    negative = clockwise. Sign is meaningful for compound paths where
    inner subpaths (holes) wind opposite to the outer boundary so the
    summed signed area equals the visually filled area."""
    n = len(pts)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return s / 2.0


def _pca_angle_ratio(pts: list[tuple[float, float]]) -> tuple[float, float]:
    n = len(pts)
    if n < 2:
        return 0.0, 1.0
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = syy = sxy = 0.0
    for x, y in pts:
        dx, dy = x - mx, y - my
        sxx += dx * dx
        syy += dy * dy
        sxy += dx * dy
    sxx /= n; syy /= n; sxy /= n
    trace = sxx + syy
    det = sxx * syy - sxy * sxy
    disc = math.sqrt(max(0.0, trace * trace / 4.0 - det))
    l1 = trace / 2.0 + disc
    l2 = trace / 2.0 - disc
    if abs(sxy) < 1e-12:
        theta = 0.0 if sxx >= syy else math.pi / 2
    else:
        theta = math.atan2(l1 - sxx, sxy)
    deg = math.degrees(theta)
    deg = ((deg % 180.0) + 180.0) % 180.0
    ratio = (l1 / l2) if l2 > 1e-9 else 1e6
    if ratio > 1e6:
        ratio = 1e6
    return deg, ratio


def _perimeter(pts: list[tuple[float, float]], closed: bool) -> float:
    n = len(pts)
    if n < 2:
        return 0.0
    s = 0.0
    limit = n if closed else n - 1
    for i in range(limit):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += math.hypot(x2 - x1, y2 - y1)
    return s


def _longest_segment(pts: list[tuple[float, float]], closed: bool) -> tuple[float, float]:
    n = len(pts)
    if n < 2:
        return 0.0, 0.0
    best_l = 0.0
    best_a = 0.0
    limit = n if closed else n - 1
    for i in range(limit):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        l = math.hypot(dx, dy)
        if l > best_l:
            best_l = l
            deg = math.degrees(math.atan2(dy, dx))
            best_a = ((deg % 180.0) + 180.0) % 180.0
    return best_a, best_l


def _color_to_rgb(c) -> tuple[str | int, str | int, str | int, str]:
    """Map PyMuPDF color (None, scalar, 1/3/4-tuple) to (r, g, b, kind)."""
    if c is None:
        return "", "", "", "none"
    if isinstance(c, (int, float)):
        v = max(0, min(255, int(round(float(c) * 255))))
        return v, v, v, "gray"
    if isinstance(c, (tuple, list)):
        if len(c) == 3:
            r, g, b = c
            return (
                max(0, min(255, int(round(r * 255)))),
                max(0, min(255, int(round(g * 255)))),
                max(0, min(255, int(round(b * 255)))),
                "rgb",
            )
        if len(c) == 1:
            v = max(0, min(255, int(round(float(c[0]) * 255))))
            return v, v, v, "gray"
        if len(c) == 4:
            cc, mm, yy, kk = c
            r = max(0, min(255, int(round(255 * (1 - cc) * (1 - kk)))))
            g = max(0, min(255, int(round(255 * (1 - mm) * (1 - kk)))))
            b = max(0, min(255, int(round(255 * (1 - yy) * (1 - kk)))))
            return r, g, b, "cmyk"
    return "", "", "", "other"


@dataclass
class _Bounds:
    left: float
    top: float
    right: float
    bottom: float

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return abs(self.top - self.bottom)


def _is_off_artboard(b: _Bounds, ab: _Bounds) -> bool:
    """JSX rule: drop if bbox is entirely outside the artboard."""
    return (
        b.right < ab.left
        or b.left > ab.right
        or b.bottom > ab.top
        or b.top < ab.bottom
    )


def _force_all_layers_visible(doc: fitz.Document) -> None:
    """Enable every OCG UI config so get_drawings() returns hidden layers too.

    Critical for labeled .ai files: the user's clean corpus saves them with
    Other / Uncertain / Lines / Text / Arrowheads layers turned OFF in the UI
    config (so the file *displays* as a clean diagram). PyMuPDF respects that
    visibility and silently drops their drawings, which would discard the
    bulk of the negative-class training data. action=0 = "set to visible".
    No-op for source PDFs that have no OCG layers at all.
    """
    try:
        configs = doc.layer_ui_configs()
    except Exception:
        return
    for c in configs:
        try:
            doc.set_layer_ui_config(c["number"], action=0)
        except Exception:
            # Some configs are radio-group-locked; skip silently.
            pass


def extract_paths(pdf_path: Path, airport: str) -> tuple[list[dict], list[dict]]:
    """Return (path rows, edge rows) for the first page of pdf_path."""
    doc = fitz.open(pdf_path)
    try:
        _force_all_layers_visible(doc)
        page = doc[0]
        page_h = float(page.rect.height)
        page_w = float(page.rect.width)
    except Exception:
        doc.close()
        raise

    # JSX uses doc.artboards[0].artboardRect — for a one-page PDF that's
    # the page rect. AI frame is y-up, so top is the larger y.
    artboard = _Bounds(left=0.0, top=page_h, right=page_w, bottom=0.0)

    rows: list[dict] = []
    edge_rows: list[dict] = []
    object_id = 0

    drawings = page.get_drawings()
    # Labeled mode: this AI/PDF carries OCG layer info on at least one
    # drawing. In that case, drawings without a layer field were left in
    # Layer 1 (the unclassified holding tank) and get UNLABELED — they
    # should be excluded from training, not lumped into the Other negative
    # class. Source FAA PDFs have no OCG layers at all; every drawing comes
    # back with layer=None and we emit source_layer="" / label=UNLABELED so
    # the inference path doesn't accidentally treat them as training data.
    has_layer_info = any(d.get("layer") for d in drawings)
    for d in drawings:
        items = d.get("items", [])
        if not items:
            continue

        # Subpaths in PDF (y-down) coords first, then flip to AI y-up.
        sps_pdf = items_to_subpaths(items)
        if not sps_pdf:
            continue
        sps_ai = [[(x, page_h - y) for (x, y) in sp] for sp in sps_pdf]

        # Layer / label. PyMuPDF surfaces the OCG layer name as d['layer']
        # for drawings that came from a layered AI/PDF (None for source FAA
        # PDFs, which have no layers). When present, map it to a canonical
        # training label via load.layer_name_to_label.
        raw_layer = d.get("layer") or ""
        if raw_layer:
            label = layer_name_to_label(raw_layer)
        elif has_layer_info:
            # Labeled file but this drawing was left in Layer 1.
            raw_layer = "Layer 1"
            label = UNLABELED
        else:
            label = UNLABELED

        # Bbox: trust PyMuPDF's reported rect (it accounts for Bezier curves).
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

        subpath_count = len(sps_ai)
        kind = "compound" if subpath_count > 1 else "path"

        # JSX records closed only when Illustrator's explicit closed flag is
        # set. PDF imports almost never set it (~0.4% of paths). PyMuPDF
        # doesn't expose this distinction, so we always emit 0.
        first_closed = False

        # Anchor list and aggregated metrics. We do NOT drop duplicate
        # close-anchors — JSX's pathPoints.length includes them.
        # Perimeter / longest_segment treat the path as open (limit=n-1) to
        # match JSX, which uses the closed flag (always false here) to gate
        # the wrap-around segment.
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
            # Signed area per subpath — outer winding adds, hole (opposite
            # winding) subtracts. abs of sum = visually filled area.
            signed_area_sum += _shoelace_signed(sp)
        num_anchors = len(all_anchors)
        # The PyMuPDF closure heuristic doesn't always agree with JSX, so
        # we compute poly_area unconditionally instead of gating on closed.
        # Letter glyphs are always closed in practice; non-glyph open paths
        # produce a near-zero or noisy area which is itself a useful signal.
        poly_area = abs(signed_area_sum)

        # Convex hull area for solidity = poly_area / hull_area. Cheap with
        # scipy. Degenerate cases (collinear points, <3 points) → 0 which
        # downstream handles as "unknown".
        hull_area = 0.0
        if num_anchors >= 3:
            try:
                hull = ConvexHull(np.asarray(all_anchors, dtype=float))
                hull_area = float(hull.volume)  # in 2D this attribute is the area
            except (QhullError, ValueError):
                hull_area = 0.0

        pca_a, pca_r = _pca_angle_ratio(all_anchors)

        width = bbox.width
        height = bbox.height
        bbox_area = abs(width * height)
        aspect = (width / height) if height > 1e-9 else 0.0
        centroid_x = (bbox.left + bbox.right) / 2.0
        centroid_y = (bbox.top + bbox.bottom) / 2.0

        # Fill / stroke. PyMuPDF drawing types: "f", "s", "fs", "n".
        dtype = d.get("type", "")
        filled = "f" in dtype
        stroked = "s" in dtype
        fr, fg, fb, fk = _color_to_rgb(d.get("fill")) if filled else ("", "", "", "none")
        sr, sg, sb, sk = _color_to_rgb(d.get("color")) if stroked else ("", "", "", "none")
        stroke_width = float(d.get("width") or 0.0) if stroked else 0.0

        rows.append({
            "airport": airport,
            "object_id": object_id,
            "kind": kind,
            "source_layer": raw_layer,
            "label": label,
            "left": round(bbox.left, 4),
            "top": round(bbox.top, 4),
            "right": round(bbox.right, 4),
            "bottom": round(bbox.bottom, 4),
            "width": round(width, 4),
            "height": round(height, 4),
            "bbox_area": round(bbox_area, 4),
            "poly_area": round(poly_area, 4),
            "perimeter": round(perim, 4),
            "centroid_x": round(centroid_x, 4),
            "centroid_y": round(centroid_y, 4),
            "aspect": round(aspect, 4),
            "num_anchors": num_anchors,
            "subpath_count": subpath_count,
            "closed": 1 if first_closed else 0,
            "filled": 1 if filled else 0,
            "fill_kind": fk,
            "fill_r": fr, "fill_g": fg, "fill_b": fb,
            "stroked": 1 if stroked else 0,
            "stroke_kind": sk,
            "stroke_r": sr, "stroke_g": sg, "stroke_b": sb,
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

        # Edges per subpath. JSX treats paths as open (closed flag is
        # always false in our output), so emit n-1 segments and skip the
        # wrap-around closing segment.
        for sp_idx, sp in enumerate(sps_ai):
            n = len(sp)
            if n < 2:
                continue
            limit = n - 1
            edge_idx = 0
            for i in range(limit):
                x1, y1 = sp[i]
                x2, y2 = sp[i + 1]
                dx = x2 - x1; dy = y2 - y1
                length = math.hypot(dx, dy)
                if length < 0.001:
                    continue
                deg = math.degrees(math.atan2(dy, dx))
                deg = ((deg % 180.0) + 180.0) % 180.0
                edge_rows.append({
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

        object_id += 1

    doc.close()
    return rows, edge_rows


def write_csvs(out_dir: Path, airport: str, rows: list[dict], edges: list[dict]) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths_csv = out_dir / f"{airport}_paths.csv"
    edges_csv = out_dir / f"{airport}_paths_edges.csv"
    with paths_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    with edges_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EDGE_COLUMNS)
        w.writeheader()
        w.writerows(edges)
    return paths_csv, edges_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=Path, required=True)
    ap.add_argument("--airport", type=str, default=None,
                    help="Airport code; defaults to '<code>' from <code>-faa.pdf")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    if args.airport is None:
        args.airport = args.pdf.stem.replace("-faa", "").lower()

    rows, edges = extract_paths(args.pdf, args.airport)
    paths_csv, edges_csv = write_csvs(args.out_dir, args.airport, rows, edges)
    print(f"[extract_paths_fitz] {args.airport}: {len(rows)} paths, {len(edges)} edges")
    print(f"  -> {paths_csv}")
    print(f"  -> {edges_csv}")


if __name__ == "__main__":
    main()
