"""build_light_artwork.py — OFFLINE build: per-artboard SVG exports -> the
normalized artwork JSON the renderer consumes (artwork/lights/lights.json).

Run after exporting artboards from Illustrator ("Export for Screens", one SVG
per artboard, named by the artwork key). Converts every drawable shape to a
Path via svgelements, applies the canonical affine transform as a Matrix (so
Bézier control points are transformed correctly, not dropped), handles SVG
primitives (rect/circle/line/polygon) that Illustrator commonly emits, and
writes lights.json.

Dependencies:
    svgelements >= 1.9 (pip install svgelements)

Key concepts:
    Canonical frame: attach point is the artboard bottom-center (W/2, H).
    The canonical transform is the affine Matrix(0, 1, -1, 0, H, -W/2),
    equivalent to the per-point formula:
        canonical (X, Y) = (H - py, px - W/2)
    So the attach point maps to (0, 0), and paths extending "up" (decreasing
    py) produce positive X values — the forward direction.

    Applying the transform via Matrix (not per-endpoint arithmetic) preserves
    Bézier control points, so curved shapes (flasher dots, rounded marks)
    survive the normalization step unchanged.

    native_width = W is the shared artboard width used by the renderer to
    compute a uniform scale factor for each airport diagram.

Usage:
    python pipeline/build_light_artwork.py artwork/lights/ artwork/lights/lights.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from svgelements import SVG, Path as SvgPath, Shape, Matrix


def _symbol_from_svg(svg_path: Path) -> tuple[float, list[dict]]:
    """Return (artboard_width, paths) with every drawable shape converted to a
    Path and mapped into the canonical frame (attach bottom-center -> origin,
    "up" -> +X) via the affine Matrix(0, 1, -1, 0, H, -W/2).

    Using Matrix multiplication rather than per-point arithmetic ensures Bézier
    control points are transformed correctly (curves preserved, not flattened).
    Filtering on Shape (not SvgPath) ensures SVG primitives emitted by
    Illustrator (rect, circle, line, polygon) are included rather than dropped.

    Args:
        svg_path: Absolute path to the artboard SVG file.

    Returns:
        A 2-tuple (W, paths) where W is the artboard width (float) and paths
        is a list of dicts with keys: d, fill, stroke, stroke_width.
    """
    svg = SVG.parse(str(svg_path))
    W = float(svg.width)
    H = float(svg.height)
    # Affine map verified: (50,120)->(0,0) and (50,20)->(100,0) for W=100,H=120.
    to_canonical = Matrix(0, 1, -1, 0, H, -W / 2.0)
    paths: list[dict] = []

    for el in svg.elements():
        # Drawable shapes only — excludes the SVG root and any group containers.
        # Shape is the common base of Path/Rect/Circle/Ellipse/SimpleLine/
        # Polyline/Polygon; SVG itself also inherits Shape so exclude it.
        if not isinstance(el, Shape) or isinstance(el, SVG):
            continue

        # abs() resolves the element's own transform and converts to absolute
        # coordinates; * to_canonical applies the canonical affine map.
        # Both operations work on the full segment sequence (including control
        # points), so Bézier curves are preserved correctly.
        path = abs(SvgPath(el)) * to_canonical
        d = path.d()
        if not d.strip():
            continue

        # Resolve fill / stroke to hex strings; Color.value is None for "none".
        stroke = el.stroke
        fill = el.fill
        fill_str = fill.hex if fill is not None and fill.value is not None else "none"
        stroke_str = stroke.hex if stroke is not None and stroke.value is not None else "none"
        stroke_width = float(el.stroke_width) if el.stroke_width is not None else 0.0

        paths.append({
            "d": d,
            "fill": fill_str,
            "stroke": stroke_str,
            "stroke_width": stroke_width,
        })

    return W, paths


def build_light_artwork(artboards_dir: Path, out_json: Path) -> None:
    """Build lights.json from a directory of per-artboard SVGs.

    Reads every *.svg in artboards_dir, normalizes each path to the canonical
    frame, asserts all artboards share one width, and writes the JSON library.

    Args:
        artboards_dir: Directory containing one SVG per artwork symbol,
            named <key>.svg (e.g. malsr.svg, odals.svg).
        out_json: Destination path for the JSON output file.

    Raises:
        ValueError: If the directory has no SVGs, or artboards differ in width.
    """
    artboards_dir = Path(artboards_dir)
    svgs = sorted(p for p in artboards_dir.glob("*.svg"))
    if not svgs:
        raise ValueError(f"No .svg artboards in {artboards_dir}")

    symbols: dict[str, dict] = {}
    widths: list[float] = []
    for svg_path in svgs:
        W, paths = _symbol_from_svg(svg_path)
        widths.append(W)
        symbols[svg_path.stem] = {"paths": paths}

    if max(widths) - min(widths) > 1e-6:
        raise ValueError(
            "Artboards must share one width; got "
            + str(dict(zip([p.stem for p in svgs], widths)))
        )

    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(
        {"native_width": widths[0], "symbols": symbols}, indent=2,
    ))


def main() -> None:
    """CLI entry point: build_light_artwork.py <artboards_dir> <out.json>."""
    if len(sys.argv) != 3:
        raise SystemExit("usage: build_light_artwork.py <artboards_dir> <out.json>")
    build_light_artwork(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    main()
