"""build_light_artwork.py — OFFLINE build: per-artboard SVG exports -> the
normalized artwork JSON the renderer consumes (artwork/lights/lights.json).

Run after exporting artboards from Illustrator ("Export for Screens", one SVG
per artboard, named by the artwork key). Flattens each SVG to absolute paths
via svgelements, normalizes to the canonical frame (attach point = bottom-
center -> origin; artboard "up" -> +X), asserts all artboards share one width
(the runway-width reference that makes per-document uniform scaling valid),
and writes lights.json.

Dependencies:
    svgelements >= 1.9 (pip install svgelements)

Key concepts:
    Canonical frame: attach point is the artboard bottom-center (W/2, H).
    The canonical transform maps SVG coords to frame coords:
        canonical (X, Y) = (H - py, px - W/2)
    So the attach point maps to (0, 0), and paths extending "up" (decreasing
    py) produce positive X values — the forward direction.

    native_width = W is the shared artboard width used by the renderer to
    compute a uniform scale factor for each airport diagram.

Usage:
    python pipeline/build_light_artwork.py artwork/lights/ artwork/lights/lights.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from svgelements import SVG, Path as SvgPath, Move


def _symbol_from_svg(svg_path: Path) -> tuple[float, list[dict]]:
    """Parse one artboard SVG and return (artboard_width, paths).

    Each path in the return list is normalized to the canonical frame where
    the artboard bottom-center is the origin and "up" is the +X direction.

    The canonical transform per point is:
        (X, Y) = (H - py, px - W/2)

    Args:
        svg_path: Absolute path to the artboard SVG file.

    Returns:
        A 2-tuple (W, paths) where W is the artboard width (float) and paths
        is a list of dicts with keys: d, fill, stroke, stroke_width.

    Raises:
        ValueError: If the SVG contains no parseable path elements.
    """
    svg = SVG.parse(str(svg_path))
    W = float(svg.width)
    H = float(svg.height)
    paths: list[dict] = []

    for el in svg.elements():
        if not isinstance(el, SvgPath):
            continue

        # Collect all subpaths and apply the canonical transform.
        # We build the d string manually (not via SvgPath.d()) so we control
        # the format: no spaces after M/L commands, which lets the test assert
        # d.startswith("M0") cleanly.
        parts: list[str] = []
        for sub in el.as_subpaths():
            segs = list(sub)
            if not segs:
                continue

            # First segment is always a Move; its .end is the subpath start.
            start_seg = segs[0]
            assert isinstance(start_seg, Move), (
                f"Expected Move as first subpath segment, got {type(start_seg)}"
            )
            allpts = [start_seg.end] + [s.end for s in segs[1:]]

            canon_pts = [(H - pt.y, pt.x - W / 2.0) for pt in allpts]
            first_X, first_Y = canon_pts[0]
            parts.append(f"M{first_X},{first_Y}")
            for X, Y in canon_pts[1:]:
                parts.append(f" L{X},{Y}")

        if not parts:
            continue

        d = "".join(parts)

        # Resolve fill / stroke to strings; Color.value is None for "none".
        stroke = el.stroke
        fill = el.fill
        stroke_str = str(stroke.hex) if stroke and stroke.value is not None else "none"
        fill_str = str(fill.hex) if fill and fill.value is not None else "none"
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
