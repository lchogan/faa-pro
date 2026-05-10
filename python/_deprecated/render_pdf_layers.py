"""
render_pdf_layers.py — Python PDF renderer for the layered AI output.

Replaces the slow JSX bbox-matching flow. Builds a PDF where each
TARGET_LAYERS entry is a PDF Optional Content Group (OCG). Illustrator
opens the PDF and presents each OCG as a native Illustrator layer:
  open the .pdf -> File > Save As ... > Adobe Illustrator (.ai) and
  the layer structure carries over directly. No JSX, no bbox matching.

Polygon geometry is re-emitted from PyMuPDF's per-drawing items, so
each polygon's fill/stroke/path is preserved exactly. Each drawing's
target layer comes from <airport>_predictions.json (one prediction
per polygon, indices aligned to PyMuPDF iteration order with the same
off-artboard / empty-subpath filter as extract_paths_fitz.py).

Usage:
    python render_pdf_layers.py \\
        --pdf /path/to/<airport>-faa.pdf \\
        --predictions /path/to/<airport>_predictions.json \\
        --out /path/to/<airport>-diagram.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import fitz

from pipeline.extract_paths_fitz import (
    _Bounds,
    _is_off_artboard,
    items_to_subpaths,
)


TARGET_LAYERS = [
    "PDF Text Tokens",
    "Lights",
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
TOKEN_COLOR = (0.8, 0.0, 0.8)  # magenta in [0, 1]


def _color_tuple(raw):
    """Normalize a PyMuPDF color value to an RGB tuple in [0, 1].
    PDFs use 1/3/4-component colors. Returns None for absent/unknown."""
    if raw is None:
        return None
    if not isinstance(raw, (tuple, list)):
        return None
    if len(raw) == 1:
        g = float(raw[0])
        return (g, g, g)
    if len(raw) == 3:
        return tuple(float(c) for c in raw)
    if len(raw) == 4:
        c, m, y, k = (float(x) for x in raw)
        return ((1 - c) * (1 - k), (1 - m) * (1 - k), (1 - y) * (1 - k))
    return None


def _draw_path(shape, items):
    """Re-emit PyMuPDF drawing items into a Shape. Mirrors
    extract_paths_fitz.items_to_subpaths' walking of the items list."""
    for item in items:
        op = item[0]
        if op == "l":
            shape.draw_line(item[1], item[2])
        elif op == "c":
            shape.draw_bezier(item[1], item[2], item[3], item[4])
        elif op == "re":
            shape.draw_rect(item[1])
        elif op == "qu":
            shape.draw_quad(item[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=Path, required=True,
                    help="Source <airport>-faa.pdf")
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True,
                    help="Output PDF (Illustrator opens it as layered AI).")
    args = ap.parse_args()

    payload = json.loads(args.predictions.read_text())
    preds = payload.get("predictions", [])
    text_tokens = payload.get("text_tokens", [])

    # Walk source PDF in the same order extract_paths_fitz emits, so
    # drawing[i] aligns with predictions[i].
    src = fitz.open(args.pdf)
    src_page = src[0]
    page_w = float(src_page.rect.width)
    page_h = float(src_page.rect.height)
    artboard = _Bounds(left=0.0, top=page_h, right=page_w, bottom=0.0)

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
            f"predictions JSON has {len(preds)}."
        )
    print(f"[render_pdf] {len(valid_drawings)} polygons aligned to predictions")

    # Build the output PDF.
    out_doc = fitz.open()
    out_page = out_doc.new_page(width=page_w, height=page_h)

    # OCG (= Illustrator layer) creation. Order matters — the first OCG
    # ends up at the top of Illustrator's Layers panel.
    # OCG creation. PyMuPDF's add_ocg() can't accept a list for `intent`
    # (its assertion requires str), so we patch each OCG's /Intent and
    # /Usage dict entries via xref_set_key after creation. Without
    # /Intent containing "Design" and /Usage with /CreatorInfo
    # /Subtype/Artwork, Illustrator treats the OCGs as inert "PDF Layers"
    # metadata and dumps everything into Layer 1.
    ocgs: dict[str, int] = {}
    ocg_xrefs_in_panel_order: list[int] = []
    for name in TARGET_LAYERS:
        xref = out_doc.add_ocg(name, on=1)
        out_doc.xref_set_key(xref, "Intent", "[/View/Design]")
        out_doc.xref_set_key(
            xref, "Usage",
            "<</CreatorInfo<</Creator(faa-pro)/Subtype/Artwork>>>>",
        )
        ocgs[name] = xref
        ocg_xrefs_in_panel_order.append(xref)

    # /OCProperties/D/Order controls the layer order in Illustrator's
    # panel. First entry = topmost layer.
    order_str = "[" + " ".join(f"{x} 0 R" for x in ocg_xrefs_in_panel_order) + "]"
    out_doc.xref_set_key(out_doc.pdf_catalog(),
                         "OCProperties/D/Order", order_str)

    layer_counts: dict[str, int] = {name: 0 for name in TARGET_LAYERS}

    # One Shape, many finishes, one commit. PyMuPDF's commit() runs an
    # O(n) parser over the page's content stream every time it's called,
    # so doing 4331 commits is O(n²) in shape count and was the dominant
    # cost. Each finish() ends a subpath with its own fill/stroke/OCG;
    # the single trailing commit() pays the parser cost once.
    shape = out_page.new_shape()
    for i, d in enumerate(valid_drawings):
        rec = preds[i]
        label = rec.get("label", "Other")
        if label not in ocgs:
            label = "Other"
        oc_xref = ocgs[label]

        dtype = d.get("type", "")
        is_filled = "f" in dtype
        is_stroked = "s" in dtype
        fill_rgb = _color_tuple(d.get("fill")) if is_filled else None
        stroke_rgb = _color_tuple(d.get("color")) if is_stroked else None
        stroke_w = float(d.get("width") or 0.5) if is_stroked else 0.0

        _draw_path(shape, d.get("items"))
        finish_kwargs = {"oc": oc_xref, "closePath": False}
        if fill_rgb is not None:
            finish_kwargs["fill"] = fill_rgb
        if stroke_rgb is not None:
            finish_kwargs["color"] = stroke_rgb
            finish_kwargs["width"] = stroke_w
        else:
            finish_kwargs["color"] = None
            finish_kwargs["width"] = 0
        shape.finish(**finish_kwargs)
        layer_counts[label] = layer_counts.get(label, 0) + 1
    shape.commit()

    # PDF Text Tokens via TextWriter — same O(n²) issue applies to
    # insert_text (each call wraps the content stream). TextWriter
    # batches all text into a single write_text() call.
    text_oc = ocgs["PDF Text Tokens"]
    tw = fitz.TextWriter(out_page.rect, color=TOKEN_COLOR)
    for tok in text_tokens:
        text = str(tok.get("text", ""))
        if not text:
            continue
        x_ai = float(tok.get("x", 0))           # AI x == PDF x
        y_ai = float(tok.get("y", 0))           # AI y-up
        pdf_y_center = page_h - y_ai
        # insert_text places the text at the baseline. Center the
        # rendered text on (x_ai, pdf_y_center): horizontally by
        # subtracting half the text width, vertically by adding
        # ~0.36 * fontsize so cap-midline lands on pdf_y_center.
        text_w = fitz.get_text_length(text, fontsize=TOKEN_FONT_SIZE,
                                       fontname="helv")
        tw.append(
            (x_ai - text_w / 2.0, pdf_y_center + TOKEN_FONT_SIZE * 0.36),
            text,
            fontsize=TOKEN_FONT_SIZE,
        )
    tw.write_text(out_page, oc=text_oc)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_doc.save(args.out)
    print(f"[render_pdf] wrote -> {args.out}")
    print("Layer distribution:")
    for name in TARGET_LAYERS:
        n = layer_counts.get(name, 0)
        if n:
            print(f"  {name:<22} {n}")


if __name__ == "__main__":
    main()
