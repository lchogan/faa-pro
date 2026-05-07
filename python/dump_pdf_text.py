"""
dump_pdf_text.py — extract every word from a FAA PDF and write a JSON file
that DebugTextLayer.jsx can read to place text items in Illustrator.

Coordinates are already flipped to Illustrator space (y-up, origin at
bottom-left of page), so they should line up with the vector paths.

Usage:
    python dump_pdf_text.py --pdf bna-faa.pdf --out bna_text_debug.json
"""

import argparse
import json
from pathlib import Path

import fitz  # PyMuPDF


def dump(pdf_path: Path, out_path: Path) -> None:
    doc = fitz.open(pdf_path)
    page = doc[0]
    ph = float(page.rect.height)
    pw = float(page.rect.width)

    items = []
    for word in page.get_text("words"):
        x0, y0, x1, y1, text, *_ = word
        if not text.strip():
            continue
        # Flip y to Illustrator space (origin bottom-left, y increases upward).
        ai_y_min = ph - float(y1)   # bottom of word in AI coords
        ai_y_max = ph - float(y0)   # top of word in AI coords
        items.append({
            "text":    text.strip(),
            "x_min":  round(float(x0), 3),
            "y_min":  round(ai_y_min, 3),   # bottom (AI)
            "x_max":  round(float(x1), 3),
            "y_max":  round(ai_y_max, 3),   # top (AI)
        })

    doc.close()

    payload = {
        "pdf":         str(pdf_path),
        "page_width":  round(pw, 3),
        "page_height": round(ph, 3),
        "words":       items,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {len(items)} words → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    dump(args.pdf, args.out)


if __name__ == "__main__":
    main()
