from __future__ import annotations

import argparse
from pathlib import Path

import fitz


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert one PDF page to SVG for Illustrator import.")
    ap.add_argument("--pdf", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--page", type=int, default=1)
    args = ap.parse_args()

    doc = fitz.open(args.pdf)
    try:
        page = doc[args.page - 1]
        svg = page.get_svg_image(text_as_path=True)
        args.out.write_text(svg, encoding="utf-8")
        print(f"wrote {args.out} from {args.pdf} page {args.page}")
    finally:
        doc.close()


if __name__ == "__main__":
    main()
