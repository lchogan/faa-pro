"""
Extract the text layer from each airport's FAA PDF and dump one CSV row per
word, with bounding boxes y-flipped so they share the Illustrator coordinate
frame used by ExportClassifiedPaths.jsx (PDF y-down → AI y-up).

Each row carries an `is_known_runway` flag. When --nasr-rwy is provided
(pointing at FAA NASR APT_RWY.csv — the authoritative per-airport runway
record from FAA's 28-day subscription), that flag is set deterministically
from the FAA designation list. Without --nasr-rwy, it falls back to a
heuristic that parses runway-pair tokens (e.g. "08L-26R") out of the chart's
own text — less reliable, since pair tokens only appear when the chart
includes a lighting box.

Usage:
    python extract_pdf_text.py \
      --root /path/to/airports-class \
      --nasr-rwy /path/to/data/nasr_apt_rwy.csv \
      --out pdf_text.csv

Schema:
    airport, text, x_min, y_min, x_max, y_max,
    page_height, page_width, is_known_runway

Each `<code>/<code>-faa.pdf` under --root contributes its first page's text.
y_min/y_max are already in Illustrator coordinates (so:
   y_path = page_height - y_pdf
is applied here once, and downstream code can compare directly to PathItem
bounds emitted by the JSX exporter).
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import fitz  # PyMuPDF


# Runway pairs in airport diagrams appear as "08L-26R", "09R/27L", "10-28",
# etc. Extracting these gives the *actual* set of runway designations for the
# airport — far cleaner than trusting every word that matches the runway
# pattern (compass labels, magnetic variations, etc. also match).
_PAIR_RE = re.compile(r"^(\d{1,2})([LRC]?)[/\-](\d{1,2})([LRC]?)$")
_DESIG_RE = re.compile(r"^(\d{1,2})([LRC]?)$")


def _normalize_designation(text: str) -> str | None:
    """Return canonical form ('8L' from '08L', '10' from '10') or None."""
    m = _DESIG_RE.match(text)
    if not m:
        return None
    return str(int(m.group(1))) + m.group(2)


def find_known_runway_designations(words):
    """Heuristic fallback: scan PDF word list for runway-pair tokens
    ("08L-26R", "10/28", etc) and extract canonical designations.

    Less reliable than NASR — only fires when the chart's lighting box is
    present, may miss unlit runways. Used when --nasr-rwy isn't provided.
    """
    desigs = set()
    for w in words:
        text = w[4] if isinstance(w, tuple) else w
        m = _PAIR_RE.match(text)
        if m:
            d1 = str(int(m.group(1))) + m.group(2)
            d2 = str(int(m.group(3))) + m.group(4)
            desigs.add(d1)
            desigs.add(d2)
    return desigs


def load_nasr_runways(csv_path: Path) -> dict[str, set[str]]:
    """Read NASR APT_RWY.csv and return {airport_code_lower: {canonical_designation, ...}}.

    The RWY_ID column carries pair form ("08L/26R", "10/28"). Each pair is
    split into its two canonical designations ("8L", "26R", "10", "28").
    The dict is keyed by FAA airport ID lowercased to match the airport
    folder names (e.g. "atl", "21d").
    """
    import csv as _csv
    out: dict[str, set[str]] = {}
    with open(csv_path, encoding="utf-8-sig") as f:
        for row in _csv.DictReader(f):
            ap = (row.get("ARPT_ID") or "").strip().lower()
            rwy_id = (row.get("RWY_ID") or "").strip()
            if not ap or not rwy_id:
                continue
            m = _PAIR_RE.match(rwy_id)
            if not m:
                # Single-end designations are rare in NASR but fall back gracefully.
                norm = _normalize_designation(rwy_id)
                if norm:
                    out.setdefault(ap, set()).add(norm)
                continue
            d1 = str(int(m.group(1))) + m.group(2)
            d2 = str(int(m.group(3))) + m.group(4)
            out.setdefault(ap, set()).update([d1, d2])
    return out


def find_pdfs(root: Path):
    """Yield (airport_code, pdf_path) for every <code>/<code>-faa.pdf under root."""
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        airport = child.name
        pdf = child / f"{airport}-faa.pdf"
        if pdf.exists():
            yield airport, pdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True,
                    help="Folder containing one subdirectory per airport (e.g. airports-class)")
    ap.add_argument("--nasr-rwy", type=Path, default=None,
                    help="FAA NASR APT_RWY.csv. Recommended — provides authoritative "
                         "per-airport runway designations. Without it, falls back "
                         "to parsing runway-pair tokens out of the chart text.")
    ap.add_argument("--out", type=Path, required=True, help="Output CSV path")
    args = ap.parse_args()

    nasr_runways: dict[str, set[str]] = {}
    if args.nasr_rwy is not None:
        nasr_runways = load_nasr_runways(args.nasr_rwy)
        print(f"[extract_pdf_text] loaded NASR runway data for {len(nasr_runways):,} airports")

    rows = 0
    skipped = 0
    with args.out.open("w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["airport", "text", "x_min", "y_min", "x_max", "y_max",
                     "page_height", "page_width", "is_known_runway"])
        for airport, pdf_path in find_pdfs(args.root):
            try:
                doc = fitz.open(pdf_path)
            except Exception as e:
                print(f"[extract_pdf_text] {airport}: failed to open ({e})")
                skipped += 1
                continue
            if doc.page_count == 0:
                doc.close()
                skipped += 1
                continue
            page = doc[0]
            ph = float(page.rect.height)
            pw = float(page.rect.width)
            words = page.get_text("words")
            # Authoritative source first; fall back to chart-text heuristic
            # only if the airport isn't present in the NASR table.
            known_runways = nasr_runways.get(airport.lower())
            source = "NASR"
            if not known_runways:
                known_runways = find_known_runway_designations(words)
                source = "chart-pair fallback"
            n = 0
            for word in words:
                x0, y0, x1, y1, text, *_ = word
                if not text or not text.strip():
                    continue
                norm = _normalize_designation(text)
                is_known = bool(norm and norm in known_runways)
                y_ai_min = ph - float(y1)
                y_ai_max = ph - float(y0)
                cw.writerow([
                    airport,
                    text,
                    f"{float(x0):.4f}",
                    f"{y_ai_min:.4f}",
                    f"{float(x1):.4f}",
                    f"{y_ai_max:.4f}",
                    f"{ph:.4f}",
                    f"{pw:.4f}",
                    int(is_known),
                ])
                n += 1
                rows += 1
            doc.close()
            print(f"[extract_pdf_text] {airport}: {n} words, {len(known_runways)} known runways via {source}: {sorted(known_runways)}")
    print(f"\n[extract_pdf_text] wrote {rows:,} rows to {args.out}  (skipped {skipped} files)")


if __name__ == "__main__":
    main()
