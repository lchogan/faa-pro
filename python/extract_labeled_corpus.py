"""
extract_labeled_corpus.py — extract labeled CSVs from labeled .ai files
across one or more corpus roots, concatenated into a single training CSV.

Each root is expected to be `<root>/<code>/<code>-diagram.ai`. The
labeling-aware mode of extract_paths_fitz.extract_paths reads each
drawing's OCG layer name (PyMuPDF surfaces it as `d['layer']`) and maps
it to a canonical label via load.layer_name_to_label, with hidden OCG
layers force-enabled so Other / Uncertain / Lines / Text / Arrowheads
content isn't silently dropped.

Usage:
    python extract_labeled_corpus.py \\
        --root /Users/lukehogan/AOA-Code/faa-downloader/airports-class \\
        --root /Users/lukehogan/Documents/startups/aoa/products/artwork/airports \\
        --out  /Users/lukehogan/AOA-Code/faa-pro/python/labeled_corpus.csv \\
        --workers 6 \\
        --us-only

`--us-only` filters legacy folders to those whose code appears in NASR
APT_RWY (drops 38 international airports — they were sourced from OSM and
have different footprint stylization, per the project memory).

Outputs `<out>` and `<out>_edges.csv` next to it.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import pandas as pd

from extract_paths_fitz import CSV_COLUMNS, EDGE_COLUMNS, extract_paths


def _walk_diagrams(root: Path) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    if not root.is_dir():
        return out
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        ai = child / f"{child.name}-diagram.ai"
        if ai.exists():
            out.append((child.name, ai))
    return out


def _process_one(args: tuple[str, str]) -> tuple[str, list[dict], list[dict], str | None]:
    airport, ai_path = args
    try:
        rows, edges = extract_paths(Path(ai_path), airport)
        return (airport, rows, edges, None)
    except Exception as e:  # noqa: BLE001
        return (airport, [], [], f"{type(e).__name__}: {e}")


def _load_us_codes(nasr_csv: Path) -> set[str]:
    df = pd.read_csv(nasr_csv, usecols=["ARPT_ID"])
    return set(df["ARPT_ID"].astype(str).str.upper().unique())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, action="append", required=True,
                    help="Repeat for each corpus root containing <code>/<code>-diagram.ai")
    ap.add_argument("--out", type=Path, required=True,
                    help="Combined labeled CSV path (edges sidecar written next to it)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--us-only", action="store_true",
                    help="Drop folders whose code isn't in NASR APT_RWY (legacy intl airports)")
    ap.add_argument("--nasr-rwy", type=Path,
                    default=_HERE.parent / "data" / "nasr_apt_rwy.csv")
    args = ap.parse_args()

    workers = args.workers or max(1, (os.cpu_count() or 4) - 1)

    us_codes: set[str] | None = None
    if args.us_only:
        us_codes = _load_us_codes(args.nasr_rwy)
        print(f"[corpus] NASR US codes loaded: {len(us_codes):,}")

    # Collect work, dedup by airport code (later roots don't override earlier).
    seen: set[str] = set()
    work: list[tuple[str, str]] = []
    skipped_intl = 0
    for root in args.root:
        for code, ai in _walk_diagrams(root):
            if code in seen:
                continue
            if us_codes is not None and code.upper() not in us_codes:
                skipped_intl += 1
                continue
            seen.add(code)
            work.append((code, str(ai)))

    print(f"[corpus] roots: {len(args.root)}  airports queued: {len(work)}"
          + (f"  skipped (non-US): {skipped_intl}" if args.us_only else "")
          + f"  workers: {workers}")
    if not work:
        print("[corpus] nothing to do.")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    edges_out = args.out.with_name(args.out.stem + "_edges" + args.out.suffix)

    t0 = time.time()
    failed: list[tuple[str, str]] = []
    n_rows_total = 0
    n_edges_total = 0

    with args.out.open("w", newline="") as fh, edges_out.open("w", newline="") as eh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        edge_writer = csv.DictWriter(eh, fieldnames=EDGE_COLUMNS)
        edge_writer.writeheader()

        with mp.Pool(workers) as pool:
            for i, (code, rows, edges, err) in enumerate(
                pool.imap_unordered(_process_one, work, chunksize=2), start=1
            ):
                if err:
                    failed.append((code, err))
                    marker = "✗"
                else:
                    writer.writerows(rows)
                    edge_writer.writerows(edges)
                    n_rows_total += len(rows)
                    n_edges_total += len(edges)
                    marker = "✓"
                elapsed = time.time() - t0
                print(f"[{i:>4}/{len(work)}  {elapsed:>6.1f}s]  "
                      f"{marker} {code}  paths={len(rows)}  edges={len(edges)}"
                      + (f"  err={err}" if err else ""))

    elapsed = time.time() - t0
    print()
    print(f"[corpus] done. rows={n_rows_total:,}  edges={n_edges_total:,}  "
          f"failed={len(failed)}  elapsed={elapsed:.1f}s")
    print(f"  -> {args.out}")
    print(f"  -> {edges_out}")
    if failed:
        print("[corpus] failures:")
        for c, e in failed[:20]:
            print(f"   {c}: {e}")
        if len(failed) > 20:
            print(f"   ... (+{len(failed) - 20} more)")


if __name__ == "__main__":
    main()
