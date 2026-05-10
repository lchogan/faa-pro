"""
extract_paths_batch.py — parallel PyMuPDF path extraction for the full
airport corpus.

Replaces the slow Illustrator-based export_paths.sh batch. Walks the
faa-downloader directories, runs extract_paths_fitz.extract_paths in
multiprocessing workers, and writes <airport>_paths.csv +
<airport>_paths_edges.csv into the corpus directory.

Resume-friendly: skips airports whose output CSVs already exist.

Usage:
    python extract_paths_batch.py
    python extract_paths_batch.py --workers 4 --limit 50
    python extract_paths_batch.py --root path/to/airports --root path/to/other
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

# Make sibling modules importable when the script runs from any CWD.
# python/ root needs to be on sys.path so `from pipeline.* import ...`
# and `from ml.* import ...` resolve.
_PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTHON_ROOT))

from pipeline.extract_paths_fitz import extract_paths, CSV_COLUMNS, EDGE_COLUMNS, write_csvs


DEFAULT_ROOTS = [
    Path("/Users/lukehogan/AOA-Code/faa-downloader/airports"),
    Path("/Users/lukehogan/AOA-Code/faa-downloader/airports-dup"),
]


def _walk_pdfs(root: Path) -> list[tuple[str, Path]]:
    """Return [(airport_code, pdf_path), ...] for <code>/<code>-faa.pdf under root."""
    out: list[tuple[str, Path]] = []
    if not root.is_dir():
        return out
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        airport = child.name
        pdf = child / f"{airport}-faa.pdf"
        if pdf.exists():
            out.append((airport, pdf))
    return out


def _process_one(args: tuple[str, str, str]) -> tuple[str, int, int, str | None]:
    """Worker: runs extract_paths and writes CSVs. Returns (airport, n_paths, n_edges, error)."""
    airport, pdf_path, out_dir = args
    try:
        rows, edges = extract_paths(Path(pdf_path), airport)
        write_csvs(Path(out_dir), airport, rows, edges)
        return (airport, len(rows), len(edges), None)
    except Exception as e:  # noqa: BLE001
        return (airport, 0, 0, f"{type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", type=Path,
                    default=Path(__file__).parent.parent / "data" / "char_corpus")
    ap.add_argument("--root", type=Path, action="append", default=None,
                    help="Repeat to add multiple roots; default = airports + airports-dup")
    ap.add_argument("--workers", type=int, default=None,
                    help="Worker count (default: cpu_count - 1)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true",
                    help="Re-extract even if CSVs already exist")
    args = ap.parse_args()

    roots = args.root if args.root else DEFAULT_ROOTS
    workers = args.workers or max(1, (os.cpu_count() or 4) - 1)

    # Collect work, dedup by airport code (later roots wouldn't overwrite,
    # but we also don't want to do the work twice).
    seen: set[str] = set()
    work: list[tuple[str, str, str]] = []
    skipped_existing = 0
    for root in roots:
        for airport, pdf in _walk_pdfs(root):
            if airport in seen:
                continue
            seen.add(airport)
            out_dir = args.corpus_dir / airport
            paths_csv = out_dir / f"{airport}_paths.csv"
            edges_csv = out_dir / f"{airport}_paths_edges.csv"
            if not args.force and paths_csv.exists() and edges_csv.exists():
                skipped_existing += 1
                continue
            work.append((airport, str(pdf), str(out_dir)))

    if args.limit is not None:
        work = work[: args.limit]

    print(f"[batch] airports queued: {len(work)}  "
          f"skipped (already extracted): {skipped_existing}  "
          f"workers: {workers}")
    if not work:
        print("[batch] nothing to do.")
        return

    args.corpus_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    failed: list[tuple[str, str]] = []
    processed = 0

    with mp.Pool(workers) as pool:
        for i, (airport, n_paths, n_edges, err) in enumerate(
            pool.imap_unordered(_process_one, work, chunksize=4), start=1
        ):
            if err:
                failed.append((airport, err))
                marker = "✗"
            else:
                marker = "✓"
                processed += 1
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0.0
            print(f"[{i:>4}/{len(work)}  {elapsed:>6.1f}s  {rate:>5.1f}/s]  "
                  f"{marker} {airport}  paths={n_paths}  edges={n_edges}"
                  f"{('  err=' + err) if err else ''}")

    elapsed = time.time() - t0
    print()
    print(f"[batch] done. processed={processed}/{len(work)}  "
          f"failed={len(failed)}  elapsed={elapsed:.1f}s")
    if failed:
        print("[batch] failures:")
        for a, e in failed[:20]:
            print(f"   {a}: {e}")
        if len(failed) > 20:
            print(f"   ... (+{len(failed) - 20} more)")


if __name__ == "__main__":
    main()
