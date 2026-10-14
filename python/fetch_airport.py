#!/usr/bin/env python3
"""Fetch the latest FAA data for one airport LID and lay it out for classify.sh.

Downloads the airport-diagram PDF, best approach chart, and this airport's
NASR runway rows into a de-duplicated folder under a destination parent,
and applies a macOS Finder color label by airspace class. Prints two
machine-readable lines the bash wrapper parses: FOLDER=<path>, PDF=<path>.

Depends on pipeline.faa_fetch. Network calls go through that module so
tests can stub them.

Usage:
    python fetch_airport.py --lid ORD --dest-parent /path/to/artwork/airports
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline import faa_fetch


def run(lid: str, dest_parent: Path, priority: list[str] | None = None,
        *, session=None) -> tuple[Path, Path]:
    """Download FAA data for one LID. Returns (folder, apd_pdf_path).

    :raises LookupError: when the LID has no published FAA airport diagram.
    """
    priority = priority or faa_fetch.APPROACH_PRIORITY
    lid_upper = lid.strip().upper()
    lid_lower = lid_upper.lower()

    cycle, date_str = faa_fetch.get_current_cycle()
    session = session or faa_fetch.make_session()

    root = faa_fetch.fetch_metafile(cycle, session)
    apd_map, iap_map = faa_fetch.parse_tpp_records(root)
    apd_pdf = apd_map.get(lid_upper)
    if not apd_pdf:
        raise LookupError(f"no FAA airport diagram for {lid_upper}")

    airports, runway_rows = faa_fetch.download_nasr(date_str, session, lid_upper)
    airspace = faa_fetch.determine_airspace_class(lid_upper, airports.get(lid_upper, {}))

    folder = faa_fetch.dedup_folder(dest_parent, lid_lower)
    folder.mkdir(parents=True)
    faa_fetch.set_finder_label(folder, airspace, faa_fetch.AIRSPACE_LABEL_COLORS)

    apd_dest = folder / f"{lid_lower}-faa.pdf"
    faa_fetch.download_pdf(
        faa_fetch.BASE_PDF_URL.format(cycle=cycle, pdf_name=apd_pdf),
        apd_dest, session, 120)

    iap = faa_fetch.select_best_approach(iap_map.get(lid_upper, []), priority)
    if iap:
        try:
            faa_fetch.download_pdf(
                faa_fetch.BASE_PDF_URL.format(cycle=cycle, pdf_name=iap[1]),
                folder / f"{lid_lower}-approach.pdf", session, 120)
        except Exception as exc:  # approach is optional; the diagram is required
            print(f"WARN: approach download failed for {lid_upper}: {exc}", file=sys.stderr)

    faa_fetch.write_runway_csv(runway_rows, folder / f"{lid_lower}-nasr-rwy.csv")

    print(f"FOLDER={folder}")
    print(f"PDF={apd_dest}")
    return folder, apd_dest


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on success, 2 when the LID has no FAA diagram."""
    ap = argparse.ArgumentParser(description="Fetch latest FAA data for one airport LID.")
    ap.add_argument("--lid", required=True, help="FAA location identifier, e.g. ORD")
    ap.add_argument("--dest-parent", required=True, type=Path,
                    help="Parent folder; the airport folder is created (deduped) inside it.")
    ap.add_argument("--priority", default="",
                    help="Comma-separated approach priority override.")
    args = ap.parse_args(argv)

    priority = [p.strip() for p in args.priority.split(",") if p.strip()] or None
    try:
        run(args.lid, args.dest_parent, priority)
    except LookupError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
