"""
build_char_training.py — Phase 2 of the character-recognition build plan.

Builds distant-supervision training data for the 36-class character classifier
(A-Z, 0-9). For each airport in data/char_corpus/, aligns PDF text tokens to
the polygons emitted by ClassifyAirport.jsx (export-only mode) and emits
(polygon_features, character) rows.

Distant-supervision premise: FAA airport diagrams use a single consistent
font, so outlined letter polygons are geometrically identical across
airports up to scale + rotation. For any character C, summing across
hundreds of airports gives many polygons matched to C-tokens — the actual
C-shaped polygons dominate the noise, and a classifier trained on them
generalizes to unseen airports.

Algorithm per airport:
  1. Load <airport>_paths.csv + <airport>_paths_edges.csv from corpus dir.
  2. Open the source -faa.pdf (search airports/, then airports-dup/).
  3. For each PDF text token T whose chars are all A-Z0-9:
       - Slice the token bbox into len(T) equal horizontal slices.
       - Build a candidate set: polygons whose centroid lies inside
         (token bbox + margin). Margin handles the 15-20pt offset
         between PDF text and Illustrator vector space mentioned in
         the plan.
       - For each char position i, pick the nearest unused candidate
         to the slice center and emit (polygon_features, T[i]).

Polygon features (all rotation+scale invariant — the same letter rendered
at any scale/orientation must produce the same vector):
  num_anchors, subpath_count, closed
  principal_ratio
  perimeter_ratio       perimeter / sqrt(bbox_area)
  seg_ratio_0..K-1      top-K segment lengths sorted desc, normalized
                        by the longest segment (so seg_ratio_0 == 1.0)

Output: one parquet at --out with one row per (airport, polygon, char).

Usage:
  python build_char_training.py \\
    --corpus-dir /Users/lukehogan/AOA-Code/faa-pro/data/char_corpus \\
    --out /Users/lukehogan/AOA-Code/faa-pro/data/char_training.parquet

Optional flags:
  --airports abi,atl,bna   process only these airports (comma-separated)
  --limit 20               process only the first N airports (debug/iteration)
  --margin 15              token-bbox margin in PDF units (default: 15)
  --top-k-segments 16      number of segment-length ratios per polygon
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
# ConvexHull is no longer needed here — extract_paths_fitz.py emits
# hull_area directly into the paths CSV (see CSV_COLUMNS).

# Hardcoded search roots for the source PDFs. Distant supervision needs the
# original PDF text, which is not preserved in the path-export CSVs.
PDF_SEARCH_ROOTS = [
    Path("/Users/lukehogan/AOA-Code/faa-downloader/airports"),
    Path("/Users/lukehogan/AOA-Code/faa-downloader/airports-dup"),
]

# A "training token" is one whose every character is A-Z or 0-9. Keeps the
# label space at 36 classes and avoids ambiguity from punctuation/spacing.
_TOKEN_RE = re.compile(r"^[A-Z0-9]+$")

# Sentinel label for the negative class — polygons that are NOT inside any
# PDF text token's bbox. Including this class teaches the model to reject
# non-letter shapes (footprints, lights, chevrons, runway markings)
# natively instead of relying on a confidence-threshold hack at inference.
NEGATIVE_CLASS = "_NOT_"


def find_source_pdf(airport: str) -> Path | None:
    """Return path to <airport>-faa.pdf under any of the search roots."""
    for root in PDF_SEARCH_ROOTS:
        p = root / airport / f"{airport}-faa.pdf"
        if p.exists():
            return p
    return None


def load_pdf_tokens(pdf_path: Path) -> list[dict]:
    """Return PDF text tokens with bboxes in Illustrator (y-up) coords.

    Mirrors the y-flip logic in python/extract_pdf_text.py so token bboxes
    can be compared directly to polygon centroids from the JSX exporter.
    """
    doc = fitz.open(pdf_path)
    page = doc[0]
    page_h = float(page.rect.height)
    words = page.get_text("words")
    out: list[dict] = []
    for word in words:
        x0, y0, x1, y1, text, *_ = word
        if not text:
            continue
        text = text.strip()
        if not _TOKEN_RE.match(text):
            continue
        out.append({
            "text": text,
            "x_min": float(x0),
            "x_max": float(x1),
            "y_min": page_h - float(y1),
            "y_max": page_h - float(y0),
        })
    doc.close()
    return out


def polygon_invariant_features(
    row: pd.Series,
    edges_for_obj: pd.DataFrame | None,
    top_k: int,
) -> dict:
    """Compute rotation+scale-invariant features for one polygon row.

    Reads precomputed poly_area + hull_area from the row (extract_paths_fitz
    emits both) and derives the standard shape ratios from them.
    """
    bbox_area = float(row["bbox_area"])
    perim = float(row["perimeter"])
    poly_area = float(row.get("poly_area", -1) or -1)
    poly_area = max(poly_area, 0.0)
    hull_area = float(row.get("hull_area", 0) or 0)

    width = float(row.get("width", 0) or 0)
    height = float(row.get("height", 0) or 0)
    artboard_w = float(row.get("artboard_right", 0) or 0) - float(row.get("artboard_left", 0) or 0)
    artboard_h = float(row.get("artboard_top", 0) or 0) - float(row.get("artboard_bottom", 0) or 0)
    artboard_diag = math.hypot(max(artboard_w, 0), max(artboard_h, 0))

    perim_ratio = perim / max(np.sqrt(max(bbox_area, 0.0)), 1e-6)

    # Extent / fill ratio: poly_area / bbox_area. Letters have characteristic
    # distributions per glyph ("I" is ~0.7, "O" outline is ~0.3).
    fill_ratio = poly_area / max(bbox_area, 1e-6) if bbox_area > 0 else 0.0
    fill_ratio = min(1.0, max(0.0, fill_ratio))

    # Compactness (Polsby-Popper score): 4π·area / perimeter². 1.0 for a
    # perfect circle, smaller for elongated or irregular shapes.
    compactness = (
        4 * math.pi * poly_area / (perim * perim)
        if perim > 1e-6 and poly_area > 0 else 0.0
    )
    compactness = min(1.0, max(0.0, compactness))

    # Solidity = poly_area / convex_hull_area. Letters with concavities
    # (E, F, U) have low solidity; round letters (O, Q) high.
    solidity = poly_area / hull_area if hull_area > 1e-6 else 0.0
    solidity = min(1.0, max(0.0, solidity))

    # Filled / stroked: massively important for distinguishing letter glyphs
    # (which are *filled outlines* in PDFs) from chevrons, runway markings,
    # and other line work (which are *stroked* paths). Without this, a
    # stroked-only chevron's geometric shoelace can give a high fill_ratio
    # value that looks letter-like — a long-standing source of confusion.
    filled = int(row.get("filled", 0) or 0)
    stroked = int(row.get("stroked", 0) or 0)
    stroke_w = float(row.get("stroke_width", 0) or 0)
    max_dim = max(width, height, 1e-6)
    stroke_width_rel = stroke_w / max_dim

    # NOTE: size relative to the artboard was tried as a feature but
    # rejected — it caused the binary detector to systematically miss
    # large display-text letters (e.g. "AIRPORT DIAGRAM" header) because
    # the training distribution skewed toward small taxiway labels. Stick
    # to scale-invariant features so any letter at any scale matches.

    feats: dict = {
        "num_anchors": int(row["num_anchors"]),
        "subpath_count": int(row["subpath_count"]),
        "closed": int(row["closed"]),
        "principal_ratio": float(row["principal_ratio"]),
        "perimeter_ratio": perim_ratio,
        "fill_ratio": fill_ratio,
        "compactness": compactness,
        "solidity": solidity,
        "filled": filled,
        "stroked": stroked,
        "stroke_width_rel": stroke_width_rel,
    }

    # Top-K segment lengths, sorted descending and normalized by the
    # longest segment. seg_ratio_0 is always 1.0 (or 0.0 if no edges).
    if edges_for_obj is not None and len(edges_for_obj) > 0:
        lens = np.sort(edges_for_obj["length"].to_numpy(dtype=float))[::-1]
        denom = lens[0] if lens[0] > 0 else 1.0
        normalized = lens / denom
    else:
        normalized = np.zeros(0, dtype=float)
    for i in range(top_k):
        feats[f"seg_ratio_{i}"] = float(normalized[i]) if i < len(normalized) else 0.0
    return feats


def process_airport(
    airport: str,
    paths_csv: Path,
    edges_csv: Path,
    margin: float,
    top_k: int,
    neg_ratio: float = 1.0,
    neg_seed: int = 42,
) -> tuple[list[dict], dict]:
    """Return (training rows, debug stats) for one airport.

    Positives: polygons greedily 1-1 matched to characters in PDF text
    tokens (existing logic).

    Negatives: polygons whose centroid does NOT fall inside any token's
    bbox + margin — sampled at neg_ratio * (#positives) per airport,
    labeled with NEGATIVE_CLASS. These teach the classifier that most
    polygons (footprints, lights, taxiway pavement, chevrons) are not
    glyphs at all, eliminating the prior failure mode where every
    polygon got forced into a 36-class softmax.
    """
    pdf_path = find_source_pdf(airport)
    if pdf_path is None:
        return [], {"airport": airport, "skipped": "pdf-not-found"}

    paths_df = pd.read_csv(paths_csv)
    edges_df = pd.read_csv(edges_csv) if edges_csv.exists() else pd.DataFrame()

    if len(edges_df) > 0:
        edges_by_obj: dict[int, pd.DataFrame] = {
            int(oid): grp for oid, grp in edges_df.groupby("object_id")
        }
    else:
        edges_by_obj = {}

    centroids = paths_df[["centroid_x", "centroid_y"]].to_numpy(dtype=float)

    tokens = load_pdf_tokens(pdf_path)

    rows: list[dict] = []
    matched_chars = 0
    skipped_no_candidates = 0
    skipped_all_used = 0
    # Indices of polygons used as positives — we exclude them from
    # negative sampling so the same polygon never ends up as both a
    # letter example and a non-letter example.
    positive_poly_indices: set[int] = set()
    # Indices of polygons that fell inside ANY token's expanded bbox
    # (whether or not they were greedily chosen as the matched polygon
    # for some character). Excluded from negatives because we have low
    # confidence they aren't glyphs.
    near_token_indices: set[int] = set()

    for tok in tokens:
        text = tok["text"]
        n = len(text)
        char_w = (tok["x_max"] - tok["x_min"]) / n
        slice_cy = (tok["y_min"] + tok["y_max"]) / 2

        bbox_mask = (
            (centroids[:, 0] >= tok["x_min"] - margin)
            & (centroids[:, 0] <= tok["x_max"] + margin)
            & (centroids[:, 1] >= tok["y_min"] - margin)
            & (centroids[:, 1] <= tok["y_max"] + margin)
        )
        cand_idx = np.where(bbox_mask)[0]
        if len(cand_idx) == 0:
            skipped_no_candidates += n
            continue
        near_token_indices.update(int(i) for i in cand_idx)

        cand_cents = centroids[cand_idx]
        used_local: set[int] = set()

        for i, ch in enumerate(text):
            slice_cx = tok["x_min"] + (i + 0.5) * char_w
            d2 = (cand_cents[:, 0] - slice_cx) ** 2 + (cand_cents[:, 1] - slice_cy) ** 2
            order = np.argsort(d2)

            chosen_local = None
            for j in order:
                if int(j) not in used_local:
                    chosen_local = int(j)
                    used_local.add(chosen_local)
                    break
            if chosen_local is None:
                skipped_all_used += 1
                continue

            poly_idx = int(cand_idx[chosen_local])
            positive_poly_indices.add(poly_idx)
            poly_row = paths_df.iloc[poly_idx]
            obj_id = int(poly_row["object_id"])
            edges_for_obj = edges_by_obj.get(obj_id)

            feats = polygon_invariant_features(poly_row, edges_for_obj, top_k)
            feats.update({
                "airport": airport,
                "object_id": obj_id,
                "token": text,
                "char_pos": i,
                "char": ch,
            })
            rows.append(feats)
            matched_chars += 1

    # ---------------- Negative sampling (size-stratified) ----------------
    # Pool: polygons NOT near any text token. We exclude near_token_indices
    # so polygons that share a token's bbox (e.g. the second char in a
    # multi-char token) don't get labeled as non-glyphs.
    #
    # Stratification: random sampling pulls heavily from big polygons
    # (footprints, taxiway pavement) since they outnumber small ones in
    # most charts. The classifier then learns "big = not letter," which
    # systematically rejects large display-text letters like "AIRPORT
    # DIAGRAM" headers. Fix: for each positive's bbox_area, sample a
    # negative whose bbox_area is within the same log-scale band, so the
    # final negatives have ~the same size distribution as positives. The
    # model cannot use size as a discriminator and must learn from shape.
    n_polys = len(paths_df)
    pool_idx = np.array(
        [i for i in range(n_polys) if i not in near_token_indices], dtype=int
    )
    n_negs_target = int(round(matched_chars * neg_ratio))
    n_negs_actual = 0
    if n_negs_target > 0 and len(pool_idx) > 0 and len(positive_poly_indices) > 0:
        rng = np.random.default_rng(seed=hash((airport, neg_seed)) & 0xFFFFFFFF)

        pos_sizes = paths_df.iloc[list(positive_poly_indices)]["bbox_area"].to_numpy(dtype=float)
        pool_sizes = paths_df.iloc[pool_idx]["bbox_area"].to_numpy(dtype=float)

        # Log-scale matching — letter sizes span a wide range (header A
        # could be 100x larger than a taxiway-label A) and absolute-diff
        # matching at small sizes would be too forgiving.
        eps = 1e-6
        pool_log = np.log(np.maximum(pool_sizes, eps))
        pos_log = np.log(np.maximum(pos_sizes, eps))

        # For each positive, draw one negative from the band [logsize-W, logsize+W]
        # without replacement. Width ~0.5 in log space ≈ a factor of 1.6
        # on either side, comfortable enough that most positives find a match.
        BAND = 0.5
        used: set[int] = set()
        sampled: list[int] = []
        # Sort positives by size — match the most-extreme sizes first so
        # they don't lose their candidates to mid-size positives.
        order = np.argsort(pos_log)
        n_target = min(n_negs_target, len(pool_idx))
        for k in order:
            if len(sampled) >= n_target:
                break
            target = pos_log[k]
            mask = (pool_log >= target - BAND) & (pool_log <= target + BAND)
            cand = np.where(mask)[0]
            cand = np.array([j for j in cand if int(j) not in used], dtype=int)
            if len(cand) == 0:
                # Nothing in band — pick the nearest unused log-size
                remaining = np.array([j for j in range(len(pool_log)) if int(j) not in used], dtype=int)
                if len(remaining) == 0:
                    break
                j = int(remaining[np.argmin(np.abs(pool_log[remaining] - target))])
            else:
                j = int(rng.choice(cand))
            used.add(j)
            sampled.append(int(pool_idx[j]))

        for poly_idx in sampled:
            poly_row = paths_df.iloc[int(poly_idx)]
            obj_id = int(poly_row["object_id"])
            edges_for_obj = edges_by_obj.get(obj_id)
            feats = polygon_invariant_features(poly_row, edges_for_obj, top_k)
            feats.update({
                "airport": airport,
                "object_id": obj_id,
                "token": "",
                "char_pos": -1,
                "char": NEGATIVE_CLASS,
            })
            rows.append(feats)
        n_negs_actual = len(sampled)

    stats = {
        "airport": airport,
        "tokens": len(tokens),
        "matched_chars": matched_chars,
        "skipped_no_candidates": skipped_no_candidates,
        "skipped_all_used": skipped_all_used,
        "negatives_sampled": n_negs_actual,
        "negative_pool_size": len(pool_idx),
    }
    return rows, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "char_corpus",
        help="Folder produced by export_paths.sh — contains <airport>/<airport>_paths.csv",
    )
    ap.add_argument("--out", type=Path, required=True, help="Output parquet path")
    ap.add_argument(
        "--airports",
        type=str,
        default=None,
        help="Comma-separated airport codes; default = all subfolders of --corpus-dir",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N airports (after sort) — for iteration",
    )
    ap.add_argument(
        "--margin",
        type=float,
        default=15.0,
        help="Token-bbox margin in PDF units. Should exceed the PDF-vs-AI "
             "coordinate offset (~15-20pt per the plan).",
    )
    ap.add_argument(
        "--top-k-segments",
        type=int,
        default=16,
        help="Number of largest segment-length ratios kept per polygon",
    )
    ap.add_argument(
        "--neg-ratio",
        type=float,
        default=1.0,
        help="Number of negative-class samples per airport, as a fraction of "
             "matched-character count (1.0 = balanced; 0 disables negatives).",
    )
    args = ap.parse_args()

    if not args.corpus_dir.is_dir():
        raise SystemExit(f"corpus dir not found: {args.corpus_dir}")

    if args.airports:
        airports = [a.strip().lower() for a in args.airports.split(",") if a.strip()]
    else:
        airports = sorted(
            d.name for d in args.corpus_dir.iterdir() if d.is_dir()
        )
    if args.limit is not None:
        airports = airports[: args.limit]

    print(f"[build_char_training] processing {len(airports)} airports from {args.corpus_dir}")

    all_rows: list[dict] = []
    all_stats: list[dict] = []
    n_pdf_missing = 0

    for airport in airports:
        airport_dir = args.corpus_dir / airport
        paths_csv = airport_dir / f"{airport}_paths.csv"
        edges_csv = airport_dir / f"{airport}_paths_edges.csv"
        if not paths_csv.exists():
            print(f"  {airport}: skip (no paths.csv)")
            continue

        rows, stats = process_airport(
            airport, paths_csv, edges_csv, args.margin, args.top_k_segments,
            neg_ratio=args.neg_ratio,
        )
        if "skipped" in stats:
            n_pdf_missing += 1
            print(f"  {airport}: skip ({stats['skipped']})")
            continue

        all_rows.extend(rows)
        all_stats.append(stats)
        print(
            f"  {airport}: tokens={stats['tokens']:>4}  "
            f"matched={stats['matched_chars']:>5}  "
            f"neg={stats.get('negatives_sampled', 0):>4}  "
            f"miss_nocand={stats['skipped_no_candidates']:>4}  "
            f"miss_used={stats['skipped_all_used']:>3}"
        )

    if not all_rows:
        raise SystemExit("No training rows produced — check corpus dir and PDF paths.")

    df = pd.DataFrame(all_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    total_chars = len(df)
    per_class = df["char"].value_counts().sort_index()
    print("")
    print(f"[build_char_training] wrote {total_chars:,} rows -> {args.out}")
    print(f"[build_char_training] PDFs missing for {n_pdf_missing} airports")
    print(f"[build_char_training] class coverage ({len(per_class)}/36 chars seen):")
    for ch in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        n = int(per_class.get(ch, 0))
        bar = "#" * min(40, n // max(1, total_chars // 400))
        print(f"   {ch}  {n:>6,}  {bar}")


if __name__ == "__main__":
    main()
