"""
runway_detection.py — deterministic runway detection (step 2 of the new pipeline).

Rule:
  N = number of runways NASR reports for this airport (helipads excluded).
  Runways = the N polygons with the largest polygon area, drawn from
            both regular paths and nested clip-group rects, after
            step-1 taxi surfaces have been removed from the pool.
            Candidates must be filled near-black or stroked. Each
            pick's PCA-derived aspect ratio must be >= a fraction of
            the airport's smallest NASR runway aspect — a sanity check
            against label boxes or other rectangles whose polygon area
            could rival a small runway.

Public API:
    count_nasr_runways(nasr_csv, airport_code) -> int
    detect_runways(all_polys, clips, airport_code, nasr_csv,
                   page_w, page_h, claimed_indices=None)
        -> set[int]    # indices into all_polys
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

# A chart's outer artwork frame is a stroked rectangle that covers
# ~75% of the page area on standard FAA diagrams. Applied to bbox area.
MAX_BBOX_FRACTION_OF_PAGE = 0.50

# Required aspect ratio (PCA principal/secondary) is a fraction of the
# airport's smallest NASR runway aspect. 0.2 = candidate must be at
# least 20% as elongated as the most square-ish real runway. This
# tolerates chart depictions where a grass strip's stroked path swings
# wider than the strict rectangle (e.g. F45 clip group ~10:1 vs NASR
# ~50:1) while still rejecting label boxes (~5:1).
ASPECT_FRACTION_OF_NASR_MIN = 0.2

# Floor on the aspect threshold. Used when NASR is missing or yields
# no usable runways. 4:1 still rejects square symbols and most label
# boxes without hard-coding a number that might exclude legitimate
# stubby runways.
ASPECT_FLOOR = 4.0

_HELIPAD_RE = re.compile(r"^H\d", re.IGNORECASE)


# --- NASR helpers --------------------------------------------------------

def _load_nasr_runways(nasr_csv: Path, airport_code: str) -> list[tuple[float, float]]:
    """Return list of (length_ft, width_ft) for each non-helipad runway
    NASR has for this airport. Match on ARPT_ID is case-insensitive."""
    target = airport_code.strip().upper()
    out: list[tuple[float, float]] = []
    with open(nasr_csv, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if (row.get("ARPT_ID") or "").strip().upper() != target:
                continue
            rwy_id = (row.get("RWY_ID") or "").strip()
            if not rwy_id or _HELIPAD_RE.match(rwy_id):
                continue
            try:
                L = float((row.get("RWY_LEN") or "0").strip() or 0)
                W = float((row.get("RWY_WIDTH") or "0").strip() or 0)
            except ValueError:
                continue
            if L <= 0 or W <= 0:
                continue
            out.append((L, W))
    return out


def count_nasr_runways(nasr_csv: Path, airport_code: str) -> int:
    """Count physical runways for an airport, excluding helipads."""
    return len(_load_nasr_runways(nasr_csv, airport_code))


def _min_nasr_aspect(nasr_runways: list[tuple[float, float]]) -> float:
    """Smallest length/width aspect across the airport's runways.
    Used as the basis for the candidate-aspect sanity check."""
    if not nasr_runways:
        return 0.0
    return min(max(L, W) / max(min(L, W), 1.0) for L, W in nasr_runways)


# --- Geometry helpers ----------------------------------------------------

def _polygon_area(subpaths) -> float:
    """Sum of absolute shoelace areas across subpaths. For clean
    rectangles equals length*width regardless of rotation."""
    total = 0.0
    for ring in subpaths or []:
        n = len(ring)
        if n < 3:
            continue
        s = 0.0
        for i in range(n):
            xi, yi = ring[i]
            xj, yj = ring[(i + 1) % n]
            s += xi * yj - xj * yi
        total += abs(s) / 2.0
    return total


def _pca_aspect(subpaths, fallback_rect) -> float:
    """PCA-based length/width aspect ratio. Robust to rotation —
    returns the polygon's true elongation regardless of how its bbox
    looks. Falls back to bbox aspect if the polygon has fewer than 3
    distinct anchor points."""
    pts: list[tuple[float, float]] = []
    for ring in subpaths or []:
        pts.extend(ring)
    x0, y0, x1, y1 = fallback_rect
    if len(pts) < 3:
        w, h = x1 - x0, y1 - y0
        return max(w, h) / max(min(w, h), 1e-9)
    arr = np.asarray(pts, dtype=float)
    centered = arr - arr.mean(axis=0)
    _, sv, _ = np.linalg.svd(centered, full_matrices=False)
    if len(sv) < 2 or sv[1] < 1e-9:
        # Degenerate (collinear) — treat as infinitely elongated.
        return float("inf")
    # SVD singular values relate to per-axis spread; their ratio is the
    # PCA aspect. Use 4 * singular_value as the full-extent proxy
    # (roughly principal length / secondary length).
    return float(sv[0] / sv[1])


def _scissor_subpaths(rect) -> list[list[tuple[float, float]]]:
    """Build a clean 4-anchor rectangular subpath from a scissor rect.
    Used for clip polygons so their PCA / shoelace measurements
    reflect the rectangle, not the underlying clip-path geometry
    (which may be a zigzag tracing the contents)."""
    x0, y0, x1, y1 = rect
    return [[(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]]


# --- Public entry point --------------------------------------------------

def detect_runways(all_polys: list[dict],
                   clips: list[dict],
                   airport_code: str,
                   nasr_csv: Path,
                   page_w: float,
                   page_h: float,
                   claimed_indices: set[int] | list[int] | None = None) -> set[int]:
    """Identify the indices in `all_polys` that are runways. Returns
    a set of indices.

    Clip rectangles can win the ranking (e.g. F45's grass-strip outline
    lives only inside a nested clip group). When a clip wins, every
    polygon in `all_polys` whose centroid sits inside the clip's
    scissor is claimed as part of that runway.
    """
    nasr_rwys = _load_nasr_runways(nasr_csv, airport_code)
    n = len(nasr_rwys)
    if n == 0:
        return set()

    claimed = set(claimed_indices or [])
    page_area = max(page_w * page_h, 1.0)
    max_bbox = page_area * MAX_BBOX_FRACTION_OF_PAGE
    aspect_threshold = max(
        ASPECT_FLOOR,
        _min_nasr_aspect(nasr_rwys) * ASPECT_FRACTION_OF_NASR_MIN,
    )

    candidates: list[dict] = []

    # Regular polygon candidates.
    for i, p in enumerate(all_polys):
        if i in claimed:
            continue
        is_paved = p.get("filled") and p.get("is_near_black")
        is_grass = p.get("stroked") and not p.get("filled")
        if not (is_paved or is_grass):
            continue
        x0, y0, x1, y1 = p["rect"]
        bbox_area = (x1 - x0) * (y1 - y0)
        if bbox_area <= 0 or bbox_area >= max_bbox:
            continue
        poly_area = _polygon_area(p.get("subpaths"))
        if poly_area <= 0:
            continue
        candidates.append({
            "kind": "poly",
            "all_polys_index": i,
            "rect": p["rect"],
            "subpaths": p.get("subpaths"),
            "score": poly_area,
        })

    # Clip-group candidates. The scissor rect is the polygon. Score by
    # its rectangular area (which is bbox area, since the scissor is
    # rectangular).
    for c in clips:
        x0, y0, x1, y1 = c["rect"]
        bbox_area = (x1 - x0) * (y1 - y0)
        if bbox_area <= 0 or bbox_area >= max_bbox:
            continue
        candidates.append({
            "kind": "clip",
            "all_polys_index": None,
            "rect": c["rect"],
            "subpaths": _scissor_subpaths(c["rect"]),
            "score": bbox_area,
        })

    # Sort by area descending, then take the first N that pass the
    # aspect-ratio sanity check.
    candidates.sort(key=lambda c: -c["score"])
    selected: list[dict] = []
    for cand in candidates:
        aspect = _pca_aspect(cand["subpaths"], cand["rect"])
        if aspect < aspect_threshold:
            continue
        selected.append(cand)
        if len(selected) == n:
            break

    # Translate selections to all_polys indices. For poly picks, claim
    # the polygon directly. For clip picks, find the single largest
    # polygon whose bbox is contained inside the clip's scissor and
    # claim that one — it's the simple-stroked-rectangle outline (or
    # the runway-shape pattern polygon, for charts like F45) that the
    # clip group wraps. Don't claim the dozens of hatching marks /
    # glyph polygons that also happen to live inside the scissor;
    # those aren't runways.
    runway_indices: set[int] = set()
    for cand in selected:
        if cand["kind"] == "poly":
            runway_indices.add(cand["all_polys_index"])
            continue
        cx0, cy0, cx1, cy1 = cand["rect"]
        # Small epsilon to absorb the tiny float drift between a path's
        # rect and the scissor of the clip that wraps it.
        eps = 0.5
        best_idx = None
        best_area = 0.0
        for i, p in enumerate(all_polys):
            if i in claimed or i in runway_indices:
                continue
            px0, py0, px1, py1 = p["rect"]
            if (px0 + eps < cx0 or px1 - eps > cx1
                    or py0 + eps < cy0 or py1 - eps > cy1):
                continue
            area = (px1 - px0) * (py1 - py0)
            if area > best_area:
                best_area = area
                best_idx = i
        if best_idx is not None:
            runway_indices.add(best_idx)

    return runway_indices


# --- CLI -----------------------------------------------------------------

def main():
    """Standalone diagnostic. Print NASR count, candidate ranking, and
    selected runway indices for one airport PDF."""
    import argparse
    from chart_scene import read_chart

    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=Path, required=True)
    ap.add_argument("--nasr-rwy", type=Path,
                    default=Path(__file__).parent.parent / "data" / "nasr_apt_rwy.csv")
    ap.add_argument("--airport", type=str, default=None)
    args = ap.parse_args()

    airport = args.airport or args.pdf.stem.replace("-faa", "").lower()
    nasr_rwys = _load_nasr_runways(args.nasr_rwy, airport)
    n = len(nasr_rwys)
    min_aspect_nasr = _min_nasr_aspect(nasr_rwys)
    threshold = max(ASPECT_FLOOR, min_aspect_nasr * ASPECT_FRACTION_OF_NASR_MIN)
    print(f"airport: {airport}")
    print(f"NASR runways (non-helipad): {n}")
    print(f"NASR runway dims: {nasr_rwys}")
    print(f"min NASR aspect: {min_aspect_nasr:.1f}, "
          f"aspect threshold: {threshold:.1f} "
          f"(= max({ASPECT_FLOOR}, {ASPECT_FRACTION_OF_NASR_MIN} * {min_aspect_nasr:.1f}))")

    scene = read_chart(args.pdf)
    all_polys = scene["all_polys"]
    clips = scene["clips"]
    print(f"polygons: {len(all_polys)}, clips: {len(clips)}")

    # Run detection without claiming anything (so taxi surfaces aren't
    # excluded). Useful for diagnostic; production calls pass claimed.
    rwy = detect_runways(all_polys, clips, airport, args.nasr_rwy,
                         page_w=scene["page_w"], page_h=scene["page_h"])
    print(f"\nselected runways: {len(rwy)} indices ({sorted(rwy)[:20]}{'...' if len(rwy)>20 else ''})")


if __name__ == "__main__":
    main()
