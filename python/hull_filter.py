"""
hull_filter.py — concave-hull rejection (step 4 of the six-step plan).

Build a concave hull over the union of rule-claimed Runway and Taxiway
polygons' anchor points. Any candidate polygon whose centroid sits
outside that hull is demoted to Other. The caller is responsible for
excluding exempt classes (Runways, Taxiways, Runway Labels, Taxi
Labels) from the candidate set — the user's plan exempts them because
they're rule-claimed and labels can legitimately sit at chart edges.

No buffer is applied: the hull is the tightest valid concave wrap
around the airport's runway/taxi network. shapely.concave_hull with
ratio=0.0 matches the user's "no buffer" intent.

Public API:
    hull_reject(all_polys, anchor_indices, candidate_indices)
        -> tuple[set[int], dict]
"""

from __future__ import annotations

import shapely
from shapely.geometry import MultiPoint, Point


def _collect_anchor_points(
    all_polys: list[dict],
    anchor_indices,
) -> list[tuple[float, float]]:
    """Flatten every anchor point across the chosen polygons' subpaths.
    Closing-anchor duplicates are harmless — concave_hull dedupes
    coincident points internally."""
    pts: list[tuple[float, float]] = []
    for i in anchor_indices:
        for ring in all_polys[i].get("subpaths") or []:
            pts.extend(ring)
    return pts


def hull_reject(
    all_polys: list[dict],
    anchor_indices,
    candidate_indices,
) -> tuple[set[int], dict]:
    """Return indices to demote to Other plus a diagnostics dict.

    Args:
        all_polys: full polygon list from chart_scene.read_chart.
        anchor_indices: indices that define the airport surface
            (rule-claimed Runways + Taxiways).
        candidate_indices: indices to test. The caller filters out
            exempt classes (Runways, Taxiways, Runway Labels, Taxi
            Labels) before passing this in.

    Returns:
        (demote_indices, diag). demote_indices is a set of indices
        whose centroid sits outside the hull. diag carries hull stats
        for the pipeline's print-out.
    """
    pts = _collect_anchor_points(all_polys, anchor_indices)
    diag: dict = {
        "hull_built": False,
        "n_anchor_points": len(pts),
        "n_candidates_tested": 0,
        "n_demoted": 0,
        "hull_area": 0.0,
    }
    # concave_hull needs at least 3 non-collinear points to enclose any
    # area. Tiny airports with no rule-claimed runway/taxi (rare) skip
    # the rejection entirely.
    if len(pts) < 3:
        return set(), diag

    hull = shapely.concave_hull(MultiPoint(pts), ratio=0.0)
    if hull.is_empty or hull.area <= 0:
        return set(), diag

    candidates = list(candidate_indices)
    diag["hull_built"] = True
    diag["hull_area"] = float(hull.area)
    diag["n_candidates_tested"] = len(candidates)

    demote: set[int] = set()
    for i in candidates:
        p = all_polys[i]
        if not hull.covers(Point(p["cx"], p["cy"])):
            demote.add(i)
    diag["n_demoted"] = len(demote)
    return demote, diag


# --- CLI -----------------------------------------------------------------

def main():
    """Standalone diagnostic. Reads a chart, runs taxi + runway
    detection, builds the hull, and reports how many polygons would be
    rejected."""
    import argparse
    from pathlib import Path

    from chart_scene import read_chart
    from runway_detection import detect_runways
    from taxi_detection import detect_taxi

    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=Path, required=True)
    ap.add_argument("--nasr-rwy", type=Path,
                    default=Path(__file__).parent.parent / "data" / "nasr_apt_rwy.csv")
    args = ap.parse_args()

    det = detect_taxi(args.pdf)
    all_polys = det["all_polys"]
    surf_set = set(det["taxi_surface_indices"])
    label_set = set(det["taxi_label_indices"])
    airport = args.pdf.stem.replace("-faa", "").lower()

    rwy_set = detect_runways(
        all_polys, det["clips"], airport, args.nasr_rwy,
        page_w=det["page_w"], page_h=det["page_h"],
        claimed_indices=surf_set | label_set,
    )
    print(f"airport: {airport}")
    print(f"  taxiways: {len(surf_set)}, taxi labels: {len(label_set)}, "
          f"runways: {len(rwy_set)}")

    claimed = surf_set | label_set | rwy_set
    candidates = [i for i in range(len(all_polys)) if i not in claimed]
    demote, diag = hull_reject(all_polys, surf_set | rwy_set, candidates)
    print(f"  hull anchor pts: {diag['n_anchor_points']}, "
          f"area: {diag['hull_area']:.0f}")
    print(f"  unclaimed candidates: {diag['n_candidates_tested']}, "
          f"would demote: {diag['n_demoted']}")


if __name__ == "__main__":
    main()
