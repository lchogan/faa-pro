"""
apply_taxi_overrides.py — post-ML override step.

Reads the predictions JSON from predict_one.py and the source PDF,
runs rule-based Taxiways + Taxiway-Labels detection (taxi_detection.py),
and overrides the ML label for any matching polygon. Step 1's
detection is authoritative for these two classes.

Polygons are matched by bbox identity. Both the ML pipeline
(extract_paths_fitz.py) and our detection (taxi_detection.py) extract
polygons from the same PDF via PyMuPDF, so the bboxes line up to PDF
unit precision modulo the y-flip:
    AI-frame top    = page_h - PDF y0
    AI-frame bottom = page_h - PDF y1

Usage:
    python apply_taxi_overrides.py \\
        --pdf /path/to/<airport>-faa.pdf \\
        --predictions /path/to/<airport>_predictions.json \\
        --out /path/to/<airport>_predictions.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from taxi_detection import detect_taxi


# Match tolerance — both extractions go through PyMuPDF on the same PDF
# so bboxes should be identical, but allow tiny float drift.
BBOX_TOLERANCE_PT = 0.5


def _build_bbox_index(predictions: list[dict],
                      page_h: float) -> dict[tuple, int]:
    """Index ML predictions by their AI-frame bbox (rounded for hash
    stability) so we can look up quickly. Multiple predictions could
    in theory share a bbox; first wins."""
    idx: dict[tuple, int] = {}
    for i, rec in enumerate(predictions):
        key = (
            round(float(rec["left"]),   1),
            round(float(rec["bottom"]), 1),
            round(float(rec["right"]),  1),
            round(float(rec["top"]),    1),
        )
        idx.setdefault(key, i)
    return idx


def _find_match(rec_idx: dict[tuple, int],
                ai_left: float, ai_bottom: float,
                ai_right: float, ai_top: float) -> int | None:
    """Look up a JSON prediction by AI-frame bbox. Tries exact rounded
    key first, then a small neighborhood for float drift."""
    key = (round(ai_left, 1), round(ai_bottom, 1),
           round(ai_right, 1), round(ai_top, 1))
    if key in rec_idx:
        return rec_idx[key]
    # Neighborhood scan in 0.1pt steps within tolerance.
    steps = int(BBOX_TOLERANCE_PT * 10) + 1
    for dl in range(-steps, steps + 1):
        for db in range(-steps, steps + 1):
            for dr in range(-steps, steps + 1):
                for dt in range(-steps, steps + 1):
                    k = (round(ai_left + dl * 0.1, 1),
                         round(ai_bottom + db * 0.1, 1),
                         round(ai_right + dr * 0.1, 1),
                         round(ai_top + dt * 0.1, 1))
                    if k in rec_idx:
                        return rec_idx[k]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True,
                    help="Input predictions JSON from predict_one.py")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output JSON (may be the same path as --predictions).")
    args = ap.parse_args()

    payload = json.loads(args.predictions.read_text())
    preds = payload.get("predictions", [])
    print(f"[apply_taxi] loaded {len(preds)} predictions")

    # Step 1: strip ML's Taxiways and Taxiway Labels predictions entirely.
    # Rule-based detection is the only source for these two classes.
    TAXI_LABELS = {"Taxiways", "Taxiway Labels",
                   "Maybe Taxiway", "Maybe Taxiway Label"}
    stripped = 0
    for rec in preds:
        if rec.get("label") in TAXI_LABELS:
            rec["label"] = "Other"
            stripped += 1
    print(f"[apply_taxi] stripped {stripped} ML taxi predictions -> Other")

    # Step 2: rule-based detection assigns Taxiways and Taxiway Labels.
    print(f"[apply_taxi] running rule-based taxi detection on {args.pdf}")
    det = detect_taxi(args.pdf)
    page_h = det["page_h"]
    surf_indices = set(det["taxi_surface_indices"])
    label_indices = set(det["taxi_label_indices"])
    print(f"[apply_taxi] detected {len(surf_indices)} taxi surfaces, "
          f"{len(label_indices)} taxi-label glyphs")

    rec_idx = _build_bbox_index(preds, page_h)

    n_taxi_surface = 0
    n_taxi_label = 0
    n_unmatched = 0
    polys = det["all_polys"]
    for i, p in enumerate(polys):
        if i not in surf_indices and i not in label_indices:
            continue
        x0, y0, x1, y1 = p["rect"]
        ai_left   = x0
        ai_right  = x1
        ai_top    = page_h - y0
        ai_bottom = page_h - y1
        match = _find_match(rec_idx, ai_left, ai_bottom, ai_right, ai_top)
        if match is None:
            n_unmatched += 1
            continue
        rec = preds[match]
        if i in surf_indices:
            rec["label"] = "Taxiways"
            n_taxi_surface += 1
        else:
            rec["label"] = "Taxiway Labels"
            n_taxi_label += 1
        rec["override_applied"] = True
        rec["override_source"] = "rule_based_taxi"

    print(f"[apply_taxi] assigned {n_taxi_surface} polygons -> Taxiways")
    print(f"[apply_taxi] assigned {n_taxi_label} polygons -> Taxiway Labels")
    if n_unmatched:
        print(f"[apply_taxi] WARNING: {n_unmatched} detected polygons not "
              f"found in predictions JSON")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"[apply_taxi] wrote -> {args.out}")


if __name__ == "__main__":
    main()
