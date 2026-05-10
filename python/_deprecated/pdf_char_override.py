"""
pdf_char_override.py — DEPRECATED. Not used by classify_pipeline.

This module belonged to the predict.py / predict_one.py inference path,
which has been replaced by classify_pipeline.py. The current pipeline
handles runway labels, taxi labels, and stroked-only sweeping
declaratively in steps 3, 4, and 7 — there's no separate
"final-label assignment" pass any more. Kept for archaeology only.

Original purpose was to replace the brittle PDF-text-bbox-overlap rule
and the 36-class char-classifier detour:

  1. Pre-classify Taxiways by color. Filled-gray polygons (RGB in
     150-235 with low channel spread) are taxiway pavement, full stop.
     They never enter the override decision.

  2. Detect runway centerlines. Heuristic: the most elongated +
     largest-area polygons. Their PCA principal angle defines a line
     through their centroid that runway labels are painted along.

  3. Map polygons to PDF text tokens via PyMuPDF's per-character
     bboxes (`get_text("rawdict")`). Each polygon's centroid lies in
     at most one character's bbox; that character's parent token (the
     space-separated text run within its span) is the polygon's
     associated word.

  4. Group polygons by token. The group's centroid is the center of
     the PyMuPDF token bbox (the union of its character bboxes).

  5. For each token group, fire an override only if BOTH:
       text matches a known pattern   AND   group centroid passes the
                                              right spatial gate
     - Runway Labels: text ∈ NASR runway designations for this airport
                      AND group centroid lateral offset to a runway
                      centerline ≤ RUNWAY_LATERAL_THRESH_PT.
     - Taxiway Labels: text matches ^[A-Z][A-Z0-9]?$ AND group centroid
                       lies inside any taxiway-pavement polygon's bbox.

  6. Anything not pre-classified or overridden defers to the v24 model
     prediction with confidence-banded "Maybe X" routing.
"""

from __future__ import annotations

import re
from pathlib import Path

import fitz
import numpy as np
import pandas as pd

from ml.load import LABELS
from pipeline.extract_paths_fitz import items_to_subpaths
from pipeline.extract_pdf_text import _normalize_designation

_RUNWAY_RE = re.compile(r"^\d{1,2}[LRC]?$")
# Taxiway designations: typically 1-2 chars (A, B1) but big hub airports
# (ORD, ATL, DFW) have 3-4 char taxiways like YY1, AA10, ECC.
_TAXIWAY_RE = re.compile(r"^[A-Z][A-Z0-9]{0,3}$")

# Reuse the gray-pavement thresholds from relational.py.
TAXIWAY_GRAY_MIN = 150
TAXIWAY_GRAY_MAX = 235
TAXIWAY_GRAY_TOL = 20

# Group-centroid lateral-offset threshold for a runway-label override to
# fire. Real runway designations sit ~0pt from the centerline, but
# parallel-runway designations (14L, 14R) are painted offset by half the
# pavement width, so the threshold needs to clear typical runway widths
# in PDF units (~30-50pt for a thicker runway). 50pt covers the full
# observed range without grabbing labels in side-info boxes.
RUNWAY_LATERAL_THRESH_PT = 50.0

MAYBE_OF = {
    "Taxiways":       "Maybe Taxiway",
    "Footprints":     "Maybe Footprint",
    "Runways":        "Maybe Runway",
    "Lights":         "Maybe Light",
    "Stars":          "Maybe Star",
    "Taxiway Labels": "Maybe Taxiway Label",
    "Runway Labels":  "Maybe Runway Label",
}


# ---------------------------------------------------------------------------
# PDF text → tokens
# ---------------------------------------------------------------------------

def parse_pdf(pdf_path: Path) -> tuple[float, float, list[dict], list[dict]]:
    """Return (page_w, page_h, tokens, drawings). Each drawing dict has:
        bbox          (left, top, right, bottom) in AI y-up
        fill_rgb      (r, g, b) in 0-255 ints, or None
        is_filled     bool
        subpaths      list of [(x, y), ...] in AI y-up

    Each token dict has:
        text          full token text
        bbox          (left, top, right, bottom) in AI y-up
        char_bboxes   list of per-character bboxes (AI y-up)
    """
    doc = fitz.open(pdf_path)
    try:
        page = doc[0]
        page_w = float(page.rect.width)
        page_h = float(page.rect.height)

        # ---- text tokens
        rd = page.get_text("rawdict")
        tokens: list[dict] = []

        def _flush(buf: list[dict]):
            if not buf:
                return
            text = "".join(c["c"] for c in buf)
            char_bboxes_ai: list[tuple[float, float, float, float]] = []
            for c in buf:
                x0, y0, x1, y1 = c["bbox"]
                char_bboxes_ai.append((x0, page_h - y0, x1, page_h - y1))
            left = min(b[0] for b in char_bboxes_ai)
            right = max(b[2] for b in char_bboxes_ai)
            top = max(b[1] for b in char_bboxes_ai)
            bottom = min(b[3] for b in char_bboxes_ai)
            tokens.append({
                "text": text,
                "bbox": (left, top, right, bottom),
                "char_bboxes": char_bboxes_ai,
            })

        for block in rd.get("blocks", []):
            if block.get("type", 0) != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    cur: list[dict] = []
                    for c in span.get("chars", []):
                        ch = c.get("c", "")
                        if ch and ch.strip():
                            cur.append(c)
                        else:
                            _flush(cur)
                            cur = []
                    _flush(cur)

        # Post-process: merge runway designations that PyMuPDF split
        # across spans. "14L" / "32R" labels render with the L/R/C in a
        # smaller-font span, so PyMuPDF emits two tokens. Taxiway labels
        # like "CC", "YY1" come as single tokens already (same span), so
        # we don't merge those — doing so would also wrongly merge
        # unrelated adjacent labels like "B1" + "B2".
        tokens = _merge_runway_designations(tokens)

        # ---- drawings with subpath geometry (for point-in-polygon tests)
        drawings: list[dict] = []
        for d in page.get_drawings():
            items = d.get("items") or []
            rect = d.get("rect")
            if not items or rect is None:
                continue
            sps_pdf = items_to_subpaths(items)
            if not sps_pdf:
                continue
            # Convert subpaths and bbox from PDF y-down to AI y-up.
            sps_ai = [[(x, page_h - y) for (x, y) in sp] for sp in sps_pdf]
            left = float(rect.x0)
            right = float(rect.x1)
            top = page_h - float(rect.y0)
            bottom = page_h - float(rect.y1)

            dtype = d.get("type", "")
            is_filled = "f" in dtype
            fill_rgb: tuple[int, int, int] | None = None
            if is_filled:
                f = d.get("fill")
                if isinstance(f, (tuple, list)) and len(f) == 3:
                    fill_rgb = (
                        int(round(f[0] * 255)),
                        int(round(f[1] * 255)),
                        int(round(f[2] * 255)),
                    )
                elif isinstance(f, (tuple, list)) and len(f) == 1:
                    v = int(round(float(f[0]) * 255))
                    fill_rgb = (v, v, v)
                elif isinstance(f, (int, float)):
                    v = int(round(float(f) * 255))
                    fill_rgb = (v, v, v)

            drawings.append({
                "bbox": (left, top, right, bottom),
                "fill_rgb": fill_rgb,
                "is_filled": is_filled,
                "subpaths": sps_ai,
            })

        return page_w, page_h, tokens, drawings
    finally:
        doc.close()


def parse_pdf_tokens(pdf_path: Path) -> tuple[float, float, list[dict]]:
    """Backward-compat shim — returns just (page_w, page_h, tokens)."""
    page_w, page_h, tokens, _ = parse_pdf(pdf_path)
    return page_w, page_h, tokens


def _merge_runway_designations(tokens: list[dict]) -> list[dict]:
    """Merge spatially-adjacent (digits, [LRC]) token pairs into a single
    runway designation token. PDFs commonly render labels like "14L" with
    the L/R/C in a separate span (smaller font / superscript / rotated),
    so PyMuPDF emits two tokens.

    Match criterion: an LRC token whose center is within 1.5 × the digit
    token's bbox-max-dimension of the digit token's center. This is
    direction-agnostic — works for horizontal, stacked-vertical, and
    rotated runway designations alike.

    Two-pass: pass 1 picks the merge pairs, pass 2 emits the output. A
    single-pass approach would emit unmerged suffix tokens to the output
    before later iterations could decide to consume them.
    """
    DIGIT_RE = re.compile(r"^\d{1,2}$")
    SUFFIX_RE = re.compile(r"^[LRC]$")

    # Pass 1: pick merge partners.
    merge_pairs: dict[int, int] = {}
    used_suffixes: set[int] = set()

    for i, t in enumerate(tokens):
        if not DIGIT_RE.match(t["text"]):
            continue
        bb_i = t["bbox"]
        ic_x = (bb_i[0] + bb_i[2]) / 2.0
        ic_y = (bb_i[1] + bb_i[3]) / 2.0
        i_w = bb_i[2] - bb_i[0]
        i_h = bb_i[1] - bb_i[3]
        i_size = max(i_w, i_h, 1.0)

        best_j = -1
        best_dist = 1e9
        for j, tj in enumerate(tokens):
            if j == i or j in used_suffixes:
                continue
            if not SUFFIX_RE.match(tj["text"]):
                continue
            bb_j = tj["bbox"]
            jc_x = (bb_j[0] + bb_j[2]) / 2.0
            jc_y = (bb_j[1] + bb_j[3]) / 2.0
            dist = ((ic_x - jc_x) ** 2 + (ic_y - jc_y) ** 2) ** 0.5
            # Tight proximity gate. Real runway-pair distances measure
            # ~0.5x the digit-token size; loosening this lets stray C/R
            # tokens from unrelated text (e.g. "RWY C", "CONCRETE") get
            # falsely glued onto digit tokens like "22" → "22C".
            if dist > i_size * 0.8:
                continue
            if dist < best_dist:
                best_dist = dist
                best_j = j
        if best_j != -1:
            merge_pairs[i] = best_j
            used_suffixes.add(best_j)

    # Pass 2: emit tokens, merging where pass 1 paired them.
    new_tokens: list[dict] = []
    for i, t in enumerate(tokens):
        if i in used_suffixes:
            continue
        if i in merge_pairs:
            tj = tokens[merge_pairs[i]]
            merged_text = t["text"] + tj["text"]
            bb = (
                min(t["bbox"][0], tj["bbox"][0]),
                max(t["bbox"][1], tj["bbox"][1]),
                max(t["bbox"][2], tj["bbox"][2]),
                min(t["bbox"][3], tj["bbox"][3]),
            )
            new_tokens.append({
                "text": merged_text,
                "bbox": bb,
                "char_bboxes": t["char_bboxes"] + tj["char_bboxes"],
            })
        else:
            new_tokens.append(t)
    return new_tokens


def _merge_adjacent_alphanumeric(tokens: list[dict]) -> list[dict]:
    """Merge adjacent alphanumeric tokens into single multi-character
    tokens. PyMuPDF often splits a visual label like "CC" or "YY1" into
    multiple single-character tokens because of span boundaries (kerning,
    style differences) — they need to be joined back so the override
    logic sees one taxiway designation, not several letters.

    Merge criteria:
      - Both tokens are alphanumeric ([A-Z0-9])
      - Center-to-center distance < 0.8 × max(token sizes)
      - Vertical centers within 0.5 × max token height (same baseline)

    Iterates union-find style: any two close-enough tokens get joined,
    transitively. So three close tokens "Y" + "Y" + "1" become one "YY1".
    Resulting tokens are sorted by document order (left-to-right within
    a horizontal label, top-to-bottom within a rotated/stacked label).
    """
    n = len(tokens)
    if n < 2:
        return list(tokens)

    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    def is_alnum(text: str) -> bool:
        return bool(text) and all(c.isalnum() and c.isascii() for c in text)

    centers = []
    sizes = []
    for t in tokens:
        bb = t["bbox"]
        cx = (bb[0] + bb[2]) / 2.0
        cy = (bb[1] + bb[3]) / 2.0
        w = bb[2] - bb[0]
        h = bb[1] - bb[3]
        centers.append((cx, cy))
        sizes.append(max(w, h, 1.0))

    for i in range(n):
        if not is_alnum(tokens[i]["text"]):
            continue
        for j in range(i + 1, n):
            if not is_alnum(tokens[j]["text"]):
                continue
            ci, cj = centers[i], centers[j]
            dist = ((ci[0] - cj[0]) ** 2 + (ci[1] - cj[1]) ** 2) ** 0.5
            if dist > 0.8 * max(sizes[i], sizes[j]):
                continue
            # Same-baseline check via vertical-center proximity. Use the
            # smaller token's height so a small "1" near a large "YY"
            # doesn't get rejected for being slightly off-baseline.
            bbi = tokens[i]["bbox"]
            bbj = tokens[j]["bbox"]
            hi = bbi[1] - bbi[3]
            hj = bbj[1] - bbj[3]
            if abs(ci[1] - cj[1]) > 0.5 * max(hi, hj):
                continue
            union(i, j)

    # Validation pattern: only KEEP a merge whose joined text would be a
    # plausible taxiway or runway designation. Otherwise emit the
    # constituent tokens individually — this prevents adjacent unrelated
    # labels (e.g. "B1" next to "B2") from collapsing into "B1B2",
    # which would then match neither pattern and leave both polygons
    # uncategorized.
    MERGE_KEEP_RE = re.compile(r"^([A-Z][A-Z0-9]{0,3}|\d{1,2}[LRC])$")

    components: dict[int, list[int]] = {}
    for i in range(n):
        components.setdefault(find(i), []).append(i)

    merged: list[dict] = []
    for indices in components.values():
        if len(indices) == 1:
            merged.append(tokens[indices[0]])
            continue
        # Order by left-x then top-y (descending) so multi-char tokens
        # read left-to-right or top-to-bottom for rotated labels.
        indices.sort(key=lambda i: (tokens[i]["bbox"][0], -tokens[i]["bbox"][1]))
        merged_text = "".join(tokens[i]["text"] for i in indices)
        if not MERGE_KEEP_RE.match(merged_text):
            # Bad merge — keep the originals separate.
            for i in indices:
                merged.append(tokens[i])
            continue
        all_chars: list[tuple[float, float, float, float]] = []
        for i in indices:
            all_chars.extend(tokens[i]["char_bboxes"])
        bb_lefts = [tokens[i]["bbox"][0] for i in indices]
        bb_rights = [tokens[i]["bbox"][2] for i in indices]
        bb_tops = [tokens[i]["bbox"][1] for i in indices]
        bb_bots = [tokens[i]["bbox"][3] for i in indices]
        merged.append({
            "text": merged_text,
            "bbox": (min(bb_lefts), max(bb_tops), max(bb_rights), min(bb_bots)),
            "char_bboxes": all_chars,
        })
    return merged


def _point_in_polygon(px: float, py: float, polygon: list[tuple[float, float]]) -> bool:
    """Standard ray-casting point-in-polygon test. Polygon is a sequence of
    (x, y) anchors; closed implicitly (first and last connect). At least 3
    points required for a meaningful answer.
    """
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        # Crossing the ray (horizontal, going right from (px, py))
        if ((yi > py) != (yj > py)):
            x_at_py = (xj - xi) * (py - yi) / (yj - yi + 1e-30) + xi
            if px < x_at_py:
                inside = not inside
        j = i
    return inside


def _point_in_any_taxiway(
    px: float, py: float, taxiway_subpaths: list[list[list[tuple[float, float]]]]
) -> bool:
    """taxiway_subpaths is a list (one per taxiway polygon) of subpath lists.
    Returns True if (px, py) is inside any subpath of any taxiway polygon.
    """
    for sps in taxiway_subpaths:
        for sp in sps:
            if _point_in_polygon(px, py, sp):
                return True
    return False


# ---------------------------------------------------------------------------
# Color-based taxiway pavement detection
# ---------------------------------------------------------------------------

def _is_taxiway_gray(r, g, b) -> bool:
    if not all(isinstance(v, (int, float)) for v in (r, g, b)):
        return False
    if any(np.isnan(v) for v in (r, g, b)):
        return False
    if not all(TAXIWAY_GRAY_MIN <= v <= TAXIWAY_GRAY_MAX for v in (r, g, b)):
        return False
    return max(abs(r - g), abs(r - b), abs(g - b)) <= TAXIWAY_GRAY_TOL


def identify_taxiway_polygons_by_color(df: pd.DataFrame) -> np.ndarray:
    """Boolean mask: True for filled-gray polygons (i.e. taxiway pavement)."""
    n = len(df)
    mask = np.zeros(n, dtype=bool)
    if "filled" not in df.columns:
        return mask
    filled = pd.to_numeric(df["filled"], errors="coerce").fillna(0).to_numpy(dtype=int)
    fr = pd.to_numeric(df.get("fill_r", pd.Series([np.nan] * n)), errors="coerce").to_numpy()
    fg = pd.to_numeric(df.get("fill_g", pd.Series([np.nan] * n)), errors="coerce").to_numpy()
    fb = pd.to_numeric(df.get("fill_b", pd.Series([np.nan] * n)), errors="coerce").to_numpy()
    for i in range(n):
        if filled[i] != 1:
            continue
        if _is_taxiway_gray(fr[i], fg[i], fb[i]):
            mask[i] = True
    return mask


# ---------------------------------------------------------------------------
# Runway centerline detection (heuristic)
# ---------------------------------------------------------------------------

def identify_runway_axes(df: pd.DataFrame) -> list[dict]:
    """Find runway-shaped polygons (most elongated, large) and return their
    centerline geometry: cx, cy, angle (rad), half_length.

    Runways in FAA charts are very elongated (principal_ratio >> 5) and
    among the largest polygons on the page, so a rank-based filter on
    those two features is reliable without needing the v24 model.
    """
    if len(df) == 0:
        return []
    bbox_area = pd.to_numeric(df["bbox_area"], errors="coerce").to_numpy(dtype=float)
    pratio = pd.to_numeric(df["principal_ratio"], errors="coerce").to_numpy(dtype=float)
    pangle_deg = pd.to_numeric(df["principal_angle"], errors="coerce").to_numpy(dtype=float)
    cx = pd.to_numeric(df["centroid_x"], errors="coerce").to_numpy(dtype=float)
    cy = pd.to_numeric(df["centroid_y"], errors="coerce").to_numpy(dtype=float)
    width = pd.to_numeric(df["width"], errors="coerce").to_numpy(dtype=float)
    height = pd.to_numeric(df["height"], errors="coerce").to_numpy(dtype=float)

    valid = (~np.isnan(bbox_area)) & (~np.isnan(pratio))
    if valid.sum() == 0:
        return []

    area_thresh = (
        np.quantile(bbox_area[valid], 0.9) if valid.sum() > 5 else 0.0
    )
    cand = valid & (pratio >= 5.0) & (bbox_area >= area_thresh)

    if cand.sum() == 0:
        # Fallback: top 5 by pratio among valid rows.
        valid_idx = np.where(valid)[0]
        cand_idx = valid_idx[np.argsort(-pratio[valid_idx])][:5]
    else:
        cand_idx = np.where(cand)[0]

    out: list[dict] = []
    for i in cand_idx:
        long_axis = float(max(width[i], height[i]))
        out.append({
            "cx": float(cx[i]),
            "cy": float(cy[i]),
            "angle": float(np.deg2rad(pangle_deg[i])),
            "half_length": long_axis / 2.0,
        })
    return out


def _lateral_offset_to_runway(px: float, py: float, runway: dict) -> float:
    """Perpendicular distance from (px,py) to the line through the runway's
    centroid in the runway's principal-angle direction."""
    dx = px - runway["cx"]
    dy = py - runway["cy"]
    a = runway["angle"]
    return abs(-dx * np.sin(a) + dy * np.cos(a))


def _min_lateral_offset_to_any_runway(px: float, py: float, runways: list[dict]) -> float:
    if not runways:
        return float("inf")
    return min(_lateral_offset_to_runway(px, py, r) for r in runways)


def _is_inside_any_taxi_bbox(
    px: float, py: float, taxi_bboxes: list[tuple[float, float, float, float]]
) -> bool:
    """taxi_bboxes are AI-frame (left, top, right, bottom) with top > bottom."""
    for (left, top, right, bottom) in taxi_bboxes:
        if left <= px <= right and bottom <= py <= top:
            return True
    return False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def apply_pdf_char_override(
    df: pd.DataFrame,
    probs: np.ndarray,
    pdf_path: Path,
    nasr_runways: set[str] | None = None,
    confidence: float = 0.60,
    runway_lateral_thresh: float = RUNWAY_LATERAL_THRESH_PT,
    verbose: bool = False,
) -> tuple[list[str], list[bool]]:
    """Return (final_label_per_row, was_override_per_row).

    `df` must have columns: centroid_x, centroid_y, filled, fill_r/g/b,
    bbox_area, principal_ratio, principal_angle, width, height, left, top,
    right, bottom — i.e. the standard load.py / extract_paths_fitz columns.
    """
    n = len(df)
    pred_cls = probs.argmax(axis=1)
    top_p = probs.max(axis=1)
    final_labels: list[str | None] = [None] * n
    overrode: list[bool] = [False] * n

    df_r = df.reset_index(drop=True)
    cx_arr = pd.to_numeric(df_r["centroid_x"], errors="coerce").to_numpy(dtype=float)
    cy_arr = pd.to_numeric(df_r["centroid_y"], errors="coerce").to_numpy(dtype=float)

    # 1a. Stroked-only polygons → Lights (deterministic). The user
    # asserts that "lines with stroke" only ever belong on Lights, so we
    # pre-classify any unfilled stroked polygon there. Filled polygons
    # (with or without an additional stroke) are eligible for any layer
    # except Lights.
    if "filled" in df_r.columns and "stroked" in df_r.columns:
        filled_col = pd.to_numeric(df_r["filled"], errors="coerce").fillna(0).to_numpy(dtype=int)
        stroked_col = pd.to_numeric(df_r["stroked"], errors="coerce").fillna(0).to_numpy(dtype=int)
        lights_mask = (filled_col == 0) & (stroked_col == 1)
    else:
        lights_mask = np.zeros(n, dtype=bool)
    n_lights = 0
    for i in range(n):
        if lights_mask[i]:
            final_labels[i] = "Lights"
            overrode[i] = True
            n_lights += 1

    # 1b. Color-classify taxiway pavement — bypasses the model entirely.
    taxi_mask = identify_taxiway_polygons_by_color(df_r)

    # Walk the PDF once: get text tokens AND drawing geometry. We need
    # the actual taxiway polygon shapes (not just bboxes) for the
    # point-in-polygon spatial gate that gates Taxiway Labels.
    page_w, page_h, tokens, pdf_drawings = parse_pdf(pdf_path)

    # Match df rows to pdf_drawings by bbox identity. Both are extracted
    # from the same PDF so left/top/right/bottom should agree to ~0.01pt.
    pdf_drawing_for_df: list[dict | None] = [None] * n
    used_pdf = [False] * len(pdf_drawings)
    for i in range(n):
        try:
            l = float(df_r.iloc[i]["left"])
            t = float(df_r.iloc[i]["top"])
            r = float(df_r.iloc[i]["right"])
            b = float(df_r.iloc[i]["bottom"])
        except (KeyError, ValueError, TypeError):
            continue
        for j, d in enumerate(pdf_drawings):
            if used_pdf[j]:
                continue
            db = d["bbox"]
            if (abs(l - db[0]) < 0.05 and abs(t - db[1]) < 0.05
                    and abs(r - db[2]) < 0.05 and abs(b - db[3]) < 0.05):
                pdf_drawing_for_df[i] = d
                used_pdf[j] = True
                break

    # Collect actual taxiway polygon subpaths for point-in-polygon tests.
    taxiway_subpaths: list[list[list[tuple[float, float]]]] = []
    for i in range(n):
        if final_labels[i] is not None:
            continue  # already pre-classified (e.g. Lights)
        if taxi_mask[i]:
            final_labels[i] = "Taxiways"
            overrode[i] = True
            d = pdf_drawing_for_df[i]
            if d is not None:
                taxiway_subpaths.append(d["subpaths"])

    # 2. Identify runway centerlines.
    runways = identify_runway_axes(df_r)

    # 3. Map polygons → PDF tokens via per-char bboxes.
    polygon_to_token: dict[int, dict] = {}
    for i in range(n):
        if final_labels[i] is not None:
            continue
        cx = cx_arr[i]
        cy = cy_arr[i]
        if np.isnan(cx) or np.isnan(cy):
            continue
        for tok in tokens:
            matched = False
            for (cl, ct, cr, cb) in tok["char_bboxes"]:
                if cl <= cx <= cr and cb <= cy <= ct:
                    polygon_to_token[i] = tok
                    matched = True
                    break
            if matched:
                break

    # 4. Group polygons by token (id() — same text in different positions
    # are different tokens with different bboxes).
    by_token: dict[int, list[int]] = {}
    for i, tok in polygon_to_token.items():
        by_token.setdefault(id(tok), []).append(i)

    # 5. Per-token override decision.
    n_runway_overrides = 0
    n_taxiway_overrides = 0
    n_text_match_no_spatial = 0
    for poly_indices in by_token.values():
        tok = polygon_to_token[poly_indices[0]]
        text = tok["text"]
        bb = tok["bbox"]
        gcx = (bb[0] + bb[2]) / 2.0
        gcy = (bb[1] + bb[3]) / 2.0

        is_runway_text = bool(_RUNWAY_RE.match(text))
        is_taxiway_text = bool(_TAXIWAY_RE.match(text))
        norm = _normalize_designation(text) if is_runway_text else None
        is_known_rwy = bool(nasr_runways and norm and norm in nasr_runways)

        if is_runway_text and is_known_rwy:
            offset = _min_lateral_offset_to_any_runway(gcx, gcy, runways)
            if offset <= runway_lateral_thresh:
                for i in poly_indices:
                    final_labels[i] = "Runway Labels"
                    overrode[i] = True
                n_runway_overrides += 1
            else:
                n_text_match_no_spatial += 1
        elif is_taxiway_text:
            # Spatial gate: group centroid must lie inside the actual
            # taxiway pavement polygon (point-in-polygon, not bbox). A
            # token like "1" or "5" sitting in white space inside the
            # bbox of an L-shaped taxiway would have passed bbox
            # containment but fails point-in-polygon.
            if _point_in_any_taxiway(gcx, gcy, taxiway_subpaths):
                for i in poly_indices:
                    final_labels[i] = "Taxiway Labels"
                    overrode[i] = True
                n_taxiway_overrides += 1
            else:
                n_text_match_no_spatial += 1

    # 6. Defer to model with confidence banding for the rest.
    #
    # Hard rules — output layers are ONLY: Footprints, Runways, Runway
    # Labels, Taxiways, Taxiway Labels, Stars, Other, Metadata.
    #
    #   - Taxiways / Taxiway Labels / Runway Labels can only be reached
    #     via the deterministic override (color or text+spatial gate).
    #   - Lights is reachable ONLY via the stroked-only pre-classification
    #     (filled-polygon "Lights" predictions are suppressed → Other).
    #   - All "Maybe X" variants are routed to Other.
    excluded_idx = {LABELS.index(c) for c in ("Taxiways", "Taxiway Labels", "Runway Labels", "Lights")}

    for i in range(n):
        if final_labels[i] is not None:
            continue
        cls_probs = probs[i].copy()
        for idx in excluded_idx:
            cls_probs[idx] = -1.0
        cls_idx = int(cls_probs.argmax())
        top_prob = float(probs[i, cls_idx])
        cls_name = LABELS[cls_idx]
        if cls_name == "Other":
            final_labels[i] = "Other"
        elif top_prob >= confidence:
            final_labels[i] = cls_name
        else:
            # Below confidence threshold → Other (no Maybe layers).
            final_labels[i] = "Other"

    if verbose:
        print(
            f"[pdf_char_override] "
            f"lights_by_stroke={n_lights}  "
            f"taxiways_by_color={int(taxi_mask.sum())}  "
            f"runway_centerlines={len(runways)}  "
            f"tokens={len(tokens)}  "
            f"polygons_in_tokens={len(polygon_to_token)}  "
            f"token_groups={len(by_token)}  "
            f"runway_label_groups={n_runway_overrides}  "
            f"taxiway_label_groups={n_taxiway_overrides}  "
            f"text_match_failed_spatial={n_text_match_no_spatial}"
        )

    return final_labels, overrode  # type: ignore[return-value]
