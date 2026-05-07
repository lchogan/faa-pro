"""
render_char_layers_charbox.py — render PDF polygons + word-level text
tokens to SVG, with taxiway-surface detection and label matching.

Output layers:
  1. Taxiways         — gray-filled polygons (~#cfcfcf, the pavement
                        surfaces). Excluded from label matching.
  2. Polygons         — every other drawing not claimed as a label.
  3. Taxiway Labels   — polygons claimed as glyphs of taxiway-label
                        tokens (red fill for visibility).
  4. PDF Text Tokens  — word strings from the PDF text stream, one
                        <text> per word, anchored at bbox center.

Label matching:
  A token qualifies as a label candidate if:
    (a) its text matches the class pattern, and
    (b) its bbox touches (any overlap with) a surface of the same class.
  For each qualifying token, the K = len(token) nearest unclaimed
  candidate polygon centroids are reassigned to the Labels layer.

  Patterns:
    Runway:  ^(0?[1-9]|[12][0-9]|3[0-6])[LRC]?$  ("9", "27R", "10L")
    Taxiway: ^([A-Z][A-Z]?[0-9]{0,2}|[0-9])$     ("C", "C1", "A11", "3")

  Runway matching runs first, so a single-digit token that sits on a
  runway is claimed as a runway label before the taxiway pass.

Output layers (SVG <g> with inkscape:groupmode="layer"):
  Taxiways, Runways, Polygons, Taxiway Labels, Runway Labels,
  PDF Text Tokens.

Usage:
    python render_char_layers_charbox.py \\
        --pdf /path/to/<airport>-faa.pdf \\
        --out /path/to/<airport>-chars.svg
"""

from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import fitz
import numpy as np

from extract_paths_fitz import _color_to_rgb, items_to_subpaths


TAXIWAY_RE = re.compile(r"^([A-Z][A-Z]?[0-9]{0,2}|[0-9])$")
RUNWAY_RE = re.compile(r"^(0?[1-9]|[12][0-9]|3[0-6])[LRC]?$")
TAXI_LABEL_FILL = "#ff0000"
RUNWAY_LABEL_FILL = "#0000ff"
# Gray-fill detection bracket around #cfcfcf = (207, 207, 207).
GRAY_MIN = 175
GRAY_MAX = 235
GRAY_CHANNEL_TOL = 20  # max R/G/B spread to still call it "gray"
# Near-black fill bracket — glyph polygons are essentially pure black.
BLACK_MAX = 60
# Runway pavement is a large, elongated, near-black filled polygon —
# elongation rules out terminal aprons / parking blocks that are also
# large + black-filled.
RUNWAY_SURFACE_AREA_MIN = 500.0
RUNWAY_SURFACE_ASPECT_MIN = 5.0
# Step-3 extended centerline matching: a polygon is a runway-label
# candidate if its bbox intersects a runway centerline extended past
# the runway endpoints by RUNWAY_CENTERLINE_PAD_PT on each side.
RUNWAY_CENTERLINE_PAD_PT = 80.0


def _items_to_svg_d(items) -> str:
    parts: list[str] = []
    last_end: tuple[float, float] | None = None

    def feq(a, b):
        return abs(a[0] - b[0]) < 1e-3 and abs(a[1] - b[1]) < 1e-3

    for item in items:
        op = item[0]
        if op == "l":
            p1 = (item[1].x, item[1].y)
            p2 = (item[2].x, item[2].y)
            if last_end is None or not feq(p1, last_end):
                parts.append(f"M{p1[0]:.3f},{p1[1]:.3f}")
            parts.append(f"L{p2[0]:.3f},{p2[1]:.3f}")
            last_end = p2
        elif op == "c":
            p1 = (item[1].x, item[1].y)
            c1 = (item[2].x, item[2].y)
            c2 = (item[3].x, item[3].y)
            p4 = (item[4].x, item[4].y)
            if last_end is None or not feq(p1, last_end):
                parts.append(f"M{p1[0]:.3f},{p1[1]:.3f}")
            parts.append(
                f"C{c1[0]:.3f},{c1[1]:.3f} {c2[0]:.3f},{c2[1]:.3f} "
                f"{p4[0]:.3f},{p4[1]:.3f}"
            )
            last_end = p4
        elif op == "re":
            r = item[1]
            parts.append(
                f"M{r.x0:.3f},{r.y0:.3f} L{r.x1:.3f},{r.y0:.3f} "
                f"L{r.x1:.3f},{r.y1:.3f} L{r.x0:.3f},{r.y1:.3f} Z"
            )
            last_end = None
        elif op == "qu":
            q = item[1]
            parts.append(
                f"M{q.ul.x:.3f},{q.ul.y:.3f} L{q.ur.x:.3f},{q.ur.y:.3f} "
                f"L{q.lr.x:.3f},{q.lr.y:.3f} L{q.ll.x:.3f},{q.ll.y:.3f} Z"
            )
            last_end = None
    return " ".join(parts)


def _color_to_hex(c) -> str | None:
    r, g, b, kind = _color_to_rgb(c)
    if kind == "none" or kind == "other" or r == "":
        return None
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


def _is_near_black(fill_rgb_kind) -> bool:
    """True if the fill is essentially black — every channel <= BLACK_MAX.
    Glyphs are filled near-black on FAA diagrams; this excludes gray
    taxiway pavement, brown markers, and pale gridlines."""
    r, g, b, kind = fill_rgb_kind
    if kind == "none" or kind == "other" or r == "":
        return False
    if not (isinstance(r, int) and isinstance(g, int) and isinstance(b, int)):
        return False
    return max(r, g, b) <= BLACK_MAX


def _is_taxiway_gray(fill_rgb_kind) -> bool:
    """True if the fill color is a light gray near #cfcfcf — the
    taxiway-pavement fill on FAA diagrams. Black, white, color, and
    'no fill' all return False."""
    r, g, b, kind = fill_rgb_kind
    if kind == "none" or kind == "other" or r == "":
        return False
    if not (isinstance(r, int) and isinstance(g, int) and isinstance(b, int)):
        return False
    if max(abs(r - g), abs(g - b), abs(r - b)) > GRAY_CHANNEL_TOL:
        return False
    avg = (r + g + b) / 3.0
    return GRAY_MIN <= avg <= GRAY_MAX


def _pca_principal_ratio(subpaths) -> float:
    """Ratio of principal-axis singular values for the polygon's anchor
    points. Long thin shapes have ratio >> 1 even when rotated, where
    bbox aspect would be ~1."""
    pts: list[tuple[float, float]] = []
    for ring in subpaths or []:
        pts.extend(ring)
    if len(pts) < 3:
        return 1.0
    arr = np.asarray(pts, dtype=float)
    centered = arr - arr.mean(axis=0)
    _, sv, _ = np.linalg.svd(centered, full_matrices=False)
    if len(sv) < 2 or sv[1] < 1e-9:
        return float("inf")
    return float(sv[0] / sv[1])


def _runway_axis_from_subpaths(subpaths, fallback_rect):
    """Compute the runway centerline geometry via PCA on the surface's
    anchor points. Returns dict with cx, cy, ux, uy (unit direction),
    half_len (longitudinal half-extent of the polygon along the axis).
    Falls back to bbox long-axis if PCA can't run."""
    pts: list[tuple[float, float]] = []
    for ring in subpaths or []:
        pts.extend(ring)
    x0, y0, x1, y1 = fallback_rect
    if len(pts) < 3:
        if (x1 - x0) >= (y1 - y0):
            return {"cx": (x0 + x1) / 2, "cy": (y0 + y1) / 2,
                    "ux": 1.0, "uy": 0.0, "half_len": (x1 - x0) / 2}
        return {"cx": (x0 + x1) / 2, "cy": (y0 + y1) / 2,
                "ux": 0.0, "uy": 1.0, "half_len": (y1 - y0) / 2}
    arr = np.asarray(pts, dtype=float)
    centroid = arr.mean(axis=0)
    centered = arr - centroid
    # Principal axis direction = first right-singular vector of centered.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    # Normalize defensively (svd already gives unit vectors).
    norm = float(np.linalg.norm(direction)) or 1.0
    ux, uy = float(direction[0]) / norm, float(direction[1]) / norm
    proj = centered @ np.array([ux, uy])
    half_len = float((proj.max() - proj.min()) / 2.0)
    return {"cx": float(centroid[0]), "cy": float(centroid[1]),
            "ux": ux, "uy": uy, "half_len": half_len}


def _segment_intersects_bbox(p1, p2, x0, y0, x1, y1) -> bool:
    """Liang-Barsky line-clipping segment-vs-axis-aligned-rect test."""
    ax, ay = p1
    bx, by = p2
    dx, dy = bx - ax, by - ay
    p_arr = (-dx, dx, -dy, dy)
    q_arr = (ax - x0, x1 - ax, ay - y0, y1 - ay)
    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p_arr, q_arr):
        if abs(pi) < 1e-12:
            if qi < 0:
                return False
        else:
            u = qi / pi
            if pi < 0:
                if u > u2:
                    return False
                if u > u1:
                    u1 = u
            else:
                if u < u1:
                    return False
                if u < u2:
                    u2 = u
    return u1 <= u2


def _point_in_subpaths(x: float, y: float,
                       subpaths: list[list[tuple[float, float]]]) -> bool:
    """Even-odd ray-cast test against a compound polygon's anchor rings.
    Curved edges are approximated by straight lines between anchors,
    which is close enough for taxiway-surface containment."""
    inside = False
    for ring in subpaths:
        n = len(ring)
        if n < 3:
            continue
        j = n - 1
        for i in range(n):
            xi, yi = ring[i]
            xj, yj = ring[j]
            if (yi > y) != (yj > y):
                xt = xi + (xj - xi) * (y - yi) / ((yj - yi) or 1e-12)
                if x < xt:
                    inside = not inside
            j = i
    return inside


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    doc = fitz.open(args.pdf)
    page = doc[0]
    page_w = float(page.rect.width)
    page_h = float(page.rect.height)

    # --- 1) All drawings → polygon records.
    # Gray-filled polygons are tagged as taxiway surfaces; their anchor
    # subpaths are kept so we can do point-in-polygon tests on them.
    drawings = page.get_drawings()
    all_polys: list[dict] = []
    for d in drawings:
        items = d.get("items") or []
        rect = d.get("rect")
        if not items or rect is None:
            continue
        cx = (float(rect.x0) + float(rect.x1)) / 2.0
        cy = (float(rect.y0) + float(rect.y1)) / 2.0
        dtype = d.get("type", "")
        filled = "f" in dtype
        stroked = "s" in dtype
        fill_rgb = _color_to_rgb(d.get("fill")) if filled else ("", "", "", "none")
        rw = float(rect.x1) - float(rect.x0)
        rh = float(rect.y1) - float(rect.y0)
        area = max(rw * rh, 0.0)
        long_side = max(rw, rh)
        short_side = max(min(rw, rh), 1e-6)
        bbox_aspect = long_side / short_side
        is_taxi_surface = filled and _is_taxiway_gray(fill_rgb)
        is_near_black = filled and _is_near_black(fill_rgb)
        # Runway pavement: large + near-black + elongated. We accept if
        # *either* bbox aspect is high (axis-aligned runway) OR PCA
        # principal-ratio is high (diagonal runway). This rules out big
        # black blocks (terminal aprons / parking ramps) that are large
        # but roughly square in both bbox AND PCA terms.
        runway_subs = items_to_subpaths(items) if (
            is_near_black and not is_taxi_surface
            and area >= RUNWAY_SURFACE_AREA_MIN
        ) else None
        is_runway_surface = False
        if runway_subs is not None:
            if bbox_aspect >= RUNWAY_SURFACE_ASPECT_MIN:
                is_runway_surface = True
            else:
                pca_ratio = _pca_principal_ratio(runway_subs)
                is_runway_surface = pca_ratio >= RUNWAY_SURFACE_ASPECT_MIN
        is_surface = is_taxi_surface or is_runway_surface
        record = {
            "d": _items_to_svg_d(items),
            "fill": _color_to_hex(d.get("fill")) if filled else None,
            "stroke": _color_to_hex(d.get("color")) if stroked else None,
            "stroke_width": float(d.get("width") or 0.0) if stroked else 0.0,
            "cx": cx,
            "cy": cy,
            "rect": (float(rect.x0), float(rect.y0),
                     float(rect.x1), float(rect.y1)),
            "filled": filled,
            "is_taxi_surface": is_taxi_surface,
            "is_runway_surface": is_runway_surface,
            "is_near_black": is_near_black,
            # Subpaths are needed only for surfaces (point-in-polygon
            # test for taxi-touch, PCA centerline for runway).
            "subpaths": (
                runway_subs if is_runway_surface
                else items_to_subpaths(items) if is_taxi_surface
                else None
            ),
        }
        all_polys.append(record)

    taxi_surfaces = [p for p in all_polys if p["is_taxi_surface"]]
    runway_surfaces = [p for p in all_polys if p["is_runway_surface"]]
    print(f"[charbox] {len(all_polys)} polygons "
          f"({len(taxi_surfaces)} taxi-surface, "
          f"{len(runway_surfaces)} runway-surface)")

    # --- 2) PDF text tokens (word-level: "RWY", "22L", etc.)
    words = page.get_text("words")  # (x0,y0,x1,y1,word,block,line,wno)
    text_tokens: list[dict] = []
    for w in words:
        x0, y0, x1, y1, txt, *_ = w
        if not txt or not txt.strip():
            continue
        text_tokens.append({
            "text": txt,
            "x0": float(x0), "y0": float(y0),
            "x1": float(x1), "y1": float(y1),
            "cx": (float(x0) + float(x1)) / 2.0,
            "cy": (float(y0) + float(y1)) / 2.0,
        })
    doc.close()
    print(f"[charbox] {len(text_tokens)} text tokens")

    # --- 3) Label matching.
    # A token qualifies if its bbox *touches* (has any overlap with) a
    # surface of the same class. K-nearest is computed against candidate
    # polygons only (filled, non-surface; taxiway adds a near-black
    # constraint, runway labels can be near-black or near-white).

    def _bbox_touches(t, surfaces):
        # 5-point sample of token bbox + check if any surface anchor
        # falls inside the bbox. Together this catches almost any
        # overlap between a small token bbox and a surface polygon
        # without a full edge-intersection test.
        sample = [
            (t["cx"], t["cy"]),
            (t["x0"], t["y0"]), (t["x1"], t["y0"]),
            (t["x1"], t["y1"]), (t["x0"], t["y1"]),
        ]
        for s in surfaces:
            sx0, sy0, sx1, sy1 = s["rect"]
            if sx1 < t["x0"] or sx0 > t["x1"] or sy1 < t["y0"] or sy0 > t["y1"]:
                continue
            subs = s["subpaths"] or []
            for px, py in sample:
                if _point_in_subpaths(px, py, subs):
                    return True
            for ring in subs:
                for ax, ay in ring:
                    if t["x0"] <= ax <= t["x1"] and t["y0"] <= ay <= t["y1"]:
                        return True
        return False

    def _run_match(name, tokens, surfaces, candidate_pred,
                   claimed_polys, claimed_token_keys):
        """Greedy K-nearest matching for one label class. Mutates
        claimed_polys and claimed_token_keys."""
        label_indices: list[int] = []
        qualifying: list[dict] = []
        for tok in tokens:
            tok_key = (tok["text"], round(tok["cx"], 3), round(tok["cy"], 3))
            if tok_key in claimed_token_keys:
                continue
            if not _bbox_touches(tok, surfaces):
                continue
            qualifying.append(tok)
            claimed_token_keys.add(tok_key)
            k = len(tok["text"])
            scored = []
            for i, p in enumerate(all_polys):
                if i in claimed_polys or not candidate_pred(p):
                    continue
                dx = tok["cx"] - p["cx"]
                dy = tok["cy"] - p["cy"]
                scored.append((dx * dx + dy * dy, i))
            scored.sort()
            for _, i in scored[:k]:
                claimed_polys.add(i)
                label_indices.append(i)
        print(f"[charbox] {name}: {len(qualifying)} tokens, "
              f"{len(label_indices)} polygons claimed")
        return label_indices

    # Candidate predicates. Both exclude surfaces and stroke-only paths.
    def _runway_candidate(p):
        return p["filled"] and not p["is_taxi_surface"] and not p["is_runway_surface"]

    def _taxi_candidate(p):
        return (p["filled"] and p["is_near_black"]
                and not p["is_taxi_surface"] and not p["is_runway_surface"])

    runway_tokens = [t for t in text_tokens if RUNWAY_RE.match(t["text"])]
    taxi_tokens = [t for t in text_tokens if TAXIWAY_RE.match(t["text"])]
    print(f"[charbox] {len(runway_tokens)} runway-pattern tokens, "
          f"{len(taxi_tokens)} taxi-pattern tokens")

    claimed_polys: set[int] = set()
    claimed_token_keys: set = set()

    runway_label_idx = _run_match(
        "runway labels (step 2)", runway_tokens, runway_surfaces,
        _runway_candidate, claimed_polys, claimed_token_keys,
    )
    taxi_label_idx = _run_match(
        "taxi labels", taxi_tokens, taxi_surfaces,
        _taxi_candidate, claimed_polys, claimed_token_keys,
    )

    # --- 4) Step 3: extended runway centerline matching.
    # For each runway surface, compute its principal axis and extend the
    # centerline past both endpoints by RUNWAY_CENTERLINE_PAD_PT. Any
    # filled, unclaimed, non-surface polygon whose bbox is intersected by
    # this segment becomes a runway label.
    centerlines = []
    for s in runway_surfaces:
        axis = _runway_axis_from_subpaths(s["subpaths"], s["rect"])
        ext = axis["half_len"] + RUNWAY_CENTERLINE_PAD_PT
        p1 = (axis["cx"] - axis["ux"] * ext, axis["cy"] - axis["uy"] * ext)
        p2 = (axis["cx"] + axis["ux"] * ext, axis["cy"] + axis["uy"] * ext)
        centerlines.append((p1, p2))

    step3_idx: list[int] = []
    for i, p in enumerate(all_polys):
        if i in claimed_polys:
            continue
        if p["is_taxi_surface"] or p["is_runway_surface"]:
            continue
        if not p["filled"]:
            continue
        x0p, y0p, x1p, y1p = p["rect"]
        for seg_p1, seg_p2 in centerlines:
            if _segment_intersects_bbox(seg_p1, seg_p2, x0p, y0p, x1p, y1p):
                claimed_polys.add(i)
                step3_idx.append(i)
                break
    print(f"[charbox] runway labels (step 3 centerline): "
          f"{len(step3_idx)} polygons claimed")

    runway_label_idx = runway_label_idx + step3_idx
    runway_label_layer = [all_polys[i] for i in runway_label_idx]
    taxi_label_layer = [all_polys[i] for i in taxi_label_idx]
    polygons_layer = [
        p for i, p in enumerate(all_polys)
        if i not in claimed_polys
        and not p["is_taxi_surface"] and not p["is_runway_surface"]
    ]
    print(f"[charbox] Layers — Taxiways: {len(taxi_surfaces)}  "
          f"Runways: {len(runway_surfaces)}  "
          f"Polygons: {len(polygons_layer)}  "
          f"Taxi Labels: {len(taxi_label_layer)}  "
          f"Runway Labels: {len(runway_label_layer)}")

    # --- 4) Emit SVG
    svg_attrs = {
        "xmlns": "http://www.w3.org/2000/svg",
        "xmlns:inkscape": "http://www.inkscape.org/namespaces/inkscape",
        "width": f"{page_w}",
        "height": f"{page_h}",
        "viewBox": f"0 0 {page_w} {page_h}",
    }
    svg = ET.Element("svg", svg_attrs)
    ET.SubElement(svg, "rect", {
        "x": "0", "y": "0",
        "width": str(page_w), "height": str(page_h),
        "fill": "#ffffff",
    })

    layers = [
        ("Taxiways", taxi_surfaces, None),
        ("Runways", runway_surfaces, None),
        ("Polygons", polygons_layer, None),
        ("Taxiway Labels", taxi_label_layer, TAXI_LABEL_FILL),
        ("Runway Labels", runway_label_layer, RUNWAY_LABEL_FILL),
    ]
    for layer_name, paths, override_fill in layers:
        if not paths:
            continue
        g = ET.SubElement(svg, "g", {
            "id": layer_name,
            "inkscape:label": layer_name,
            "inkscape:groupmode": "layer",
        })
        for p in paths:
            attrs = {"d": p["d"], "fill": override_fill or p["fill"] or "none"}
            if p["stroke"]:
                attrs["stroke"] = override_fill or p["stroke"]
                attrs["stroke-width"] = f"{p['stroke_width']:.3f}"
            ET.SubElement(g, "path", attrs)

    if text_tokens:
        # Fixed font-size; each <text> anchored at bbox center so (x, y)
        # IS the analyzable center of the original PDF token.
        g = ET.SubElement(svg, "g", {
            "id": "PDF Text Tokens",
            "inkscape:label": "PDF Text Tokens",
            "inkscape:groupmode": "layer",
            "fill": "#cc00cc",
            "font-family": "Helvetica, Arial, sans-serif",
            "font-size": "4",
            "text-anchor": "middle",
            "dominant-baseline": "middle",
        })
        for t in text_tokens:
            te = ET.SubElement(g, "text", {
                "x": f"{t['cx']:.3f}",
                "y": f"{t['cy']:.3f}",
            })
            te.text = t["text"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(svg).write(args.out, encoding="utf-8", xml_declaration=True)
    print(f"[charbox] wrote -> {args.out}")


if __name__ == "__main__":
    main()
