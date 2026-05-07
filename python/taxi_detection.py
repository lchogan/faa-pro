"""
taxi_detection.py — rule-based taxiway-surface + taxiway-label detection.

This is the authoritative source for Taxiways and Taxiway Labels in the
new pipeline. The ML model is *not* used for these two classes.

Detection logic:
  - Taxiway surface: filled polygon with gray fill (~#cfcfcf, with
    leeway for chart variation). Detection of the gray fill itself
    happens in chart_scene.read_chart and is exposed via the
    `is_taxi_surface` flag on each polygon dict.
  - Taxiway label: a word token matching `^([A-Z][A-Z]?[0-9]{0,2}|[0-9])$`
    (e.g. "C", "C1", "A11", "3") whose bbox touches a taxi surface
    polygon. The K = len(token) nearest unclaimed near-black filled
    glyph polygons are claimed for that token.

The pipeline calls `match_taxi_labels` AFTER runway detection +
runway-label matching, so digit-glyph polygons that belong to a
runway designator are already claimed and unavailable to the
taxi-label K-nearest. This avoids the failure mode where APF runway
"5" sits over a taxiway and gets claimed as a taxi label, leaving
the digit-5 polygon unavailable when step 5 looks for runway labels.

Coordinates are PDF y-down (matching PyMuPDF). Convert to AI y-up at
the consumer boundary if needed.

Public:
    detect_taxi(pdf_path) -> dict with all_polys, clips,
        taxi_surface_indices, taxi_label_indices, text_tokens,
        page_w, page_h. Convenience wrapper that runs both steps.
    match_taxi_labels(all_polys, taxi_surfaces, text_tokens,
                      claimed_polys=None) -> list[int]
        Run only the K-nearest claim, given an externally-loaded
        scene and a set of polygon indices already claimed by other
        steps. Production pipeline calls this directly.
"""

from __future__ import annotations

import re
from pathlib import Path

from chart_scene import read_chart


TAXIWAY_RE = re.compile(r"^([A-Z][A-Z]?[0-9]{0,2}|[0-9])$")


def _point_in_subpaths(x: float, y: float,
                       subpaths: list[list[tuple[float, float]]]) -> bool:
    """Even-odd ray-cast test against a compound polygon's anchor rings."""
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


def _bbox_touches(t: dict, surfaces: list[dict]) -> bool:
    """True if the token's center sits inside any surface polygon.

    Earlier this was a 5-sample bbox-corners test; OGG showed it was
    too lenient — a runway-slope annotation like "UP" painted just off
    pavement still passed because its bbox corners reached into the
    adjacent taxiway. Real taxi-letter labels are painted *on* the
    pavement, so requiring the centroid to be inside is both more
    conservative and more semantically correct.
    """
    cx, cy = t["cx"], t["cy"]
    for s in surfaces:
        sx0, sy0, sx1, sy1 = s["rect"]
        if not (sx0 <= cx <= sx1 and sy0 <= cy <= sy1):
            continue
        if _point_in_subpaths(cx, cy, s["subpaths"] or []):
            return True
    return False


def match_taxi_labels(all_polys: list[dict],
                      taxi_surfaces: list[dict],
                      text_tokens: list[dict],
                      claimed_polys: set[int] | None = None,
                      excluded_token_ids: set[int] | None = None) -> list[int]:
    """For each token matching the taxi-label regex whose bbox touches
    a taxi-surface polygon, claim the K = len(token.text) nearest
    near-black filled polygons (excluding ones already in
    `claimed_polys`) as taxiway-label glyphs.

    Production pipeline passes a `claimed_polys` set that already
    contains taxi surfaces, runways, and runway labels — so a digit
    glyph that belongs to a runway designator is unavailable here.

    `excluded_token_ids` is a set of id(token) values for tokens
    already consumed by an earlier label step (e.g. runway-label
    matching). Each token can identify polygons at most once across
    the pipeline; without this, a runway designator that overlaps a
    taxi surface would re-qualify here and pull arrowheads or other
    near-black symbols into Taxiway Labels via the K-nearest claim.

    Returns the list of claimed indices in claim order.
    """
    claimed: set[int] = set(claimed_polys or [])
    excluded_ids: set[int] = set(excluded_token_ids or [])
    pattern_tokens = [
        t for t in text_tokens
        if TAXIWAY_RE.match(t["text"]) and id(t) not in excluded_ids
    ]
    qualifying = [t for t in pattern_tokens if _bbox_touches(t, taxi_surfaces)]

    def _is_taxi_candidate(p):
        return p["filled"] and p["is_near_black"] and not p["is_taxi_surface"]

    label_indices: list[int] = []
    for tok in qualifying:
        k = len(tok["text"])
        scored = []
        for i, p in enumerate(all_polys):
            if i in claimed or not _is_taxi_candidate(p):
                continue
            dx = tok["cx"] - p["cx"]
            dy = tok["cy"] - p["cy"]
            scored.append((dx * dx + dy * dy, i))
        scored.sort()
        for _, i in scored[:k]:
            claimed.add(i)
            label_indices.append(i)
    return label_indices


def detect_taxi(pdf_path: Path) -> dict:
    """Run rule-based taxiway-surface and taxiway-label detection on a
    single-page FAA airport diagram PDF. Convenience wrapper that
    runs surface detection + label matching back-to-back. The
    production pipeline calls `read_chart` and `match_taxi_labels`
    directly so it can interleave runway and runway-label claims
    between the two steps.
    """
    scene = read_chart(pdf_path)
    all_polys = scene["all_polys"]
    text_tokens = scene["text_tokens"]
    clips = scene["clips"]
    page_w = scene["page_w"]
    page_h = scene["page_h"]

    taxi_surface_indices = [i for i, p in enumerate(all_polys)
                            if p["is_taxi_surface"]]
    taxi_surfaces = [all_polys[i] for i in taxi_surface_indices]

    label_indices = match_taxi_labels(
        all_polys, taxi_surfaces, text_tokens,
        claimed_polys=set(taxi_surface_indices),
    )

    return {
        "all_polys": all_polys,
        "clips": clips,
        "taxi_surface_indices": taxi_surface_indices,
        "taxi_label_indices": label_indices,
        "text_tokens": text_tokens,
        "page_w": page_w,
        "page_h": page_h,
    }
