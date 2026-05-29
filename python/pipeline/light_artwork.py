"""light_artwork.py — runtime loader for the normalized approach-light artwork
library (artwork/lights/lights.json), produced offline by build_light_artwork.

The library is already normalized to the canonical frame (attach point at the
origin, forward = +X) and all symbols share one `native_width` (the runway-
width reference). Runtime does NO SVG/.ai parsing — it reads JSON and caches.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class LightArtwork:
    native_width: float
    symbols: dict  # {key: {"paths": [{"d", "fill", "stroke", "stroke_width"}]}}


@lru_cache(maxsize=8)
def load_light_artwork(json_path: Path) -> LightArtwork:
    """Load and cache the artwork library. Raises ValueError if native_width
    is missing or non-positive (an unbuilt/corrupt library)."""
    data = json.loads(Path(json_path).read_text())
    native_width = data.get("native_width")
    if not native_width or float(native_width) <= 0.0:
        raise ValueError(
            f"lights.json missing positive native_width: {json_path}"
        )
    return LightArtwork(
        native_width=float(native_width),
        symbols=dict(data.get("symbols") or {}),
    )
