"""
test_render_lights.py — Unit tests for emit_approach_lights in render_svg_layers.

Tests that approach-light symbols are correctly scaled, rotated, translated, and
injected into an "Approach Lights" SVG layer <g>.
"""
import xml.etree.ElementTree as ET

from render_svg_layers import emit_approach_lights
from pipeline.light_artwork import LightArtwork


def _artwork():
    return LightArtwork(
        native_width=100.0,
        symbols={"malsr": {"paths": [
            {"d": "M0 0 L10 0", "fill": "#000000", "stroke": "none", "stroke_width": 0.0},
        ]}},
    )


def test_emit_creates_layer_with_scaled_rotated_translated_symbol():
    svg = ET.Element("svg")
    groups = [{
        "artwork_key": "malsr", "x": 700.0, "y": 300.0,
        "dir_x": 1.0, "dir_y": 0.0, "target_width": 50.0,
    }]
    emit_approach_lights(svg, groups, _artwork())

    layer = svg.find("g")
    assert layer is not None
    assert layer.get("id") == "Approach Lights"
    sym = layer.find("g")
    # scale = target_width / native_width = 50/100 = 0.5; dir = (1,0).
    assert sym.get("transform") == "matrix(0.500000,0.000000,-0.000000,0.500000,700.0000,300.0000)"
    path = sym.find("path")
    assert path.get("d") == "M0 0 L10 0"
    assert path.get("fill") == "#000000"


def test_emit_skips_unknown_key_and_emits_nothing_for_empty():
    svg = ET.Element("svg")
    emit_approach_lights(svg, [], _artwork())
    assert svg.find("g") is None  # no layer when there are no lights

    svg2 = ET.Element("svg")
    emit_approach_lights(svg2, [{"artwork_key": "nope", "x": 0, "y": 0,
                                 "dir_x": 1.0, "dir_y": 0.0, "target_width": 50.0}],
                         _artwork())
    layer = svg2.find("g")
    assert layer is not None and layer.find("g") is None  # layer, no symbols
