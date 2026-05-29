from pipeline.approach_light_layout import (
    _reciprocal_designation,
    compute_approach_light_placements,
    DEFAULT_K,
)


def test_numeric_reciprocals_swap_lr_and_add_18():
    assert _reciprocal_designation("17R") == "35L"
    assert _reciprocal_designation("35L") == "17R"
    assert _reciprocal_designation("9L") == "27R"
    assert _reciprocal_designation("18") == "36"
    assert _reciprocal_designation("36") == "18"
    assert _reciprocal_designation("4C") == "22C"


def test_compass_reciprocals():
    assert _reciprocal_designation("NE") == "SW"
    assert _reciprocal_designation("N") == "S"
    assert _reciprocal_designation("W") == "E"


def test_unrecognized_returns_none():
    assert _reciprocal_designation("XYZ") is None


def _rect_runway(cx, cy, length, width):
    """Axis-aligned runway polygon: long axis along x, width along y."""
    hl, hw = length / 2.0, width / 2.0
    ring = [
        (cx - hl, cy - hw), (cx + hl, cy - hw),
        (cx + hl, cy + hw), (cx - hl, cy + hw),
    ]
    return {
        "cx": cx, "cy": cy,
        "rect": (cx - hl, cy - hw, cx + hl, cy + hw),
        "subpaths": [ring],
        "filled": True, "stroked": False, "stroke_width": 0.0,
        "is_taxi_surface": False, "is_near_black": False,
    }


def _diag(runway_idx, end_sign, token):
    return {
        "runway_idx": runway_idx, "end_sign": end_sign,
        "matched": token is not None, "token": token, "claimed": [],
    }


def test_places_one_symbol_per_end_with_als_oriented_and_scaled():
    # One horizontal runway, idx 0, length 400 width 40 -> half_wid 20.
    polys = [_rect_runway(500.0, 300.0, 400.0, 40.0)]
    step3 = [_diag(0, +1, "16"), _diag(0, -1, "34")]
    als = {"16": "MALSR", "34": "ALSF2"}
    groups, diags = compute_approach_light_placements(
        polys, step3, runway_indices=[0], als_by_end=als,
    )
    assert len(groups) == 2
    by_key = {g["artwork_key"]: g for g in groups}
    assert set(by_key) == {"malsr", "alsf2"}
    # Uniform target width = k * (2 * max_half_wid) = 0.5 * 40 = 20.
    for g in groups:
        assert g["target_width"] == DEFAULT_K * 40.0
    # Horizontal runway: outward unit vector is +/-(1, 0).
    assert {round(g["dir_x"]) for g in groups} == {-1, 1}
    assert all(round(g["dir_y"]) == 0 for g in groups)
    # Attach points sit beyond each end (|x - 500| > half_len 200) on the
    # centerline (y == 300).
    for g in groups:
        assert abs(g["x"] - 500.0) > 200.0
        assert abs(g["y"] - 300.0) < 1e-6


def test_scale_is_keyed_to_widest_runway_across_the_document():
    # Wide runway idx 0 (width 60) and narrow runway idx 1 (width 20).
    polys = [
        _rect_runway(500.0, 200.0, 400.0, 60.0),   # max half_wid = 30
        _rect_runway(500.0, 600.0, 400.0, 20.0),
    ]
    step3 = [_diag(1, +1, "09"), _diag(1, -1, "27")]   # lights only on narrow
    als = {"9": "MALSR", "27": "MALSR"}
    groups, _ = compute_approach_light_placements(
        polys, step3, runway_indices=[0, 1], als_by_end=als,
    )
    # target width uses the WIDEST runway (half_wid 30), not the narrow one.
    assert all(g["target_width"] == DEFAULT_K * 60.0 for g in groups)


def test_reciprocal_inference_when_one_end_unmatched():
    polys = [_rect_runway(500.0, 300.0, 400.0, 40.0)]
    step3 = [_diag(0, +1, "16"), _diag(0, -1, None)]   # only +1 matched
    als = {"34": "ALSF2"}                              # ALS on the recip end
    groups, _ = compute_approach_light_placements(
        polys, step3, runway_indices=[0], als_by_end=als,
    )
    assert [g["artwork_key"] for g in groups] == ["alsf2"]
    assert round(groups[0]["dir_x"]) == -1   # the -1 (reciprocal) end


def test_none_and_unmapped_codes_emit_no_symbol():
    polys = [_rect_runway(500.0, 300.0, 400.0, 40.0)]
    step3 = [_diag(0, +1, "16"), _diag(0, -1, "34")]
    als = {"16": "NONE", "34": "NSTD"}   # one NONE, one out-of-scope
    groups, _ = compute_approach_light_placements(
        polys, step3, runway_indices=[0], als_by_end=als,
    )
    assert groups == []


def test_runway_with_no_matched_number_is_skipped():
    polys = [_rect_runway(500.0, 300.0, 400.0, 40.0)]
    step3 = [_diag(0, +1, None), _diag(0, -1, None)]
    als = {"16": "MALSR"}
    groups, diags = compute_approach_light_placements(
        polys, step3, runway_indices=[0], als_by_end=als,
    )
    assert groups == []
    assert any(d.get("skipped") == "no_matched_number" for d in diags)
