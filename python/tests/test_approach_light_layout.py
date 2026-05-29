from pipeline.approach_light_layout import _reciprocal_designation


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
