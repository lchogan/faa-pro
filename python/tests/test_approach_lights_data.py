from pipeline.approach_lights_data import als_code_to_artwork_key


def test_mapped_codes_resolve_to_artwork_keys():
    assert als_code_to_artwork_key("ALSF2") == "alsf2"
    assert als_code_to_artwork_key("MALSR") == "malsr"
    assert als_code_to_artwork_key("SSALR") == "malsr"   # shares MALSR artwork
    assert als_code_to_artwork_key("ODALS") == "odals"


def test_code_matching_is_case_and_space_insensitive():
    assert als_code_to_artwork_key(" malsr ") == "malsr"
    assert als_code_to_artwork_key("Alsf1") == "alsf1"


def test_none_blank_and_unmapped_codes_return_none():
    assert als_code_to_artwork_key("NONE") is None
    assert als_code_to_artwork_key("") is None
    assert als_code_to_artwork_key(None) is None
    assert als_code_to_artwork_key("NSTD") is None   # out of scope -> no symbol
    assert als_code_to_artwork_key("RLLS") is None


from pathlib import Path
from pipeline.approach_lights_data import load_approach_lights

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "nasr_apt_rwy_end_sample.csv"


def test_load_returns_normalized_end_to_code_for_airport():
    result = load_approach_lights(FIXTURE, "den")
    # Designators are normalized the same way runway-number tokens are:
    # leading zeros stripped, L/R/C kept, compass uppercased.
    assert result == {"16": "MALSR", "34": "NONE", "8": "ALSF2", "26": "ODALS"}


def test_load_is_airport_case_insensitive_and_handles_compass_and_suffix():
    assert load_approach_lights(FIXTURE, "APF") == {"NE": "MALSF"}
    atl = load_approach_lights(FIXTURE, "atl")
    assert atl["9L"] == "MALSR"      # "09L" -> "9L"
    assert "27R" not in atl          # blank code row dropped


def test_load_unknown_airport_returns_empty():
    assert load_approach_lights(FIXTURE, "zzz") == {}
