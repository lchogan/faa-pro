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
