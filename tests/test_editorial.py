"""Tests for the editorial gates."""


def test_editorial_blocks_low_value_content(config):
    e = config["editorial"]
    joined = " ".join(
        e["celebrity_terms"]
        + e["gossip_terms"]
        + e["astrology_terms"]
        + e["rumour_terms"]
    ).lower()
    for needle in ("bollywood", "gossip", "horoscope", "kundli"):
        assert needle in joined, f"editorial gates must block '{needle}'"


def test_editorial_blocks_clickbait_and_routine(config):
    e = config["editorial"]
    joined = " ".join(e["clickbait_patterns"] + e["routine_official_patterns"]).lower()
    for needle in ("shocking", "tender"):
        assert needle in joined, f"editorial gates must handle '{needle}'"


def test_editorial_markers_present(config):
    e = config["editorial"]
    assert e["opinion_markers"], "no opinion markers"
    assert e["filler_phrases"], "no filler phrases"
    assert e["sports_minor_patterns"], "no minor-sports patterns"
