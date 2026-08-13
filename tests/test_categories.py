"""Tests for the category taxonomy."""

EXPECTED_CATEGORIES = {
    "national-politics",
    "government-policy",
    "economy-finance",
    "business-industry",
    "defence-security",
    "technology-digital",
    "science-space",
    "weather-disasters",
    "courts-law",
    "health",
    "education",
    "jobs-employment",
    "transport-infrastructure",
    "sports",
    "international-affairs",
    "agriculture-rural",
    "energy-environment",
    "crime-public-safety",
}


def test_all_expected_categories_present(config):
    ids = {c["id"] for c in config["categories"]["categories"]}
    assert EXPECTED_CATEGORIES.issubset(ids)


def test_category_weights(config):
    by_id = {c["id"]: c for c in config["categories"]["categories"]}
    assert by_id["weather-disasters"]["importance_weight"] >= 1.0
    assert by_id["defence-security"]["importance_weight"] >= 1.0
    assert by_id["sports"]["importance_weight"] < 1.0
    assert by_id["sports"]["urgent"] is False
    assert by_id["weather-disasters"]["importance_weight"] > by_id["sports"]["importance_weight"]


def test_categories_have_terms(config):
    for c in config["categories"]["categories"]:
        assert len(c["terms"]) > 0, f"{c['id']} has no terms"
        assert c["default_cap_per_day"] >= 0
