"""Tests for the central config loader + schema validation."""
import pytest

from src import config_loader as cl
from src.config_loader import CONFIG_DIR, CONFIG_FILES, ConfigError

# Sources approved for collection.  Exactly these may be enabled; every
# enabled source must be verified, and verified sources must be enabled.
# Canonical definition lives in test_sources.py.
from test_sources import APPROVED_ENABLED


def test_every_expected_config_loaded(config):
    for key in CONFIG_FILES:
        assert key.replace(".json", "") in config, key


def test_core_pipeline_settings(config):
    cfg = config["config"]
    assert cfg["english_only"] is True
    assert cfg["translate_non_english"] is False
    assert cfg["min_score_to_queue"] > 0
    assert cfg["min_india_relevance"] >= 0
    assert cfg["telegram"]["channel_id"] == ""


def test_source_registry_shape(config):
    sources = config["sources"]["sources"]
    assert len(sources) >= 40
    ids = {s["id"] for s in sources}
    assert len(ids) == len(sources), "duplicate source ids"
    for s in sources:
        assert isinstance(s["verified"], bool), f"{s['id']} verified must be boolean"
        assert "verified" in s and "enabled" in s, f"{s['id']} missing flags"
    assert ids >= APPROVED_ENABLED, "approved sources missing"

    enabled = {s["id"] for s in sources if s["enabled"]}
    assert enabled == APPROVED_ENABLED, (
        f"enabled sources {sorted(enabled)} != approved "
        f"{sorted(APPROVED_ENABLED)}"
    )
    for s in sources:
        if s["enabled"]:
            assert s["verified"] is True, f"{s['id']} enabled but unverified"


def test_categories_shape(config):
    cats = config["categories"]["categories"]
    assert len(cats) >= 15
    ids = {c["id"] for c in cats}
    assert len(ids) == len(cats), "duplicate category ids"


def test_editorial_shape(config):
    editorial = config["editorial"]
    for key in (
        "clickbait_patterns", "gossip_terms", "celebrity_terms",
        "astrology_terms", "rumour_terms", "routine_official_patterns",
        "sports_minor_patterns", "opinion_markers", "filler_phrases",
        "excluded_topics",
    ):
        assert editorial[key], f"editorial.{key} must not be empty"


def test_all_source_categories_exist(config):
    cat_ids = {c["id"] for c in config["categories"]["categories"]}
    for s in config["sources"]["sources"]:
        for c in s["categories"]:
            assert c in cat_ids, f"{s['id']} references unknown category {c}"


@pytest.mark.parametrize("file_name", list(CONFIG_FILES))
def test_each_config_validates_against_its_schema(file_name):
    data = cl._load_json(CONFIG_DIR / file_name)
    schema = cl._load_json(CONFIG_DIR / "schemas" / CONFIG_FILES[file_name])
    cl._validate(data, schema, file_name)  # should not raise


def test_bad_value_raises_validation_error():
    data = {"english_only": "yes"}
    schema = cl._load_json(CONFIG_DIR / "schemas" / "config.schema.json")
    with pytest.raises(ConfigError):
        cl._validate(data, schema, "config.json")
