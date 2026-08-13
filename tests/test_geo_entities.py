"""Tests for the India geo and entity config."""


def test_all_states_present(config):
    ent = config["india_entities"]
    assert len(ent["states"]) == 28, "expected 28 states"
    assert "telangana" in ent["states"]
    assert "delhi" in ent["union_territories"]


def test_geo_scope(config):
    geo = config["india_geo"]
    assert geo["phase_1_scope"] == "india-wide"
    assert geo["include_state_only_stories"] is False
    assert geo["allow_state_with_national_significance"] is True
    assert geo["scope_labels"] == ["national", "state", "local"]


def test_geo_markers(config):
    geo = config["india_geo"]
    assert "prime minister" in geo["national_significance_markers"]


def test_phase2_states_disabled(config):
    geo = config["india_geo"]
    for state, block in geo["phase_2_states"].items():
        assert block["enabled"] is False, f"{state} must start disabled"


def test_aliases_map_to_canonical(config):
    ent = config["india_entities"]
    for alias, canonical in ent["entity_aliases"].items():
        assert canonical and isinstance(canonical, str)
