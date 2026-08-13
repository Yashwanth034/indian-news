"""Tests for the source registry."""

# The approved six-source validation configuration.  Exactly these
# sources may be enabled; every other source must stay disabled and
# all sources must stay unverified.
APPROVED_ENABLED = {
    "the-hindu",
    "indian-express",
    "ndtv",
    "economic-times",
    "bbc-india",
    "media-nama",
}


def test_source_enablement_matches_approved_set(config):
    by_id = {s["id"]: s for s in config["sources"]["sources"]}
    assert set(by_id) >= APPROVED_ENABLED, "approved sources missing"

    enabled = {s["id"] for s in config["sources"]["sources"] if s["enabled"]}
    assert enabled == APPROVED_ENABLED, (
        f"enabled sources {sorted(enabled)} != approved "
        f"{sorted(APPROVED_ENABLED)}"
    )


def test_all_sources_unverified(config):
    for s in config["sources"]["sources"]:
        assert s["verified"] is False, f"{s['id']} must start unverified"


def test_source_ids_unique(config):
    ids = [s["id"] for s in config["sources"]["sources"]]
    assert len(set(ids)) == len(ids)


def test_discovery_sources_flagged_correctly(config):
    for s in config["sources"]["sources"]:
        if s["method"] == "discovery":
            assert s["discovery"] is True, f"{s['id']} should be flagged discovery"
            assert s["tier"] == 4, f"{s['id']} should be tier 4"
        elif s["discovery"]:
            assert s["method"] == "discovery", f"{s['id']} inconsistent discovery flag"


def test_allowlisted_discovery_source_restricts_domains(config):
    allowlisted = [s for s in config["sources"]["sources"]
                   if s["method"] == "discovery" and s["allow_domains"]]
    assert allowlisted, "expected at least one allowlisted discovery source"
    for s in allowlisted:
        assert all(d for d in s["allow_domains"])


def test_primary_official_sources_present(config):
    official = [s for s in config["sources"]["sources"] if s["type"] == "official"]
    assert official, "no official sources"
    for s in official:
        assert s["primary"] is True, f"{s['id']} should be primary"
        assert s["news"] is True, f"{s['id']} should be a news source"
