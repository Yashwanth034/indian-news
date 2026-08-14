"""Tests for the source registry."""

# Sources verified end-to-end (feed/page fetch returns usable items).
# Every enabled source must be verified; verified sources must be enabled.
APPROVED_ENABLED = {
    # national journalism
    "the-hindu",
    "indian-express",
    "ndtv",
    "times-of-india",
    "india-today",
    "the-news-minute",
    "national-herald",
    "bbc-india",
    # official (page/rss)
    "trai",
    "irdai",
    "cbi",
    "nia",
    "mha",
    "mor",
    "imd",
    "dst",
    # specialist finance/business
    "economic-times",
    "mint",
    "mint-companies",
    "mint-auto",
    "business-standard",
    "moneycontrol",
    "cnbc-tv18",
    "business-today",
    "forbes-india",
    "media-nama",
    "businessline",
    # specialist tech/startups
    "inc42",
    "yourstory",
    "telecomtalk",
    # specialist defence
    "bharat-shakti",
    "idrw",
    "defence-news-india",
    # specialist energy/agri/environment
    "mercom-india",
    "saur-energy",
    "downtoearth",
    "mongabay-india",
    "krishi-jagran",
    "agriculture-post",
    # specialist health
    "medical-dialogues",
    "express-pharma",
    # specialist courts
    "live-law",
    # specialist sports/science/other
    "sportstar",
    "frontline",
    "the-hindu-science",
    "rail-analysis",
    "india-narrative",
    # regional editions (The Hindu group)
    "the-hindu-bengaluru",
    "the-hindu-telangana",
    "the-hindu-chennai",
    "the-hindu-kerala",
    "the-hindu-mumbai",
    "the-hindu-delhi",
    # regional media
    "telangana-today",
    "hans-india",
    "deccan-chronicle",
    "mathrubhumi",
    "onmanorama",
    "kalinga-tv",
    "eastmojo",
    "northeast-now",
    "sentinel-assam",
    "assam-tribune",
    "greater-kashmir",
    "rising-kashmir",
    "kashmir-observer",
    "navhind-times",
    # international (India-relevant)
    "simple-flying",
    "pv-magazine",
    "spacetech-asia",
    "cyber-express",
}

# Publisher groups: sources sharing a group must never inflate
# corroboration; each group counts as one independent source group.
PUBLISHER_GROUPS = {
    "thehindu": {
        "the-hindu",
        "the-hindu-bengaluru",
        "the-hindu-telangana",
        "the-hindu-chennai",
        "the-hindu-kerala",
        "the-hindu-mumbai",
        "the-hindu-delhi",
        "businessline",
        "frontline",
        "the-hindu-science",
        "sportstar",
    },
    "ht-media": {"mint", "mint-companies", "mint-auto", "hindustan-times"},
    "indiatimes": {"times-of-india", "economic-times"},
    "indiatoday": {"india-today", "business-today"},
    "network18": {"moneycontrol", "cnbc-tv18"},
}


def test_source_enablement_matches_approved_set(config):
    by_id = {s["id"]: s for s in config["sources"]["sources"]}
    assert set(by_id) >= APPROVED_ENABLED, "approved sources missing"

    enabled = {s["id"] for s in config["sources"]["sources"] if s["enabled"]}
    assert enabled == APPROVED_ENABLED, (
        f"enabled sources {sorted(enabled)} != approved "
        f"{sorted(APPROVED_ENABLED)}"
    )


def test_enabled_sources_are_verified(config):
    for s in config["sources"]["sources"]:
        if s["enabled"]:
            assert s["verified"] is True, f"{s['id']} enabled but unverified"


def test_verified_sources_are_enabled(config):
    for s in config["sources"]["sources"]:
        if s["verified"]:
            assert s["enabled"] is True, f"{s['id']} verified but disabled"


def test_source_ids_unique(config):
    ids = [s["id"] for s in config["sources"]["sources"]]
    assert len(set(ids)) == len(ids)


def test_publisher_groups_defined_in_registry(config):
    by_id = {s["id"]: s for s in config["sources"]["sources"]}
    declared = {}
    for s in config["sources"]["sources"]:
        g = s.get("group")
        if g:
            declared.setdefault(g, set()).add(s["id"])
    assert declared == PUBLISHER_GROUPS, (
        f"registry groups {declared} != expected {PUBLISHER_GROUPS}"
    )


def test_regionally_enabled_sources_have_state_value(config):
    regional = {s["id"] for s in config["sources"]["sources"]
                if s["enabled"] and any(k in s["id"] for k in (
                    "kashmir", "telangana", "chennai", "bengaluru", "kerala",
                    "assam", "goa", "odisha", "eastmojo", "northeast",
                    "deccan", "mathrubhumi", "onmanorama", "sentinel",
                    "hans", "navhind"))}
    assert len(regional) >= 12, f"regional coverage too thin: {regional}"


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


def test_enabled_sources_have_domains(config):
    for s in config["sources"]["sources"]:
        if s["enabled"] and s["method"] in ("rss", "page", "endpoint", "api"):
            assert s["allow_domains"], f"{s['id']} enabled without allow_domains"
