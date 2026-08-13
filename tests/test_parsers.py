"""Tests for the parsers: RSS, JSON/API, page listing."""
import json
from pathlib import Path

import pytest

from src.ingest.parsers import ParseError, parse_for_method

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name):
    return (FIXTURES / name).read_bytes()


def _rss_source():
    return {"id": "test-rss", "method": "rss", "url": "https://example.com/feed"}


def test_parse_rss_valid_feed():
    items = parse_for_method("rss", _load("sample.rss"), _rss_source())
    assert len(items) == 2
    assert items[0]["title"] == "India launches new communications satellite"
    assert items[0]["published"]  # raw timestamp string kept for later normalization
    assert items[0]["category_hints"] == ["Science"]
    assert items[1]["author"] is None  # missing author stays None


def test_parse_rss_keeps_summary_and_url():
    items = parse_for_method("rss", _load("sample.rss"), _rss_source())
    assert "communications satellite" in items[0]["summary"]
    assert items[0]["url"].startswith("https://example.com/stories/")


def test_parse_rss_malformed_raises():
    with pytest.raises(ParseError):
        parse_for_method("rss", _load("malformed.rss"), _rss_source())


def test_parse_rss_raises_on_non_xml():
    with pytest.raises(ParseError):
        parse_for_method("rss", b"this is not xml at all", _rss_source())


def test_parse_json_valid_response():
    source = {"id": "test-api", "method": "endpoint", "url": "https://api.example.com/list"}
    items = parse_for_method("endpoint", _load("api_items.json"), source)
    assert len(items) == 2
    assert items[0]["title"] == "SEBI tightens disclosure rules"
    assert items[0]["published"] == "2026-08-11T09:15:00+05:30"
    assert items[0]["category_hints"] == ["sebi", "markets"]


def test_parse_json_author_missing_is_none():
    source = {"id": "test-api", "method": "endpoint", "url": "https://api.example.com/list"}
    items = parse_for_method("endpoint", _load("api_items.json"), source)
    assert items[1]["author"] is None


def test_parse_json_malformed_raises():
    source = {"id": "test-api", "method": "endpoint", "url": "https://api.example.com/list"}
    with pytest.raises(ParseError):
        parse_for_method("endpoint", b"{not json", source)


def test_parse_json_handles_top_level_list():
    source = {"id": "test-api", "method": "api", "url": "https://api.example.com/list"}
    payload = json.dumps(
        [{"title": "T1", "url": "https://example.com/1"}, {"title": "T2", "url": "https://example.com/2"}]
    ).encode()
    items = parse_for_method("api", payload, source)
    assert len(items) == 2


def test_parse_page_extracts_article_links_within_allow_domains():
    source = {
        "id": "test-page",
        "method": "page",
        "url": "https://www.example.gov.in/",
        "allow_domains": ["example.gov.in"],
    }
    items = parse_for_method("page", _load("listing.html"), source)
    assert len(items) == 2  # anchor links only, fragment/privacy excluded
    assert items[0]["title"] == "India launches new satellite"
    assert items[0]["url"] == "https://www.example.gov.in/releases/2026/08/10/satellite"


def test_parse_page_dedupes_repeat_titles_with_different_query():
    source = {
        "id": "test-page",
        "method": "page",
        "url": "https://www.example.gov.in/",
        "allow_domains": ["example.gov.in"],
    }
    items = parse_for_method("page", _load("listing.html"), source)
    urls = [i["url"] for i in items]
    assert len(urls) == len(set(urls))
    assert "print=true" not in " ".join(urls)


def test_parse_page_filters_out_of_domain_links():
    source = {
        "id": "test-page",
        "method": "page",
        "url": "https://www.example.gov.in/",
        "allow_domains": ["example.gov.in"],
    }
    payload = (
        b'<html><body><a href="https://other-site.com/x">X</a>'
        b'<a href="/releases/ok">OK</a></body></html>'
    )
    items = parse_for_method("page", payload, source)
    assert len(items) == 1
    assert items[0]["url"] == "https://www.example.gov.in/releases/ok"


def test_discovery_method_parses_as_rss():
    source = {"id": "google-news", "method": "discovery", "url": "https://news.google.com/rss"}
    items = parse_for_method("discovery", _load("sample.rss"), source)
    assert len(items) == 2


def test_unknown_method_raises():
    source = {"id": "x", "method": "carrier-pigeon", "url": "https://x.in"}
    with pytest.raises(ParseError):
        parse_for_method("carrier-pigeon", b"<rss>", source)
