"""Tests for the collector: isolation, dedupe, normalization, health."""
from datetime import datetime, timezone

import pytest

from src.ingest.collect import SourceResult, collect_sources
from src.ingest.fetch import FetchError, FetchTimeout
from src.ingest.health import HealthStore

RSS_OK = b"""<?xml version="1.0"?>
<rss version="2.0"><channel>
<item><title>Headline one</title><link>https://example.com/a?utm_source=rss</link>
<pubDate>Mon, 10 Aug 2026 06:30:00 +0530</pubDate><description>Summary one.</description></item>
<item><title>Headline two</title><link>https://example.com/b</link></item>
</channel></rss>"""

RSS_DUPES = b"""<?xml version="1.0"?>
<rss version="2.0"><channel>
<item><title>Same story</title><link>https://example.com/s</link></item>
<item><title>Same story again</title><link>https://example.com/s?utm_source=rss</link></item>
</channel></rss>"""


def _source(sid, method="rss", url="https://example.com/feed", enabled=True):
    return {
        "id": sid,
        "name": sid.title(),
        "type": "journalism",
        "tier": 2,
        "method": method,
        "url": url,
        "categories": ["national-politics"],
        "language": "en",
        "primary": False,
        "news": True,
        "discovery": False,
        "allow_domains": ["example.com"],
        "enabled": enabled,
        "verified": False,
        "settings": {},
    }


def _fetcher(contents, errors=None):
    errors = errors or {}

    def _get(url, timeout=None, headers=None):
        if url in errors:
            raise errors[url]
        return contents[url]

    return _get


def test_collect_builds_normalized_articles():
    now = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)
    source = _source("s1")
    report = collect_sources([source], fetcher=_fetcher({source["url"]: RSS_OK}), now=now)
    assert report.status("s1") == "ok"
    articles = report.articles
    assert len(articles) == 2
    a = articles[0]
    assert a.source_id == "s1"
    assert a.tier == 2
    assert a.url == "https://example.com/a"  # tracking stripped
    assert a.published == datetime(2026, 8, 10, 1, 0, tzinfo=timezone.utc)
    assert a.summary == "Summary one."
    assert a.fetched_at == now


def test_collect_dedupes_within_feed_by_canonical_url():
    source = _source("s2")
    report = collect_sources([source], fetcher=_fetcher({source["url"]: RSS_DUPES}))
    assert len(report.articles) == 1
    assert report.articles[0].title == "Same story"


def test_collect_isolates_failed_source():
    ok = _source("ok", url="https://example.com/ok")
    bad = _source("bad", url="https://example.com/bad")
    fetcher = _fetcher(
        {ok["url"]: RSS_OK},
        errors={bad["url"]: FetchTimeout("timeout")},
    )
    report = collect_sources([ok, bad], fetcher=fetcher)
    assert report.status("ok") == "ok"
    assert report.status("bad") == "error"
    assert len(report.articles) == 2  # good source still yields articles
    assert "bad" in report.failed_sources


def test_collect_handles_http_error():
    source = _source("s3")
    report = collect_sources(
        [source],
        fetcher=_fetcher({}, errors={source["url"]: FetchError("HTTP 500")}),
    )
    assert report.status("s3") == "error"
    assert report.articles == []


def test_collect_skips_disabled_sources():
    source = _source("disabled", enabled=False)

    def _explode(url, timeout=None, headers=None):
        raise AssertionError("disabled source must not be fetched")

    report = collect_sources([source], fetcher=_explode)
    assert report.status("disabled") is None
    assert report.articles == []


def test_collect_records_health_on_success_and_failure(tmp_path):
    ok = _source("ok", url="https://example.com/ok")
    bad = _source("bad", url="https://example.com/bad")
    store = HealthStore(tmp_path / "health.json")
    report = collect_sources(
        [ok, bad],
        fetcher=_fetcher(
            {ok["url"]: RSS_OK},
            errors={bad["url"]: FetchTimeout("timeout")},
        ),
        health=store,
    )
    assert store.get("ok").status == "ok"
    assert store.get("ok").items_found == 2
    assert store.get("bad").status == "error"
    assert store.get("bad").timeouts == 1
    assert store.get("bad").consecutive_failures == 1
    assert report.status("ok") == "ok"


def test_health_persists_and_reloads(tmp_path):
    store = HealthStore(tmp_path / "health.json")
    store.record_success("s1", items_found=3, fetched_at="2026-08-11T12:00:00Z")
    store.record_failure("s2", error="boom", timeout=True)
    store.save()

    reloaded = HealthStore(tmp_path / "health.json")
    reloaded.load()
    assert reloaded.get("s1").status == "ok"
    assert reloaded.get("s2").timeouts == 1
    assert reloaded.get("missing") is None
