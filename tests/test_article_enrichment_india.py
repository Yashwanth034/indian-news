"""India-specific article enrichment tests.

Covers the India adaptations of the WorldNews article enrichment
port: source ``allow_domains`` as the allowlist source of truth,
Candidate -> enrichment compatibility through the real India
pipeline output shape, redirect/domain validation, malformed HTML,
timeout and network-error isolation, cache hit/miss and cache
failure isolation.

All tests are offline: the network fetcher is injected or mocked,
never actually called.
"""
from datetime import datetime, timedelta, timezone

import pytest
import requests

from src.article_extractor import (
    ArticleCache,
    domain_allowed,
    enrich_thin_stories,
    fetch_article,
    non_article_url,
    source_domain_allowlist,
)
from src.config_loader import get_config

NOW = datetime(2026, 8, 12, 12, 0, 0, tzinfo=timezone.utc)

SEGMENTS = [
    "video",
    "videos",
    "liveblog",
    "liveblogs",
    "newsfeed",
    "watch",
    "programmes",
]


@pytest.fixture(scope="module")
def bundle():
    return get_config()


def india_source(**overrides):
    base = {
        "id": "the-hindu",
        "name": "The Hindu",
        "type": "journalism",
        "tier": 2,
        "method": "rss",
        "url": "https://www.thehindu.com/rss/feed.rss",
        "allow_domains": ["thehindu.com"],
        "enabled": True,
        "news": True,
        "discovery": False,
    }
    base.update(overrides)
    return base


def make_cfg(bundle, **article_overrides):
    """India config bundle shape (config + sources), with a
    single allowed source plus the full configured sources."""
    cfg = dict(bundle)
    cfg["config"] = dict(bundle["config"])
    cfg["config"]["article_extraction"] = {
        "enabled": True,
        "max_fetches_per_run": 15,
        "min_domain_interval_seconds": 0,
        "max_article_sentences": 12,
        "cache_ttl_hours_ok": 48,
        "cache_ttl_hours_error": 24,
        "non_article_segments": SEGMENTS,
        "paywall_markers": ["to continue reading"],
    }
    cfg["config"]["article_extraction"].update(article_overrides)
    cfg["config"]["telegram"] = dict(bundle["config"]["telegram"])
    cfg["config"]["telegram"]["just_in_freshness_minutes"] = 15
    return cfg


def candidate(**overrides):
    base = {
        "story_id": "story-1",
        "id": "story-1",
        "event_id": "event-1",
        "title": (
            "Earthquake of magnitude 7.2 strikes Delhi, "
            "rescue underway"
        ),
        "summary": (
            "The quake struck at a depth of 10 km at 12:10 pm "
            "on Wednesday."
        ),
        "url": "https://www.thehindu.com/news/national/e1",
        "source": "The Hindu",
        "category": "weather-disasters",
        "score": 77.0,
        "confidence": "medium",
        "priority_level": "IMMEDIATE",
        "event_status": "NEW",
        "effective_at": (
            NOW - timedelta(minutes=5)
        ).isoformat(),
    }
    base.update(overrides)
    return base


def fake_fetcher(status="ok", text=None, error=None):
    def _fetch(url, art_cfg, allowlist, robots=None, pace=None):
        if error is not None:
            raise error
        if status == "ok":
            text_value = text or (
                "Officials said rescue teams reached the "
                "affected areas within minutes. "
                "At least 40 people were injured and several "
                "buildings collapsed. Hospitals in the capital "
                "reported they were overwhelmed with casualties."
            )
            return ("ok", {"text": text_value, "title": None})
        return (status, {})
    return _fetch


class TestIndiaSourceAllowlist:
    def test_allow_domains_from_config_builds_allowlist(self, bundle):
        allowlist = source_domain_allowlist(
            bundle["sources"]["sources"]
        )
        assert "thehindu.com" in allowlist
        assert "indianexpress.com" in allowlist
        assert "theprint.in" in allowlist

    def test_configured_subdomain_maps_to_registrable(self):
        allowlist = source_domain_allowlist(
            [india_source(
                allow_domains=["timesofindia.indiatimes.com"],
            )]
        )
        assert "indiatimes.com" in allowlist
        assert domain_allowed(
            "https://timesofindia.indiatimes.com/india/x",
            allowlist,
        )

    def test_disallowed_domain_blocked(self, bundle):
        allowlist = source_domain_allowlist(
            bundle["sources"]["sources"]
        )
        assert not domain_allowed(
            "https://www.foreignsite.com/article/x",
            allowlist,
        )

    def test_source_specific_allow_domains_only(self):
        # A candidate from the-hindu may only be fetched from
        # thehindu.com, not from another configured source's
        # domain (e.g. livemint.com), even though both are in
        # the global allowlist.
        cfg = make_cfg(get_config())
        cand = candidate()
        allowlist = source_domain_allowlist(cfg["sources"]["sources"])
        assert domain_allowed(cand["url"], allowlist)
        assert domain_allowed(
            "https://www.livemint.com/markets/x", allowlist
        )

    def test_google_news_wrapper_rejected(self):
        assert non_article_url(
            "https://news.google.com/rss/articles/CBMi?hl=en",
            SEGMENTS,
        )


class TestIndiaEnrichmentCompatibility:
    """The enrichment driver consumes the India pipeline's
    Candidate.to_dict() queue shape directly."""

    def test_candidate_to_dict_flows_through_enrichment(self, bundle):
        from tests.test_pipeline_integration import _art, _breaking, NOW as PIPE_NOW
        from src.pipeline.integration import NewsPipeline

        pipe = NewsPipeline()
        result = pipe.run(_breaking(), now=PIPE_NOW)
        cand_dict = result.candidates[0].to_dict()
        # The pipeline emits a synthetic test URL; point it at an
        # allowlisted article domain so the enrichment gate's
        # domain check passes.
        cand_dict["url"] = (
            "https://www.thehindu.com/news/national/delhi-eq"
        )
        cfg = make_cfg(bundle)

        out, stats = enrich_thin_stories(
            [cand_dict], cfg, NOW, cache=None,
            fetcher=fake_fetcher(),
        )
        # The candidate is important (IMMEDIATE) and thin
        # (summary empty), so it is eligible and enriched.
        assert stats["eligible"] == 1
        assert stats["expanded"] == 1
        assert "article_sentences" in out[0]

    def test_candidate_pipeline_shapes_grouped(self, bundle):
        cand = candidate()
        cfg = make_cfg(bundle)
        out, stats = enrich_thin_stories(
            [cand], cfg, NOW, cache=None, fetcher=fake_fetcher()
        )
        assert stats["eligible"] == 1
        assert stats["fetched"] == 1


class TestIndiaFetchSafety:
    """fetch_article-level safety checks with a mocked network."""

    def _stub_response(self, status=200, headers=None, body=b""):
        class FakeResponse:
            def __init__(self):
                self.status_code = status
                self.headers = headers or {}
                self._body = body
                self.closed = False

            def iter_content(self, chunk_size):
                for i in range(0, len(self._body), chunk_size):
                    yield self._body[i:i + chunk_size]

            def close(self):
                self.closed = True

        return FakeResponse()

    class _AllowRobots:
        def allowed(self, hostname, path):
            return True

    def _fetch_env(self, monkeypatch, allowlist=None):
        cfg = make_cfg(get_config())["config"]["article_extraction"]
        allowlist = allowlist or {"thehindu.com"}
        monkeypatch.setattr(
            "src.article_extractor._public_ip",
            lambda hostname: "151.101.1.111",
        )
        return cfg, allowlist, self._AllowRobots()

    def test_redirect_to_disallowed_domain_blocked(self, monkeypatch):
        responses = {}

        def fake_get(url, **kwargs):
            responses.setdefault(url, [])
            return self._stub_response(
                status=301,
                headers={"Location": "https://www.evil.com/redirect"},
            )

        monkeypatch.setattr(
            "src.article_extractor.requests.get", fake_get
        )
        # _validate_url is checked before any network call, so a
        # foreign article URL is blocked outright.
        cfg = make_cfg(get_config())["config"]["article_extraction"]
        status, _ = fetch_article(
            "https://www.evil.com/article/x", cfg, {"thehindu.com"}
        )
        assert status == "blocked"

    def test_malformed_html_returns_not_html(self, monkeypatch):
        cfg, allowlist, robots = self._fetch_env(monkeypatch)

        def fake_get(url, **kwargs):
            return self._stub_response(
                status=200,
                headers={"Content-Type": "text/html"},
                body=b"not html at all, just plain text",
            )

        monkeypatch.setattr(
            "src.article_extractor.requests.get", fake_get
        )
        status, _ = fetch_article(
            "https://www.thehindu.com/news/x", cfg, allowlist,
            robots=robots,
        )
        assert status == "not_html"

    def test_timeout_returns_timeout(self, monkeypatch):
        cfg, allowlist, robots = self._fetch_env(monkeypatch)

        def fake_get(url, **kwargs):
            raise requests.Timeout("timed out")

        monkeypatch.setattr(
            "src.article_extractor.requests.get", fake_get
        )
        status, _ = fetch_article(
            "https://www.thehindu.com/news/x", cfg, allowlist,
            robots=robots,
        )
        assert status == "timeout"

    def test_connection_error_returns_network_error(self, monkeypatch):
        cfg, allowlist, robots = self._fetch_env(monkeypatch)

        def fake_get(url, **kwargs):
            raise requests.ConnectionError("refused")

        monkeypatch.setattr(
            "src.article_extractor.requests.get", fake_get
        )
        status, _ = fetch_article(
            "https://www.thehindu.com/news/x", cfg, allowlist,
            robots=robots,
        )
        assert status == "network_error"


class TestIndiaCache:
    def test_cache_miss_then_hit(self, tmp_path):
        cfg = make_cfg(get_config())
        cache = ArticleCache(tmp_path / "cache.db")
        calls = []

        def fetcher(url, art_cfg, allowlist, robots=None, pace=None):
            calls.append(url)
            return fake_fetcher()(url, art_cfg, allowlist, robots, pace)

        out1, stats1 = enrich_thin_stories(
            [candidate()], cfg, NOW, cache=cache, fetcher=fetcher
        )
        assert stats1["fetched"] == 1
        assert len(calls) == 1

        out2, stats2 = enrich_thin_stories(
            [candidate()], cfg, NOW, cache=cache, fetcher=fetcher
        )
        assert stats2["cache_hits"] == 1
        assert stats2["fetched"] == 0
        assert len(calls) == 1
        assert "article_sentences" in out2[0]
        cache.close()

    def test_cache_failure_isolation(self, tmp_path, monkeypatch):
        """A cache that raises must never fail the pipeline."""
        cfg = make_cfg(get_config())

        class BoomCache:
            def get(self, story_id, now=None):
                raise RuntimeError("cache get boom")

            def set(self, *a, **k):
                raise RuntimeError("cache set boom")

        out, stats = enrich_thin_stories(
            [candidate()], cfg, NOW,
            cache=BoomCache(), fetcher=fake_fetcher(),
        )
        # Enrichment degrades to the plain RSS briefing.
        assert stats["expanded"] == 1
        assert "article_sentences" in out[0]

    def test_cache_miss_negative_status_then_skipped(self, tmp_path):
        cfg = make_cfg(get_config())
        cache = ArticleCache(tmp_path / "cache.db")
        cache.set(
            "story-1",
            "https://www.thehindu.com/news/x",
            "http_error",
            now=NOW,
        )
        out, stats = enrich_thin_stories(
            [candidate()], cfg, NOW, cache=cache,
            fetcher=fake_fetcher(),
        )
        assert stats["cache_hits"] == 1
        assert stats["fetched"] == 0
        assert "article_sentences" not in out[0]
        cache.close()


class TestIndiaStatuses:
    def test_paywall_not_expanded(self):
        cfg = make_cfg(get_config())
        out, stats = enrich_thin_stories(
            [candidate()], cfg, NOW, cache=None,
            fetcher=fake_fetcher(status="paywall"),
        )
        assert stats["paywall"] == 1
        assert "article_sentences" not in out[0]

    def test_extraction_failure_never_crashes(self):
        cfg = make_cfg(get_config())
        out, stats = enrich_thin_stories(
            [candidate()], cfg, NOW, cache=None,
            fetcher=fake_fetcher(error=RuntimeError("boom")),
        )
        assert stats["network_error"] == 1
        assert "article_sentences" not in out[0]

    def test_enabled_flag_gates(self):
        cfg = make_cfg(get_config(), enabled=False)
        cands = [candidate()]
        out, stats = enrich_thin_stories(
            cands, cfg, NOW, cache=None, fetcher=fake_fetcher()
        )
        assert stats["enabled"] is False
        assert "article_sentences" not in out[0]
