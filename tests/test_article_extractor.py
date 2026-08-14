"""Unit tests for the article extraction module.

Run with:  .venv/bin/python -m pytest src/test_article_extractor.py -q

All tests are offline: the network fetcher is injected or
mocked, never actually called.
"""
import json
import time
from datetime import datetime, timedelta, timezone

import pytest

from src.article_extractor import (
    ArticleCache,
    USER_AGENT,
    _RobotCache,
    article_sentences,
    count_useful,
    domain_allowed,
    enrich_thin_stories,
    feed_domain_allowlist,
    non_article_url,
    _public_ip,
)

NOW = datetime(2026, 8, 9, 7, 3, 10, tzinfo=timezone.utc)

SEGMENTS = [
    "video",
    "videos",
    "liveblog",
    "liveblogs",
    "newsfeed",
    "watch",
    "programmes",
]


def make_cfg(**article_overrides):
    cfg = {
        "article_extraction": {
            "enabled": True,
            "max_fetches_per_run": 15,
            "min_domain_interval_seconds": 0,
            "max_article_sentences": 12,
            "cache_ttl_hours_ok": 48,
            "cache_ttl_hours_error": 24,
            "non_article_segments": SEGMENTS,
            "paywall_markers": ["to continue reading"],
        },
        "telegram": {"just_in_freshness_minutes": 15},
        "feeds": [
            {
                "name": "BBC World",
                "url": "https://feeds.bbci.co.uk/news/world/rss.xml",
            }
        ],
    }
    cfg["article_extraction"].update(article_overrides)
    return cfg


def candidate(**overrides):
    base = {
        "story_id": "story-1",
        "id": "story-1",
        "event_id": "event-1",
        "title": (
            "State of emergency declared as fast-moving "
            "Canada wildfire doubles in size"
        ),
        "summary": (
            "The Bald Range wildfire in British Columbia, "
            "still considered out of control, has spread over "
            "more than 36 sq miles."
        ),
        "url": "https://www.bbc.co.uk/news/articles/c1234",
        "source": "BBC World",
        "score": 70,
        "confidence": "medium",
        "priority_level": "HIGH",
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
                "Officials in British Columbia said the "
                "situation remained extremely dangerous. "
                "More than 20,000 residents have been forced "
                "from their homes so far. Fire crews are "
                "working to protect the town of Ashcroft."
            )
            return ("ok", {"text": text_value, "title": None})
        return (status, {})
    return _fetch


# ---------------------------------------------------------------------------
# URL classification
# ---------------------------------------------------------------------------

class TestNonArticleUrl:
    def test_plain_article_accepted(self):
        assert not non_article_url(
            "https://www.bbc.co.uk/news/articles/c1234",
            SEGMENTS,
        )

    def test_video_rejected(self):
        assert non_article_url(
            "https://www.aljazeera.com/video/xyz",
            SEGMENTS,
        )

    def test_videos_plural_rejected(self):
        assert non_article_url(
            "https://www.bbc.co.uk/news/videos/c987",
            SEGMENTS,
        )

    def test_liveblog_rejected(self):
        assert non_article_url(
            "https://www.aljazeera.com/news/liveblog/xyz",
            SEGMENTS,
        )

    def test_newsfeed_rejected(self):
        assert non_article_url(
            "https://www.aljazeera.com/newsfeed/xyz",
            SEGMENTS,
        )

    def test_programmes_rejected(self):
        assert non_article_url(
            "https://www.bbc.co.uk/programmes/p0001",
            SEGMENTS,
        )

    def test_watch_rejected(self):
        assert non_article_url(
            "https://www.bbc.co.uk/watch/live",
            SEGMENTS,
        )

    def test_google_news_wrapper_rejected(self):
        assert non_article_url(
            "https://news.google.com/rss/articles/CBMi?hl=en",
            SEGMENTS,
        )

    def test_missing_url_rejected(self):
        assert non_article_url(None, SEGMENTS)


class TestDomainAllowlist:
    def test_feeds_build_allowlist(self):
        allowlist = feed_domain_allowlist(make_cfg()["feeds"])
        assert "bbci.co.uk" in allowlist
        assert "bbc.co.uk" in allowlist

    def test_article_domain_allowed_via_alias(self):
        allowlist = feed_domain_allowlist(make_cfg()["feeds"])
        assert domain_allowed(
            "https://www.bbc.co.uk/news/articles/c1234",
            allowlist,
        )

    def test_foreign_domain_blocked(self):
        allowlist = feed_domain_allowlist(make_cfg()["feeds"])
        assert not domain_allowed(
            "https://www.evil-example.com/news/x",
            allowlist,
        )

    def test_empty_allowlist_blocks_everything(self):
        assert not domain_allowed(
            "https://www.bbc.co.uk/news/articles/c1",
            set(),
        )

    def test_non_http_scheme_blocked(self):
        allowlist = feed_domain_allowlist(make_cfg()["feeds"])
        assert not domain_allowed(
            "file:///etc/passwd",
            allowlist,
        )

    def test_no_feeds_yields_empty_allowlist(self):
        assert feed_domain_allowlist([]) == set()


class TestPublicIp:
    def test_loopback_rejected(self, monkeypatch):
        monkeypatch.setattr(
            "src.article_extractor.socket.getaddrinfo",
            lambda *a, **k: [
                (
                    2,
                    1,
                    6,
                    "",
                    ("127.0.0.1", 443),
                )
            ],
        )
        assert _public_ip("localhost") is None

    def test_private_rejected(self, monkeypatch):
        monkeypatch.setattr(
            "src.article_extractor.socket.getaddrinfo",
            lambda *a, **k: [
                (
                    2,
                    1,
                    6,
                    "",
                    ("10.0.0.5", 443),
                )
            ],
        )
        assert _public_ip("intranet.example") is None

    def test_link_local_rejected(self, monkeypatch):
        monkeypatch.setattr(
            "src.article_extractor.socket.getaddrinfo",
            lambda *a, **k: [
                (
                    2,
                    1,
                    6,
                    "",
                    ("169.254.10.10", 443),
                )
            ],
        )
        assert _public_ip("mystery.example") is None

    def test_public_accepted(self, monkeypatch):
        monkeypatch.setattr(
            "src.article_extractor.socket.getaddrinfo",
            lambda *a, **k: [
                (
                    2,
                    1,
                    6,
                    "",
                    ("151.101.1.111", 443),
                )
            ],
        )
        assert _public_ip("www.bbc.co.uk") == "151.101.1.111"


# ---------------------------------------------------------------------------
# Robots.txt fetching uses the configured USER_AGENT
# ---------------------------------------------------------------------------

class TestRobotCacheUserAgent:
    """The robots.txt request must use the project USER_AGENT, not
    urllib's default Python-urllib/x.x (which many source WAFs 403)."""

    def _stub_urlopen(self, monkeypatch, body, headers=None):
        captured = {}

        class _Response:
            def __init__(self):
                self.status = 200

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return body.encode("utf-8")

        def fake_urlopen(request, timeout=None):
            captured["user_agent"] = (
                request.get_header("User-agent")
                or request.get_header("User-Agent")
            )
            return _Response()

        monkeypatch.setattr(
            "src.article_extractor.urllib.request.urlopen",
            fake_urlopen,
        )
        return captured

    def test_robots_fetch_uses_configured_user_agent(self, monkeypatch):
        captured = self._stub_urlopen(
            monkeypatch,
            "User-agent: *\nDisallow: /private/\n",
        )
        cache = _RobotCache()
        assert cache.allowed("example.com", "/public/article") is True
        assert cache.allowed("example.com", "/private/x") is False
        assert captured["user_agent"] == USER_AGENT

    def test_robots_fetch_cached_per_host(self, monkeypatch):
        captured = self._stub_urlopen(
            monkeypatch,
            "User-agent: *\nDisallow: /\n",
        )
        cache = _RobotCache()
        assert cache.allowed("example.com", "/x") is False
        assert cache.allowed("example.com", "/y") is False
        assert cache.allowed("other.example", "/x") is False
        assert captured["user_agent"] == USER_AGENT


# ---------------------------------------------------------------------------
# Sentence helpers
# ---------------------------------------------------------------------------

class TestCountUseful:
    def test_rich_summary_counts_two(self):
        assert count_useful(
            "Officials said the fire had grown. "
            "Residents were told to evacuate.",
            "Canada wildfire doubles in size",
        ) == 2

    def test_headline_paraphrase_only_counts_zero(self):
        assert count_useful(
            "A fast-moving Canada wildfire doubles in size.",
            (
                "State of emergency declared as fast-moving "
                "Canada wildfire doubles in size"
            ),
        ) == 0

    def test_empty_summary_counts_zero(self):
        assert count_useful("", "Some headline") == 0

    def test_filler_dropped(self):
        assert count_useful(
            "This is a developing story.",
            "Some headline",
        ) == 0

    def test_duplicates_count_once(self):
        assert count_useful(
            "The fire has spread. The fire has spread.",
            "Some headline",
        ) == 1


class TestArticleSentences:
    def test_extracts_verbatim_sentences(self):
        out = article_sentences(
            "Officials in British Columbia said the situation "
            "remained extremely dangerous. More than 20,000 "
            "residents have been forced from their homes.",
            "Canada wildfire doubles in size",
        )
        assert len(out) == 2
        assert out[0].startswith("Officials in British Columbia")

    def test_headline_paraphrase_dropped(self):
        out = article_sentences(
            "A fast-moving Canada wildfire doubles in size. "
            "Evacuation orders were issued for three towns.",
            (
                "State of emergency declared as fast-moving "
                "Canada wildfire doubles in size"
            ),
        )
        assert len(out) == 1
        assert "Evacuation" in out[0]

    def test_filler_dropped(self):
        out = article_sentences(
            "This is a developing story. Officials spoke "
            "at a press conference.",
            "Some headline",
        )
        assert len(out) == 1

    def test_capped_at_max_sentences(self):
        body = "Sentence number one here. " * 30
        out = article_sentences(body, "Headline", max_sentences=12)
        assert len(out) <= 12

    def test_headline_line_dropped(self):
        out = article_sentences(
            "Helicopter crash kills pilot and crew member "
            "amid Utah wildfire battle\n"
            "Utah wildfire response continues despite loss "
            "of helicopter crew, with containment standing "
            "at 24 percent.",
            "Helicopter crash kills pilot and crew member "
            "amid Utah wildfire battle",
        )
        assert all(
            "amid Utah wildfire battle" not in s
            for s in out
        )

    def test_nav_junk_lines_dropped(self):
        out = article_sentences(
            "Officials confirmed the crash on Saturday.\n"
            "Recommended Stories\n"
            "list of 3 items\n"
            "- list 1 of 3Vance says US destroyed Iran's "
            "nuclear programme\n"
            "Related topics\n"
            "The bodies were recovered on Sunday.",
            "Some headline",
        )
        assert len(out) == 2
        assert all(
            "Recommended" not in s
            and "list of 3" not in s
            and "Related" not in s
            and "Vance" not in s
            for s in out
        )

    def test_section_header_line_dropped(self):
        out = article_sentences(
            "Serbian-made artillery shells\n"
            "The president left Belgrade on Sunday evening.",
            "Some headline",
        )
        assert len(out) == 1
        assert out[0].startswith("The president")

    def test_two_sentences_on_one_line_split(self):
        out = article_sentences(
            "The fire spread quickly. Crews arrived by "
            "dawn.",
            "Some headline",
        )
        assert len(out) == 2


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class TestArticleCache:
    def test_round_trip(self, tmp_path):
        cache = ArticleCache(tmp_path / "cache.db")
        cache.set(
            "story-1",
            "https://www.bbc.co.uk/news/articles/c1",
            "ok",
            text="body text",
            sentences=["s1", "s2"],
            now=NOW,
        )
        entry = cache.get("story-1", now=NOW)
        assert entry["status"] == "ok"
        assert entry["text"] == "body text"
        assert json.loads(entry["sentences_json"]) == [
            "s1",
            "s2",
        ]
        assert entry["ttl_hours"] == 48
        cache.close()

    def test_positive_ttl_expiry(self, tmp_path):
        cache = ArticleCache(tmp_path / "cache.db")
        cache.set(
            "story-1",
            "https://www.bbc.co.uk/news/articles/c1",
            "ok",
            sentences=["s1", "s2"],
            now=NOW,
        )
        later = NOW + timedelta(hours=49)
        assert cache.get("story-1", now=later) is None
        cache.close()

    def test_negative_ttl_shorter(self, tmp_path):
        cache = ArticleCache(tmp_path / "cache.db")
        cache.set(
            "story-1",
            "https://www.bbc.co.uk/news/articles/c1",
            "http_error",
            now=NOW,
        )
        entry = cache.get("story-1", now=NOW)
        assert entry["status"] == "http_error"
        assert entry["ttl_hours"] == 24
        later = NOW + timedelta(hours=25)
        assert cache.get("story-1", now=later) is None
        cache.close()

    def test_error_result_has_no_text(self, tmp_path):
        cache = ArticleCache(tmp_path / "cache.db")
        cache.set(
            "story-1",
            "https://www.bbc.co.uk/news/articles/c1",
            "blocked",
            text=None,
            sentences=None,
            now=NOW,
        )
        entry = cache.get("story-1", now=NOW)
        assert entry["text"] is None
        assert entry["sentences_json"] is None
        cache.close()

    def test_missing_entry_returns_none(self, tmp_path):
        cache = ArticleCache(tmp_path / "cache.db")
        assert cache.get("nope", now=NOW) is None
        cache.close()


# ---------------------------------------------------------------------------
# Enrichment driver (offline, injected fetcher)
# ---------------------------------------------------------------------------

class TestEnrichThinStories:
    def test_disabled_config_noop(self):
        cfg = make_cfg(enabled=False)
        cands = [candidate()]
        out, stats = enrich_thin_stories(
            cands, cfg, NOW, cache=None, fetcher=fake_fetcher()
        )
        assert out == cands
        assert stats["enabled"] is False
        assert "article_sentences" not in out[0]

    def test_unimportant_thin_story_not_fetched(self):
        cfg = make_cfg()
        cands = [
            candidate(score=50, priority_level="NORMAL")
        ]
        out, stats = enrich_thin_stories(
            cands, cfg, NOW, cache=None, fetcher=fake_fetcher()
        )
        assert stats["eligible"] == 0
        assert "article_sentences" not in out[0]

    def test_thin_important_story_fetched(self):
        cfg = make_cfg()
        cands = [candidate()]
        out, stats = enrich_thin_stories(
            cands, cfg, NOW, cache=None, fetcher=fake_fetcher()
        )
        assert stats["eligible"] == 1
        assert stats["fetched"] == 1
        assert stats["expanded"] == 1
        assert len(out[0]["article_sentences"]) >= 2

    def test_not_thin_story_skipped(self):
        cfg = make_cfg()
        cands = [
            candidate(
                summary=(
                    "Officials said the fire had grown. "
                    "Residents were told to evacuate. "
                    "The town of Ashcroft was threatened."
                )
            )
        ]
        out, stats = enrich_thin_stories(
            cands, cfg, NOW, cache=None, fetcher=fake_fetcher()
        )
        assert stats["thin"] == 0
        assert stats["fetched"] == 0

    def test_just_in_story_fetched(self):
        cfg = make_cfg()
        cands = [
            candidate(
                score=60,
                confidence="high",
                effective_at=(
                    NOW - timedelta(minutes=5)
                ).isoformat(),
            )
        ]
        out, stats = enrich_thin_stories(
            cands, cfg, NOW, cache=None, fetcher=fake_fetcher()
        )
        assert stats["eligible"] == 1
        assert stats["fetched"] == 1

    def test_update_story_fetched_at_low_score(self):
        cfg = make_cfg()
        cands = [
            candidate(
                score=55,
                priority_level="NORMAL",
                event_status="UPDATE",
            )
        ]
        out, stats = enrich_thin_stories(
            cands, cfg, NOW, cache=None, fetcher=fake_fetcher()
        )
        assert stats["eligible"] == 1

    def test_event_dedup_one_fetch_per_event(self):
        cfg = make_cfg()
        cands = [
            candidate(story_id="story-2", score=70),
            candidate(
                story_id="story-3",
                score=60,
                url="https://www.bbc.co.uk/news/articles/c999",
            ),
        ]
        out, stats = enrich_thin_stories(
            cands, cfg, NOW, cache=None, fetcher=fake_fetcher()
        )
        assert stats["fetched"] == 1
        assert "article_sentences" in out[0]
        assert "article_sentences" not in out[1]

    def test_non_article_url_never_fetched(self):
        cfg = make_cfg()
        cands = [
            candidate(
                url="https://www.aljazeera.com/video/xyz",
            )
        ]
        out, stats = enrich_thin_stories(
            cands, cfg, NOW, cache=None, fetcher=fake_fetcher()
        )
        assert stats["non_article"] == 1
        assert stats["fetched"] == 0

    def test_foreign_domain_never_fetched(self):
        cfg = make_cfg()
        cands = [
            candidate(
                url="https://www.foreignnews.example/story",
            )
        ]
        out, stats = enrich_thin_stories(
            cands, cfg, NOW, cache=None, fetcher=fake_fetcher()
        )
        assert stats["domain_blocked"] == 1
        assert stats["fetched"] == 0

    def test_cache_hit_skips_fetch(self, tmp_path):
        cfg = make_cfg()
        cache = ArticleCache(tmp_path / "cache.db")
        cache.set(
            "story-1",
            "https://www.bbc.co.uk/news/articles/c1234",
            "ok",
            sentences=["a", "b"],
            now=NOW,
        )
        out, stats = enrich_thin_stories(
            candidates=[candidate()],
            cfg=cfg,
            now_dt=NOW,
            cache=cache,
            fetcher=fake_fetcher(),
        )
        assert stats["cache_hits"] == 1
        assert stats["fetched"] == 0
        assert len(out[0]["article_sentences"]) == 2
        cache.close()

    def test_negative_cache_hit_skips_fetch(self, tmp_path):
        cfg = make_cfg()
        cache = ArticleCache(tmp_path / "cache.db")
        cache.set(
            "story-1",
            "https://www.bbc.co.uk/news/articles/c1234",
            "http_error",
            now=datetime.now(timezone.utc),
        )
        out, stats = enrich_thin_stories(
            candidates=[candidate()],
            cfg=cfg,
            now_dt=NOW,
            cache=cache,
            fetcher=fake_fetcher(),
        )
        assert stats["cache_hits"] == 1
        assert stats["fetched"] == 0
        assert "article_sentences" not in out[0]
        cache.close()

    def test_fetch_ok_persists_cache(self, tmp_path):
        cfg = make_cfg()
        cache = ArticleCache(tmp_path / "cache.db")
        out, stats = enrich_thin_stories(
            candidates=[candidate()],
            cfg=cfg,
            now_dt=NOW,
            cache=cache,
            fetcher=fake_fetcher(),
        )
        entry = cache.get("story-1", now=NOW)
        assert entry is not None
        assert entry["status"] == "ok"
        assert len(json.loads(entry["sentences_json"])) >= 2
        cache.close()

    def test_fetch_ok_but_too_few_sentences_not_expanded(self):
        cfg = make_cfg()
        out, stats = enrich_thin_stories(
            candidates=[candidate()],
            cfg=cfg,
            now_dt=NOW,
            cache=None,
            fetcher=fake_fetcher(
                text=(
                    "Officials in British Columbia confirmed "
                    "the fire had spread across the region."
                )
            ),
        )
        assert stats["expanded"] == 0
        assert stats["not_expanded"] == 1
        assert "article_sentences" not in out[0]

    def test_http_error_not_expanded(self):
        cfg = make_cfg()
        out, stats = enrich_thin_stories(
            candidates=[candidate()],
            cfg=cfg,
            now_dt=NOW,
            cache=None,
            fetcher=fake_fetcher(status="http_error"),
        )
        assert stats["http_error"] == 1
        assert "article_sentences" not in out[0]

    def test_paywall_not_expanded(self):
        cfg = make_cfg()
        out, stats = enrich_thin_stories(
            candidates=[candidate()],
            cfg=cfg,
            now_dt=NOW,
            cache=None,
            fetcher=fake_fetcher(status="paywall"),
        )
        assert stats["paywall"] == 1
        assert "article_sentences" not in out[0]

    def test_no_text_not_expanded(self):
        cfg = make_cfg()
        out, stats = enrich_thin_stories(
            candidates=[candidate()],
            cfg=cfg,
            now_dt=NOW,
            cache=None,
            fetcher=fake_fetcher(status="no_text"),
        )
        assert stats["no_text"] == 1
        assert "article_sentences" not in out[0]

    def test_timeout_not_expanded(self):
        cfg = make_cfg()
        out, stats = enrich_thin_stories(
            candidates=[candidate()],
            cfg=cfg,
            now_dt=NOW,
            cache=None,
            fetcher=fake_fetcher(status="timeout"),
        )
        assert stats["timeout"] == 1
        assert "article_sentences" not in out[0]

    def test_budget_cap(self):
        cfg = make_cfg(max_fetches_per_run=2)
        cands = [
            candidate(
                story_id="s%d" % i,
                score=80 - i,
                event_id="event-s%d" % i,
                title="Distinct wildfire story number %d" % i,
                summary=(
                    "The fire continued to spread through "
                    "the night."
                ),
            )
            for i in range(4)
        ]
        out, stats = enrich_thin_stories(
            cands, cfg, NOW, cache=None, fetcher=fake_fetcher()
        )
        assert stats["fetched"] == 2
        assert stats["budget_exhausted"] >= 1

    def test_fetcher_exception_never_fails_pipeline(self):
        cfg = make_cfg()
        out, stats = enrich_thin_stories(
            candidates=[candidate()],
            cfg=cfg,
            now_dt=NOW,
            cache=None,
            fetcher=fake_fetcher(error=RuntimeError("boom")),
        )
        assert stats["network_error"] == 1
        assert "article_sentences" not in out[0]

    def test_mixed_failures_isolated(self):
        cfg = make_cfg()
        good = candidate(
            story_id="story-a",
            title="Evacuations ordered in western Oregon",
            summary=(
                "The fire continued to spread through "
                "the night."
            ),
        )
        bad = candidate(
            story_id="story-b",
            title="Flood warnings issued across Iowa",
            summary=(
                "The river was expected to crest on "
                "Sunday evening."
            ),
            url="https://www.bbc.co.uk/news/articles/c-bad",
        )
        bad["event_id"] = "event-b"

        calls = []

        def fetcher(url, art_cfg, allowlist, robots=None, pace=None):
            calls.append(url)
            if "c-bad" in url:
                return ("paywall", {})
            return (
                "ok",
                {
                    "text": (
                        "Officials in British Columbia said the "
                        "situation remained extremely dangerous. "
                        "More than 20,000 residents have been "
                        "forced from their homes so far."
                    ),
                    "title": None,
                },
            )

        out, stats = enrich_thin_stories(
            [good, bad], cfg, NOW, cache=None, fetcher=fetcher
        )
        assert len(calls) == 2
        assert stats["paywall"] == 1
        assert stats["expanded"] == 1


class TestEnrichmentFallbackToCoMembers:
    """When the event primary's article cannot supply >= 2 usable
    sentences, enrichment falls back to the event's other members in
    deterministic group order (primary first, then group order) and
    attaches the first usable extraction to the primary, recording
    which source actually provided the article text.
    """

    _OK_TEXT = (
        "The Economic Times said an earthquake of magnitude 5.5 "
        "struck Leh in Ladakh early on Thursday. "
        "Tremors were felt across the region for several seconds. "
        "Officials said no damage was reported so far. "
        "The quake was the strongest to hit Ladakh this year."
    )

    def _pair(self, primary_status="blocked",
              member_url=(
                  "https://www.bbc.co.uk/news/articles/c-et"
              ),
              member_text=None, **member_overrides):
        primary = candidate(
            story_id="primary-story",
            id="primary-story",
            event_id="event-ladakh",
            score=80,
            url="https://www.bbc.co.uk/news/articles/c-ndtv",
            source="NDTV",
            title=(
                "5.5 Magnitude Earthquake Strikes Ladakh's "
                "Leh Early Thursday"
            ),
            summary=(
                "An earthquake of magnitude 5.5 struck Leh in "
                "Ladakh early on Thursday morning."
            ),
        )
        member = candidate(
            story_id="member-story",
            id="member-story",
            event_id="event-ladakh",
            score=60,
            url=member_url,
            source="The Economic Times",
            title=(
                "Ladakh earthquake: 5.5 magnitude quake jolts "
                "Leh early Thursday"
            ),
            summary="",
            **member_overrides,
        )
        calls = []

        def fetcher(url, art_cfg, allowlist, robots=None, pace=None):
            calls.append(url)
            if url == primary["url"]:
                return (primary_status, {})
            return (
                ("ok",
                 {"text": member_text or self._OK_TEXT,
                  "title": None})
            )

        return [primary, member], calls, fetcher

    def test_blocked_primary_falls_back_to_co_member(self):
        cands, calls, fetcher = self._pair()
        out, stats = enrich_thin_stories(
            cands, make_cfg(), NOW, cache=None, fetcher=fetcher
        )
        assert stats["blocked"] == 1
        assert stats["expanded"] == 1
        assert len(out[0]["article_sentences"]) >= 2
        assert out[0]["enrichment_source"] == (
            "The Economic Times"
        )

    def test_non_article_primary_falls_back(self):
        cands, calls, fetcher = self._pair()
        cands[0]["url"] = "https://www.aljazeera.com/video/xyz"
        out, stats = enrich_thin_stories(
            cands, make_cfg(), NOW, cache=None, fetcher=fetcher
        )
        assert stats["non_article"] == 1
        assert stats["fetched"] == 1
        assert stats["expanded"] == 1
        assert len(out[0]["article_sentences"]) >= 2

    def test_domain_blocked_primary_falls_back(self):
        cands, calls, fetcher = self._pair()
        cands[0]["url"] = (
            "https://www.foreignnews.example/story"
        )
        out, stats = enrich_thin_stories(
            cands, make_cfg(), NOW, cache=None, fetcher=fetcher
        )
        assert stats["domain_blocked"] == 1
        assert stats["expanded"] == 1
        assert len(out[0]["article_sentences"]) >= 2

    def test_primary_too_few_sentences_falls_back(self):
        cands, calls, fetcher = self._pair(
            primary_status="ok",
        )

        def fetcher(url, art_cfg, allowlist, robots=None,
                    pace=None):
            calls.append(url)
            if url == cands[0]["url"]:
                return ("ok", {"text": "A single lone sentence.",
                               "title": None})
            return ("ok", {"text": self._OK_TEXT, "title": None})

        out, stats = enrich_thin_stories(
            cands, make_cfg(), NOW, cache=None, fetcher=fetcher
        )
        assert stats["not_expanded"] == 1
        assert stats["expanded"] == 1
        assert len(out[0]["article_sentences"]) >= 2
        assert out[0]["enrichment_source"] == (
            "The Economic Times"
        )

    def test_blocked_member_skipped_next_member_used(self):
        cands, calls, fetcher = self._pair()
        third = candidate(
            story_id="third-story",
            id="third-story",
            event_id="event-ladakh",
            score=60,
            url="https://www.bbc.co.uk/news/articles/c-hindu",
            source="The Hindu",
            title=(
                "Ladakh earthquake: 5.5 magnitude quake jolts "
                "Leh early Thursday"
            ),
            summary="",
        )
        cands.append(third)

        def fetcher(url, art_cfg, allowlist, robots=None,
                    pace=None):
            calls.append(url)
            if url == cands[0]["url"]:
                return ("blocked", {})
            if "c-hindu" in url:
                return ("ok",
                        {"text": self._OK_TEXT, "title": None})
            return ("blocked", {})

        out, stats = enrich_thin_stories(
            cands, make_cfg(), NOW, cache=None, fetcher=fetcher
        )
        assert stats["blocked"] == 2
        assert stats["expanded"] == 1
        # The Economic Times member was tried before The Hindu
        # and skipped; The Hindu supplied the text.
        assert "c-et" in calls[1]
        assert "c-hindu" in calls[2]
        assert len(out[0]["article_sentences"]) >= 2
        assert out[0]["enrichment_source"] == "The Hindu"

    def test_all_members_fail_stays_thin(self):
        cands, calls, fetcher = self._pair()

        def fetcher(url, art_cfg, allowlist, robots=None,
                    pace=None):
            calls.append(url)
            return ("blocked", {})

        out, stats = enrich_thin_stories(
            cands, make_cfg(), NOW, cache=None, fetcher=fetcher
        )
        # Primary and the single co-member are both attempted.
        assert stats["blocked"] == 2
        assert stats["expanded"] == 0
        assert "article_sentences" not in out[0]
        assert "enrichment_source" not in out[0]

    def test_non_article_member_skipped(self):
        cands, calls, fetcher = self._pair(
            member_url="https://www.aljazeera.com/video/xyz"
        )
        out, stats = enrich_thin_stories(
            cands, make_cfg(), NOW, cache=None, fetcher=fetcher
        )
        assert stats["non_article"] == 1
        assert stats["blocked"] == 1
        assert stats["expanded"] == 0
        assert "article_sentences" not in out[0]

    def test_budget_respected_across_fallback(self):
        cands, calls, fetcher = self._pair()
        out, stats = enrich_thin_stories(
            cands,
            make_cfg(max_fetches_per_run=1),
            NOW,
            cache=None,
            fetcher=fetcher,
        )
        # The primary consumed the only fetch; the co-member is
        # never fetched once the budget is exhausted.
        assert stats["fetched"] == 1
        assert len(calls) == 1
        assert stats["budget_exhausted"] == 0
        assert "article_sentences" not in out[0]

    def test_cache_blocked_primary_falls_back_offline(self, tmp_path):
        from src.article_extractor import ArticleCache

        cache = ArticleCache(tmp_path / "cache.db")
        cands, calls, fetcher = self._pair()
        primary = cands[0]
        cache.set(
            primary["story_id"],
            primary["url"],
            "blocked",
            now=NOW,
        )
        out, stats = enrich_thin_stories(
            cands, make_cfg(), NOW, cache=cache, fetcher=fetcher
        )
        assert stats["cache_hits"] == 1
        # Primary blocked in cache, so only the member is fetched.
        assert stats["fetched"] == 1
        assert stats["expanded"] == 1
        assert len(out[0]["article_sentences"]) >= 2
        assert out[0]["enrichment_source"] == (
            "The Economic Times"
        )
        cache.close()


class TestMassCasualtyEnrichmentGate:
    """The enrichment gate is aligned with the Telegram candidate
    queue gate: every thin candidate already accepted at HIGH /
    URGENT / IMMEDIATE is eligible for article enrichment.

    The mass-casualty track remains as a precise observability
    counter: it increments only for thin HIGH stories scoring
    60-64 that carry strong casualty evidence in a serious
    category (the Sudan mass-grave case).  A bare
    "death"/"die"/"bodies" is never enough: individual deaths and
    human-interest stories that were NOT accepted into the
    telegram queue (NORMAL/LOW priority) stay excluded from
    enrichment, so the pipeline never fetches every thin
    candidate.
    """

    def make_candidate(self, **overrides):
        base = candidate()
        base.update(
            {
                "score": 62,
                "priority_level": "HIGH",
                "priority_score": 62,
                "category": "world",
                "confidence": "high",
                "effective_at": (
                    NOW - timedelta(minutes=30)
                ).isoformat(),
            }
        )
        base.update(overrides)
        return base

    def test_sudan_mass_grave_story_enrichment_eligible(self):
        cfg = make_cfg()
        cands = [
            self.make_candidate(
                story_id="sudan-grave",
                id="sudan-grave",
                event_id="event-sudan",
                url="https://www.bbc.co.uk/news/articles/c-sudan",
                title=(
                    "Mass grave with 25 bodies found in Sudan's "
                    "Kurmuk after army retakes town"
                ),
                summary=(
                    "Officials accuse the Rapid Support Forces "
                    "of carrying out the killings, saying grave "
                    "held remains of children and women."
                ),
            )
        ]
        out, stats = enrich_thin_stories(
            cands, cfg, NOW, cache=None, fetcher=fake_fetcher()
        )
        assert stats["mass_casualty"] == 1
        assert stats["eligible"] == 1
        assert stats["fetched"] == 1
        assert len(out[0]["article_sentences"]) >= 2

    def test_nba_individual_drug_death_not_eligible(self):
        cfg = make_cfg()
        cands = [
            self.make_candidate(
                story_id="nba-death",
                id="nba-death",
                event_id="event-nba",
                priority_level="NORMAL",
                title="NBA forward Clarke's death due to drugs",
                summary=(
                    "Memphis Grizzlies forward Brandon Clarke's "
                    "death was due to the effects of heroin and "
                    "cocaine, the Los Angeles medical authority "
                    "reveals."
                ),
            )
        ]
        out, stats = enrich_thin_stories(
            cands, cfg, NOW, cache=None, fetcher=fake_fetcher()
        )
        assert stats["mass_casualty"] == 0
        assert stats["eligible"] == 0
        assert stats["fetched"] == 0
        assert "article_sentences" not in out[0]

    def test_ordinary_single_death_not_eligible(self):
        cfg = make_cfg()
        cands = [
            self.make_candidate(
                story_id="single-death",
                id="single-death",
                event_id="event-death",
                priority_level="NORMAL",
                title="Man dies after being hit by a car",
                summary=(
                    "A 65-year-old man died in hospital after "
                    "the crash."
                ),
            )
        ]
        out, stats = enrich_thin_stories(
            cands, cfg, NOW, cache=None, fetcher=fake_fetcher()
        )
        assert stats["mass_casualty"] == 0
        assert stats["eligible"] == 0
        assert stats["fetched"] == 0
        assert "article_sentences" not in out[0]

    def test_genuine_mass_casualty_story_eligible(self):
        cfg = make_cfg()
        cands = [
            self.make_candidate(
                story_id="market-bomb",
                id="market-bomb",
                event_id="event-bomb",
                category="conflict",
                title="At least 30 killed in market bombing",
                summary=(
                    "Officials said at least 30 people were "
                    "killed and dozens more injured in the blast."
                ),
            )
        ]
        out, stats = enrich_thin_stories(
            cands, cfg, NOW, cache=None, fetcher=fake_fetcher()
        )
        assert stats["mass_casualty"] == 1
        assert stats["eligible"] == 1
        assert stats["fetched"] == 1
        assert len(out[0]["article_sentences"]) >= 2

    def test_score_65_plus_behavior_unchanged(self):
        cfg = make_cfg()
        cands = [
            self.make_candidate(
                story_id="no-casualty-70",
                id="no-casualty-70",
                event_id="event-70",
                score=70,
                priority_score=70,
                title="Central bank raises interest rates",
                summary=(
                    "The central bank announced a surprise "
                    "rate increase on Thursday morning."
                ),
            )
        ]
        out, stats = enrich_thin_stories(
            cands, cfg, NOW, cache=None, fetcher=fake_fetcher()
        )
        assert stats["mass_casualty"] == 0
        assert stats["eligible"] == 1
        assert stats["fetched"] == 1

    def test_score_63_without_casualty_signal_not_eligible(self):
        cfg = make_cfg()
        cands = [
            self.make_candidate(
                story_id="no-casualty-63",
                id="no-casualty-63",
                event_id="event-63",
                priority_level="NORMAL",
                title="Trade delegation visits the capital",
                summary=(
                    "A trade delegation arrived to discuss "
                    "export quotas."
                ),
            )
        ]
        out, stats = enrich_thin_stories(
            cands, cfg, NOW, cache=None, fetcher=fake_fetcher()
        )
        assert stats["mass_casualty"] == 0
        assert stats["eligible"] == 0
        assert stats["fetched"] == 0

    def test_non_urgent_category_not_eligible(self):
        cfg = make_cfg()
        cands = [
            self.make_candidate(
                story_id="sports-casualty",
                id="sports-casualty",
                event_id="event-sports",
                priority_level="NORMAL",
                category="sports",
                title="Crowd trampled in stadium stampede",
                summary=(
                    "Several people were killed in the crush "
                    "outside the stadium."
                ),
            )
        ]
        out, stats = enrich_thin_stories(
            cands, cfg, NOW, cache=None, fetcher=fake_fetcher()
        )
        assert stats["mass_casualty"] == 0
        assert stats["eligible"] == 0
        assert stats["fetched"] == 0

    def test_no_fatalities_denial_not_eligible(self):
        cfg = make_cfg()
        cands = [
            self.make_candidate(
                story_id="no-fat",
                id="no-fat",
                event_id="event-fat",
                priority_level="NORMAL",
                title="Warehouse fire under control",
                summary="No fatalities reported, officials say.",
            )
        ]
        out, stats = enrich_thin_stories(
            cands, cfg, NOW, cache=None, fetcher=fake_fetcher()
        )
        assert stats["mass_casualty"] == 0
        assert stats["eligible"] == 0
        assert stats["fetched"] == 0

    def test_not_thin_mass_casualty_still_skipped(self):
        cfg = make_cfg()
        cands = [
            self.make_candidate(
                story_id="not-thin",
                id="not-thin",
                event_id="event-notthin",
                title="At least 20 killed in bridge collapse",
                summary=(
                    "At least 20 people were killed when the "
                    "bridge gave way. Rescuers searched the "
                    "river through the night. Officials said "
                    "the death toll could rise."
                ),
            )
        ]
        out, stats = enrich_thin_stories(
            cands, cfg, NOW, cache=None, fetcher=fake_fetcher()
        )
        assert stats["mass_casualty"] == 1
        assert stats["thin"] == 0
        assert stats["fetched"] == 0


class TestEnrichmentGateQueueAlignment:
    """Regression tests for the enrichment-gate alignment: a
    candidate already accepted into the Telegram candidate queue
    at HIGH / URGENT / IMMEDIATE is eligible for article
    enrichment when its briefing is thin.

    Covers the confirmed production gap: HIGH stories scoring
    38-64 (e.g. the Supreme Court food-regulator stories at
    score 44 with empty or one-sentence RSS) were queued but
    never article-fetched, so they were rejected as
    rejected_insufficient.

    The thinness gate, the fetch budget, the blocked-source
    protections and the 2-5 sentence summary requirement are all
    unchanged and are re-pinned below.
    """

    def make_candidate(self, **overrides):
        base = candidate()
        base.update(
            {
                "score": 44.0,
                "priority_score": 44.0,
                "priority_level": "HIGH",
                "event_status": "NEW",
                "effective_at": (
                    NOW - timedelta(minutes=5)
                ).isoformat(),
            }
        )
        base.update(overrides)
        return base

    def test_high_score_38_to_64_thin_rss_fetchable_enriched(self):
        # Production case: Supreme Court gives food regulator
        # 2-week notice (HIGH, score 44, one-sentence RSS, article
        # fetchable).  The enrichment gate now admits it and the
        # 2-sentence summary can be produced.
        from tests.telegram_pipeline_helpers import (
            build_telegram_stories,
        )

        cfg = make_cfg()
        cands = [
            self.make_candidate(
                story_id="sc-food-notice",
                id="sc-food-notice",
                event_id="event-sc-food",
                summary=(
                    "The Supreme Court asked the food regulator "
                    "to respond within two weeks."
                ),
            )
        ]
        out, stats = enrich_thin_stories(
            cands, cfg, NOW, cache=None, fetcher=fake_fetcher()
        )
        assert stats["eligible"] == 1
        assert stats["fetched"] == 1
        assert stats["expanded"] == 1
        assert len(out[0]["article_sentences"]) >= 2

        story = build_telegram_stories(
            out, cfg["telegram"], NOW
        )[0]
        rows = story["briefing"]["sentences"]
        assert 2 <= len(rows) <= 5
        assert all(r["text"] for r in rows)

    def test_high_empty_rss_fetchable_article_enriched(self):
        # Production case: Supreme Court environmental-law story
        # (HIGH, score 44, empty RSS description, article
        # fetchable).  Enrichment runs and the summary succeeds.
        cfg = make_cfg()
        cands = [
            self.make_candidate(
                story_id="sc-env-law",
                id="sc-env-law",
                event_id="event-sc-env",
                summary="",
            )
        ]
        out, stats = enrich_thin_stories(
            cands, cfg, NOW, cache=None, fetcher=fake_fetcher()
        )
        assert stats["eligible"] == 1
        assert stats["expanded"] == 1
        assert len(out[0]["article_sentences"]) >= 2

        from tests.telegram_pipeline_helpers import (
            build_telegram_stories,
        )

        story = build_telegram_stories(
            out, cfg["telegram"], NOW
        )[0]
        rows = story["briefing"]["sentences"]
        assert 2 <= len(rows) <= 5
        assert all(r["text"] for r in rows)

    def test_high_one_sentence_rss_fetchable_article_enriched(self):
        # Production case: Supreme Court slams FSSAI (HIGH,
        # score 44, one-sentence RSS, article fetchable).
        cfg = make_cfg()
        cands = [
            self.make_candidate(
                story_id="sc-fssai",
                id="sc-fssai",
                event_id="event-sc-fssai",
                summary=(
                    "The Bench asks the country's food regulator "
                    "whether it does not want children to grow up "
                    "healthy."
                ),
            )
        ]
        out, stats = enrich_thin_stories(
            cands, cfg, NOW, cache=None, fetcher=fake_fetcher()
        )
        assert stats["eligible"] == 1
        assert stats["expanded"] == 1
        assert len(out[0]["article_sentences"]) >= 2

    def test_normal_thin_candidate_not_enriched(self):
        # A NORMAL/LOW candidate is never article-enriched merely
        # because it is thin: enrichment only follows the telegram
        # queue gate (HIGH / URGENT / IMMEDIATE).
        cfg = make_cfg()
        cands = [
            self.make_candidate(
                story_id="normal-thin",
                id="normal-thin",
                event_id="event-normal",
                priority_level="NORMAL",
                priority_score=30.0,
                score=30.0,
                summary="A minor incident was reported.",
            )
        ]
        out, stats = enrich_thin_stories(
            cands, cfg, NOW, cache=None, fetcher=fake_fetcher()
        )
        assert stats["eligible"] == 0
        assert stats["fetched"] == 0
        assert "article_sentences" not in out[0]

    def test_non_thin_high_candidate_not_fetched(self):
        # A HIGH candidate with a usable RSS briefing does not
        # consume an unnecessary article-fetch slot.
        cfg = make_cfg()
        cands = [
            self.make_candidate(
                story_id="high-not-thin",
                id="high-not-thin",
                event_id="event-hightthin",
                summary=(
                    "The court asked for a reply within two "
                    "weeks. The regulator was told to file its "
                    "response in writing. The next hearing was "
                    "scheduled for September."
                ),
            )
        ]
        out, stats = enrich_thin_stories(
            cands, cfg, NOW, cache=None, fetcher=fake_fetcher()
        )
        assert stats["thin"] == 0
        assert stats["fetched"] == 0
        assert "article_sentences" not in out[0]

    def test_high_blocked_article_no_alternate_stays_unenriched(self):
        # Production case: Karnataka Cabinet story (HIGH,
        # score 44, enrichment score 72) whose article is blocked
        # by robots.txt.  The enrichment fetch is attempted and
        # counted as blocked; the story is never padded from the
        # headline or the blocked fetch.
        cfg = make_cfg()
        cands = [
            self.make_candidate(
                story_id="karnataka-bill",
                id="karnataka-bill",
                event_id="event-karnataka",
                url="https://www.bbc.co.uk/news/articles/c-blocked",
                summary=(
                    "Priyank Kharge said the government has no "
                    "specific institution in mind."
                ),
            )
        ]
        out, stats = enrich_thin_stories(
            cands, cfg, NOW, cache=None,
            fetcher=fake_fetcher(status="blocked"),
        )
        assert stats["eligible"] == 1
        assert stats["blocked"] == 1
        assert stats["fetched"] == 1
        assert "article_sentences" not in out[0]
