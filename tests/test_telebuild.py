"""Integration tests: the India Telegram queue builder / orchestrator.

Covers the full Phase 2C flow: Candidate -> queue dict (India
metadata preserved), non-news source gate, freshness window,
max_candidates cap, best-effort article enrichment, event grouping,
briefing + source-grounded summarization, and the final
data/telegram_queue.json shape.
"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.config_loader import get_config
from src.pipeline.integration import NewsPipeline
from src.telebuild import (
    build_telegram_queue,
    build_telegram_stories,
    candidate_to_dict,
    filter_telegram_candidates,
    telegram_ineligible_sources,
)

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    from src.config_loader import get_config

    cfg = get_config()
    cfg = json.loads(json.dumps(cfg))
    cfg["config"]["telegram"] = dict(
        cfg["config"]["telegram"]
    )
    cfg["config"]["telegram"]["telegram_state_file"] = str(
        tmp_path_factory.mktemp("telebuild_state")
        / "telegram_state.json"
    )
    return cfg


@pytest.fixture
def tmp_cache(tmp_path):
    from src.article_extractor import ArticleCache
    return ArticleCache(tmp_path / "test_cache.db")


def _art(source_id, title, summary="", *, tier=2, role="journalism",
         published=NOW, url=None, source_name=None, author=None):
    from src.models.article import Article
    return Article(
        source_id=source_id,
        source_name=source_name or source_id.replace("-", " ").title(),
        tier=tier,
        source_role=role,
        url=url or f"https://{source_id}.example.in/{abs(hash(title))}.html",
        title=title,
        summary=summary or None,
        author=author,
        published=published,
    )


def _pipeline_candidates(*articles):
    return NewsPipeline().run(list(articles), now=NOW).queue


_EARTHQUAKE_SUMMARY = (
    "A powerful earthquake of magnitude 7.2 struck Delhi on "
    "Wednesday morning. Rescue teams reached the affected areas "
    "within minutes and evacuated hundreds of residents. Officials "
    "reported at least 40 people injured and several buildings "
    "collapsed across the city."
)


def _real_story(bundle, *, published=NOW, url=None):
    """One real pipeline candidate converted to a queue dict."""
    c = _pipeline_candidates(
        _art("the-hindu", "Major earthquake of magnitude 7.2 strikes Delhi, rescue underway",
             summary=_EARTHQUAKE_SUMMARY, published=published, url=url),
    )[0]
    return candidate_to_dict(c)


# --- candidate -> queue dict ------------------------------------------------

def test_candidate_to_dict_preserves_india_metadata(bundle):
    cand = _pipeline_candidates(
        _art("the-hindu", "Major earthquake of magnitude 7.2 strikes Delhi, rescue underway"),
    )[0]
    d = candidate_to_dict(cand)
    assert d["event_id"] == cand.event_id
    assert d["story_id"] == cand.event_id
    assert d["headline"] == cand.title
    assert d["title"] == cand.title
    assert d["summary"] == cand.summary
    assert d["url"] == cand.representative.url
    assert d["source"] == cand.representative.source_name
    assert d["primary_source"] is False
    assert d["tier"] == cand.representative.tier
    assert d["category"] == cand.category
    assert d["secondary_categories"] == list(cand.secondary)
    assert d["confidence"] == cand.confidence
    assert d["score"] == cand.relevance.score
    assert d["priority_score"] == cand.priority.score
    assert d["priority_level"] == cand.priority.priority
    assert d["effective_at"] == cand.effective_at
    assert d["event_time"] == cand.event_time.isoformat()
    assert d["published_at"] == cand.representative.published.isoformat()
    assert d["source_groups"] == cand.source_groups
    assert d["is_wire_echo"] is False
    assert d["status"] == "queued"


def test_candidate_to_dict_adds_geo_metadata(bundle):
    cand = _pipeline_candidates(
        _art("the-hindu", "Major earthquake of magnitude 7.2 strikes Delhi, rescue underway"),
    )[0]
    d = candidate_to_dict(cand)
    assert d["geo_scope"] == cand.geo.scope
    assert d["state"] == cand.geo.state
    assert d["national_significance"] == cand.geo.national_significance


def test_candidate_to_dict_passthrough_for_dicts():
    raw = {"story_id": "s1", "effective_at": (NOW - timedelta(minutes=5)).isoformat()}
    assert candidate_to_dict(raw) == raw


# --- non-news source gate ---------------------------------------------------

def test_telegram_ineligible_sources_uses_news_flag(bundle):
    no_news = telegram_ineligible_sources(bundle)
    assert isinstance(no_news, set)
    # All configured India sources are currently news sources.
    assert no_news == set()


def test_non_news_source_excluded_from_queue(bundle):
    cfg = dict(bundle)
    cfg["sources"] = {
        "sources": [
            {
                "id": "press-bureau",
                "name": "Press Bureau",
                "news": False,
            }
        ]
    }
    cand = _real_story(bundle)
    cand["source"] = "Press Bureau"
    kept, stats = filter_telegram_candidates([cand], cfg, NOW)
    assert kept == []
    assert stats["non_news_filtered"] == 1


# --- freshness window -------------------------------------------------------

def test_fresh_candidate_kept(bundle):
    cand = _real_story(bundle, published=NOW - timedelta(hours=1))
    kept, stats = filter_telegram_candidates([cand], bundle, NOW)
    assert len(kept) == 1
    assert stats["fresh"] == 1


def test_stale_candidate_excluded(bundle):
    cand = _real_story(bundle, published=NOW - timedelta(hours=20))
    kept, stats = filter_telegram_candidates([cand], bundle, NOW)
    assert kept == []
    assert stats["stale"] == 1


def test_candidate_without_effective_at_excluded(bundle):
    cand = _real_story(bundle)
    cand["effective_at"] = None
    kept, stats = filter_telegram_candidates([cand], bundle, NOW)
    assert kept == []
    assert stats["no_effective_at"] == 1


# --- max_candidates cap -----------------------------------------------------

def test_max_candidates_cap_applied(bundle):
    cfg = dict(bundle)
    cfg["config"] = dict(bundle["config"])
    cfg["config"]["telegram"] = dict(bundle["config"]["telegram"])
    cfg["config"]["telegram"]["max_candidates"] = 1
    cands = [_real_story(bundle, published=NOW - timedelta(minutes=i))
             for i in range(3)]
    kept, stats = filter_telegram_candidates(cands, cfg, NOW)
    assert len(kept) == 1


def test_scheduled_story_survives_candidate_truncation(bundle):
    # An already-scheduled story (an owed obligation) must survive
    # the max_candidates cap: even when it falls beyond the normal
    # selection boundary, it is preserved in the queue.
    cfg = dict(bundle)
    cfg["config"] = dict(bundle["config"])
    cfg["config"]["telegram"] = dict(bundle["config"]["telegram"])
    cfg["config"]["telegram"]["max_candidates"] = 50

    cands = [
        {
            "story_id": "story-{:03d}".format(i),
            "id": "story-{:03d}".format(i),
            "status": "queued",
            "source": "The Hindu",
            "effective_at": (
                NOW - timedelta(minutes=1)
            ).isoformat(),
        }
        for i in range(60)
    ]

    # story-059 sits beyond the 50-candidate boundary but is a
    # committed scheduled obligation.
    protected = {"story-059"}
    kept, stats = filter_telegram_candidates(
        cands,
        cfg,
        NOW,
        protected_story_ids=protected,
    )

    assert len(kept) == 51
    assert stats["protected_kept"] == 1
    ids = [c["story_id"] for c in kept]
    assert "story-059" in ids
    assert ids[:50] == [
        "story-{:03d}".format(i)
        for i in range(50)
    ]
    assert ids.count("story-059") == 1


def test_scheduled_story_within_cap_no_duplicate(bundle):
    # When the scheduled story is already inside the normal cap it
    # must not be duplicated by the protection mechanism.
    cfg = dict(bundle)
    cfg["config"] = dict(bundle["config"])
    cfg["config"]["telegram"] = dict(bundle["config"]["telegram"])
    cfg["config"]["telegram"]["max_candidates"] = 50

    cands = [
        {
            "story_id": "story-{:03d}".format(i),
            "id": "story-{:03d}".format(i),
            "status": "queued",
            "source": "The Hindu",
            "effective_at": (
                NOW - timedelta(minutes=1)
            ).isoformat(),
        }
        for i in range(5)
    ]

    kept, stats = filter_telegram_candidates(
        cands,
        cfg,
        NOW,
        protected_story_ids={"story-003"},
    )

    assert len(kept) == 5
    assert stats["protected_kept"] == 0
    ids = [c["story_id"] for c in kept]
    assert ids.count("story-003") == 1


def test_scheduled_story_protection_no_duplicate_ids(bundle):
    # A protected story that is not among the fresh candidates must
    # not be invented or duplicated -- protection only preserves
    # candidates that actually passed the gate.
    cfg = dict(bundle)
    cfg["config"] = dict(bundle["config"])
    cfg["config"]["telegram"] = dict(bundle["config"]["telegram"])
    cfg["config"]["telegram"]["max_candidates"] = 50

    cands = [
        {
            "story_id": "story-{:03d}".format(i),
            "id": "story-{:03d}".format(i),
            "status": "queued",
            "source": "The Hindu",
            "effective_at": (
                NOW - timedelta(minutes=1)
            ).isoformat(),
        }
        for i in range(55)
    ]

    kept, stats = filter_telegram_candidates(
        cands,
        cfg,
        NOW,
        protected_story_ids={
            "story-054",
            "does-not-exist",
        },
    )

    assert stats["protected_kept"] == 1
    ids = [c["story_id"] for c in kept]
    assert ids.count("story-054") == 1
    assert "does-not-exist" not in ids


# --- non-queued candidates --------------------------------------------------

def test_held_candidate_excluded(bundle):
    cand = _real_story(bundle)
    cand["status"] = "held"
    kept, stats = filter_telegram_candidates([cand], bundle, NOW)
    assert kept == []
    assert stats["candidates"] == 1
    assert stats["kept"] == 0


def test_rejected_and_filler_candidates_excluded(bundle):
    cands = []
    for status in ("rejected", "filler"):
        cand = _real_story(bundle)
        cand["status"] = status
        cands.append(cand)
    kept, stats = filter_telegram_candidates(cands, bundle, NOW)
    assert kept == []
    assert stats["kept"] == 0


# --- story building ---------------------------------------------------------

def test_build_telegram_stories_groups_multiple_candidates_into_one_story(bundle):
    cand = _real_story(bundle)
    cand["id"] = "event-1"
    cand["story_id"] = "event-1"
    dup = dict(cand)
    dup["id"] = "event-1dup"
    dup["story_id"] = "event-1dup"
    dup["score"] = cand.get("score") - 1
    stories = build_telegram_stories([cand, dup], bundle["config"]["telegram"], NOW)
    assert len(stories) == 1


def test_build_telegram_stories_emits_briefing_metadata(bundle):
    story = build_telegram_stories(
        [_real_story(bundle)],
        bundle["config"]["telegram"],
        NOW,
    )[0]
    briefing = story["briefing"]
    assert set(briefing) >= {
        "opening", "body", "bullets", "sentences",
        "source", "corroborating", "url",
    }
    assert len(briefing["sentences"]) >= 2
    assert story["public_label"]
    assert story["headline"]
    assert story["story_id"]
    assert story["group_size"] >= 1
    assert story["enrichment"] in ("rss", "article")


def test_build_telegram_stories_preserves_candidate_fields(bundle):
    cand = _real_story(bundle)
    story = build_telegram_stories([cand], bundle["config"]["telegram"], NOW)[0]
    for key in (
        "event_id", "story_id", "url", "source", "tier",
        "category", "confidence", "score", "priority_score",
        "priority_level", "effective_at", "geo_scope", "state",
        "national_significance",
    ):
        assert key in story


def test_wire_echo_collapses_to_single_candidate(bundle):
    a = _art("the-hindu", "Cyclone devastates coastal Odisha, thousands evacuated",
             summary="A powerful cyclone hit the Odisha coast. Thousands of people were evacuated to safety.")
    b = _art("economictimes", "Cyclone devastates coastal Odisha, thousands evacuated",
             summary="A powerful cyclone hit the Odisha coast. Thousands of people were evacuated to safety.",
             author="PTI")
    queue = _pipeline_candidates(a, b)
    assert len(queue) == 1
    assert queue[0].is_wire_echo is True
    stories = build_telegram_stories(
        [candidate_to_dict(c) for c in queue],
        bundle["config"]["telegram"],
        NOW,
    )
    assert len(stories) == 1


def test_empty_candidates_produce_no_stories(bundle):
    stories = build_telegram_stories([], bundle["config"]["telegram"], NOW)
    assert stories == []


# --- full orchestrator ------------------------------------------------------

def test_build_telegram_queue_writes_queue_file(bundle, tmp_path, tmp_cache):
    result = NewsPipeline().run(
        [_art("the-hindu", "Major earthquake of magnitude 7.2 strikes Delhi, rescue underway",
              summary=_EARTHQUAKE_SUMMARY)],
        now=NOW,
    )
    out = tmp_path / "telegram_queue.json"
    stories, stats = build_telegram_queue(result, bundle, now_dt=NOW, queue_path=out, cache=tmp_cache)
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "generated_at" in payload
    assert payload["count"] == len(stories)
    assert isinstance(payload["stories"], list)
    assert len(payload["stories"]) >= 1


def test_build_telegram_queue_accepts_candidate_list(bundle, tmp_path, tmp_cache):
    queue = _pipeline_candidates(
        _art("the-hindu", "Major earthquake of magnitude 7.2 strikes Delhi, rescue underway",
             summary=_EARTHQUAKE_SUMMARY),
    )
    out = tmp_path / "q.json"
    stories, stats = build_telegram_queue(queue, bundle, now_dt=NOW, queue_path=out, cache=tmp_cache)
    assert len(stories) == len(queue)


def test_build_telegram_queue_empty_queue_writes_empty_file(bundle, tmp_path, tmp_cache):
    out = tmp_path / "q.json"
    stories, stats = build_telegram_queue([], bundle, now_dt=NOW, queue_path=out, cache=tmp_cache)
    assert stories == []
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["count"] == 0
    assert payload["stories"] == []


def test_build_telegram_queue_filter_stats(bundle, tmp_path, tmp_cache):
    out = tmp_path / "q.json"
    stories, stats = build_telegram_queue([], bundle, now_dt=NOW, queue_path=out, cache=tmp_cache)
    assert stats["filter"]["candidates"] == 0
    assert "article_extraction" in stats
    assert "summarization" in stats


def test_scheduled_story_ids_reads_state_file(bundle, tmp_path):
    from src.telebuild import scheduled_story_ids

    state_file = tmp_path / "telegram_state.json"
    state_file.write_text(
        json.dumps(
            {
                "scheduled": [
                    {
                        "story_id": "owed-1",
                        "scheduled_at": "2026-08-12T12:30:00+00:00",
                    },
                    {
                        "story_id": "owed-2",
                        "scheduled_at": "2026-08-12T12:31:00+00:00",
                    },
                ],
                "posted": [],
                "failures": [],
            }
        ),
        encoding="utf-8",
    )

    cfg = dict(bundle)
    cfg["config"] = dict(bundle["config"])
    cfg["config"]["telegram"] = dict(
        bundle["config"]["telegram"]
    )
    cfg["config"]["telegram"]["telegram_state_file"] = str(
        state_file
    )

    assert scheduled_story_ids(cfg) == {
        "owed-1",
        "owed-2",
    }


def test_scheduled_story_ids_missing_state_is_empty(
    bundle,
    tmp_path,
):
    from src.telebuild import scheduled_story_ids

    cfg = dict(bundle)
    cfg["config"] = dict(bundle["config"])
    cfg["config"]["telegram"] = dict(
        bundle["config"]["telegram"]
    )
    cfg["config"]["telegram"]["telegram_state_file"] = str(
        tmp_path / "does-not-exist.json"
    )

    assert scheduled_story_ids(cfg) == set()


def test_build_telegram_queue_preserves_protected_story(
    bundle,
    tmp_path,
    tmp_cache,
):
    # The full orchestrator path: a scheduled story beyond the
    # normal cap survives queue regeneration via protected ids,
    # and protected ids are auto-derived from the state file.
    cfg = dict(bundle)
    cfg["config"] = dict(bundle["config"])
    cfg["config"]["telegram"] = dict(
        bundle["config"]["telegram"]
    )
    cfg["config"]["telegram"]["max_candidates"] = 2

    cands = [
        _real_story(
            bundle,
            published=NOW - timedelta(minutes=i),
        )
        for i in range(5)
    ]
    for i, cand in enumerate(cands):
        cand["story_id"] = "event-{:02d}".format(i)
        cand["id"] = "event-{:02d}".format(i)
        cand["event_id"] = "event-{:02d}".format(i)
        cand["title"] = "Distinct event {:02d} unfolds".format(
            i
        )
        cand["headline"] = cand["title"]
        cand["url"] = "https://example.in/event-{:02d}".format(
            i
        )

    out = tmp_path / "q.json"
    stories, stats = build_telegram_queue(
        cands,
        cfg,
        now_dt=NOW,
        queue_path=out,
        cache=tmp_cache,
        protected_story_ids={"event-04"},
    )

    story_ids = [s.get("story_id") for s in stories]
    assert "event-04" in story_ids
    assert stats["filter"]["protected_kept"] == 1
    assert story_ids.count("event-04") == 1


def test_priority_order_preserved_in_queue(bundle, tmp_path, tmp_cache):
    urgent = _art("the-hindu", "Major terror attack in Mumbai, multiple casualties reported",
                  summary="Terrorists attacked Mumbai on Wednesday. Multiple people were killed. Security forces responded.")
    breaking = _art("the-hindu", "Major earthquake of magnitude 7.2 strikes Delhi, rescue underway",
                    summary=_EARTHQUAKE_SUMMARY)
    result = NewsPipeline().run([urgent, breaking], now=NOW)
    assert len(result.queue) == 2
    out = tmp_path / "q.json"
    stories, _ = build_telegram_queue(result, bundle, now_dt=NOW, queue_path=out, cache=tmp_cache)
    # Priority ordering: the higher-priority event renders first.
    scores = [s.get("priority_score", 0) or s.get("score", 0) for s in stories]
    assert scores == sorted(scores, reverse=True)


def test_three_story_briefing_renders_concise_verbatim(bundle, tmp_path, tmp_cache):
    """End-to-end regression for the n=3 briefing: every story rendered
    from the queue is concise, every body sentence is verbatim in the
    source text, no filler/boilerplate sentence survives, and nothing
    exceeds the hard message-size budget.
    """
    import re

    from src.telegram_briefing import is_filler
    from src.telegram_formatter import (
        build_message,
        telegram_visible_len,
    )

    quake = _art(
        "the-hindu",
        "Major earthquake of magnitude 7.2 strikes Delhi, rescue underway",
        summary=_EARTHQUAKE_SUMMARY,
    )
    cyclone = _art(
        "indian-express",
        "Cyclone makes landfall on Odisha coast, thousands evacuated",
        summary=(
            "A powerful cyclone hit the Odisha coast on Wednesday. "
            "Thousands of people were evacuated to safety before "
            "landfall. Rescue teams reached the affected villages "
            "and restored road links. Officials said the storm "
            "weakened after crossing the coast."
        ),
    )
    terror = _art(
        "ndtv",
        "Major terror attack in Mumbai, multiple casualties reported",
        summary=(
            "Terrorists attacked Mumbai on Wednesday. Multiple "
            "people were killed and dozens were injured. Security "
            "forces cordoned off the area and launched an "
            "operation. Officials said an investigation was "
            "underway."
        ),
    )

    out = tmp_path / "q.json"
    stories, _ = build_telegram_queue(
        NewsPipeline().run([quake, cyclone, terror], now=NOW),
        bundle,
        now_dt=NOW,
        queue_path=out,
        cache=tmp_cache,
    )

    assert len(stories) == 3

    cfg = (bundle["config"].get("telegram") or {})
    max_chars = int(cfg.get("max_message_chars", 3000))

    def norm(text):
        return re.sub(r"\s+", " ", (text or "").strip()).lower()

    for story in stories:
        msg = build_message(story, dict(cfg), NOW)
        assert msg is not None, "story must render: " + story.get("headline")
        assert telegram_visible_len(msg["text"]) <= max_chars

        evidence = " ".join(story.get("article_sentences") or [])
        if story.get("summary"):
            evidence += " " + story["summary"]

        for row in (story.get("briefing") or {}).get("sentences") or []:
            sentence = (row or {}).get("text") or ""
            assert sentence, "no empty sentences in a briefing"
            assert not is_filler(sentence), "filler leaked: " + sentence[:60]
            assert norm(sentence) in norm(evidence), (
                "not verbatim: " + sentence[:60]
            )


def test_stale_candidates_dropped_before_story_building(bundle, tmp_path, tmp_cache):
    stale = _pipeline_candidates(
        _art("the-hindu", "Major earthquake of magnitude 7.2 strikes Delhi, rescue underway",
             published=NOW - timedelta(hours=20)),
    )
    out = tmp_path / "q.json"
    stories, stats = build_telegram_queue(stale, bundle, now_dt=NOW, queue_path=out, cache=tmp_cache)
    assert stories == []
    assert stats["filter"]["stale"] == 1


def test_one_bad_candidate_does_not_fail_the_run(bundle, tmp_path, tmp_cache):
    good = _real_story(bundle)
    bad = _real_story(bundle)
    bad["effective_at"] = "not-a-timestamp"
    out = tmp_path / "q.json"
    stories, stats = build_telegram_queue(
        [good, bad], bundle, now_dt=NOW, queue_path=out, cache=tmp_cache
    )
    assert stats["filter"]["no_effective_at"] == 1
    assert len(stories) >= 1


# --- enrichment through the orchestrator ------------------------------------

def _thin_pipeline_result():
    """A real pipeline result whose candidate is important but thin."""
    return NewsPipeline().run(
        [_art("the-hindu", "Major earthquake of magnitude 7.2 strikes Delhi, rescue underway",
              summary=None,
              url="https://www.thehindu.com/news/national/delhi-earthquake")],
        now=NOW,
    )


def _fake_fetcher(text):
    def _fetch(url, art_cfg, allowlist, robots=None, pace=None):
        return ("ok", {"text": text, "title": None})
    return _fetch


_ARTICLE_TEXT = (
    "Rescue teams reached the affected areas within minutes. "
    "At least 40 people were injured and several buildings "
    "collapsed. Hospitals in the capital reported they were "
    "overwhelmed with casualties."
)


def test_thin_story_enriched_through_orchestrator(bundle, tmp_path, tmp_cache):
    result = _thin_pipeline_result()
    out = tmp_path / "q.json"
    stories, stats = build_telegram_queue(
        result, bundle, now_dt=NOW, queue_path=out,
        cache=tmp_cache,
        fetcher=_fake_fetcher(_ARTICLE_TEXT),
    )
    assert len(stories) == 1
    assert stats["article_extraction"]["expanded"] == 1
    assert stories[0]["enrichment"] == "article"
    assert len(stories[0]["briefing"]["sentences"]) >= 2


def test_enrichment_failure_degrades_to_rss(bundle, tmp_path, tmp_cache):
    result = _thin_pipeline_result()

    def _boom(url, art_cfg, allowlist, robots=None, pace=None):
        raise RuntimeError("network down")

    out = tmp_path / "q.json"
    stories, stats = build_telegram_queue(
        result, bundle, now_dt=NOW, queue_path=out,
        cache=tmp_cache, fetcher=_boom,
    )
    # The pipeline must not fail because of enrichment; the thin
    # story has no RSS briefing of its own, so it is rejected
    # (never padded or invented).
    assert stats["article_extraction"]["network_error"] == 1
    assert stories == []


def test_cache_hit_skips_fetch_through_orchestrator(bundle, tmp_path):
    from src.article_extractor import ArticleCache

    db = tmp_path / "cache.db"
    cache = ArticleCache(db)
    result = _thin_pipeline_result()
    cand = result.queue[0].to_dict()
    cache.set(cand["story_id"], cand["url"], "ok",
              text=_ARTICLE_TEXT, ttl_ok=48)

    calls = []

    def _spy(url, art_cfg, allowlist, robots=None, pace=None):
        calls.append(url)
        return ("ok", {"text": _ARTICLE_TEXT, "title": None})

    out = tmp_path / "q.json"
    stories, stats = build_telegram_queue(
        result, bundle, now_dt=NOW, queue_path=out,
        cache=cache, fetcher=_spy,
    )
    assert stats["article_extraction"]["cache_hits"] == 1
    assert calls == []

def test_enrichment_disabled_no_fetch(bundle, tmp_path, tmp_cache):
    cfg = dict(bundle)
    cfg["config"] = dict(bundle["config"])
    cfg["config"]["article_extraction"] = dict(
        bundle["config"]["article_extraction"]
    )
    cfg["config"]["article_extraction"]["enabled"] = False

    result = _thin_pipeline_result()
    calls = []

    def _spy(url, art_cfg, allowlist, robots=None, pace=None):
        calls.append(url)
        return ("ok", {"text": _ARTICLE_TEXT, "title": None})

    out = tmp_path / "q.json"
    stories, stats = build_telegram_queue(
        result, cfg, now_dt=NOW, queue_path=out,
        cache=tmp_cache, fetcher=_spy,
    )
    assert calls == []
    assert stats["article_extraction"]["enabled"] is False
    # Without enrichment the thin story has no usable RSS briefing,
    # so it is rejected rather than padded.
    assert stories == []


def test_queue_output_is_deterministic(bundle, tmp_path, tmp_cache):
    articles = [
        _art("the-hindu", "Major earthquake of magnitude 7.2 strikes Delhi, rescue underway",
             summary=_EARTHQUAKE_SUMMARY),
        _art("the-hindu", "Major terror attack in Mumbai, multiple casualties reported",
             summary="Terrorists attacked Mumbai on Wednesday. Multiple people were killed. Security forces responded."),
    ]
    r1 = NewsPipeline().run(list(articles), now=NOW)
    r2 = NewsPipeline().run(list(articles), now=NOW)
    out1, out2 = tmp_path / "a.json", tmp_path / "b.json"
    s1, _ = build_telegram_queue(r1, bundle, now_dt=NOW, queue_path=out1, cache=tmp_cache)
    s2, _ = build_telegram_queue(r2, bundle, now_dt=NOW, queue_path=out2, cache=tmp_cache)
    assert json.loads(out1.read_text(encoding="utf-8")) == json.loads(
        out2.read_text(encoding="utf-8")
    )
    assert [s["story_id"] for s in s1] == [s["story_id"] for s in s2]


# --- co-member enrichment fallback through the orchestrator ------------------

def test_blocked_primary_enriched_from_co_member(bundle, tmp_path, tmp_cache):
    """Ladakh repro: the event primary (NDTV) blocks bots, but a
    co-member (The Economic Times) article extracts cleanly.  The
    event must still enrich from the co-member and attribute the
    article text to the real source, not the primary.
    """
    base = _real_story(bundle)

    primary = dict(base)
    primary.update({
        "story_id": "ndtv-ladakh",
        "id": "ndtv-ladakh",
        "event_id": "event-ladakh",
        "source": "NDTV",
        "title": (
            "5.5 Magnitude Earthquake Strikes Ladakh's "
            "Leh Early Thursday"
        ),
        "summary": (
            "An earthquake of magnitude 5.5 struck Leh in "
            "Ladakh early on Thursday morning."
        ),
        "url": (
            "https://www.ndtv.com/india-news/"
            "5-5-magnitude-earthquake-strikes-ladakhs-leh-11902766"
        ),
        "score": 75,
        "tier": 2,
    })

    member = dict(base)
    member.update({
        "story_id": "et-ladakh",
        "id": "et-ladakh",
        "event_id": "event-ladakh",
        "source": "The Economic Times",
        "title": (
            "Ladakh earthquake: 5.5 magnitude quake jolts "
            "Leh early Thursday"
        ),
        "summary": "",
        "url": (
            "https://economictimes.indiatimes.com/news/india/"
            "ladakh-earthquake-5-5-magnitude-quake-jolts-leh-"
            "early-thursday/articleshow/133195506.cms"
        ),
        "score": 70,
        "tier": 3,
    })

    _ET_ARTICLE_TEXT = (
        "An earthquake of magnitude 5.5 struck Leh in Ladakh "
        "early on Thursday. "
        "Tremors were felt across the region for several seconds. "
        "Officials said no damage was reported so far. "
        "The quake was the strongest to hit Ladakh this year."
    )

    def fetcher(url, art_cfg, allowlist, robots=None, pace=None):
        if "ndtv.com" in url:
            return ("blocked", {})
        return ("ok",
                {"text": _ET_ARTICLE_TEXT, "title": None})

    out = tmp_path / "q.json"
    stories, stats = build_telegram_queue(
        [primary, member], bundle, now_dt=NOW,
        queue_path=out, cache=tmp_cache, fetcher=fetcher,
    )

    assert len(stories) == 1
    assert stats["article_extraction"]["blocked"] == 1
    assert stats["article_extraction"]["expanded"] == 1

    story = stories[0]
    assert story["enrichment"] == "article"
    assert story["story_id"] == "ndtv-ladakh"
    assert story["briefing"]["source"] == "NDTV"
    assert len(story["briefing"]["sentences"]) >= 2

    # The article facts are attributed to the co-member source,
    # never to the robots-blocked primary.
    sources = {
        s.get("source")
        for s in story["briefing"]["sentences"]
    }
    assert "The Economic Times" in sources
    assert "NDTV" not in sources


def test_blocked_high_singleton_rejected_insufficient(bundle, tmp_path, tmp_cache):
    """Karnataka repro: a thin HIGH story whose only article is
    robots-blocked and which has no alternate source stays
    rejected.  Enrichment is attempted and counted, but the story
    is never padded or invented, and the blocked-source
    protection is never bypassed."""
    base = _real_story(bundle)
    cand = dict(base)
    cand.update({
        "story_id": "ndtv-karnataka",
        "id": "ndtv-karnataka",
        "event_id": "event-karnataka",
        "source": "NDTV",
        "title": (
            "Karnataka Cabinet Approves Public Property Bill "
            "Amid RSS Registration Row"
        ),
        "summary": (
            "Priyank Kharge said the government has no "
            "specific institution in mind."
        ),
        "url": (
            "https://www.ndtv.com/india-news/"
            "karnataka-cabinet-approves-public-property-bill"
        ),
        "score": 44.0,
        "priority_score": 44.0,
        "priority_level": "HIGH",
    })

    def fetcher(url, art_cfg, allowlist, robots=None, pace=None):
        return ("blocked", {})

    out = tmp_path / "q.json"
    stories, stats = build_telegram_queue(
        [cand], bundle, now_dt=NOW,
        queue_path=out, cache=tmp_cache, fetcher=fetcher,
    )

    assert stats["article_extraction"]["eligible"] == 1
    assert stats["article_extraction"]["fetched"] == 1
    assert stats["article_extraction"]["blocked"] == 1
    assert "article_sentences" not in stories
    assert stats["summarization"]["rejected_insufficient"] == 1
    assert stories == []
