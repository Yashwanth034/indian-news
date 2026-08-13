"""Regression tests for Telegram queue-build observability.

``python -m src.main`` now prints the existing
``build_telegram_queue`` stats (filter / article extraction /
summarization) so production logs explain why a run produced zero
stories.  These tests prove:

- every required category is surfaced (received, freshness,
  enrichment, summarization, final story count)
- each rejection category is visible enough to diagnose a
  zero-story run
- ``main()`` prints the observability block without changing the
  queue file written by ``build_telegram_queue``
- the underlying ``build_telegram_queue`` behavior is unchanged
"""
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from src.config_loader import get_config
from src.main import (
    _telegram_observability_lines,
)
from src.pipeline.integration import NewsPipeline
from src.telebuild import build_telegram_queue

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def bundle():
    return get_config()


@pytest.fixture
def tmp_cache(tmp_path):
    from src.article_extractor import ArticleCache
    return ArticleCache(tmp_path / "test_cache.db")


def _art(source_id, title, summary="", *, tier=2, role="journalism",
         published=NOW, url=None):
    from src.models.article import Article
    return Article(
        source_id=source_id,
        source_name=source_id.replace("-", " ").title(),
        tier=tier,
        source_role=role,
        url=url or f"https://{source_id}.example.in/{abs(hash(title))}.html",
        title=title,
        summary=summary or None,
        published=published,
    )


def _thin_pipeline_result():
    return NewsPipeline().run(
        [_art("the-hindu", "Major earthquake of magnitude 7.2 strikes Delhi, rescue underway",
              summary=None,
              url="https://www.thehindu.com/news/national/delhi-earthquake")],
        now=NOW,
    )


_ARTICLE_TEXT = (
    "Rescue teams reached the affected areas within minutes. "
    "At least 40 people were injured and several buildings "
    "collapsed. Hospitals in the capital reported they were "
    "overwhelmed with casualties."
)


# ---------------------------------------------------------------------------
# unit: the observability formatter
# ---------------------------------------------------------------------------

def _full_stats():
    """A stats dict shaped exactly like build_telegram_queue returns."""
    return {
        "filter": {
            "candidates": 3,
            "non_news_filtered": 1,
            "no_effective_at": 1,
            "stale": 1,
            "fresh": 2,
            "kept": 2,
        },
        "article_extraction": {
            "enabled": True,
            "candidates": 3,
            "eligible": 2,
            "thin": 2,
            "important": 2,
            "mass_casualty": 0,
            "non_article": 0,
            "domain_blocked": 0,
            "cache_hits": 1,
            "fetched": 1,
            "ok": 0,
            "http_error": 0,
            "blocked": 1,
            "paywall": 0,
            "no_text": 0,
            "timeout": 0,
            "too_large": 0,
            "not_html": 0,
            "network_error": 0,
            "expanded": 1,
            "not_expanded": 1,
            "budget_exhausted": 0,
        },
        "summarization": {
            "stories_considered": 2,
            "summarized": 1,
            "article_source": 1,
            "rss_source": 0,
            "rejected_insufficient": 1,
            "rejected_verification": 0,
            "rejected_quality": 0,
            "sentences_composed": 3,
            "sentences_verify_dropped": 0,
            "sentences_quality_dropped": 0,
            "problems": [{
                "story": "abc12345",
                "stage": "quality",
                "text": "The NCS also directed users to the app",
                "problems": ["filler sentence"],
            }],
        },
    }


def test_observability_reports_all_categories():
    lines = _telegram_observability_lines(_full_stats())
    joined = "\n".join(lines)

    assert "telegram candidates received=3" in joined
    assert "telegram freshness fresh=2 stale=1 no_effective_at=1 non_news=1 kept=2" in joined
    assert (
        "telegram enrichment eligible=2 expanded=1 cache_hits=1 "
        "fetched=1 errors=1" in joined
    )
    assert "blocked=1" in joined
    assert (
        "telegram summarization considered=2 summarized=1 "
        "article_source=1 rss_source=0 "
        "rejected_insufficient=1 rejected_verification=0 "
        "rejected_quality=0" in joined
    )
    assert "telegram summarization problem story=abc12345" in joined
    assert "stage=quality" in joined
    assert "reasons='filler sentence'" in joined


def test_observability_makes_zero_story_run_diagnosable():
    stats = _full_stats()
    stats["filter"].update({
        "candidates": 2, "fresh": 0, "stale": 2, "kept": 0,
        "non_news_filtered": 0, "no_effective_at": 0,
    })
    stats["summarization"].update({
        "stories_considered": 0, "summarized": 0,
        "rejected_insufficient": 0,
        "rejected_verification": 0, "rejected_quality": 0,
    })
    lines = _telegram_observability_lines(stats)
    joined = "\n".join(lines)

    assert "fresh=0 stale=2" in joined
    assert "kept=0" in joined
    assert "summarized=0" in joined
    assert "rejected_insufficient=0" in joined


def test_observability_shows_enrichment_error_detail_only_when_present():
    stats = _full_stats()
    lines = _telegram_observability_lines(stats)
    assert any(line.startswith("telegram enrichment errors: ") for line in lines)
    assert any("blocked=1" in line and line.startswith("telegram enrichment errors:")
               for line in lines)


def test_observability_handles_missing_sections():
    lines = _telegram_observability_lines({})
    joined = "\n".join(lines)
    assert "telegram candidates received=0" in joined
    assert "telegram enrichment eligible=0" in joined
    assert "telegram summarization considered=0" in joined


def test_observability_surfaces_enrichment_fatal_error():
    lines = _telegram_observability_lines({
        "article_extraction": {"error": "boom"},
    })
    assert any(line == "telegram enrichment error=boom" for line in lines)


# ---------------------------------------------------------------------------
# integration: real build_telegram_queue stats drive the formatter
# ---------------------------------------------------------------------------

def test_zero_story_run_stats_render_rejection_reasons(bundle, tmp_path, tmp_cache):
    result = _thin_pipeline_result()

    def _boom(url, art_cfg, allowlist, robots=None, pace=None):
        raise RuntimeError("network down")

    out = tmp_path / "q.json"
    stories, stats = build_telegram_queue(
        result, bundle, now_dt=NOW, queue_path=out,
        cache=tmp_cache, fetcher=_boom,
    )
    assert stories == []

    lines = _telegram_observability_lines(stats)
    joined = "\n".join(lines)
    assert "network_error=1" in joined
    assert "summarized=0" in joined


def test_success_run_stats_render_enrichment_and_summary(bundle, tmp_path, tmp_cache):
    result = _thin_pipeline_result()

    def _ok(url, art_cfg, allowlist, robots=None, pace=None):
        return ("ok", {"text": _ARTICLE_TEXT, "title": None})

    out = tmp_path / "q.json"
    stories, stats = build_telegram_queue(
        result, bundle, now_dt=NOW, queue_path=out,
        cache=tmp_cache, fetcher=_ok,
    )
    assert len(stories) == 1

    lines = _telegram_observability_lines(stats)
    joined = "\n".join(lines)
    assert "telegram enrichment eligible=1 expanded=1 cache_hits=0 fetched=1 errors=0" in joined
    assert "telegram summarization considered=1 summarized=1" in joined


def test_observability_does_not_change_queue_output(bundle, tmp_path, tmp_cache):
    """build_telegram_queue writes the same queue file with and without
    the observability formatter involved."""
    result = _thin_pipeline_result()

    def _ok(url, art_cfg, allowlist, robots=None, pace=None):
        return ("ok", {"text": _ARTICLE_TEXT, "title": None})

    out = tmp_path / "q.json"
    stories, stats = build_telegram_queue(
        result, bundle, now_dt=NOW, queue_path=out,
        cache=tmp_cache, fetcher=_ok,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["count"] == len(stories)
    assert payload["stories"] == stories
    # The observability lines are derived from stats after the fact and
    # never mutate them.
    before = json.dumps(stats, sort_keys=True)
    _telegram_observability_lines(stats)
    after = json.dumps(stats, sort_keys=True)
    assert before == after


# ---------------------------------------------------------------------------
# integration: main() prints the block via its normal call path
# ---------------------------------------------------------------------------

def test_main_prints_observability_block(capsys, monkeypatch, tmp_path):
    import src.main as main_mod

    stats = _full_stats()

    class _FakeReport:
        articles = []
        ok_sources = ["the-hindu"]
        failed_sources = []

        def status(self, source_id):
            return "ok"

    class _FakePipeline:
        def __init__(self, bundle):
            pass

        def run(self, articles, now=None):
            return SimpleNamespace(
                collected=5,
                normalized=5,
                relevant=3,
                events=2,
                candidates=[],
                queue=[],
            )

    monkeypatch.setattr(main_mod, "collect_sources", lambda sources, health=None: _FakeReport())
    monkeypatch.setattr(main_mod, "NewsPipeline", _FakePipeline)
    monkeypatch.setattr(main_mod, "build_telegram_queue", lambda result, bundle, now_dt=None: ([], stats))
    monkeypatch.setattr(main_mod, "ROOT", tmp_path)

    assert main_mod.main() == 0

    out = capsys.readouterr().out
    assert "main: telegram candidates received=3" in out
    assert "main: telegram freshness" in out
    assert "main: telegram enrichment" in out
    assert "main: telegram summarization" in out
    assert "main: telegram queue written with 0 stories" in out


def test_main_preserves_existing_output_lines(capsys, monkeypatch, tmp_path):
    import src.main as main_mod

    class _FakeReport:
        articles = []
        ok_sources = ["the-hindu"]
        failed_sources = []

        def status(self, source_id):
            return "ok"

    class _FakePipeline:
        def __init__(self, bundle):
            pass

        def run(self, articles, now=None):
            return SimpleNamespace(
                collected=5,
                normalized=5,
                relevant=3,
                events=2,
                candidates=[],
                queue=[],
            )

    monkeypatch.setattr(main_mod, "collect_sources", lambda sources, health=None: _FakeReport())
    monkeypatch.setattr(main_mod, "NewsPipeline", _FakePipeline)
    monkeypatch.setattr(main_mod, "build_telegram_queue",
                        lambda result, bundle, now_dt=None: ([], _full_stats()))
    monkeypatch.setattr(main_mod, "ROOT", tmp_path)

    assert main_mod.main() == 0

    out = capsys.readouterr().out
    assert "main: collecting from" in out
    assert "main: collected 0 articles" in out
    assert "main: pipeline collected=5 normalized=5 relevant=3 events=2 candidates=0 queued=0" in out
    assert "main: telegram queue written with 0 stories" in out
