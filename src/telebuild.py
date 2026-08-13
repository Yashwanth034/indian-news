"""India Telegram queue builder / orchestrator (Phase 2C).

Consumes the India pipeline queue (``PipelineResult.queue`` of
``Candidate`` objects) and emits ``data/telegram_queue.json`` in the
shape consumed by the Telegram formatter and scheduler:

    {
        "generated_at": "2026-08-12T12:00:00+00:00",
        "count": 1,
        "stories": [ ... ]
    }

Flow (mirrors the proven WorldNews ``main.py`` orchestration):

    candidates -> Candidate.to_dict() (India metadata preserved)
    -> non-news source gate (sources.json "news": false)
    -> freshness window (0 <= age <= freshness_hours)
    -> max_candidates cap (priority order preserved)
    -> best-effort article enrichment (ArticleCache SQLite cache)
    -> group into events -> briefing -> source-grounded summarization
    -> telegram queue JSON

Nothing is generated or invented: the story representation keeps the
India candidate metadata (event_id, story_id, headline/title, summary,
url, source, tier, primary_source, category, secondary categories,
geo_scope, state, national_significance, scores, priority level,
confidence, source groups, effective_at, event time, published time)
and adds briefing/summarization/enrichment metadata.  Every failure
degrades to the plain RSS briefing; the pipeline never fails because
of the orchestrator.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from src.article_extractor import (
    ArticleCache,
    enrich_thin_stories,
)
from src.config_loader import ROOT
from src.telegram_briefing import (
    build_briefing,
    group_items,
)
from src.telegram_summarizer import (
    ARTICLE_ITEM_SUFFIX,
    summarize_rows,
)


def candidate_to_dict(candidate):
    """Convert a Candidate (or dict) to a Telegram queue dict.

    Preserves the WorldNews-shaped fields produced by
    ``Candidate.to_dict()`` and adds the India geo metadata that
    lives on the candidate's GeoResult.
    """
    if isinstance(candidate, dict):
        return dict(candidate)

    d = candidate.to_dict()

    geo = getattr(candidate, "geo", None)
    if geo is not None:
        d["geo_scope"] = geo.scope
        d["state"] = geo.state
        d["national_significance"] = geo.national_significance

    return d


def telegram_ineligible_sources(config):
    """Source names flagged non-news ("news": false) in sources.json.

    These sources never produce Telegram posts, even when their
    items otherwise pass the pipeline queue.
    """
    sources = (
        (config.get("sources") or {})
        .get("sources")
        or []
    )
    return {
        source.get("name") or source.get("id")
        for source in sources
        if not source.get("news", True)
    }


def _parse_effective_at(value):
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def filter_telegram_candidates(candidates, config, now_dt):
    """Apply the Telegram candidate gate, preserving priority order.

    Returns (kept, stats).  Kept candidates are dicts with a
    normalized ``story_id`` (``story_id`` or ``id``), in the same
    relative order as the input queue.  No re-sorting happens: the
    pipeline already emits a deterministic priority-ordered queue.
    """
    telegram_cfg = (
        (config.get("config") or {})
        .get("telegram")
        or {}
    )

    freshness_hours = float(
        telegram_cfg.get("freshness_hours", 6)
    )

    max_candidates = int(
        telegram_cfg.get("max_candidates", 50)
    )

    no_news_sources = telegram_ineligible_sources(config)

    stats = {
        "candidates": 0,
        "non_news_filtered": 0,
        "no_effective_at": 0,
        "stale": 0,
        "fresh": 0,
        "kept": 0,
    }

    kept = []

    for x in candidates:
        stats["candidates"] += 1

        # Only publishable (queued) candidates ever enter the
        # Telegram layer.
        if x.get("status") not in (None, "queued"):
            continue

        if x.get("source") in no_news_sources:
            stats["non_news_filtered"] += 1
            continue

        effective_dt = _parse_effective_at(
            x.get("effective_at")
        )

        if effective_dt is None:
            stats["no_effective_at"] += 1
            continue

        age_seconds = (
            now_dt - effective_dt
        ).total_seconds()

        if not (
            0
            <= age_seconds
            <= freshness_hours * 3600
        ):
            stats["stale"] += 1
            continue

        stats["fresh"] += 1

        candidate = dict(x)

        # Normalize the dedup key: the pipeline stores it as
        # "id"; the telegram layer keys off "story_id".
        candidate["story_id"] = (
            candidate.get("story_id")
            or candidate.get("id")
        )

        kept.append(candidate)

    if len(kept) > max_candidates:
        kept = kept[:max_candidates]

    stats["kept"] = len(kept)

    return kept, stats


def build_telegram_stories(
    candidates,
    telegram_cfg,
    now_dt,
    summarization_stats=None,
):
    """Group candidates into events and build one story per event.

    Mirrors the WorldNews ``build_telegram_stories`` behavior: the
    primary candidate is chosen by score/primary_source/tier, the
    article text (when >= 2 useful sentences) is attached as a
    higher-scoring event member, the source-grounded briefing is
    built, and the summarizer composes + verifies + quality-gates the
    2-5 sentence summary.  Stories with fewer than two genuinely
    useful sentences are rejected, never padded.
    """
    if summarization_stats is not None:
        summarization_stats.update(
            {
                "stories_considered": 0,
                "summarized": 0,
                "article_source": 0,
                "rss_source": 0,
                "rejected_insufficient": 0,
                "rejected_verification": 0,
                "rejected_quality": 0,
                "sentences_composed": 0,
                "sentences_verify_dropped": 0,
                "sentences_quality_dropped": 0,
                "problems": [],
            }
        )

    groups = group_items(candidates)
    telegram_stories = []

    for group in groups:
        primary = sorted(
            group,
            key=lambda x: (
                x.get("score", 0)
                or x.get("priority_score", 0)
                or 0,
                int(bool(x.get("primary_source"))),
                -int(x.get("tier", 4)),
            ),
            reverse=True,
        )[0]

        article_sentences = primary.get("article_sentences") or []
        article_item_ids = set()
        briefing_group = group

        if len(article_sentences) >= 2:
            article_member = dict(primary)
            enrichment_source = primary.get(
                "enrichment_source"
            )
            if enrichment_source:
                article_member["source"] = enrichment_source
            article_member["id"] = (
                primary.get("id")
                or primary.get("story_id")
            ) + ARTICLE_ITEM_SUFFIX
            article_member["story_id"] = article_member["id"]
            article_member["summary"] = " ".join(article_sentences)
            article_member["score"] = (
                primary.get("score")
                or primary.get("priority_score")
                or 0
            ) + 1
            briefing_group = group + [article_member]
            article_item_ids.add(article_member["id"])

        briefing = build_briefing(
            primary,
            briefing_group,
            int(telegram_cfg.get("just_in_freshness_minutes", 15)),
            now_dt,
            max_sentences=(
                int(telegram_cfg.get("max_briefing_sentences", 10))
                if len(article_sentences) >= 2
                else None
            ),
        )

        if not briefing["sentences"]:
            if summarization_stats is not None:
                summarization_stats["rejected_insufficient"] += 1
            continue

        source_parts = []
        if len(article_sentences) >= 2:
            source_parts.extend(article_sentences)
        for member in group:
            summary = member.get("summary")
            if summary:
                source_parts.append(summary)

        summarization_cfg = telegram_cfg.get("summarization", {})

        if summarization_stats is not None:
            summarization_stats["stories_considered"] += 1

        summary_rows, summary_stats = summarize_rows(
            briefing["sentences"],
            " ".join(source_parts),
            briefing["headline"],
            article_item_ids=article_item_ids,
            cfg=summarization_cfg,
        )

        if summary_rows is None:
            if summarization_stats is not None:
                reason = summary_stats.get("rejected") or "insufficient_information"
                if reason == "verification":
                    summarization_stats["rejected_verification"] += 1
                elif reason == "quality":
                    summarization_stats["rejected_quality"] += 1
                else:
                    summarization_stats["rejected_insufficient"] += 1
            continue

        if summarization_stats is not None:
            summarization_stats["summarized"] += 1
            if len(article_sentences) >= 2:
                summarization_stats["article_source"] += 1
            else:
                summarization_stats["rss_source"] += 1
            summarization_stats["sentences_composed"] += len(summary_rows)
            summarization_stats["sentences_verify_dropped"] += len(
                summary_stats["verify_problems"]
            )
            summarization_stats["sentences_quality_dropped"] += len(
                summary_stats["quality_problems"]
            )
            for problem in summary_stats["verify_problems"]:
                summarization_stats["problems"].append(
                    {
                        "story": primary.get("story_id", "?")[:8],
                        "stage": "verify",
                        "text": problem.get("text"),
                        "problems": problem.get("problems"),
                    }
                )
            for problem in summary_stats["quality_problems"]:
                summarization_stats["problems"].append(
                    {
                        "story": primary.get("story_id", "?")[:8],
                        "stage": "quality",
                        "text": problem.get("text"),
                        "problems": problem.get("problems"),
                    }
                )

        enriched = dict(primary)
        enriched["story_id"] = primary.get("story_id") or primary.get("id")
        enriched["public_label"] = briefing["label"]
        enriched["headline"] = briefing["headline"]
        enriched["briefing"] = {
            "opening": briefing["opening"],
            "body": briefing["body"],
            "bullets": briefing["bullets"],
            "sentences": summary_rows,
            "source": briefing["source"],
            "corroborating": briefing["corroborating"],
            "url": briefing["url"],
        }
        enriched["group_size"] = len(group)
        enriched["label"] = briefing["label"]
        enriched["enrichment"] = (
            "article"
            if len(article_sentences) >= 2
            else "rss"
        )
        telegram_stories.append(enriched)

    return telegram_stories


def _enrich_stories(candidates, config, now_dt, cache=None, fetcher=None):
    """Best-effort article enrichment.

    Every failure degrades to the plain RSS briefing; the pipeline
    never fails because of extraction.
    """
    if cache is None:
        db_path = ROOT / (
            (config.get("config") or {})
            .get("database")
            or "data/news.db"
        )
        db_path.parent.mkdir(parents=True, exist_ok=True)
        cache = ArticleCache(db_path)

    return enrich_thin_stories(
        candidates,
        config,
        now_dt,
        cache=cache,
        fetcher=fetcher,
    )


def _queue_path(config, queue_path=None):
    if queue_path is not None:
        return Path(queue_path)

    telegram_cfg = (
        (config.get("config") or {})
        .get("telegram")
        or {}
    )

    rel = (
        telegram_cfg.get("telegram_queue_file")
        or (config.get("config") or {}).get(
            "telegram_queue_file",
            "data/telegram_queue.json",
        )
    )

    return ROOT / rel


def build_telegram_queue(
    pipeline_result,
    config,
    now_dt=None,
    cache=None,
    fetcher=None,
    queue_path=None,
):
    """Build the India Telegram queue and write it to disk.

    Accepts a ``PipelineResult`` (its ``queue`` is consumed) or a
    plain list of Candidate objects / dicts.  Returns
    ``(stories, stats)``.  Writes ``data/telegram_queue.json`` as
    ``{"generated_at": ..., "count": N, "stories": [...]}`` unless a
    ``queue_path`` is given.
    """
    now_dt = now_dt or datetime.now(timezone.utc)

    if hasattr(pipeline_result, "queue"):
        candidates = list(pipeline_result.queue)
    else:
        candidates = list(pipeline_result)

    candidates = [candidate_to_dict(c) for c in candidates]

    candidates, filter_stats = filter_telegram_candidates(
        candidates, config, now_dt
    )

    try:
        candidates, article_extraction_stats = _enrich_stories(
            candidates, config, now_dt,
            cache=cache,
            fetcher=fetcher,
        )
    except Exception as exc:
        article_extraction_stats = {
            "error": str(exc),
        }

    telegram_cfg = (
        (config.get("config") or {})
        .get("telegram")
        or {}
    )

    telegram_cfg_briefing = dict(telegram_cfg)
    telegram_cfg_briefing["max_briefing_sentences"] = int(
        (config.get("config") or {})
        .get("article_extraction", {})
        .get("max_briefing_sentences", 10)
    )
    telegram_cfg_briefing["summarization"] = (
        (config.get("config") or {})
        .get("summarization", {})
    )

    summarization_stats = {}

    stories = build_telegram_stories(
        candidates,
        telegram_cfg_briefing,
        now_dt,
        summarization_stats=summarization_stats,
    )

    path = _queue_path(config, queue_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            {
                "generated_at": now_dt.isoformat(),
                "count": len(stories),
                "stories": stories,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    stats = {
        "filter": filter_stats,
        "article_extraction": article_extraction_stats,
        "summarization": summarization_stats,
    }

    return stories, stats
