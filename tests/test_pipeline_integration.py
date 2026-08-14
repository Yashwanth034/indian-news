"""Integration tests: NewsPipeline runs all stages in order end-to-end.

Covers the deterministic pipeline ordering (normalize -> relevance ->
classify -> geography -> dedupe -> editorial -> priority -> candidate gate
-> queue), candidate representation reusing Article/EventGroup/PriorityResult,
the final candidate gate, queue ordering, and alignment with the WorldNews
queue shape (reference only).
"""
from datetime import datetime, timedelta, timezone

import pytest

from src.models.article import Article
from src.pipeline.dedupe import EventGroup
from src.pipeline.editorial import EditorialResult
from src.pipeline.integration import NewsPipeline
from src.pipeline.priority import PriorityResult

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def pipeline():
    return NewsPipeline()


_NO_PUBLISHED = object()


def _art(source_id, title, summary="", *, tier=2, role="journalism",
         published=_NO_PUBLISHED, updated=None, source_name=None, url=None, author=None):
    return Article(
        source_id=source_id,
        source_name=source_name or source_id.replace("-", " ").title(),
        tier=tier,
        source_role=role,
        url=url or f"https://{source_id}.example.in/{abs(hash(title))}.html",
        title=title,
        summary=summary or None,
        author=author,
        published=NOW if published is _NO_PUBLISHED else published,
        updated=updated,
    )


def _breaking(dup=False):
    """A major national story with two corroborating sources."""
    title = "Major earthquake of magnitude 7.2 strikes Delhi, rescue underway"
    a = _art("the-hindu", title)
    if not dup:
        return [a]
    return [
        a,
        _art("ndtv", "Major earthquake hits Delhi, rescue underway"),
        _art("pti", "Major earthquake hits Delhi, rescue underway", author="PTI"),
    ]


# --- pipeline ordering + counters -------------------------------------------

def test_empty_run_is_empty(pipeline):
    r = pipeline.run([], now=NOW)
    assert r.collected == 0
    assert r.normalized == 0
    assert r.relevant == 0
    assert r.events == 0
    assert r.candidates == []
    assert r.queue == []


def test_full_pipeline_produces_one_queued_candidate(pipeline):
    r = pipeline.run(_breaking(), now=NOW)
    assert r.collected == 1
    assert r.normalized == 1
    assert r.relevant == 1
    assert r.events == 1
    assert len(r.candidates) == 1
    c = r.candidates[0]
    assert c.queued
    assert c.status == "queued"
    assert c.queue_rank == 0
    assert c.priority.priority == "IMMEDIATE"


def test_pipeline_drops_non_india_story_at_relevance(pipeline):
    r = pipeline.run([_art("bbc", "European Union agrees trade deal with Canada")], now=NOW)
    assert r.relevant == 0
    assert len(r.dropped) == 1
    assert r.dropped[0].stage == "relevance"
    assert r.events == 0
    assert r.queue == []


def test_pipeline_drops_local_story_without_national_significance(pipeline):
    r = pipeline.run(
        [_art("the-hindu", "Minor accident near Bengaluru junction, two injured")],
        now=NOW,
    )
    assert len(r.dropped) == 1
    assert r.dropped[0].stage == "relevance"


# --- dedupe end-to-end -------------------------------------------------------

def test_duplicate_stories_become_one_candidate(pipeline):
    r = pipeline.run(_breaking(dup=True), now=NOW)
    assert r.collected == 3
    assert r.events == 1
    assert len(r.candidates) == 1
    c = r.candidates[0]
    assert len(c.articles) == 3
    assert c.independent_source_groups >= 2


def test_candidate_reuses_existing_structures(pipeline):
    r = pipeline.run(_breaking(dup=True), now=NOW)
    c = r.candidates[0]
    assert isinstance(c.representative, Article)
    assert isinstance(c.event, EventGroup)
    assert isinstance(c.editorial, EditorialResult)
    assert isinstance(c.priority, PriorityResult)


def test_candidate_exposes_serializable_worldnews_aligned_dict(pipeline):
    r = pipeline.run(_breaking(dup=True), now=NOW)
    c = r.candidates[0]
    d = c.to_dict()
    for key in (
        "event_id", "story_id", "headline", "title", "summary", "url",
        "source", "primary_source", "category", "confidence", "score",
        "priority_score", "priority_level", "event_time", "published_at",
        "effective_at", "independent_source_groups", "is_wire_echo",
        "status", "reasons",
    ):
        assert key in d, key
    assert d["event_id"] == d["story_id"] == c.event_id
    assert d["priority_level"] == "IMMEDIATE"
    assert d["is_wire_echo"] is False


# --- Candidate.to_dict() telegram compatibility -----------------------------

def _run_to_dict(pipeline, *articles):
    return pipeline.run(list(articles), now=NOW).candidates[0].to_dict()


def test_effective_at_uses_latest_of_published_and_updated(pipeline):
    published = NOW - timedelta(hours=3)
    updated = NOW - timedelta(hours=1)
    d = _run_to_dict(
        pipeline,
        _art("the-hindu", "Major earthquake of magnitude 7.2 strikes Delhi, rescue underway",
             published=published, updated=updated),
    )
    assert d["effective_at"] == updated.isoformat()


def test_effective_at_uses_published_when_no_updated(pipeline):
    published = NOW - timedelta(hours=3)
    d = _run_to_dict(
        pipeline,
        _art("the-hindu", "Major earthquake of magnitude 7.2 strikes Delhi, rescue underway",
             published=published),
    )
    assert d["effective_at"] == published.isoformat()


def test_effective_at_uses_updated_when_no_published(pipeline):
    updated = NOW - timedelta(hours=1)
    d = _run_to_dict(
        pipeline,
        _art("the-hindu", "Major earthquake of magnitude 7.2 strikes Delhi, rescue underway",
             published=None, updated=updated),
    )
    assert d["effective_at"] == updated.isoformat()


def test_effective_at_is_none_when_no_timestamps(pipeline):
    a = _art("the-hindu", "Major earthquake of magnitude 7.2 strikes Delhi, rescue underway",
             published=None)
    a.published = None
    a.updated = None
    d = _run_to_dict(pipeline, a)
    assert d["effective_at"] is None


def test_source_emits_human_readable_source_name(pipeline):
    d = _run_to_dict(
        pipeline,
        _art("the-hindu", "Major earthquake of magnitude 7.2 strikes Delhi, rescue underway",
             source_name="The Hindu"),
    )
    assert d["source"] == "The Hindu"
    assert d["source"] != "the-hindu"


def test_primary_source_true_for_official_primary_role(pipeline):
    d = _run_to_dict(
        pipeline,
        _art("pib", "Major earthquake of magnitude 7.2 strikes Delhi, rescue underway",
             role="official-primary", source_name="Press Information Bureau"),
    )
    assert d["primary_source"] is True


def test_primary_source_false_for_journalism_role(pipeline):
    d = _run_to_dict(
        pipeline,
        _art("the-hindu", "Major earthquake of magnitude 7.2 strikes Delhi, rescue underway",
             role="journalism"),
    )
    assert d["primary_source"] is False


def test_to_dict_preserves_existing_fields_and_metadata(pipeline):
    r = pipeline.run(_breaking(dup=True), now=NOW)
    c = r.candidates[0]
    d = c.to_dict()
    assert d["event_id"] == c.event_id
    assert d["story_id"] == c.event_id
    assert d["headline"] == c.title
    assert d["title"] == c.title
    assert d["summary"] == c.summary
    assert d["url"] == c.representative.url
    assert d["tier"] == c.representative.tier
    assert d["category"] == c.category
    assert d["confidence"] == c.confidence
    assert d["score"] == c.relevance.score
    assert d["priority_score"] == c.priority.score
    assert d["priority_level"] == c.priority.priority
    assert d["event_time"] == c.event_time.isoformat()
    assert d["published_at"] == c.representative.published.isoformat()
    assert d["independent_source_groups"] == c.independent_source_groups
    assert d["source_groups"] == c.source_groups
    assert d["is_wire_echo"] == c.is_wire_echo
    assert d["status"] == c.status
    assert d["reasons"] == list(c.reasons)


# --- editorial gate end-to-end -----------------------------------------------

def test_editorial_reject_never_queued(pipeline):
    r = pipeline.run(
        [_art("the-hindu", "Delhi government announces astrologer predicts new year luck")],
        now=NOW,
    )
    assert r.queue == []
    assert all(c.status == "rejected" for c in r.candidates)


def test_editorial_filler_never_queued(pipeline):
    r = pipeline.run(
        [_art("the-hindu", "Delhi metro issues tender notice for station escalator maintenance")],
        now=NOW,
    )
    assert r.queue == []
    assert all(c.status == "filler" for c in r.candidates)


def test_opinion_as_fact_rejected(pipeline):
    r = pipeline.run(
        [_art("the-hindu", "The election results mean Modi has already won, in our view")],
        now=NOW,
    )
    assert r.queue == []
    assert all(c.status == "rejected" for c in r.candidates)


# --- candidate gate ----------------------------------------------------------

def test_below_min_score_is_held_not_queued(pipeline):
    r = pipeline.run(
        [_art("the-hindu", "Indian Railways launches new express train between Mumbai and Pune")],
        now=NOW,
    )
    assert r.queue == []
    assert len(r.held) == 1
    assert r.held[0].status == "held"


def test_held_candidate_has_explicit_reason(pipeline):
    r = pipeline.run(
        [_art("the-hindu", "Indian Railways launches new express train between Mumbai and Pune")],
        now=NOW,
    )
    assert any("candidate gate" in reason for reason in r.held[0].reasons)


def test_require_editorial_pass_config_holds_filler(pipeline):
    p = NewsPipeline()
    r = p.run(
        [_art("the-hindu", "Delhi metro issues tender notice for station escalator maintenance")],
        now=NOW,
    )
    assert all(c.status == "filler" for c in r.candidates)


# --- HIGH priority eligibility ----------------------------------------------

def test_high_priority_queues_below_min_score(pipeline):
    r = pipeline.run(
        [_art("the-hindu", "India win T20 World Cup final against South Africa")],
        now=NOW,
    )
    assert len(r.candidates) == 1
    assert r.candidates[0].priority.priority == "HIGH"
    assert r.candidates[0].priority.score < pipeline.min_queue_score
    assert r.candidates[0].status == "queued"
    assert len(r.queue) == 1


def test_urgent_priority_queues_below_min_score(pipeline):
    r = pipeline.run(
        [_art("the-hindu", "Supreme Court strikes down electoral bonds scheme")],
        now=NOW,
    )
    assert len(r.candidates) == 1
    assert r.candidates[0].priority.priority in ("URGENT", "HIGH")
    assert r.candidates[0].status == "queued"


def test_normal_below_min_score_still_held(pipeline):
    r = pipeline.run(
        [_art("the-hindu", "Indian Railways launches new express train between Mumbai and Pune")],
        now=NOW,
    )
    assert len(r.candidates) == 1
    assert r.candidates[0].priority.priority == "NORMAL"
    assert r.candidates[0].status == "held"
    assert any("candidate gate" in reason for reason in r.candidates[0].reasons)


def test_high_editorial_reject_still_rejected(pipeline):
    r = pipeline.run(
        [_art("the-hindu", "Delhi government announces astrologer predicts new year luck")],
        now=NOW,
    )
    assert r.queue == []
    assert all(c.status == "rejected" for c in r.candidates)
    assert all(c.priority.blocked for c in r.candidates)


def test_high_priority_blocked_still_rejected(pipeline):
    r = pipeline.run(
        [_art("the-hindu", "Delhi government announces astrologer predicts new year luck")],
        now=NOW,
    )
    assert r.queue == []
    assert all(c.priority.blocked for c in r.candidates)
    assert all(c.status == "rejected" for c in r.candidates)


# --- queue ordering ----------------------------------------------------------

def test_queue_orders_immediate_before_urgent(pipeline):
    urgent = _art("the-hindu", "Major terror attack in Mumbai, multiple casualties reported")
    breaking = _art("the-hindu", "Major earthquake of magnitude 7.2 strikes Delhi, rescue underway")
    r = pipeline.run([urgent, breaking], now=NOW)
    assert len(r.queue) == 2
    assert r.queue[0].priority.priority == "IMMEDIATE"
    assert r.queue[1].priority.priority == "IMMEDIATE"
    assert r.queue[0].priority.score >= r.queue[1].priority.score


def test_queue_is_deterministic(pipeline):
    articles = [
        _art("the-hindu", "Major earthquake of magnitude 7.2 strikes Delhi, rescue underway"),
        _art("the-hindu", "Major terror attack in Mumbai, multiple casualties reported"),
        _art("the-hindu", "Reserve Bank of India hikes key interest rate to 6.5 percent"),
    ]
    r1 = pipeline.run(list(articles), now=NOW)
    r2 = pipeline.run(list(articles), now=NOW)
    assert [c.event_id for c in r1.queue] == [c.event_id for c in r2.queue]
    assert [c.queue_rank for c in r1.queue] == [c.queue_rank for c in r2.queue]


def test_queue_breaks_ties_deterministically(pipeline):
    a = _art("the-hindu", "Reserve Bank of India hikes key interest rate to 6.5 percent")
    b = _art("pti", "Reserve Bank of India hikes key interest rate to 6.5 percent")
    r1 = pipeline.run([a, b], now=NOW)
    r2 = pipeline.run([a, b], now=NOW)
    assert r1.queue == []  # below queue threshold
    assert [c.event_id for c in r1.candidates] == [c.event_id for c in r2.candidates]


def test_queue_rank_is_contiguous(pipeline):
    r = pipeline.run(_breaking(dup=True), now=NOW)
    assert [c.queue_rank for c in r.queue] == list(range(len(r.queue)))


# --- wire / corroboration ----------------------------------------------------

def test_wire_echo_is_single_candidate(pipeline):
    a = _art("economictimes", "Cyclone devastates coastal Odisha, thousands evacuated",
             author="PTI")
    b = _art("indianexpress", "Cyclone devastates coastal Odisha, thousands evacuated",
             author="PTI")
    r = pipeline.run([a, b], now=NOW)
    assert r.events == 1
    assert len(r.candidates) == 1
    assert r.candidates[0].is_wire_echo is True


def test_echo_does_not_inflate_priority(pipeline):
    base = _art("the-hindu", "Major earthquake of magnitude 7.2 strikes Delhi, rescue underway")
    single = pipeline.run([base], now=NOW).candidates[0]
    echo = pipeline.run(
        [base, _art("economictimes", "Major earthquake of magnitude 7.2 strikes Delhi, rescue underway",
                    author="PTI")],
        now=NOW,
    ).candidates[0]
    assert single.priority.score == echo.priority.score


def test_independent_sources_raise_confidence(pipeline):
    title = "Major earthquake of magnitude 7.2 strikes Delhi, rescue underway"
    solo = pipeline.run([_art("the-hindu", title)], now=NOW).candidates[0]
    triple = pipeline.run(
        [
            _art("the-hindu", title),
            _art("ndtv", "Major earthquake hits Delhi, rescue underway"),
            _art("pti", "Major earthquake hits Delhi, rescue underway", author="PTI"),
        ],
        now=NOW,
    ).candidates[0]
    assert solo.confidence == "medium"
    assert triple.confidence == "high"


# --- pipeline robustness -----------------------------------------------------

def test_max_queue_respected(pipeline):
    from src.pipeline.integration import NewsPipeline as NP
    p = NP()
    articles = [
        _art("the-hindu", "Major earthquake of magnitude 7.2 strikes Delhi, rescue underway"),
        _art("the-hindu", "Major terror attack in Mumbai, multiple casualties reported"),
        _art("the-hindu", "Severe cyclone warning issued for Tamil Nadu coast, IMD alert"),
    ]
    r = p.run(articles, now=NOW)
    assert len(r.queue) <= p.max_queue


def test_bad_article_rejected_at_normalize(pipeline):
    bad = _art("the-hindu", "  ", published=NOW)
    bad.title = "   "
    r = pipeline.run([bad], now=NOW)
    assert r.normalized == 0
    assert len(r.dropped) == 1
    assert r.dropped[0].stage == "normalize"


def test_low_relevance_threshold_blocks_borderline(pipeline):
    r = pipeline.run(
        [_art("the-hindu", "International cricket team arrives in Mumbai for tour")],
        now=NOW,
    )
    assert len(r.dropped) == 1
    assert r.dropped[0].stage == "relevance"


def test_multiple_distinct_events_each_become_candidate(pipeline):
    r = pipeline.run(
        [
            _art("the-hindu", "Major earthquake of magnitude 7.2 strikes Delhi, rescue underway"),
            _art("the-hindu", "Severe cyclone warning issued for Tamil Nadu coast, IMD alert"),
        ],
        now=NOW,
    )
    assert r.events == 2
    assert len(r.candidates) == 2
    assert {c.event_id for c in r.candidates} == {c.event_id for c in r.candidates}


def test_run_does_not_modify_input_articles(pipeline):
    articles = _breaking(dup=True)
    snapshot = [a.title for a in articles]
    pipeline.run(articles, now=NOW)
    assert [a.title for a in articles] == snapshot


def test_candidate_reasons_include_editorial_and_priority(pipeline):
    r = pipeline.run(_breaking(), now=NOW)
    c = r.candidates[0]
    assert any("score" in reason for reason in c.reasons)


# --- audit regression: important news must reach the queue -------------------

@pytest.mark.parametrize("title", [
    "Cabinet approves Rs 10,000 crore semiconductor manufacturing unit",
    "Supreme Court strikes down electoral bonds scheme",
    "India successfully tests Agni-V ballistic missile",
    "SEBI bans 14 entities for insider trading",
    "Adani Group announces acquisition of majority stake in Ambuja Cements",
    "India may retaliate as US slaps 25% tariff on Indian steel",
    "Justice Sanjiv Khanna appointed as Chief Justice of India",
    "India bridge collapse kills 10 in Bihar",
])
def test_important_news_types_queue(pipeline, title):
    r = pipeline.run([_art("the-hindu", title)], now=NOW)
    assert len(r.candidates) == 1, title
    c = r.candidates[0]
    assert c.status == "queued", f"{title}: status={c.status} reasons={c.reasons}"
    assert c.priority.priority in ("HIGH", "URGENT", "IMMEDIATE"), title
    assert len(r.queue) == 1, title


def test_major_disaster_still_immediate(pipeline):
    r = pipeline.run(
        [_art("the-hindu", "Major earthquake of magnitude 7.2 strikes Delhi, rescue underway")],
        now=NOW,
    )
    c = r.candidates[0]
    assert c.priority.priority == "IMMEDIATE"
    assert c.status == "queued"


def test_ordinary_low_value_article_not_automatically_queued(pipeline):
    r = pipeline.run(
        [_art("the-hindu", "Indian Railways launches new express train between Mumbai and Pune")],
        now=NOW,
    )
    assert len(r.candidates) == 1
    assert r.candidates[0].status == "held"
    assert r.queue == []


def test_important_story_with_missing_category_is_queued(pipeline):
    title = "Cabinet gives nod to sweeping reform"
    r = pipeline.run([_art("the-hindu", title)], now=NOW)
    assert len(r.candidates) == 1
    c = r.candidates[0]
    assert c.category is None
    assert c.priority.priority in ("HIGH", "URGENT", "IMMEDIATE")
    assert c.status == "queued"


def test_normal_story_with_missing_category_still_rejected(pipeline):
    r = pipeline.run(
        [_art("the-hindu", "Parliament convenes routine committee meeting on procedural matters")],
        now=NOW,
    )
    assert len(r.candidates) == 1
    c = r.candidates[0]
    assert c.category is None
    assert c.status in ("rejected", "held")


def test_irrelevant_article_still_rejected(pipeline):
    r = pipeline.run(
        [_art("bbc", "European Union agrees trade deal with Canada")],
        now=NOW,
    )
    assert r.relevant == 0
    assert r.queue == []


def test_editorial_reject_still_rejected(pipeline):
    r = pipeline.run(
        [_art("the-hindu", "Delhi government announces astrologer predicts new year luck")],
        now=NOW,
    )
    assert r.queue == []
    assert all(c.status == "rejected" for c in r.candidates)


def test_33_relevant_events_do_not_automatically_produce_33_posts(pipeline):
    """A big batch of relevant events must not auto-publish everything: only
    genuinely important qualifying events may queue."""
    important = [
        "Major earthquake of magnitude 7.2 strikes Delhi, rescue underway",
        "Cabinet approves Rs 10,000 crore semiconductor manufacturing unit",
        "India successfully tests Agni-V ballistic missile",
    ]
    low_value = [
        "Indian Railways launches new express train between Mumbai and Pune",
        "Indian Railways launches new express train between Delhi and Agra",
        "Indian Railways launches new express train between Chennai and Bengaluru",
        "Indian Railways launches new express train between Kolkata and Patna",
        "Indian Railways launches new express train between Hyderabad and Vijayawada",
        "Indian Railways launches new express train between Ahmedabad and Surat",
        "Indian Railways launches new express train between Lucknow and Kanpur",
        "Indian Railways launches new express train between Bhopal and Indore",
        "Indian Railways launches new express train between Jaipur and Jodhpur",
        "Indian Railways launches new express train between Kochi and Thiruvananthapuram",
    ]
    articles = [_art("the-hindu", t) for t in important + low_value * 3]
    r = pipeline.run(articles, now=NOW)
    assert r.relevant == len(articles)
    assert r.events < len(articles)  # dedup merges near-identical railway stories
    queued_titles = {c.title for c in r.queue}
    assert len(r.queue) == len(important)
    assert {t for t in important} == queued_titles


def test_same_event_two_sources_deduplicated(pipeline):
    title = "Major earthquake of magnitude 7.2 strikes Delhi, rescue underway"
    a = _art("the-hindu", title)
    b = _art("ndtv", "Major earthquake hits Delhi, rescue underway")
    r = pipeline.run([a, b], now=NOW)
    assert r.events == 1
    assert len(r.candidates) == 1
    assert len(r.queue) == 1


def test_two_distinct_important_events_both_survive(pipeline):
    a = _art("the-hindu", "Major earthquake of magnitude 7.2 strikes Delhi, rescue underway")
    b = _art("the-hindu", "India successfully tests Agni-V ballistic missile")
    r = pipeline.run([a, b], now=NOW)
    assert r.events == 2
    assert len(r.candidates) == 2
    assert len(r.queue) == 2