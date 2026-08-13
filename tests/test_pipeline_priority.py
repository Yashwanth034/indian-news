"""Tests for India importance / priority scoring."""
from datetime import datetime, timedelta, timezone

import pytest

from src.models.article import Article
from src.pipeline.classify import CategoryClassifier
from src.pipeline.editorial import EditorialGate
from src.pipeline.geography import GeoClassifier
from src.pipeline.priority import PriorityScorer
from src.pipeline.dedupe import EventGroup

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def scorer():
    return PriorityScorer()


@pytest.fixture(scope="module")
def classifier():
    return CategoryClassifier()


@pytest.fixture(scope="module")
def geo():
    return GeoClassifier()


@pytest.fixture(scope="module")
def editorial():
    return EditorialGate()


def _art(source_id, title, summary="", published=None):
    return Article(
        source_id=source_id,
        source_name=source_id.replace("-", " ").title(),
        tier=2,
        source_role="journalism",
        url=f"https://{source_id}.example.in/p/{len(title)}.html",
        title=title,
        summary=summary or None,
        published=published or NOW,
    )


def _event(indep=1, wire_group="", echo=False, members=1, primary=None,
           event_time=None, confidence="medium", category=None):
    return EventGroup(
        event_id="ev1",
        member_indices=list(range(members)),
        representative_index=0,
        category=category,
        states=[],
        entities=[],
        event_time=event_time or NOW,
        primary_source=primary or "the-hindu",
        first_source="the-hindu",
        wire_group=wire_group,
        wire_kind="wire" if wire_group else "independent",
        source_groups={},
        independent_source_groups=indep,
        is_wire_echo=echo,
        confidence=confidence,
    )


def _score(scorer, title, summary="", *, category=None, geo=None,
           event=None, editorial=None, published=None, source_id="the-hindu"):
    return scorer.score_text(
        title,
        summary,
        source_id=source_id,
        published=published,
        category=category,
        geo=geo,
        event=event,
        editorial=editorial,
        now=NOW,
    )


# --- source / corroboration -------------------------------------------------

def test_major_event_single_reliable_source(scorer, classifier):
    title = "Major terror attack in Mumbai, multiple casualties reported"
    cat = classifier.classify_text(title)
    r = _score(scorer, title, category=cat, event=_event(indep=1))
    assert r.priority in ("IMMEDIATE", "URGENT")
    assert r.confidence == "medium"


def test_major_event_multiple_independent_sources(scorer, classifier):
    title = "Major terror attack in Mumbai, multiple casualties reported"
    cat = classifier.classify_text(title)
    r = _score(scorer, title, category=cat, event=_event(indep=4))
    assert r.priority in ("IMMEDIATE", "URGENT")
    assert r.confidence == "high"
    assert any("independent" in s.term for s in r.signals)


def test_wire_copies_do_not_inflate_importance(scorer, classifier):
    title = "Major terror attack in Mumbai, multiple casualties reported"
    cat = classifier.classify_text(title)
    single = _score(scorer, title, category=cat, event=_event(indep=1))
    wire = _score(scorer, title, category=cat, event=_event(indep=1, wire_group="pti", echo=True, members=10))
    assert wire.score <= single.score
    assert wire.confidence == "low"
    assert any("wire" in s.term for s in wire.signals)


def test_strong_event_beats_keyword_noise(scorer, classifier):
    noisy = "Company announces partnership, new investment, product launch and expansion"
    r = _score(scorer, noisy, category=classifier.classify_text(noisy))
    assert r.priority == "NORMAL"

    strong = "Earthquake strikes Assam, tremors felt in six districts"
    s = _score(scorer, strong, category=classifier.classify_text(strong))
    assert s.score > r.score


# --- geography / scale ------------------------------------------------------

def test_major_state_event_with_national_significance(scorer, classifier, geo):
    title = "Cyclone Dana makes landfall in Odisha, lakhs evacuated"
    cat = classifier.classify_text(title)
    g = geo.classify_text(title)
    r = _score(scorer, title, category=cat, geo=g, event=_event(indep=1))
    assert r.priority in ("IMMEDIATE", "URGENT")
    assert any("national significance" in s for s in r.reasons)


def test_state_event_without_national_significance_is_normal(scorer, classifier, geo):
    title = "District panchayat approves new park in Nashik"
    cat = classifier.classify_text(title)
    g = geo.classify_text(title)
    r = _score(scorer, title, category=cat, geo=g)
    assert r.priority == "NORMAL"


# --- national / routine -----------------------------------------------------

def test_minor_national_announcement_is_normal(scorer, classifier):
    title = "Government releases revised list of approved generic medicines"
    cat = classifier.classify_text(title)
    r = _score(scorer, title, category=cat)
    assert r.priority == "NORMAL"


def test_major_national_decision_is_high(scorer, classifier):
    title = "Cabinet approves Rs 1 lakh crore defence procurement deal"
    cat = classifier.classify_text(title)
    r = _score(scorer, title, category=cat, event=_event(indep=2))
    assert r.priority in ("HIGH", "URGENT", "IMMEDIATE")


# --- sports ----------------------------------------------------------------

def test_major_sports_event_high(scorer, classifier):
    title = "India win T20 World Cup final against South Africa"
    cat = classifier.classify_text(title)
    r = _score(scorer, title, category=cat)
    assert r.priority == "HIGH"


def test_routine_cricket_update_normal(scorer, classifier):
    title = "IPL match preview: Mumbai Indians vs Chennai Super Kings"
    cat = classifier.classify_text(title)
    r = _score(scorer, title, category=cat)
    assert r.priority == "NORMAL"


# --- disaster / weather -----------------------------------------------------

def test_major_disaster_immediate(scorer, classifier, geo):
    title = "Massive earthquake strikes Delhi-NCR, buildings collapse"
    cat = classifier.classify_text(title)
    g = geo.classify_text(title)
    r = _score(scorer, title, category=cat, geo=g, event=_event(indep=2))
    assert r.priority == "IMMEDIATE"
    assert r.major_event is True


def test_routine_weather_update_normal(scorer, classifier):
    title = "Light rain expected in Delhi tomorrow, says IMD"
    cat = classifier.classify_text(title)
    r = _score(scorer, title, category=cat)
    assert r.priority == "NORMAL"


# --- RBI -------------------------------------------------------------------

def test_major_rbi_decision_high(scorer, classifier):
    title = "RBI cuts repo rate by 25 basis points to boost economy"
    cat = classifier.classify_text(title)
    r = _score(scorer, title, category=cat, event=_event(indep=2))
    assert r.priority in ("HIGH", "URGENT")


def test_minor_rbi_announcement_normal(scorer, classifier):
    title = "RBI to issue new Rs 50 note with updated features"
    cat = classifier.classify_text(title)
    r = _score(scorer, title, category=cat)
    assert r.priority == "NORMAL"


# --- courts -----------------------------------------------------------------

def test_major_supreme_court_ruling_high(scorer, classifier):
    title = "Supreme Court strikes down electoral bonds scheme"
    cat = classifier.classify_text(title)
    r = _score(scorer, title, category=cat, event=_event(indep=2))
    assert r.priority in ("HIGH", "URGENT")


def test_minor_court_hearing_normal(scorer, classifier):
    title = "Delhi court adjourns hearing in property dispute"
    cat = classifier.classify_text(title)
    r = _score(scorer, title, category=cat)
    assert r.priority == "NORMAL"


# --- international ----------------------------------------------------------

def test_international_event_with_major_india_impact(scorer, classifier):
    title = "Chinese troops clash at LAC, India mobilises forces"
    cat = classifier.classify_text(title)
    r = _score(scorer, title, category=cat)
    assert r.priority in ("IMMEDIATE", "URGENT")


def test_international_event_with_weak_india_connection_normal(scorer, classifier):
    title = "Ronaldo signs new contract with Saudi club"
    cat = classifier.classify_text(title)
    r = _score(scorer, title, category=cat)
    assert r.priority == "NORMAL"


# --- recency ----------------------------------------------------------------

def test_breaking_update_more_urgent_than_old(scorer, classifier):
    title = "Major terror attack in Mumbai, multiple casualties reported"
    cat = classifier.classify_text(title)
    breaking = _score(scorer, title, category=cat,
                      published=NOW - timedelta(minutes=30), event=_event(indep=1))
    old = _score(scorer, title, category=cat,
                 published=NOW - timedelta(days=3), event=_event(indep=1, event_time=NOW - timedelta(days=3)))
    assert breaking.score >= old.score
    rec_breaking = next(s.weight for s in breaking.signals if s.family == "recency")
    rec_old = next(s.weight for s in old.signals if s.family == "recency")
    assert rec_breaking > rec_old


def test_older_story_not_automatically_high(scorer, classifier):
    title = "IPL match preview: Mumbai Indians vs Chennai Super Kings"
    cat = classifier.classify_text(title)
    r = _score(scorer, title, category=cat, published=NOW - timedelta(days=4))
    assert r.priority == "NORMAL"
    rec = next(s.weight for s in r.signals if s.family == "recency")
    assert rec <= 0


# --- editorial --------------------------------------------------------------

def _editorial_result(decision):
    from src.pipeline.editorial import EditorialResult
    return EditorialResult(
        decision=decision,
        score=60.0 if decision == "filler" else 0.0,
        pass_threshold=60.0,
        filler_threshold=35.0,
        category=None,
        major=False,
        strong_major=False,
        reasons=[f"test editorial decision: {decision}"],
    )


def test_editorial_filler_capped_at_normal(scorer, classifier):
    title = "Major terror attack in Mumbai, multiple casualties reported"
    cat = classifier.classify_text(title)
    r = _score(scorer, title, category=cat, event=_event(indep=2),
               editorial=_editorial_result("filler"))
    assert r.priority == "NORMAL"
    assert any("filler" in s for s in r.reasons)


def test_rejected_article_cannot_be_publishable(scorer, classifier):
    title = "Major terror attack in Mumbai, multiple casualties reported"
    cat = classifier.classify_text(title)
    r = _score(scorer, title, category=cat, event=_event(indep=3),
               editorial=_editorial_result("reject"))
    assert r.blocked is True
    assert r.score == 0.0
    assert r.priority == "NORMAL"
    assert r.major_event is False


# --- conflicting sources ----------------------------------------------------

def test_conflicting_source_information_low_confidence(scorer, classifier):
    title = "Major terror attack in Mumbai, multiple casualties reported"
    cat = classifier.classify_text(title)
    r = _score(scorer, title, category=cat,
               event=_event(indep=0, confidence="low"))
    assert r.confidence == "low"
    assert r.priority in ("IMMEDIATE", "URGENT")


# --- explainability / thresholds --------------------------------------------

def test_result_explainable_and_thresholds_configurable(scorer):
    r = _score(scorer, "Company opens new branch in Pune")
    assert r.reasons
    assert r.signals
    assert set(r.thresholds) == {"immediate", "urgent", "high"}
    assert scorer.thresholds["immediate"] > scorer.thresholds["urgent"] > scorer.thresholds["high"]


def test_priority_order():
    from src.pipeline.priority import _PRIORITY_ORDER
    assert _PRIORITY_ORDER == ["IMMEDIATE", "URGENT", "HIGH", "NORMAL"]
