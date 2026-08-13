"""Tests for India relevance detection."""
from datetime import datetime, timezone

import pytest

from src.models.article import Article
from src.pipeline.relevance import IndiaRelevance

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def detector():
    return IndiaRelevance()


def _article(title, summary=""):
    return Article(
        source_id="the-hindu",
        source_name="The Hindu",
        tier=2,
        source_role="journalism",
        url="https://www.thehindu.com/news/a1.html",
        title=title,
        summary=summary or None,
        published=NOW,
        language="en",
    )


def test_clearly_india_focused_story(detector):
    r = detector.score(
        _article("India launches new communications satellite", "ISRO successfully launched the satellite today.")
    )
    assert r.is_india is True


def test_indian_government_announcement(detector):
    r = detector.score(_article("Union Cabinet approves new farm subsidy scheme"))
    assert r.is_india is True


def test_indian_company_story(detector):
    r = detector.score(
        _article("Reliance Jio posts record quarterly profit", "India's biggest telecom operator surprised analysts.")
    )
    assert r.is_india is True


def test_india_related_international_story(detector):
    r = detector.score(
        _article("G20 summit: India pushes climate agenda", "New Delhi sought common ground on emissions.")
    )
    assert r.is_india is True


def test_india_mentioned_only_incidentally(detector):
    r = detector.score(
        _article("Global tech conference draws delegates from India, China, US", "Over 5,000 attendees joined the event.")
    )
    assert r.is_india is False


def test_foreign_story_with_no_india_connection(detector):
    r = detector.score(
        _article("NATO expands in Eastern Europe", "Alliance leaders met in Brussels.")
    )
    assert r.is_india is False


def test_state_story_with_national_significance(detector):
    r = detector.score(
        _article("Kerala flood: 20 dead, Army called in for rescue", "Landslides cut off several villages.")
    )
    assert r.is_india is True
    assert r.geo_scope == "state"


def test_local_story_without_national_significance(detector):
    r = detector.score(
        _article("Mysuru gets new city park", "The civic body approved the project.")
    )
    assert r.is_india is False
    assert r.geo_scope in ("state", "local")


def test_false_positive_entity_name(detector):
    r = detector.score(
        _article("Indian Wells tennis tournament begins", "Top seeds advanced to round two.")
    )
    assert r.is_india is False


def test_headline_signal_stronger_than_summary_signal(detector):
    weak = detector.score(
        _article("Global markets tumble", "RBI's surprise rate hike spooked investors.")
    )
    assert weak.is_india is False
    strong = detector.score(
        _article("RBI keeps repo rate unchanged", "Markets cheered the decision.")
    )
    assert strong.is_india is True


def test_india_specific_international_term_is_strong(detector):
    r = detector.score(_article("LAC standoff: India, China hold talks"))
    assert r.is_india is True


def test_general_international_without_india_anchor_excluded(detector):
    r = detector.score(_article("G20 summit concludes in Rio de Janeiro"))
    assert r.is_india is False


def test_result_is_explainable(detector):
    r = detector.score(
        _article("India launches new communications satellite", "ISRO launched it today.")
    )
    assert 0 <= r.score <= 100
    assert r.decision in ("include", "exclude")
    assert r.signals
    assert r.reasons
    assert isinstance(r.is_india, bool)
