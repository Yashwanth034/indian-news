"""Tests for the normalized Article model."""
from datetime import datetime, timezone

from src.models.article import Article


def test_article_has_all_required_fields():
    a = Article(
        source_id="the-hindu",
        source_name="The Hindu",
        tier=2,
        source_role="journalism",
        url="https://www.thehindu.com/news/a1/article1.html",
        title="A headline",
    )
    assert a.source_id == "the-hindu"
    assert a.source_name == "The Hindu"
    assert a.tier == 2
    assert a.source_role == "journalism"
    assert a.url.startswith("https://")
    assert a.canonical_url is None
    assert a.title == "A headline"
    assert a.summary is None
    assert a.published is None
    assert a.updated is None
    assert a.author is None
    assert a.category_hints == []
    assert a.language == "en"
    assert isinstance(a.raw, dict)
    assert isinstance(a.fetched_at, datetime)


def test_article_defaults():
    a = Article(
        source_id="x",
        source_name="X",
        tier=1,
        source_role="official-primary",
        url="https://x.in/a",
        title="T",
        published=datetime(2026, 8, 11, tzinfo=timezone.utc),
    )
    assert a.language == "en"
    assert a.category_hints == []
    assert a.raw == {}
    assert a.fetched_at.tzinfo is not None


def test_article_language_and_category_hints_populated():
    a = Article(
        source_id="x",
        source_name="X",
        tier=2,
        source_role="journalism",
        url="https://x.in/a",
        title="T",
        language="hi",
        category_hints=["politics", "economy"],
    )
    assert a.language == "hi"
    assert a.category_hints == ["politics", "economy"]
