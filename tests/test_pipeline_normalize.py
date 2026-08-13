"""Tests for the article normalization/validation layer."""
from datetime import datetime, timedelta, timezone

import pytest

from src.models.article import Article
from src.pipeline.normalize import normalize_article

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


def _article(**over):
    base = {
        "source_id": "the-hindu",
        "source_name": "The Hindu",
        "tier": 2,
        "source_role": "journalism",
        "url": "https://www.thehindu.com/news/national/a1/article.html?utm_source=rss",
        "title": "Headline",
        "summary": "Summary text",
        "published": datetime(2026, 8, 11, 8, 0, tzinfo=timezone.utc),
        "updated": None,
        "author": None,
        "category_hints": [],
        "language": "en",
        "raw": {"guid": "x"},
        "fetched_at": NOW,
    }
    base.update(over)
    return Article(**base)


# --- title cleanup ---


def test_title_whitespace_collapsed():
    r = normalize_article(_article(title="  ISRO   to launch  satellite  "), now=NOW)
    assert r.rejected is False
    assert r.article.title == "ISRO to launch satellite"


def test_title_strips_mint_suffix():
    r = normalize_article(_article(title="RBI keeps repo rate unchanged | Mint"), now=NOW)
    assert r.article.title == "RBI keeps repo rate unchanged"


def test_title_strips_the_hindu_suffix():
    r = normalize_article(
        _article(title="Vande Bharat to run on new route - The Hindu"), now=NOW
    )
    assert r.article.title == "Vande Bharat to run on new route"


def test_title_html_entities_unescaped():
    r = normalize_article(_article(title="Sensex &amp; Nifty hit record highs"), now=NOW)
    assert r.article.title == "Sensex & Nifty hit record highs"


def test_title_empty_rejected():
    r = normalize_article(_article(title="   "), now=NOW)
    assert r.rejected is True
    assert any("title" in reason for reason in r.reasons)


# --- summary cleanup ---


def test_summary_html_tags_stripped():
    r = normalize_article(
        _article(summary="<p>New <b>disclosure norms</b> for listed firms.</p>"), now=NOW
    )
    assert r.article.summary == "New disclosure norms for listed firms."


def test_summary_whitespace_collapsed_and_empty_to_none():
    r = normalize_article(_article(summary="   Line one\n\n   line two  "), now=NOW)
    assert r.article.summary == "Line one line two"
    r2 = normalize_article(_article(summary="   "), now=NOW)
    assert r2.article.summary is None


# --- author cleanup ---


def test_author_by_prefix_stripped():
    r = normalize_article(_article(author="By Reuters"), now=NOW)
    assert r.article.author == "Reuters"


def test_author_whitespace_collapsed():
    r = normalize_article(_article(author="  By  The Hindu   Bureau "), now=NOW)
    assert r.article.author == "The Hindu Bureau"


def test_author_empty_becomes_none():
    r = normalize_article(_article(author="   "), now=NOW)
    assert r.article.author is None


# --- date/time consistency ---


def test_naive_published_assumed_utc():
    r = normalize_article(
        _article(published=datetime(2026, 8, 10, 6, 30)), now=NOW
    )
    assert r.article.published == datetime(2026, 8, 10, 6, 30, tzinfo=timezone.utc)


def test_aware_published_kept_in_utc():
    published = datetime(2026, 8, 11, 9, 30, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    r = normalize_article(_article(published=published), now=NOW)
    assert r.article.published == datetime(2026, 8, 11, 4, 0, tzinfo=timezone.utc)


def test_updated_before_published_dropped():
    published = datetime(2026, 8, 11, 9, 0, tzinfo=timezone.utc)
    updated = datetime(2026, 8, 11, 8, 0, tzinfo=timezone.utc)
    r = normalize_article(_article(published=published, updated=updated), now=NOW)
    assert r.article.updated is None
    assert any("updated" in w for w in r.warnings)


def test_implausible_future_published_dropped():
    future = NOW + timedelta(days=3)
    r = normalize_article(_article(published=future), now=NOW)
    assert r.article.published is None
    assert any("published" in w for w in r.warnings)


def test_valid_updated_kept():
    published = datetime(2026, 8, 11, 9, 0, tzinfo=timezone.utc)
    updated = datetime(2026, 8, 11, 10, 0, tzinfo=timezone.utc)
    r = normalize_article(_article(published=published, updated=updated), now=NOW)
    assert r.article.updated == updated


# --- URL / canonical URL consistency ---


def test_url_tracking_stripped():
    r = normalize_article(
        _article(url="https://www.thehindu.com/news/national/a1?utm_source=rss&fbclid=123")
    )
    assert r.article.url == "https://www.thehindu.com/news/national/a1"


def test_canonical_url_defaults_to_url():
    r = normalize_article(_article(canonical_url=None))
    assert r.article.canonical_url == r.article.url


def test_canonical_url_kept_when_valid_and_different():
    r = normalize_article(
        _article(
            url="https://www.thehindu.com/news/national/a1",
            canonical_url="https://www.thehindu.com/news/national/a1#frag",
        )
    )
    assert r.article.canonical_url == "https://www.thehindu.com/news/national/a1"


def test_canonical_url_falls_back_when_invalid():
    r = normalize_article(_article(canonical_url="not a url"))
    assert r.article.canonical_url == r.article.url
    assert any("canonical" in w for w in r.warnings)


def test_invalid_url_rejected():
    r = normalize_article(_article(url="javascript:alert(1)"))
    assert r.rejected is True
    assert any("url" in reason for reason in r.reasons)


# --- source identity ---


def test_invalid_source_id_rejected():
    r = normalize_article(_article(source_id="Bad ID!"))
    assert r.rejected is True
    assert any("source_id" in reason for reason in r.reasons)


def test_empty_source_name_rejected():
    r = normalize_article(_article(source_name="   "))
    assert r.rejected is True


def test_invalid_source_role_rejected():
    r = normalize_article(_article(source_role="cartoonist"))
    assert r.rejected is True
    assert any("source_role" in reason for reason in r.reasons)


def test_string_tier_coerced_to_int():
    r = normalize_article(_article(tier="2"))
    assert r.rejected is False
    assert r.article.tier == 2


def test_zero_tier_rejected():
    r = normalize_article(_article(tier=0))
    assert r.rejected is True


# --- language normalization ---


def test_language_en_in_normalized():
    r = normalize_article(_article(language="en-IN"))
    assert r.article.language == "en"


def test_language_english_word_normalized():
    r = normalize_article(_article(language="English"))
    assert r.article.language == "en"


def test_language_hindi_kept_without_require_english():
    r = normalize_article(_article(language="hi"))
    assert r.rejected is False
    assert r.article.language == "hi"


def test_language_hindi_rejected_when_require_english():
    r = normalize_article(_article(language="hi"), require_english=True, now=NOW)
    assert r.rejected is True
    assert any("language" in reason for reason in r.reasons)


def test_missing_language_defaults_to_en():
    r = normalize_article(_article(language=None))
    assert r.article.language == "en"


# --- category hints ---


def test_category_hints_normalized_and_deduped():
    r = normalize_article(
        _article(category_hints=[" Politics ", "politics", "Economy", "economy"])
    )
    assert r.article.category_hints == ["politics", "economy"]


def test_category_hints_dropped_empties_and_capped():
    r = normalize_article(
        _article(category_hints=["a", "  ", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"])
    )
    assert len(r.article.category_hints) <= 10
    assert " " not in r.article.category_hints


# --- overall validity ---


def test_valid_article_passes_clean():
    r = normalize_article(_article(), now=NOW)
    assert r.rejected is False
    assert r.article is not None
    assert r.article.source_id == "the-hindu"
    assert r.article.language == "en"


def test_multiple_reasons_reported():
    r = normalize_article(_article(title="", url="bad", source_id="x y", source_role="bad"))
    assert r.rejected is True
    assert len(r.reasons) >= 3
