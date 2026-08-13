"""Tests for timestamp and URL normalization."""
from datetime import datetime, timezone

from src.ingest.normalize import canonicalize_url, is_valid_url, normalize_timestamp


def test_normalize_timestamp_rss_rfc822():
    dt = normalize_timestamp("Mon, 10 Aug 2026 06:30:00 +0530")
    assert dt == datetime(2026, 8, 10, 1, 0, tzinfo=timezone.utc)


def test_normalize_timestamp_iso_utc():
    dt = normalize_timestamp("2026-08-11T09:15:00+05:30")
    assert dt == datetime(2026, 8, 11, 3, 45, tzinfo=timezone.utc)


def test_normalize_timestamp_naive_assumed_utc():
    dt = normalize_timestamp("2026-08-11 07:00:00")
    assert dt == datetime(2026, 8, 11, 7, 0, tzinfo=timezone.utc)


def test_normalize_timestamp_unix_epoch():
    dt = normalize_timestamp(1765500000)
    assert dt.tzinfo is not None


def test_normalize_timestamp_none_and_invalid():
    assert normalize_timestamp(None) is None
    assert normalize_timestamp("not-a-date") is None
    assert normalize_timestamp("") is None


def test_canonicalize_url_strips_tracking_and_fragment():
    assert (
        canonicalize_url("https://Example.com/stories/sat?utm_source=rss&utm_medium=feed&fbclid=123#top")
        == "https://example.com/stories/sat"
    )


def test_canonicalize_url_normalizes_scheme_and_host_case():
    assert canonicalize_url("HTTPS://News.Example.com/Story") == "https://news.example.com/Story"


def test_canonicalize_url_resolves_relative_against_base():
    assert (
        canonicalize_url("/stories/rbi", base_url="https://www.thehindu.com/")
        == "https://www.thehindu.com/stories/rbi"
    )


def test_canonicalize_url_removes_trailing_slash_but_keeps_root():
    assert canonicalize_url("https://example.com/path/") == "https://example.com/path"
    assert canonicalize_url("https://example.com/") == "https://example.com/"


def test_is_valid_url():
    assert is_valid_url("https://www.thehindu.com/news/a.html")
    assert not is_valid_url("not a url")
    assert not is_valid_url("ftp://example.com/file")
    assert not is_valid_url("")


def test_canonicalize_url_returns_none_for_invalid():
    assert canonicalize_url("not a url") is None
    assert canonicalize_url("javascript:void(0)") is None
