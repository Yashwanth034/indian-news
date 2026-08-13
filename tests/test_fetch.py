"""Tests for the fetch layer (mocked, never hits network)."""
import requests
import pytest

from src.ingest.fetch import FetchError, FetchTimeout, fetch_bytes


class FakeResponse:
    def __init__(self, status_code, content=b""):
        self.status_code = status_code
        self.content = content


class FakeSession:
    def __init__(self, response):
        self.response = response

    def get(self, url, timeout=None, headers=None):
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_fetch_bytes_returns_content():
    session = FakeSession(FakeResponse(200, b"<rss>"))
    assert fetch_bytes("https://example.com/feed", session=session) == b"<rss>"


def test_fetch_bytes_raises_on_http_error():
    session = FakeSession(FakeResponse(404))
    with pytest.raises(FetchError):
        fetch_bytes("https://example.com/feed", session=session)


def test_fetch_bytes_raises_on_timeout():
    session = FakeSession(requests.exceptions.Timeout())
    with pytest.raises(FetchTimeout):
        fetch_bytes("https://example.com/feed", session=session)


def test_fetch_bytes_raises_on_connection_error():
    session = FakeSession(requests.exceptions.ConnectionError("boom"))
    with pytest.raises(FetchError):
        fetch_bytes("https://example.com/feed", session=session)


def test_fetch_bytes_raises_on_invalid_url():
    with pytest.raises(FetchError):
        fetch_bytes("not a url")


def test_fetch_timeout_is_subclass_of_fetch_error():
    assert issubclass(FetchTimeout, FetchError)
