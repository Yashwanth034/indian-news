"""HTTP fetch layer for source ingestion. Network happens only here."""
from typing import Optional

import requests

from src.ingest.normalize import is_valid_url


class FetchError(Exception):
    """Base error for any failed network fetch."""


class FetchTimeout(FetchError):
    """Raised when a source request times out."""


DEFAULT_TIMEOUT = 20
DEFAULT_MAX_BYTES = 2 * 1024 * 1024

DEFAULT_HEADERS = {
    # Many news sites 403 the default python-requests UA; use a plain
    # browser-like identity only when the caller supplied no headers.
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/xml, text/xml, text/html;q=0.9, */*;q=0.8",
    "Accept-Language": "en-IN,en;q=0.9",
}


def fetch_bytes(
    url: str,
    *,
    session=None,
    timeout: int = DEFAULT_TIMEOUT,
    max_bytes: int = DEFAULT_MAX_BYTES,
    headers: Optional[dict] = None,
) -> bytes:
    """Fetch url and return raw bytes. Raises FetchError/FetchTimeout."""
    if not is_valid_url(url):
        raise FetchError(f"invalid url: {url}")
    session = session or requests
    if not headers:
        headers = dict(DEFAULT_HEADERS)
    try:
        response = session.get(url, timeout=timeout, headers=headers)
    except requests.exceptions.Timeout as exc:
        raise FetchTimeout(f"timeout fetching {url}") from exc
    except requests.exceptions.RequestException as exc:
        raise FetchError(f"request failed for {url}: {exc}") from exc
    if response.status_code >= 400:
        raise FetchError(f"HTTP {response.status_code} for {url}")
    body = response.content
    if len(body) > max_bytes:
        raise FetchError(f"response too large ({len(body)} bytes) for {url}")
    return body
