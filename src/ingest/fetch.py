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
