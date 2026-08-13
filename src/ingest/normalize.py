"""Timestamp and URL normalization helpers shared by ingestion stages."""
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

from dateutil.parser import ParserError, parse as dateutil_parse

TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "utm_id", "fbclid", "gclid", "mc_cid", "mc_eid", "igshid", "ref",
}


def normalize_timestamp(value) -> Optional[datetime]:
    """Parse a timestamp into a timezone-aware UTC datetime, or None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc)
        except (ValueError, OSError, OverflowError):
            return None
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value:
        return None
    try:
        dt = dateutil_parse(value)
    except (ParserError, ValueError, OverflowError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def is_valid_url(url: Optional[str]) -> bool:
    """A URL is usable only if it is absolute http(s) with a host."""
    if not url or not isinstance(url, str):
        return False
    try:
        parts = urlparse(url.strip())
    except ValueError:
        return False
    return parts.scheme in ("http", "https") and bool(parts.netloc)


def canonicalize_url(url: Optional[str], base_url: Optional[str] = None) -> Optional[str]:
    """Normalize a URL: resolve relative, lowercase host, strip tracking + fragment."""
    if not url:
        return None
    url = url.strip()
    if base_url:
        url = urljoin(base_url, url)
    if not is_valid_url(url):
        return None
    parts = urlparse(url)
    query = _strip_tracking(parts.query)
    path = parts.path
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    return urlunparse(
        (parts.scheme.lower(), parts.netloc.lower(), path, parts.params, query, "")
    )


def _strip_tracking(query: str) -> str:
    if not query:
        return ""
    kept = [(k, v) for k, v in parse_qsl(query, keep_blank_values=True) if k not in TRACKING_PARAMS]
    return urlencode(kept)
