"""Parsers: turn raw bytes from a source into normalized raw items.

Each parser returns a list of dicts with a common set of keys:
title, url, summary, published, updated, author, category_hints, raw.
These raw items are later converted to Article objects by the builder.
Parsing stays separate from classification/scoring.
"""
import json
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import feedparser

from src.ingest.normalize import canonicalize_url, is_valid_url


class ParseError(Exception):
    """Raised when source content cannot be parsed."""


def parse_for_method(method, content: bytes, source: dict) -> list[dict]:
    """Dispatch to the right parser based on the source's method."""
    if method in ("rss", "discovery"):
        return parse_rss(content)
    if method in ("json", "api", "endpoint"):
        return parse_json(content, source.get("settings") or {})
    if method == "page":
        return parse_page(content, source)
    raise ParseError(f"unsupported ingestion method: {method}")


def parse_rss(content: bytes) -> list[dict]:
    """Parse an RSS feed into raw items."""
    feed = feedparser.parse(content)
    if feed.bozo and not feed.entries:
        reason = getattr(feed, "bozo_exception", None)
        raise ParseError(f"malformed RSS feed: {reason}")
    items = []
    for entry in feed.entries:
        title = _clean(entry.get("title"))
        url = _clean(entry.get("link"))
        if not title or not is_valid_url(url):
            continue
        tags = [t.get("term") for t in entry.get("tags", []) if t.get("term")]
        items.append(
            {
                "title": title,
                "url": url,
                "summary": _clean(entry.get("summary") or entry.get("description")),
                "published": _clean(entry.get("published") or entry.get("pubDate")),
                "updated": _clean(entry.get("updated")),
                "author": _clean(entry.get("author")),
                "category_hints": tags,
                "raw": {
                    "guid": entry.get("id") or entry.get("guid"),
                    "feed_entry": dict(entry),
                },
            }
        )
    return items


def parse_json(content: bytes, settings: dict) -> list[dict]:
    """Parse a JSON/API response into raw items.

    Supports top-level lists, or dicts wrapping a list under common keys.
    Field names can be customized per source via settings.json_item_fields.
    """
    try:
        data = json.loads(content.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ParseError(f"malformed JSON: {exc}") from exc

    items = _extract_list(data)
    field_map = {
        "title": settings.get("title_field", "title"),
        "url": settings.get("url_field", "url"),
        "summary": settings.get("summary_field", "summary"),
        "published": settings.get("published_field", "published_at"),
        "updated": settings.get("updated_field", "updated_at"),
        "author": settings.get("author_field", "author"),
        "categories": settings.get("categories_field", "tags"),
    }
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        raw_item = _json_item(item, field_map)
        if raw_item is not None:
            out.append(raw_item)
    return out


def _extract_list(data):
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("articles", "data", "results", "items", "rows", "entries"):
            value = data.get(key)
            if isinstance(value, list):
                return value
        if isinstance(data.get("data"), dict):
            for key in ("articles", "results", "items"):
                value = data["data"].get(key)
                if isinstance(value, list):
                    return value
    return []


def _json_item(item: dict, field_map: dict) -> dict | None:
    title = _clean(item.get(field_map["title"]))
    url = _clean(item.get(field_map["url"]))
    if not title or not is_valid_url(url):
        return None
    categories = item.get(field_map["categories"])
    if isinstance(categories, str):
        categories = [categories]
    elif not isinstance(categories, list):
        categories = []
    return {
        "title": title,
        "url": url,
        "summary": _clean(item.get(field_map["summary"])),
        "published": _clean(item.get(field_map["published"])),
        "updated": _clean(item.get(field_map["updated"])),
        "author": _clean(item.get(field_map["author"])),
        "category_hints": [_clean(c) for c in categories if _clean(c)],
        "raw": {"api_item": item},
    }


class _PageLinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.current_href = None
        self.current_text = []
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        attrs = dict(attrs)
        href = attrs.get("href")
        if href:
            self.current_href = href.strip()
            self.current_text = []

    def handle_data(self, data):
        if self.current_href is not None:
            self.current_text.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self.current_href is not None:
            text = _clean(" ".join(self.current_text))
            if text:
                self.links.append((self.current_href, text))
            self.current_href = None
            self.current_text = []


def parse_page(content: bytes, source: dict) -> list[dict]:
    """Parse an HTML listing page into raw items.

    Extracts anchor links within the source's allow_domains, dedupes by
    normalized URL, and uses the anchor text as the title. Parser-level
    extraction only; per-site selectors can be added via source settings later.
    """
    base_url = source.get("url")
    allow_domains = source.get("allow_domains") or []
    try:
        html = content.decode("utf-8", errors="replace")
    except Exception as exc:
        raise ParseError(f"cannot decode page: {exc}") from exc
    parser = _PageLinkParser()
    try:
        parser.feed(html)
    except Exception as exc:
        raise ParseError(f"malformed HTML: {exc}") from exc

    seen = set()
    out = []
    for href, text in parser.links:
        if not href or href.startswith(("#", "mailto:", "javascript:")):
            continue
        url = canonicalize_url(href, base_url=base_url)
        if not url or url in seen:
            continue
        if allow_domains and not _matches_allow_domains(url, allow_domains):
            continue
        seen.add(url)
        out.append(
            {
                "title": text,
                "url": url,
                "summary": None,
                "published": None,
                "updated": None,
                "author": None,
                "category_hints": [],
                "raw": {"page_href": href},
            }
        )
    return out


def _matches_allow_domains(url: str, allow_domains: list[str]) -> bool:
    host = urlparse(url).netloc.lower()
    return any(host == d.lower() or host.endswith("." + d.lower()) for d in allow_domains)


def _clean(value) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    return text or None
