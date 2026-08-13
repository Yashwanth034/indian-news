"""Convert parsed raw items into normalized Article objects."""
from datetime import datetime, timezone
from typing import Optional

from src.ingest.normalize import canonicalize_url
from src.models.article import Article

VALID_ROLES = {"official-primary", "official", "journalism", "specialist", "international", "discovery"}


def source_role(source: dict) -> str:
    """Derive the source's role from its config flags."""
    if source.get("discovery"):
        return "discovery"
    if source.get("primary") and source.get("type") == "official":
        return "official-primary"
    stype = source.get("type")
    return stype if stype in VALID_ROLES else "journalism"


def build_article(source: dict, raw_item: dict, *, fetched_at: Optional[datetime] = None) -> Optional[Article]:
    """Build a normalized Article from a raw item, or None if unusable."""
    url = canonicalize_url(raw_item.get("url"))
    if not url:
        return None
    title = (raw_item.get("title") or "").strip()
    if not title:
        return None
    from src.ingest.normalize import normalize_timestamp

    return Article(
        source_id=source["id"],
        source_name=source["name"],
        tier=source.get("tier", 0),
        source_role=source_role(source),
        url=url,
        canonical_url=url,
        title=title,
        summary=(raw_item.get("summary") or "").strip() or None,
        published=normalize_timestamp(raw_item.get("published")),
        updated=normalize_timestamp(raw_item.get("updated")),
        author=(raw_item.get("author") or "").strip() or None,
        category_hints=list(raw_item.get("category_hints") or []),
        language=source.get("language", "en"),
        raw=dict(raw_item.get("raw") or {}),
        fetched_at=fetched_at or datetime.now(timezone.utc),
    )
