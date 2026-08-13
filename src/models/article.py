"""Normalized article representation used by the whole pipeline."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass
class Article:
    """A single normalized news article after source ingestion."""

    source_id: str
    source_name: str
    tier: int
    source_role: str
    url: str
    title: str
    canonical_url: Optional[str] = None
    summary: Optional[str] = None
    published: Optional[datetime] = None
    updated: Optional[datetime] = None
    author: Optional[str] = None
    category_hints: list[str] = field(default_factory=list)
    language: str = "en"
    raw: dict[str, Any] = field(default_factory=dict)
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
