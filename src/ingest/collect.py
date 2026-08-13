"""Orchestrate ingestion across sources with failure isolation.

Each source is fetched, parsed and built independently. A failure in one
source never stops the others. Results are deduped within a feed by
canonical URL; cross-source dedup happens in a later pipeline stage.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

from src.ingest.builder import build_article
from src.ingest.fetch import FetchError, FetchTimeout, fetch_bytes
from src.ingest.health import HealthStore
from src.ingest.parsers import ParseError, parse_for_method
from src.models.article import Article

Fetcher = Callable[..., bytes]


@dataclass
class SourceResult:
    source_id: str
    status: str  # ok | error
    articles: list[Article] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class CollectReport:
    results: list[SourceResult] = field(default_factory=list)

    @property
    def articles(self) -> list[Article]:
        return [a for r in self.results for a in r.articles]

    def status(self, source_id: str) -> Optional[str]:
        for r in self.results:
            if r.source_id == source_id:
                return r.status
        return None

    @property
    def failed_sources(self) -> list[str]:
        return [r.source_id for r in self.results if r.status == "error"]

    @property
    def ok_sources(self) -> list[str]:
        return [r.source_id for r in self.results if r.status == "ok"]


def collect_sources(
    sources: list[dict],
    *,
    fetcher: Fetcher = fetch_bytes,
    health: Optional[HealthStore] = None,
    now: Optional[datetime] = None,
    fetch_kwargs: Optional[dict] = None,
) -> CollectReport:
    """Collect articles from all enabled sources, isolating failures."""
    report = CollectReport()
    now = now or datetime.now(timezone.utc)
    fetch_kwargs = fetch_kwargs or {}

    for source in sources:
        if not source.get("enabled"):
            continue
        sid = source["id"]
        try:
            content = fetcher(source["url"], **fetch_kwargs)
            raw_items = parse_for_method(source.get("method"), content, source)
            seen = set()
            articles = []
            for raw in raw_items:
                article = build_article(source, raw, fetched_at=now)
                if article is None or article.url in seen:
                    continue
                seen.add(article.url)
                articles.append(article)
            report.results.append(SourceResult(sid, "ok", articles))
            if health:
                health.record_success(sid, items_found=len(articles), fetched_at=_iso(now))
        except FetchTimeout as exc:
            report.results.append(SourceResult(sid, "error", error=str(exc)))
            if health:
                health.record_failure(sid, error=str(exc), timeout=True)
        except (FetchError, ParseError) as exc:
            report.results.append(SourceResult(sid, "error", error=str(exc)))
            if health:
                malformed = isinstance(exc, ParseError)
                health.record_failure(sid, error=str(exc), malformed=malformed)
        except Exception as exc:  # last-resort isolation barrier
            report.results.append(SourceResult(sid, "error", error=f"unexpected: {exc}"))
            if health:
                health.record_failure(sid, error=f"unexpected: {exc}")

    if health:
        health.save()
    return report


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()
