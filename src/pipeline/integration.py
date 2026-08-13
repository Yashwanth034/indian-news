"""Pipeline integration: runs all stages in order and emits candidates.

Stage order (deterministic):

    ingest -> normalize -> relevance -> classify -> geography
           -> dedupe -> editorial -> priority -> candidate gate -> queue

The orchestrator is pure: it takes already-collected Article objects and a
``now`` timestamp and returns a PipelineResult.  Ingestion (fetching) is a
separate step that feeds articles into :func:`NewsPipeline.run`.

Every stage is executed for every article in the same order, so identical
inputs + config + ``now`` always produce identical candidates and an
identical queue.  No stage performs I/O.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from src.config_loader import get_config
from src.models.article import Article
from src.pipeline.classify import CategoryClassifier
from src.pipeline.dedupe import Deduplicator
from src.pipeline.editorial import EditorialGate
from src.pipeline.geography import GeoClassifier
from src.pipeline.normalize import normalize_article
from src.pipeline.priority import PriorityResult, PriorityScorer
from src.pipeline.relevance import IndiaRelevance

CONFIDENCE_RANK = {"high": 0, "medium": 1, "low": 2}


@dataclass
class Drop:
    """An article excluded at an early stage, with the reason."""

    article: Article
    stage: str  # normalize | relevance | classify | geography
    reasons: list = field(default_factory=list)


@dataclass
class Candidate:
    """One event that passed the candidate gate.

    Reuses existing pipeline results instead of duplicating them:
    ``representative`` is the Article chosen by dedupe, ``event`` the
    EventGroup, and ``relevance/editorial/priority`` the gate results.
    """

    event_id: str
    representative: Article
    articles: list = field(default_factory=list)
    event: object = None
    relevance: object = None
    classification: object = None
    geo: object = None
    editorial: object = None
    priority: Optional[PriorityResult] = None
    category: Optional[str] = None
    secondary: list = field(default_factory=list)
    confidence: str = "low"
    event_time: Optional[datetime] = None
    source_groups: dict = field(default_factory=dict)
    independent_source_groups: int = 0
    is_wire_echo: bool = False
    status: str = "queued"  # queued | held | rejected | filler
    reasons: list = field(default_factory=list)
    queue_rank: Optional[int] = None

    @property
    def queued(self) -> bool:
        return self.status == "queued"

    @property
    def title(self) -> str:
        return self.representative.title or ""

    @property
    def summary(self) -> str:
        return self.representative.summary or ""

    def to_dict(self) -> dict:
        """Serializable representation aligned with the WorldNews queue shape."""
        return {
            "event_id": self.event_id,
            "story_id": self.event_id,
            "headline": self.title,
            "title": self.title,
            "summary": self.summary,
            "url": self.representative.url,
            "source": self.representative.source_name or self.representative.source_id,
            "primary_source": self.primary_source,
            "tier": self.representative.tier,
            "category": self.category,
            "secondary_categories": list(self.secondary),
            "confidence": self.confidence,
            "score": self.relevance.score if self.relevance else 0.0,
            "priority_score": self.priority.score if self.priority else 0.0,
            "priority_level": self.priority.priority if self.priority else "NORMAL",
            "max_delay_minutes": (
                (self.priority.thresholds or {}).get("max_delay_minutes")
                if self.priority and hasattr(self.priority, "thresholds")
                else None
            ),
            "event_time": self.event_time.isoformat() if self.event_time else None,
            "published_at": (
                self.representative.published.isoformat()
                if self.representative.published
                else None
            ),
            "effective_at": self.effective_at,
            "independent_source_groups": self.independent_source_groups,
            "source_groups": self.source_groups,
            "is_wire_echo": self.is_wire_echo,
            "status": self.status,
            "reasons": list(self.reasons),
        }

    @property
    def primary_source(self) -> bool:
        """True when the representative article's source role is a primary source."""
        return self.representative.source_role == "official-primary"

    @property
    def effective_at(self) -> Optional[str]:
        """Latest meaningful timestamp of the representative article, UTC ISO.

        Prefers the newest of ``published`` and ``updated`` so freshness and
        story-age handling can rely on a single authoritative timestamp.  Naive
        timestamps are treated as UTC, matching pipeline normalization.
        """
        latest = None
        for value in (self.representative.published, self.representative.updated):
            if value is None:
                continue
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            else:
                value = value.astimezone(timezone.utc)
            if latest is None or value > latest:
                latest = value
        return latest.isoformat() if latest else None


@dataclass
class PipelineResult:
    """Everything the pipeline produced for one run."""

    collected: int = 0
    normalized: int = 0
    relevant: int = 0
    events: int = 0
    candidates: list = field(default_factory=list)
    queue: list = field(default_factory=list)
    dropped: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    now: Optional[datetime] = None

    @property
    def held(self) -> list:
        return [c for c in self.candidates if c.status == "held"]

    @property
    def rejected(self) -> list:
        return [c for c in self.candidates if c.status == "rejected"]

    @property
    def filler(self) -> list:
        return [c for c in self.candidates if c.status == "filler"]


class NewsPipeline:
    """Run all pipeline stages in order over a batch of articles."""

    def __init__(self, config=None):
        config = config or get_config()
        self.config = config
        cfg = config["config"]

        self.relevance = IndiaRelevance(config)
        self.classifier = CategoryClassifier(config)
        self.geo = GeoClassifier(config)
        self.dedup = Deduplicator(config)
        self.editorial = EditorialGate(config)
        self.priority = PriorityScorer(config)

        c = cfg.get("candidate", {})
        self.min_queue_score = float(c.get("min_score", cfg.get("min_score_to_queue", 55)))
        self.require_category = bool(c.get("require_category", True))
        self.require_geo = bool(c.get("require_geo_identifiable", True))
        self.require_editorial_pass = bool(c.get("require_editorial_pass", True))
        self.max_queue = int(c.get("max_queue", cfg.get("max_stories_per_run", 5)))
        self.now = None

    # -- public -------------------------------------------------------------

    def run(self, articles, *, now: Optional[datetime] = None) -> PipelineResult:
        """Run the full pipeline over ``articles`` and return the result."""
        self.now = now or datetime.now(timezone.utc)
        result = PipelineResult(collected=len(articles), now=self.now)

        normalized = self._normalize(articles, result)
        relevant = self._relevance(normalized, result)

        classifications = []
        geos = []
        kept = []
        for article, rel in relevant:
            classification = self.classifier.classify(article)
            geo_result = self.geo.classify(article)
            classifications.append(classification)
            geos.append(geo_result)
            kept.append(article)

        events = self.dedup.dedupe(
            kept,
            categories=[c.primary for c in classifications],
            states=[g.states for g in geos],
        )
        result.events = len(events.events)
        result.warnings.extend(events.warnings)

        for ev in events.events:
            rep = kept[ev.representative_index]
            classification = classifications[ev.representative_index]
            geo_result = geos[ev.representative_index]
            rel = self._relevance_of(kept, relevant, ev.representative_index)

            editorial = self.editorial.evaluate(
                rep,
                category=classification,
                geo=geo_result,
                relevance=rel,
                event=ev,
            )
            priority = self.priority.score(
                rep,
                category=classification,
                geo=geo_result,
                relevance=rel,
                event=ev,
                editorial=editorial,
                now=self.now,
            )

            candidate = self._decide(
                ev, rep, kept, classification, geo_result, rel,
                editorial, priority,
            )
            result.candidates.append(candidate)

        result.queue = self._order(result.candidates)
        for rank, candidate in enumerate(result.queue):
            candidate.queue_rank = rank
        return result

    # -- stages -------------------------------------------------------------

    def _normalize(self, articles, result) -> list:
        kept = []
        for article in articles:
            vr = normalize_article(article, now=self.now, require_english=self.config["config"].get("english_only", False))
            if vr.rejected:
                result.dropped.append(Drop(article, "normalize", vr.reasons))
                continue
            kept.append((vr.article, vr))
        result.normalized = len(kept)
        return kept

    def _relevance(self, normalized, result) -> list:
        kept = []
        for article, vr in normalized:
            rel = self.relevance.score(article)
            if rel.decision != "include":
                result.dropped.append(Drop(article, "relevance", rel.reasons))
                continue
            kept.append((article, rel))
        result.relevant = len(kept)
        return kept

    def _relevance_of(self, kept, relevant, index):
        for i, (article, rel) in enumerate(relevant):
            if i == index:
                return rel
        return None

    def _decide(self, ev, rep, kept, classification, geo_result, rel,
                editorial, priority) -> Candidate:
        reasons = list(editorial.reasons) + list(priority.reasons)
        status = "queued"

        members = [kept[i] for i in ev.member_indices]

        if editorial.decision == "reject":
            status = "rejected"
        elif self.require_editorial_pass and editorial.decision != "pass":
            status = "filler"
        elif priority.blocked:
            status = "rejected"
        elif self.require_category and classification.primary is None:
            status = "rejected"
            reasons.append("candidate gate: no category classified")
        elif self.require_geo and not geo_result.state_identifiable and geo_result.scope == "local":
            status = "rejected"
            reasons.append("candidate gate: local scope, state not identifiable (Phase 1)")
        elif priority.priority in ("HIGH", "URGENT", "IMMEDIATE"):
            status = "queued"
        elif priority.score < self.min_queue_score:
            status = "held"
            reasons.append(
                f"candidate gate: priority score {priority.score:g} < {self.min_queue_score:g}"
            )
        else:
            status = "queued"

        return Candidate(
            event_id=ev.event_id,
            representative=rep,
            articles=members,
            event=ev,
            relevance=rel,
            classification=classification,
            geo=geo_result,
            editorial=editorial,
            priority=priority,
            category=classification.primary,
            secondary=list(classification.secondary),
            confidence=ev.confidence,
            event_time=ev.event_time,
            source_groups=ev.source_groups,
            independent_source_groups=ev.independent_source_groups,
            is_wire_echo=ev.is_wire_echo,
            status=status,
            reasons=reasons,
        )

    # -- queue ordering -----------------------------------------------------

    def _order(self, candidates) -> list:
        """Deterministic queue: strongest, most urgent, most reliable first."""
        queued = [c for c in candidates if c.status == "queued"]
        queued.sort(key=self._queue_key)
        return queued

    @staticmethod
    def _queue_key(candidate):
        priority = candidate.priority
        level = priority.priority if priority else "NORMAL"
        score = priority.score if priority else 0.0
        confidence = candidate.confidence or "low"
        event_time = candidate.event_time
        ts = event_time.timestamp() if event_time else 0.0
        return (
            _LEVEL_ORDER.get(level, 99),
            -score,
            CONFIDENCE_RANK.get(confidence, 9),
            -ts,
            candidate.event_id,
        )


_LEVEL_ORDER = {"IMMEDIATE": 0, "URGENT": 1, "HIGH": 2, "NORMAL": 3}
