"""India importance / priority scoring.

Combines the outputs of the earlier pipeline stages (relevance, category,
geography, dedup/echo, editorial) to answer one question:

    "How important is this India-related news event to our audience?"

It deliberately does NOT answer whether something is India-related, what its
category is, where it happened, whether it is a duplicate, or whether it is
acceptable news -- those are owned by other stages and are only *consumed*
here as context.

Priorities (four levels, configurable thresholds):

    IMMEDIATE  >= immediate_threshold
    URGENT     >= urgent_threshold
    HIGH       >= high_threshold
    NORMAL     otherwise

Scoring model (designed against keyword-noise inflation)
---------------------------------------------------------
A single strong signal must be able to beat many weak ones, so the score is
not a plain sum:

    score = min(max_score, max(dominant_event, stack))

  * dominant_event = the strongest single event term matched (e.g. a major
    terror attack term carries 84 points on its own).
  * stack = min(stack_cap, source + corroboration + scale + recency +
    category + geo + weak_event_terms)

Weak keywords accumulate only up to ``stack_cap`` (a config value), so a
pile-up of minor terms can reach at most HIGH -- reaching URGENT/IMMEDIATE
requires a genuinely strong event term.

Families (each capped):
  * source         -- official / tier / specialist / international; a wire
                      event counts its PRIMARY source only, so 10 wire copies
                      score exactly like one.
  * corroboration  -- only ``independent_source_groups`` (dedup already
                      strips wire echoes); wire copies never add here.
  * scale          -- multi-state, nationwide, people-affected numbers.
  * recency        -- breaking/fresh bonuses; old stories are not boosted.
  * category       -- category importance weight scaled; sports is damped so
                      a flood of sports stories cannot dominate.
  * geo            -- national vs state vs local potential.

Confidence is separate from score: it comes from corroboration, so multiple
independent sources raise confidence (not the score) while wire echoes keep
confidence low.

Editorial is respected: rejected content is blocked (score 0, priority
NORMAL, major_event False) and filler content is capped at NORMAL, so an
article can never become publishable or high priority via scoring alone.

The result is explainable: score, priority, confidence, major_event flag,
signals, and reasons.
"""
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from src.config_loader import get_config
from src.models.article import Article

DEFAULT_PRIORITY = {
    "thresholds": {"immediate": 78, "urgent": 60, "high": 38},
    "weights": {
        "official_source": 8,
        "specialist_source": 5,
        "international_source": 4,
        "tier1_source": 5,
        "tier2_source": 2,
        "tier3_source": 1,
        "tier4_source": 0,
        "independent_corroboration": 8,
        "single_source": 2,
        "national_scope": 6,
        "multi_state": 6,
        "state_scope": 2,
        "local_scope": 0,
        "people_affected": 6,
        "breaking": 8,
        "fresh": 4,
        "stale": 0,
        "very_old": -2,
        "category_scale": 20,
    },
    "caps": {
        "source": 10,
        "corroboration": 8,
        "scale": 10,
        "recency": 8,
        "category": 10,
        "geo": 6,
        "weak_events": 24,
        "stack": 45,
    },
    "max_score": 100,
    "recency_hours": {"breaking": 3, "fresh": 24, "old": 48},
    "urgent_event_min": 60,
    "major_event_min": 55,
    "event_weights": {},
}

# Default event-term weights when not present in config. Only a small number
# of genuinely significant terms carry high weight; everything else is small
# so keyword pile-ups are structurally capped.
#
# Weights follow a deliberate ladder:
#   IMMEDIATE/URGENT range (60-84): disasters, war, pandemic, national-security
#   HIGH range             (38-59): major but non-catastrophic national events
#                                    (cabinet decisions, court rulings, missile
#                                    tests, regulatory action, acquisitions,
#                                    tariff/trade moves, major appointments,
#                                    infrastructure collapses, health crises)
#   low range              (<38)   : routine terms that must not push NORMAL
#   sports                  (4-44) : damped so volume cannot dominate
#
# Multi-word phrases are preferred over bare single words so that ordinary
# stories (a company merely mentioning "investment" or "launches") do not get
# boosted; a bare term is used only when the word itself is a strong signal
# (e.g. "acquisition", "tariff").
DEFAULT_EVENT_WEIGHTS = {
    "terror attack": 84,
    "war": 82,
    "pandemic": 82,
    "earthquake": 82,
    "epidemic": 70,
    "air crash": 72,
    "cyclone": 72,
    "national security": 70,
    "cyber attack": 66,
    "flood": 64,
    "floods": 64,
    "train accident": 64,
    "cloudburst": 48,
    "avalanche": 44,
    "landslide": 40,
    "defence deal": 44,
    "defence procurement": 44,
    "union budget": 44,
    "repo rate": 42,
    "monetary policy": 40,
    "ceasefire": 40,
    "sanctions": 40,
    "heatwave": 40,
    "heat wave": 40,
    "drought": 36,
    "gdp": 38,
    "inflation": 36,
    "lac": 62,
    "loc": 62,
    "election results": 64,
    "election": 58,
    "exit poll": 46,
    "supreme court": 44,
    "strikes down": 40,
    "unconstitutional": 40,
    "verdict": 22,
    "judgment": 18,
    "world cup": 44,
    "olympics": 40,
    "ipl": 14,
    "match": 8,
    "warm-up": 6,
    "probable xi": 6,
    "fantasy": 4,
    "final": 10,
    # --- cabinet / government decisions -----------------------------------
    "cabinet approves": 44,
    "cabinet clears": 44,
    "cabinet approval": 42,
    "cabinet nod": 40,
    "gives nod": 40,
    "government approves": 40,
    "government clears": 40,
    "parliament passes": 44,
    "parliament clears": 42,
    "bill passed": 44,
    "law passed": 42,
    # --- courts -----------------------------------------------------------
    "supreme court ruling": 46,
    "supreme court judgment": 46,
    "supreme court verdict": 46,
    # --- defence ----------------------------------------------------------
    "ballistic missile": 46,
    "missile test": 46,
    "nuclear test": 50,
    "drone attack": 44,
    "border clash": 46,
    "ceasefire violation": 42,
    # --- regulatory / markets ---------------------------------------------
    "sebi bans": 44,
    "sebi bars": 42,
    "insider trading": 40,
    "market regulator": 40,
    "regulator bans": 42,
    "regulatory action": 40,
    # --- corporate --------------------------------------------------------
    "acquisition": 42,
    "acquires": 40,
    "takeover": 42,
    "merger": 40,
    # --- trade ------------------------------------------------------------
    "tariff": 40,
    "tariffs": 40,
    "trade war": 46,
    "trade barrier": 40,
    "export ban": 42,
    "import ban": 42,
    "customs duty": 40,
    # --- appointments -----------------------------------------------------
    "appointed as": 40,
    "appointed": 36,
    "resigns as": 40,
    "steps down": 40,
    "chief justice": 42,
    "chief justice of india": 46,
    "sworn in": 40,
    "takes charge": 40,
    # --- infrastructure / accidents --------------------------------------
    "bridge collapse": 48,
    "building collapse": 46,
    "tunnel collapse": 44,
    "train derailment": 48,
    "aviation accident": 46,
    "plane crash": 50,
    "helicopter crash": 46,
    "gas leak": 42,
    "boiler blast": 44,
    "hospital fire": 46,
    "factory fire": 44,
    "mass casualty": 44,
    # --- health -----------------------------------------------------------
    "disease outbreak": 46,
    "contamination": 40,
    # --- strikes / public order -------------------------------------------
    "nationwide strike": 44,
    "general strike": 42,
    "bandh": 40,
    "curfew": 40,
    "lockdown": 42,
    # --- diplomacy --------------------------------------------------------
    "diplomatic row": 42,
    "expels diplomats": 44,
    "recalls ambassador": 42,
    # --- energy / prices --------------------------------------------------
    "fuel price": 40,
    "petrol price": 40,
    "electricity tariff": 40,
    "power outage": 40,
    "grid failure": 44,
    "blackout": 40,
}

_PRIORITY_ORDER = ["IMMEDIATE", "URGENT", "HIGH", "NORMAL"]
_CONFIDENCE_ORDER = ["high", "medium", "low"]
_PEOPLE_RE = re.compile(
    r"(?<!\w)(\d+(?:[.,]\d+)?)\s*(lakh|crore|million|billion|thousand)"
    r"\s*(people|persons|affected|evacuated|residents|workers)(?!\w)",
    re.IGNORECASE,
)


@dataclass
class PrioritySignal:
    family: str  # source | event | scale | recency | corroboration | category | geo | editorial
    term: str
    weight: float


@dataclass
class PriorityResult:
    score: float
    priority: str  # IMMEDIATE | URGENT | HIGH | NORMAL
    confidence: str  # high | medium | low
    major_event: bool
    blocked: bool
    thresholds: dict = field(default_factory=dict)
    reasons: list = field(default_factory=list)
    signals: list = field(default_factory=list)

    @property
    def order(self) -> int:
        return _PRIORITY_ORDER.index(self.priority)

    def __lt__(self, other):
        return self.order < other.order


class PriorityScorer:
    """Compute importance/priority for an accepted event or article."""

    def __init__(self, config=None):
        config = config or get_config()
        cfg = config["config"].get("priority", {})
        pr = dict(DEFAULT_PRIORITY)
        # nested dicts (thresholds/weights/caps/recency_hours) merge over defaults
        for key, sub in pr.items():
            if isinstance(sub, dict):
                merged = dict(sub)
                merged.update(cfg.get(key, {}))
                pr[key] = merged
        for key, val in cfg.items():
            if key not in pr:
                pr[key] = val
        event_weights = dict(DEFAULT_EVENT_WEIGHTS)
        event_weights.update(cfg.get("event_weights", {}))

        self.thresholds = {k: float(v) for k, v in pr["thresholds"].items()}
        self.weights = {k: float(v) for k, v in pr["weights"].items()}
        self.caps = {k: float(v) for k, v in pr["caps"].items()}
        self.max_score = float(pr["max_score"])
        self.recency_hours = {k: float(v) for k, v in pr["recency_hours"].items()}
        self.urgent_event_min = float(pr["urgent_event_min"])
        self.major_event_min = float(pr["major_event_min"])
        self.event_weights = event_weights

        self._event_regex = self._compile(event_weights)

        # source metadata: id -> (type, tier)
        self._source_meta = {}
        for s in config["sources"]["sources"]:
            sid = s.get("id")
            if sid:
                self._source_meta[sid] = (s.get("type", ""), int(s.get("tier", 4) or 4))

        # category importance metadata
        self._category_meta = {}
        for cat in config["categories"]["categories"]:
            cid = cat.get("id")
            if cid:
                self._category_meta[cid] = cat

    # -- public API ---------------------------------------------------------

    def score(
        self,
        article: Article,
        *,
        category=None,
        geo=None,
        relevance=None,
        event=None,
        editorial=None,
        now=None,
    ) -> PriorityResult:
        """Score an Article with optional downstream context.

        ``category`` may be a ClassificationResult, a category id string, or a
        dict with importance_weight/urgent keys. ``geo`` is a GeoResult,
        ``event`` an EventGroup from the dedup stage, ``editorial`` an
        EditorialResult.
        """
        return self.score_text(
            article.title or "",
            article.summary or "",
            source_id=article.source_id,
            published=article.published,
            category=category,
            geo=geo,
            relevance=relevance,
            event=event,
            editorial=editorial,
            now=now,
        )

    def score_text(
        self,
        title,
        summary="",
        *,
        source_id=None,
        published=None,
        category=None,
        geo=None,
        relevance=None,
        event=None,
        editorial=None,
        now=None,
    ) -> PriorityResult:
        now = now or datetime.now(timezone.utc)
        title = title or ""
        summary = summary or ""
        reasons = []
        signals = []
        major_event = False

        # -- editorial gate: rejected/filler can never be high priority ------
        blocked = False
        if editorial is not None:
            if getattr(editorial, "decision", "") == "reject":
                blocked = True
                reasons.append("editorial: rejected -> blocked from publishing")
            elif getattr(editorial, "decision", "") == "filler":
                reasons.append("editorial: filler -> capped at NORMAL")
        if blocked:
            return PriorityResult(
                score=0.0,
                priority="NORMAL",
                confidence="low",
                major_event=False,
                blocked=True,
                thresholds=self.thresholds,
                reasons=reasons,
            )

        # -- source family --------------------------------------------------
        source_contribution, source_reasons = self._source_signal(source_id, event)
        reasons.extend(source_reasons)
        signals.extend(source_contribution)
        source_total = min(self.caps["source"], sum(s.weight for s in source_contribution))

        # -- corroboration family -------------------------------------------
        corrob_total, corrob_reasons, confidence = self._corroboration(event)
        reasons.extend(corrob_reasons)
        corrob_total = min(self.caps["corroboration"], corrob_total)

        # -- event family ---------------------------------------------------
        dominant, event_reasons, weak_sum = self._event_signal(title, summary)
        major_event = dominant >= self.major_event_min
        reasons.extend(event_reasons)
        weak_total = min(self.caps["weak_events"], weak_sum)

        # -- scale family ---------------------------------------------------
        scale_total, scale_reasons = self._scale_signal(geo, title, summary)
        reasons.extend(scale_reasons)
        scale_total = min(self.caps["scale"], scale_total)

        # -- recency family -------------------------------------------------
        recency_total, recency_reasons = self._recency_signal(event, published, now)
        reasons.extend(recency_reasons)
        recency_total = max(-self.caps["recency"], min(self.caps["recency"], recency_total))

        # -- category family ------------------------------------------------
        cat_total, cat_reasons = self._category_signal(category)
        reasons.extend(cat_reasons)
        cat_total = min(self.caps["category"], cat_total)

        # -- geo family -----------------------------------------------------
        geo_total, geo_reasons = self._geo_signal(geo)
        reasons.extend(geo_reasons)
        geo_total = min(self.caps["geo"], geo_total)

        signals.append(PrioritySignal("event", "dominant", dominant))
        corrob_term = "wire_echo" if (event is not None and getattr(event, "is_wire_echo", False)) else f"independent_groups"
        signals.append(PrioritySignal("corroboration", corrob_term, corrob_total))
        signals.append(PrioritySignal("recency", "age", recency_total))
        signals.append(PrioritySignal("category", "importance", cat_total))
        signals.append(PrioritySignal("geo", "scope", geo_total))
        signals.append(PrioritySignal("source", "capped_total", source_total))
        signals.append(PrioritySignal("scale", "scale", scale_total))

        stack = source_total + corrob_total + scale_total + recency_total + cat_total + geo_total + weak_total
        stack = min(self.caps["stack"], stack)
        score = min(self.max_score, max(dominant, stack))
        score = round(score, 1)

        priority = self._priority(score)
        if editorial is not None and getattr(editorial, "decision", "") == "filler":
            if priority != "NORMAL":
                priority = "NORMAL"
                reasons.append("priority capped to NORMAL (editorial filler)")

        reasons.append(
            f"dominant event {dominant:g}, stack {stack:g}, score {score:g} -> {priority}"
        )

        return PriorityResult(
            score=score,
            priority=priority,
            confidence=confidence,
            major_event=major_event,
            blocked=False,
            thresholds=self.thresholds,
            reasons=reasons,
            signals=signals,
        )

    # -- families -----------------------------------------------------------

    def _source_signal(self, source_id, event):
        reasons = []
        contributions = []
        # wire events use their primary source exactly once
        sid = source_id
        if event is not None and getattr(event, "primary_source", None):
            sid = event.primary_source

        if not sid:
            return [], []
        stype, tier = self._source_meta.get(sid, ("", 4))

        w = self.weights.get("official_source", 0)
        if stype == "official":
            contributions.append(PrioritySignal("source", "official", w))
            reasons.append(f"source: official primary ({sid}) +{w:g}")
        w = self.weights.get(f"tier{tier}_source", 0)
        contributions.append(PrioritySignal("source", f"tier{tier}", w))
        reasons.append(f"source: tier {tier} ({sid}) +{w:g}")
        if stype == "specialist":
            w = self.weights.get("specialist_source", 0)
            contributions.append(PrioritySignal("source", "specialist", w))
            reasons.append(f"source: specialist ({sid}) +{w:g}")
        elif stype == "international":
            w = self.weights.get("international_source", 0)
            contributions.append(PrioritySignal("source", "international", w))
            reasons.append(f"source: international ({sid}) +{w:g}")
        if event is not None and getattr(event, "member_indices", None) and len(event.member_indices) > 1:
            reasons.append(
                f"source: event has {len(event.member_indices)} copies, primary source counted once"
            )
        return contributions, reasons

    def _corroboration(self, event):
        if event is None:
            return self.weights.get("single_source", 2), ["corroboration: single article, default medium"], "medium"
        indep = getattr(event, "independent_source_groups", 0) or 0
        is_echo = getattr(event, "is_wire_echo", False)
        reasons = []
        if is_echo:
            reasons.append(
                f"corroboration: wire echo ({getattr(event, 'wire_group', '')}) -> no independent boost"
            )
            return 0.0, reasons, "low"
        if indep >= 2:
            reasons.append(f"corroboration: {indep} independent source groups -> high confidence")
            return self.weights.get("independent_corroboration", 8), reasons, "high"
        if indep == 1:
            reasons.append("corroboration: single independent source -> medium confidence")
            return self.weights.get("single_source", 2), reasons, "medium"
        reasons.append("corroboration: no independent confirmation -> low confidence")
        return 0.0, reasons, "low"

    def _event_signal(self, title, summary):
        reasons = []
        matched = []
        for text, mult in ((title, 1.0), (summary, 0.6)):
            if not text:
                continue
            for term, weight in self.event_weights.items():
                regex = self._event_regex.get(term)
                if regex is None:
                    continue
                if regex.search(text):
                    scaled = weight * mult
                    matched.append((term, scaled))
                    reasons.append(
                        f"event term: '{term}' in {'title' if mult == 1.0 else 'summary'} "
                        f"weight {scaled:g}"
                    )
        if not matched:
            return 0.0, reasons, 0.0
        matched.sort(key=lambda x: x[1], reverse=True)
        dominant = matched[0][1]
        weak_sum = sum(w for _, w in matched) - dominant
        weak_sum = max(weak_sum, 0.0)
        return dominant, reasons, weak_sum

    def _scale_signal(self, geo, title, summary):
        reasons = []
        total = 0.0
        if geo is not None:
            if getattr(geo, "is_national_story", False):
                total += self.weights.get("national_scope", 6)
                reasons.append("scale: national scope")
            elif getattr(geo, "state_identifiable", False):
                total += self.weights.get("state_scope", 2)
                reasons.append(f"scale: state scope ({getattr(geo, 'state', '')})")
            states = getattr(geo, "states", []) or []
            if len(states) >= 2:
                total += self.weights.get("multi_state", 6)
                reasons.append(f"scale: multi-state ({len(states)} states)")
        people = 0
        for text in (title, summary):
            if text:
                people += len(_PEOPLE_RE.findall(text))
        if people:
            total += self.weights.get("people_affected", 6) * min(people, 2)
            reasons.append(f"scale: {people} people-affected figures")
        return total, reasons

    def _recency_signal(self, event, published, now):
        reasons = []
        ts = None
        if event is not None:
            ts = getattr(event, "event_time", None)
        if ts is None and published is not None:
            ts = published
        if ts is None:
            return self.weights.get("fresh", 4), ["recency: published time unknown -> fresh"]
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        hours = (now - ts).total_seconds() / 3600.0
        if hours <= self.recency_hours["breaking"]:
            w = self.weights.get("breaking", 8)
            reasons.append(f"recency: breaking ({hours:.1f}h ago) +{w:g}")
        elif hours <= self.recency_hours["fresh"]:
            w = self.weights.get("fresh", 4)
            reasons.append(f"recency: fresh ({hours:.1f}h ago) +{w:g}")
        elif hours <= self.recency_hours["old"]:
            w = self.weights.get("stale", 0)
            reasons.append(f"recency: {hours:.1f}h old, no boost")
        else:
            w = self.weights.get("very_old", -2)
            reasons.append(f"recency: very old ({hours:.1f}h) {w:g}")
        return w, reasons

    def _category_signal(self, category):
        cat_id = None
        importance = None
        if isinstance(category, str):
            cat_id = category
        elif isinstance(category, dict):
            cat_id = category.get("id")
            importance = category.get("importance_weight")
        elif category is not None:
            cat_id = getattr(category, "primary", None)
            importance = getattr(category, "primary_importance_weight", None)

        if cat_id is None:
            return 0.0, []
        if importance is None:
            meta = self._category_meta.get(cat_id, {})
            importance = meta.get("importance_weight", 1.0)

        # damp sports so a flood of sports stories cannot dominate
        base = max(0.0, (float(importance) - 0.9)) * self.weights.get("category_scale", 20)
        if cat_id == "sports":
            base = min(base, 4.0)
        return base, [f"category: {cat_id} (importance {float(importance):g}) +{base:g}"]

    def _geo_signal(self, geo):
        if geo is None:
            return 0.0, []
        if getattr(geo, "is_national_story", False):
            return self.weights.get("national_scope", 6), ["geo: national"]
        if getattr(geo, "national_significance", False):
            return 4.0, ["geo: national significance (local/state event)"]
        if getattr(geo, "state_identifiable", False):
            return self.weights.get("state_scope", 2), ["geo: state"]
        return 0.0, ["geo: local"]

    # -- helpers ------------------------------------------------------------

    def _priority(self, score):
        if score >= self.thresholds["immediate"]:
            return "IMMEDIATE"
        if score >= self.thresholds["urgent"]:
            return "URGENT"
        if score >= self.thresholds["high"]:
            return "HIGH"
        return "NORMAL"

    @staticmethod
    def _compile(terms):
        out = {}
        for term, weight in terms.items():
            term = (term or "").strip().lower()
            if not term:
                continue
            out[term] = re.compile(r"(?<!\w)(?:%s)(?!\w)" % re.escape(term), re.IGNORECASE)
        return out
