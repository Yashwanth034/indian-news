"""Editorial gate layer.

Decides whether a fully-normalised article (already India-relevant,
classified, geo-scoped and deduplicated) is suitable for the main news queue.

The layer is deliberately *not* a replacement for relevance, categorisation,
geography, deduplication or importance scoring -- those remain separate
stages. It evaluates editorial quality signals that are configured in
config/editorial.json (the pattern lists) and tuned in the ``editorial``
block of config/config.json (weights, thresholds, override rules).

Four concepts are kept separate:

  * hard exclusion    -- astrology/horoscope, unverified rumours: never enter
                         the queue unless a strong major signal applies
  * soft penalty      -- clickbait, celebrity mention, minor-sports wording:
                         lowers the score but does not automatically reject
  * filler            -- technically news but low-value (routine notices,
                         match previews, "watch this space" filler): normally
                         excluded from the main channel
  * major override    -- genuinely significant events override weak
                         low-value signals so a major story is not dropped
                         just because it mentions a celebrity or uses a
                         clickbait-ish phrase

Scoring model
-------------
score = base_score + major_boost(if any major signal) - penalties

Penalties come from matched gate patterns (title weighted higher than
summary). Override rules are per gate:

  * override="any"    -- a weak or strong major signal drops this penalty
  * override="strong" -- only a strong major signal drops this penalty
                         (significance event, urgent/important category,
                         confirmed/breaking marker)
  * override="none"   -- penalty is never dropped (labelled opinion)

Major signals
-------------
weak:   category importance weight >= importance_major_min, geo
        national significance, corroboration (2+ independent sources),
        confirmed/breaking marker
strong: a configured significance event term (cyclone, earthquake, terror
        attack, ...), an urgent or high-importance category
        (>= importance_strong_min)

Decisions
---------
  pass   -- score >= pass_threshold           (enter main queue)
  filler -- score >= filler_threshold         (low value; not main queue)
  reject -- hard exclusion or score below     (do not queue)

The result is explainable: decision, score, matched signals, matched gates,
major signals, overridden gates, and reasons.
"""
import re
from dataclasses import dataclass, field
from typing import Optional

from src.config_loader import get_config
from src.models.article import Article

# editorial.json key used by each gate.
GATE_SOURCES = {
    "clickbait": "clickbait_patterns",
    "gossip": "gossip_terms",
    "celebrity": "celebrity_terms",
    "astrology": "astrology_terms",
    "rumour": "rumour_terms",
    "opinion": "opinion_markers",
    "filler": "filler_phrases",
    "routine_official": "routine_official_patterns",
    "sports_minor": "sports_minor_patterns",
    "excluded_topics": "excluded_topics",
}

# Gates whose markers are only trusted in the headline (labelled opinion
# markers like "Opinion:" / "i believe" would false-positive on quotes).
TITLE_ONLY_GATES = {"opinion"}

DEFAULT_EDITORIAL = {
    "base_score": 70,
    "pass_threshold": 60,
    "filler_threshold": 35,
    "title_multiplier": 1.5,
    "summary_multiplier": 0.5,
    "major_boost": 30,
    "importance_major_min": 1.1,
    "importance_strong_min": 1.2,
    "major_markers": ["breaking", "confirmed"],
    "weak_term_weights": {"reportedly": 6},
    "gates": {
        "clickbait": {"weight": 25, "override": "any"},
        "gossip": {"weight": 35, "override": "any"},
        "celebrity": {"weight": 15, "override": "any"},
        "astrology": {"weight": 100, "override": "strong"},
        "rumour": {"weight": 25, "override": "strong"},
        "opinion": {"weight": 20, "override": "none"},
        "filler": {"weight": 18, "override": "strong"},
        "routine_official": {"weight": 15, "override": "strong"},
        "sports_minor": {"weight": 15, "override": "any"},
        "excluded_topics": {"weight": 30, "override": "strong"},
    },
}


@dataclass
class EditorialSignal:
    gate: str
    term: str
    location: str  # title | summary
    penalty: float
    overridden: bool = False


@dataclass
class EditorialResult:
    decision: str  # pass | filler | reject
    score: float
    pass_threshold: float
    filler_threshold: float
    category: Optional[str]
    major: bool
    strong_major: bool
    reasons: list = field(default_factory=list)
    signals: list = field(default_factory=list)
    gates: list = field(default_factory=list)
    major_signals: list = field(default_factory=list)
    overridden_gates: list = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return self.decision == "pass"


class EditorialGate:
    """Evaluate editorial suitability of a normalised article."""

    def __init__(self, config=None):
        config = config or get_config()
        cfg = config["config"].get("editorial", {})
        ed = dict(DEFAULT_EDITORIAL)
        ed.update({k: v for k, v in cfg.items() if k != "gates"})
        gates_cfg = dict(DEFAULT_EDITORIAL["gates"])
        gates_cfg.update(cfg.get("gates", {}))

        self.base_score = float(ed["base_score"])
        self.pass_threshold = float(ed["pass_threshold"])
        self.filler_threshold = float(ed["filler_threshold"])
        self.title_multiplier = float(ed["title_multiplier"])
        self.summary_multiplier = float(ed["summary_multiplier"])
        self.major_boost = float(ed["major_boost"])
        self.importance_major_min = float(ed["importance_major_min"])
        self.importance_strong_min = float(ed["importance_strong_min"])

        self.weak_term_weights = {k.strip().lower(): float(v) for k, v in ed.get("weak_term_weights", {}).items()}
        self.major_markers = [m.lower() for m in ed.get("major_markers", [])]

        ed_json = config["editorial"]
        self._gate_regexes = {}
        self._gate_overrides = {}
        self._gate_weights = {}
        for gate, source_key in GATE_SOURCES.items():
            terms = [t for t in ed_json.get(source_key, []) if t and str(t).strip()]
            self._gate_regexes[gate] = self._compile(terms)
            meta = gates_cfg.get(gate, {})
            self._gate_weights[gate] = float(meta.get("weight", 0))
            self._gate_overrides[gate] = meta.get("override", "any")

        self._major_marker_regex = self._compile(self.major_markers)
        self._significance_terms = {
            (t or "").strip().lower()
            for t in config["india_geo"].get("significance_events", [])
            if t and str(t).strip()
        }
        self._significance_regex = self._compile(self._significance_terms)

        self._category_meta = {}
        for cat in config["categories"]["categories"]:
            cid = cat.get("id")
            if cid:
                self._category_meta[cid] = cat

    # -- public API ---------------------------------------------------------

    def evaluate(
        self,
        article: Article,
        *,
        category=None,
        geo=None,
        relevance=None,
        event=None,
    ) -> EditorialResult:
        """Evaluate an Article with optional downstream context.

        ``category`` may be a ClassificationResult, a category id string, or a
        dict with importance_weight/urgent keys. ``geo`` is a GeoResult,
        ``event`` an EventGroup from the dedup stage.
        """
        return self.evaluate_text(
            article.title or "",
            article.summary or "",
            category=category,
            geo=geo,
            relevance=relevance,
            event=event,
        )

    def evaluate_text(self, title, summary="", *, category=None, geo=None,
                      relevance=None, event=None) -> EditorialResult:
        title = title or ""
        summary = summary or ""

        cat_id, importance, urgent = self._category_info(category)
        weak_major, strong_major, major_reasons = self._major(
            title, summary, importance, urgent, geo=geo, event=event
        )

        signals, gates = self._match(title, summary)
        overridden = []
        if weak_major:
            for sig in signals:
                override = self._gate_overrides[sig.gate]
                if override == "any":
                    sig.overridden = True
                    overridden.append(sig.gate)
                elif override == "strong" and strong_major:
                    sig.overridden = True
                    overridden.append(sig.gate)
        elif strong_major:
            for sig in signals:
                if self._gate_overrides[sig.gate] in ("any", "strong"):
                    sig.overridden = True
                    overridden.append(sig.gate)
        overridden = list(dict.fromkeys(overridden))

        # a gate penalises at most once: use the largest penalty across its signals
        active = {s.gate: max((x.penalty for x in signals if x.gate == s.gate and not x.overridden), default=0.0) for s in signals}
        active_penalty = sum(active.values())
        score = self.base_score - active_penalty
        if weak_major or strong_major:
            score += self.major_boost
        score = round(min(max(score, 0.0), 100.0), 1)

        poison = any(
            not s.overridden and self._gate_overrides[s.gate] == "none"
            for s in signals
        )
        if poison:
            decision = "reject"
        elif score >= self.pass_threshold:
            decision = "pass"
        elif score >= self.filler_threshold:
            decision = "filler"
        else:
            decision = "reject"

        reasons = []
        for sig in sorted(signals, key=lambda s: (s.location, s.gate, s.term)):
            status = "overridden" if sig.overridden else "active"
            reasons.append(
                f"{sig.gate}: '{sig.term}' in {sig.location} "
                f"(penalty {sig.penalty:g}, {status})"
            )
        reasons.extend(major_reasons)
        if cat_id:
            reasons.append(f"category: {cat_id} (importance {importance:g}, urgent={urgent})")
        if overridden:
            reasons.append(f"major override dropped penalties: {', '.join(overridden)}")
        reasons.append(
            f"score {score} -> {decision} "
            f"(pass >= {self.pass_threshold:g}, filler >= {self.filler_threshold:g})"
        )

        return EditorialResult(
            decision=decision,
            score=score,
            pass_threshold=self.pass_threshold,
            filler_threshold=self.filler_threshold,
            category=cat_id,
            major=bool(weak_major or strong_major),
            strong_major=bool(strong_major),
            reasons=reasons,
            signals=signals,
            gates=gates,
            major_signals=major_reasons,
            overridden_gates=overridden,
        )

    # -- internals ----------------------------------------------------------

    def _match(self, title, summary):
        signals = []
        gates = []
        for gate, regex in self._gate_regexes.items():
            if regex is None:
                continue
            base = self._gate_weights[gate]
            if base <= 0:
                continue
            locations = [("title", title, self.title_multiplier)]
            if gate not in TITLE_ONLY_GATES:
                locations.append(("summary", summary, self.summary_multiplier))
            for location, text, mult in locations:
                if not text:
                    continue
                for m in regex.finditer(text):
                    term = m.group(0).strip()
                    key = term.lower()
                    weight = self.weak_term_weights.get(key, base)
                    penalty = round(weight * mult, 1)
                    signals.append(
                        EditorialSignal(
                            gate=gate,
                            term=term,
                            location=location,
                            penalty=penalty,
                        )
                    )
                    if gate not in gates:
                        gates.append(gate)
        return signals, gates

    def _major(self, title, summary, importance, urgent, *, geo=None, event=None):
        reasons = []
        strong = False
        weak = False

        if geo is not None and getattr(geo, "national_significance", False):
            weak = True
            reasons.append("major: geo national significance")

        if event is not None and getattr(event, "independent_source_groups", 0) >= 2:
            weak = True
            reasons.append(
                "major: corroborated by "
                f"{getattr(event, 'independent_source_groups', 0)} independent sources"
            )

        if importance is not None and importance >= self.importance_strong_min:
            strong = True
            reasons.append(
                f"major: category importance {importance:g} >= {self.importance_strong_min:g}"
            )
        elif importance is not None and importance >= self.importance_major_min:
            weak = True
            reasons.append(
                f"major: category importance {importance:g} >= {self.importance_major_min:g}"
            )

        if urgent:
            strong = True
            reasons.append("major: urgent category")

        for text in (title, summary):
            if not text:
                continue
            if self._significance_regex is not None:
                for m in self._significance_regex.finditer(text):
                    strong = True
                    reasons.append(f"major: significance event '{m.group(0).strip()}'")
            if self._major_marker_regex is not None:
                for m in self._major_marker_regex.finditer(text):
                    weak = True
                    reasons.append(f"major: marker '{m.group(0).strip()}'")

        return weak, strong, reasons

    def _category_info(self, category):
        if category is None:
            return None, None, False
        if isinstance(category, str):
            cid = category
        elif isinstance(category, dict):
            cid = category.get("id")
        else:
            cid = getattr(category, "primary", None)
            if not cid:
                return None, None, False

        meta = self._category_meta.get(cid, {})
        importance = meta.get("importance_weight")
        if importance is None and isinstance(category, dict):
            importance = category.get("importance_weight")
        if importance is None and not isinstance(category, str) and not isinstance(category, dict):
            importance = getattr(category, "primary_importance_weight", None)
        urgent = bool(
            getattr(category, "primary_urgent", None)
            if hasattr(category, "primary_urgent")
            else meta.get("urgent", False)
        )
        if isinstance(category, dict):
            urgent = bool(category.get("urgent", urgent))
        return cid, float(importance) if importance is not None else None, urgent

    @staticmethod
    def _compile(terms):
        ordered = sorted({t.lower() for t in terms if t and str(t).strip()}, key=len, reverse=True)
        if not ordered:
            return None
        pattern = r"(?<!\w)(?:%s)(?!\w)" % "|".join(re.escape(t) for t in ordered)
        return re.compile(pattern, re.IGNORECASE)
