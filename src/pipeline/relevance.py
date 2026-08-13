"""India relevance detection.

Determines whether an article is genuinely about India or materially affects
India, using config/india_entities.json (entities, ministries, agencies,
cities, states, companies, people/roles, international context) and
config/india_geo.json national-significance markers.

Scoring model:
  * per-signal weighted contributions, scaled by location (title 1.0x,
    summary 0.6x); headlines must carry the weight, not just summaries
  * subject bonus when the headline opens with an India anchor term
  * significance bonus when a national-significance marker is in the headline
  * international terms only count when an India anchor is present
    (general international terms), plus an intl bonus for anchored stories;
    India-specific international terms (LAC, LOC, Kashmir, galwan, ...) are
    strong on their own and get an anchor bonus in the headline
  * false-positive phrases (e.g. "Indian Wells", "West Indies") suppress any
    overlapping signal, so incidental foreign uses of "india"/"indian" do not
    count
  * state/local stories pass Phase 1 only when they carry national
    significance (explicit gate + score threshold)

The result is explainable: score, decision, matched signals, and reasons.
"""
import re
from dataclasses import dataclass, field

from src.config_loader import get_config
from src.models.article import Article

DEFAULT_WEIGHTS = {
    "core_weight": 20,
    "institution_weight": 25,
    "figure_weight": 20,
    "company_weight": 25,
    "geo_weight": 12,
    "significance_weight": 25,
    "intl_weight": 15,
    "india_specific_intl_weight": 25,
    "subject_bonus": 35,
    "intl_bonus": 30,
    "significance_bonus": 20,
    "intl_anchor_bonus": 20,
    "title_multiplier": 1.0,
    "summary_multiplier": 0.6,
    "max_score": 100,
}

STRONG_TYPES = {
    "core", "institution", "figure", "company", "significance", "india_specific_intl",
}

# International-context terms that are inherently about India's interests,
# from config/india_entities.json international_context_terms.
INDIA_SPECIFIC_INTL = {
    "lac", "loc", "siachen", "kashmir", "galwan", "chinese troops",
    "india-pakistan", "india-china", "india-us", "india-russia",
    "india-eu", "diaspora", "overseas indians", "indian-origin",
}

# Government bodies / national institutions that are not already captured by
# the ministries + agencies lists. Used to separate institution-level
# national-significance markers from event markers.
CURATED_INSTITUTIONS = [
    "parliament", "lok sabha", "rajya sabha", "supreme court", "supreme court of india",
    "chief justice", "chief justice of india", "election commission",
    "election commission of india", "reserve bank", "reserve bank of india", "rbi",
    "sebi", "trai", "niti aayog", "npci", "centre", "central government",
    "union government", "government of india", "cabinet", "union cabinet",
    "cbi", "enforcement directorate", "nia", "isro", "drdo", "imd", "ndma", "ndrf",
    "icmr", "pib", "nta", "upsc", "cbse", "ugc", "cag", "cvc", "ncert",
    "indian railways", "railways", "defence research and development organisation",
]

# Generic marker terms in india_geo.json that should not drive relevance alone.
WEAK_MARKERS = {"national"}


@dataclass
class Signal:
    type: str
    term: str
    location: str  # title | summary
    weight: float


@dataclass
class RelevanceResult:
    is_india: bool
    score: float
    threshold: float
    decision: str  # include | exclude
    reasons: list = field(default_factory=list)
    signals: list = field(default_factory=list)
    geo_scope: str = "unknown"  # state | local | national | unknown


class IndiaRelevance:
    """Detect India relevance for an Article using configured signals."""

    def __init__(self, config=None):
        config = config or get_config()
        cfg = config["config"]
        ent = config["india_entities"]
        geo = config["india_geo"]

        rel = cfg.get("relevance", {})
        self.weights = dict(DEFAULT_WEIGHTS)
        self.weights.update({k: v for k, v in rel.items() if k in self.weights})
        self.threshold = rel.get("threshold", cfg.get("min_india_relevance", 55))
        self.max_score = float(self.weights["max_score"])

        self.core_terms = self._terms(ent.get("india_signals", []))
        self.core_terms.discard("indigenous")
        self.core_terms.update({"india", "indian", "indians", "bharat"})
        self.institution_terms = self._terms(
            ent.get("ministries", []), ent.get("agencies", []), CURATED_INSTITUTIONS
        )
        self.figure_terms = self._terms(ent.get("people_roles", []))
        self.figure_terms.update({"chief justice", "president of india", "vice president"})
        self.company_terms = self._terms(ent.get("companies", []))
        self.geo_terms = self._terms(
            ent.get("states", []), ent.get("union_territories", []), ent.get("major_cities", [])
        )
        self.state_or_ut_terms = self._terms(ent.get("states", []), ent.get("union_territories", []))
        self.fp_terms = self._terms(ent.get("false_positive_phrases", []))

        intl_all = self._terms(ent.get("international_context_terms", []))
        self.india_specific_intl_terms = {
            t for t in intl_all if t in INDIA_SPECIFIC_INTL
            or any(x in t for x in ("india", "indian", "kashmir"))
        }
        self.general_intl_terms = intl_all - self.india_specific_intl_terms

        marker_terms = self._terms(geo.get("national_significance_markers", []))
        self.significance_terms = (
            marker_terms
            - self.institution_terms
            - self.figure_terms
            - self.india_specific_intl_terms
            - self.general_intl_terms
            - WEAK_MARKERS
        )

        self._group_regexes = {
            "core": self._compile(self.core_terms),
            "institution": self._compile(self.institution_terms),
            "figure": self._compile(self.figure_terms),
            "company": self._compile(self.company_terms),
            "geo": self._compile(self.geo_terms),
            "significance": self._compile(self.significance_terms),
            "india_specific_intl": self._compile(self.india_specific_intl_terms),
            "intl": self._compile(self.general_intl_terms),
        }
        self._fp_regex = self._compile(self.fp_terms)

    # -- public API ---------------------------------------------------------

    def score(self, article: Article) -> RelevanceResult:
        return self.score_text(article.title or "", article.summary or "")

    def score_text(self, title: str, summary: str = "") -> RelevanceResult:
        title_signals, title_subject = self._match(title or "", "title")
        summary_signals, _ = self._match(summary or "", "summary")
        signals = title_signals + summary_signals

        has_anchor = any(s.type != "intl" for s in signals)
        general_intl = any(s.type == "intl" for s in signals)

        score = 0.0
        for s in signals:
            if s.type == "intl" and not has_anchor:
                continue
            mult = self.weights["title_multiplier"] if s.location == "title" else self.weights["summary_multiplier"]
            score += s.weight * mult

        if title_subject:
            score += self.weights["subject_bonus"] * self.weights["title_multiplier"]
        if general_intl and has_anchor:
            score += self.weights["intl_bonus"] * self.weights["title_multiplier"]
        if any(s.type == "significance" and s.location == "title" for s in signals):
            score += self.weights["significance_bonus"] * self.weights["title_multiplier"]
        if any(s.type == "india_specific_intl" and s.location == "title" for s in signals):
            score += self.weights["intl_anchor_bonus"] * self.weights["title_multiplier"]

        score = min(round(score, 1), self.max_score)
        scope = self._geo_scope(signals)

        reasons = []
        for s in signals:
            if s.type == "intl" and not has_anchor:
                continue
            reasons.append(f"{s.type}: '{s.term}' in {s.location} (weight {s.weight})")
        if title_subject:
            reasons.append("subject bonus: title begins with an India anchor")
        if general_intl and has_anchor:
            reasons.append("international bonus: general international term with India anchor")
        if any(s.type == "significance" and s.location == "title" for s in signals):
            reasons.append("significance bonus: national-significance marker in title")
        if any(s.type == "india_specific_intl" and s.location == "title" for s in signals):
            reasons.append("anchor bonus: India-specific international term in title")

        gate = False
        if scope in ("state", "local") and not any(s.type in STRONG_TYPES for s in signals):
            gate = True
            reasons.append("state/local story without national significance (Phase 1 gate)")

        is_india = score >= self.threshold and not gate
        reasons.append(f"score {score} {'>=' if is_india else '<'} threshold {self.threshold}")

        return RelevanceResult(
            is_india=is_india,
            score=score,
            threshold=self.threshold,
            decision="include" if is_india else "exclude",
            reasons=reasons,
            signals=signals,
            geo_scope=scope,
        )

    # -- internals ----------------------------------------------------------

    def _match(self, text: str, location: str):
        fp_spans = []
        if self._fp_regex is not None:
            fp_spans = [m.span() for m in self._fp_regex.finditer(text)]

        def blocked(start, end):
            return any(start < fe and end > fs for fs, fe in fp_spans)

        first_content_offset = len(text) - len(text.lstrip())
        signals = []
        seen = set()
        subject_ok = False

        for stype, regex in self._group_regexes.items():
            if regex is None:
                continue
            weight = float(self.weights.get(f"{stype}_weight", 0))
            for m in regex.finditer(text):
                if blocked(m.start(), m.end()):
                    continue
                if stype != "intl" and location == "title" and m.start() == first_content_offset:
                    subject_ok = True
                key = (stype, m.group(0).strip().lower())
                if key in seen:
                    continue
                seen.add(key)
                signals.append(
                    Signal(type=stype, term=m.group(0).strip(), location=location, weight=weight)
                )
        return signals, subject_ok

    def _geo_scope(self, signals):
        if any(s.type == "geo" and s.term.lower() in self.state_or_ut_terms for s in signals):
            return "state"
        if any(s.type == "geo" for s in signals):
            return "local"
        if any(s.type != "intl" for s in signals):
            return "national"
        return "unknown"

    @staticmethod
    def _terms(*groups) -> set:
        out = set()
        for group in groups:
            for term in group or []:
                term = (term or "").strip().lower()
                if term:
                    out.add(term)
        return out

    @staticmethod
    def _compile(terms):
        if not terms:
            return None
        ordered = sorted(terms, key=len, reverse=True)
        pattern = r"(?<!\w)(?:%s)(?!\w)" % "|".join(re.escape(t) for t in ordered)
        return re.compile(pattern, re.IGNORECASE)
