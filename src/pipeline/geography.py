"""India geo classification (scope, state, national significance).

Determines the geographic scope of an article -- national, state, or local --
along with the identifiable Indian state/UT and whether the story carries
national significance. All rules live in config/india_geo.json plus the
states/union_territories lists in config/india_entities.json:

  * state/UT names resolve to ``state`` scope; union territories are treated
    as state nodes, so "Delhi", "Chandigarh", and city-derived UTs such as
    "Port Blair" (Andaman & Nicobar) are state scope
  * cities resolve to their state/UT via ``city_to_state`` and produce
    ``local`` scope unless the resolved place is a union territory
  * two or more distinct states/UTs (direct or city-derived) -> ``national``
    (multi-state / India-wide stories)
  * a national-scope term in the TITLE with no significance event promotes an
    otherwise state/local story to ``national`` (state is then context only,
    e.g. "PM Modi in Bihar", "RBI ... Telangana")
  * no geo signal at all -> ``national`` scope by default
  * significance events (cyclone, earthquake, train accident, election, ...)
    keep the scope but set ``national_significance = True``; a national scope
    is itself nationally significant
  * false-positive patterns (e.g. "Surat Thani", "Hyderabad ... Pakistan",
    "Indian Wells") suppress overlapping geo, scope, and event matches so
    foreign places that look Indian do not count

The result is explainable: scope, states, state, national_significance,
matched signals, and reasons.
"""
import re
from dataclasses import dataclass, field
from typing import Optional

from src.config_loader import get_config
from src.models.article import Article


@dataclass
class GeoSignal:
    group: str  # state | city | national_scope | significance | false_positive
    term: str
    location: str  # title | summary
    state: Optional[str] = None


@dataclass
class GeoResult:
    scope: str  # national | state | local
    states: list  # distinct states/UTs (direct + city-derived), title + summary
    state: Optional[str]  # single distinct state/UT, title-first, else None
    state_identifiable: bool
    national_significance: bool
    is_national_story: bool
    has_title_geo: bool
    reasons: list = field(default_factory=list)
    signals: list = field(default_factory=list)


class GeoClassifier:
    """Classify articles by geographic scope and Indian state/UT."""

    def __init__(self, config=None):
        config = config or get_config()
        ent = config["india_entities"]
        geo = config["india_geo"]

        self.state_terms = self._terms(ent.get("states", []), ent.get("union_territories", []))
        self.ut_terms = self._terms(ent.get("union_territories", []))
        self.city_to_state = {
            (k or "").strip().lower(): (v or "").strip().lower()
            for k, v in geo.get("city_to_state", {}).items()
            if k and v
        }
        self.national_scope_terms = self._terms(geo.get("national_scope_terms", []))
        self.significance_terms = self._terms(geo.get("significance_events", []))
        self.fp_patterns = [p for p in geo.get("false_positive_patterns", []) if p]

        self._group_regexes = {
            "state": self._compile(self.state_terms),
            "city": self._compile(set(self.city_to_state)),
            "national_scope": self._compile(self.national_scope_terms),
            "significance": self._compile(self.significance_terms),
        }
        self._fp_regexes = [re.compile(p, re.IGNORECASE) for p in self.fp_patterns]

    # -- public API ---------------------------------------------------------

    def classify(self, article: Article) -> GeoResult:
        return self.classify_text(article.title or "", article.summary or "")

    def classify_text(self, title: str, summary: str = "") -> GeoResult:
        title_signals = self._match(title or "", "title")
        summary_signals = self._match(summary or "", "summary")
        signals = title_signals + summary_signals

        direct_title = {s.state for s in title_signals if s.group == "state"} - {None}
        city_title = {s.state for s in title_signals if s.group == "city"} - {None}
        direct_all = {s.state for s in signals if s.group == "state"} - {None}
        city_all = {s.state for s in signals if s.group == "city"} - {None}

        # city-derived union territories count as direct state nodes
        node_title = direct_title | {s for s in city_title if s in self.ut_terms}
        node_all = direct_all | {s for s in city_all if s in self.ut_terms}

        distinct_title = direct_title | city_title
        distinct_all = direct_all | city_all

        national_title = any(s.group == "national_scope" for s in title_signals)
        event = any(s.group == "significance" for s in signals)

        if len(distinct_all) >= 2:
            scope = "national"
        elif national_title and not event:
            scope = "national"
        elif len(node_all) >= 2:
            scope = "national"
        elif node_title:
            scope = "state"
        elif node_all:
            scope = "state"
        elif city_all:
            scope = "local"
        else:
            scope = "national"

        if len(distinct_title) == 1:
            state = sorted(distinct_title)[0]
        elif len(distinct_title) == 0 and len(distinct_all) == 1:
            state = sorted(distinct_all)[0]
        else:
            state = None

        multi_state = len(distinct_all) >= 2
        national_scope_matched = any(s.group == "national_scope" for s in signals)
        national_significance = event or national_scope_matched or multi_state
        has_title_geo = any(s.group in ("state", "city", "national_scope") for s in title_signals)

        reasons = []
        for s in signals:
            if s.group == "false_positive":
                reasons.append(f"false-positive: '{s.term}' ({s.location})")
                continue
            suffix = f" -> {s.state}" if s.state else ""
            reasons.append(f"{s.group}: '{s.term}'{suffix} ({s.location})")
        if len(distinct_all) >= 2:
            reasons.append(f"multi-state ({len(distinct_all)} states/UTs) -> national")
        elif national_title and not event and scope == "national":
            reasons.append("national-scope term in title, no significance event -> national")
        if event:
            reasons.append("significance event matched -> nationally significant")
        elif national_scope_matched:
            reasons.append("national-scope term matched -> nationally significant")
        elif multi_state:
            reasons.append("multi-state story -> nationally significant")
        reasons.append(f"scope: {scope}")
        reasons.append(f"state: {state}" if state else "state: not identifiable")

        return GeoResult(
            scope=scope,
            states=sorted(distinct_all),
            state=state,
            state_identifiable=state is not None,
            national_significance=national_significance,
            is_national_story=scope == "national",
            has_title_geo=has_title_geo,
            reasons=reasons,
            signals=signals,
        )

    # -- internals ----------------------------------------------------------

    def _match(self, text: str, location: str):
        fp_spans = [m.span() for r in self._fp_regexes for m in r.finditer(text)]

        def blocked(start, end):
            return any(start < fe and end > fs for fs, fe in fp_spans)

        signals = []
        seen = set()
        for group, regex in self._group_regexes.items():
            if regex is None:
                continue
            for m in regex.finditer(text):
                if blocked(m.start(), m.end()):
                    continue
                term = m.group(0).strip().lower()
                key = (group, term)
                if key in seen:
                    continue
                seen.add(key)
                state = None
                if group == "state":
                    state = term
                elif group == "city":
                    state = self.city_to_state.get(term)
                signals.append(
                    GeoSignal(group=group, term=m.group(0).strip(), location=location, state=state)
                )
        seen_fp = set()
        for fs, fe in fp_spans:
            if (fs, fe) in seen_fp:
                continue
            seen_fp.add((fs, fe))
            signals.append(
                GeoSignal(group="false_positive", term=text[fs:fe], location=location)
            )
        return signals

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
