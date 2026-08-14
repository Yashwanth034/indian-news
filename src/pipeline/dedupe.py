"""Cross-source deduplication + wire/echo detection.

Groups articles that report the *same underlying event* and recognises when
outlets are echoing a single wire (PTI/ANI/Reuters/AFP/IANS) or an official
press release, so echoes are not counted as independent confirmations.

Deduplication is deliberately conservative: false merging is treated as worse
than an occasional duplicate. Two articles merge only when the evidence is
strong:

  * exact canonical URL  -> same article (updates/corrections)
  * identical normalised headline within the dedup window -> same story
  * near-duplicate text (title/content Jaccard over stopword-free tokens)
    -> wire / syndication echo
  * multi-signal same-event match -> entities + event terms + corroborating
    numbers + actions + category + geography, guarded by hard vetoes
    (time window, number conflict, location conflict, action conflict,
    category conflict)

Candidate comparison is not O(n^2): articles are processed in time order and
each new article only compares against a small set of candidate *event
representatives* found through a token inverted index, bounded by
``dedupe.max_candidates``.

Wire / echo metadata is exposed per event: source groups, number of genuinely
independent source groups, wire group, primary/official source, and confidence
-- enough for a later scoring stage, but no scoring is done here.
"""
import hashlib
import re
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.config_loader import get_config
from src.ingest.normalize import canonicalize_url
from src.models.article import Article
from src.pipeline.normalize import clean_title

REL_NEW = "new"
REL_EXACT_URL = "exact_url"
REL_SAME_HEADLINE = "same_headline"
REL_NEAR_DUPLICATE = "near_duplicate"
REL_SAME_EVENT = "same_event"

FIRST_HAND = {REL_NEW, REL_SAME_EVENT}

KIND_WIRE = "wire"
KIND_OFFICIAL = "official"
KIND_INDEPENDENT = "independent"

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_NUM_CLEAN = re.compile(r"[,\u066B]")


@dataclass
class ArticleDecision:
    index: int
    event_id: str
    relationship: str
    same_source: bool
    similarity: float
    is_representative: bool


@dataclass
class EventGroup:
    event_id: str
    member_indices: list
    representative_index: int
    category: Optional[str]
    states: list
    entities: list
    event_time: Optional[datetime]
    primary_source: str
    first_source: str
    wire_group: str
    wire_kind: str
    source_groups: dict
    independent_source_groups: int
    is_wire_echo: bool
    confidence: str
    reasons: list = field(default_factory=list)


@dataclass
class DedupResult:
    events: list
    decisions: dict
    warnings: list = field(default_factory=list)

    def event_of(self, index):
        return self.decisions[index].event_id

    def relationship(self, index):
        return self.decisions[index].relationship


@dataclass
class _Facts:
    url_key: str
    title_key: str
    title_tokens: frozenset
    content_tokens: frozenset
    entities: frozenset
    key_numbers: frozenset
    event_terms: frozenset
    actions: frozenset
    states: frozenset
    category: Optional[str]
    wire_group: str
    wire_kind: str
    source_id: str
    tier: int
    role: str
    published: Optional[datetime]


@dataclass
class _Event:
    event_id: str
    member_indices: list
    rep_facts: object
    last_seen: Optional[datetime]


class Deduplicator:
    """Group articles into events; detect wire/echo relationships."""

    def __init__(self, config=None):
        config = config or get_config()
        cfg = config["config"]
        ent = config["india_entities"]
        geo = config["india_geo"]
        sources = config["sources"]["sources"]

        d = cfg.get("dedupe", {})
        self.window_hours = float(d.get("window_hours", 48))
        self.major_window_hours = float(d.get("major_window_hours", 168))
        self.echo_title = float(
            d.get("echo_title_threshold", cfg.get("echo_similarity_threshold", 0.85))
        )
        self.echo_content = float(d.get("echo_content_threshold", 0.8))
        self.event_content = float(d.get("event_content_threshold", 0.5))
        self.strong_content = float(d.get("strong_content_threshold", 0.6))
        self.max_candidates = int(d.get("max_candidates", 40))
        self.min_token_len = int(d.get("min_token_len", 3))
        self.stopwords = set(w for w in d.get("stopwords", []) if w)

        self.wire_attributions = d.get("wire_attributions", {})
        self.wire_source_groups = d.get("wire_source_groups", {})
        self.event_actions = d.get("event_actions", {})
        self.extra_aliases = {k.lower(): v.lower() for k, v in d.get("entity_aliases", {}).items()}
        self.international_entities = {t.lower() for t in d.get("international_entities", [])}
        self.number_units = d.get("number_units", [])

        # ---- entity + geo term sets ------------------------------------
        aliases = {str(k).lower(): str(v).lower() for k, v in ent.get("entity_aliases", {}).items()}
        self.aliases = dict(aliases)
        self.aliases.update(self.extra_aliases)

        state_terms = {t.lower() for t in ent.get("states", [])}
        ut_terms = {t.lower() for t in ent.get("union_territories", [])}
        self.state_or_ut_terms = state_terms | ut_terms
        self.city_to_state = {
            str(k).lower(): str(v).lower()
            for k, v in geo.get("city_to_state", {}).items()
        }

        entity_groups = [
            ent.get("people_roles", []),
            ent.get("companies", []),
            ent.get("agencies", []),
            ent.get("ministries", []),
            ent.get("states", []),
            ent.get("union_territories", []),
            ent.get("major_cities", []),
            ent.get("international_context_terms", []),
            self._curated_institutions(),
            self.international_entities,
        ]
        terms = set()
        for group in entity_groups:
            for t in group or []:
                t = (t or "").strip().lower()
                if t:
                    terms.add(t)
        terms |= set(self.aliases.keys())
        self._entity_regex = self._compile(terms)

        self.significance_terms = {t.lower() for t in geo.get("significance_events", [])}
        self._significance_regex = self._compile(self.significance_terms)

        action_terms = {t.lower() for t in self.event_actions}
        self._action_regex = self._compile(action_terms)

        # false positives (entity + geo suppression)
        fp = set(ent.get("false_positive_phrases", []))
        fp |= set(geo.get("false_positive_patterns", []))
        self._fp_regexes = [re.compile(p, re.IGNORECASE) for p in fp if p]

        # wire attribution keyword -> wire name
        self._wire_keywords = {}
        for wire, keywords in self.wire_attributions.items():
            for kw in keywords:
                self._wire_keywords[kw.lower()] = wire
        self._wire_regex = self._compile(set(self._wire_keywords))

        # source -> tier/role/group (from the registry); falls back to article fields
        self._source_tier = {s["id"]: s.get("tier") for s in sources}
        self._source_role = {s["id"]: self._derive_role(s) for s in sources}
        # publisher group (optional): same-publisher feeds share one independent
        # group so multi-feed publishers cannot inflate corroboration.
        self._source_group = {s["id"]: (s.get("group") or s["id"]) for s in sources}

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def dedupe(self, articles, *, categories=None, states=None) -> DedupResult:
        """Group ``articles`` into events.

        ``categories`` and ``states`` are optional parallel lists aligned with
        ``articles`` (category string / list of state strings per article);
        missing values are treated as unknown, which is conservative.
        """
        n = len(articles)
        warnings = []
        facts = []
        for i in range(n):
            cat = None
            if categories and i < len(categories):
                cat = categories[i]
            st = None
            if states and i < len(states):
                st = states[i]
            facts.append(self._facts(articles[i], cat, st))

        order = sorted(
            range(n),
            key=lambda i: (facts[i].published is None, facts[i].published or _EPOCH, facts[i].source_id),
        )

        exact_url = {}
        title_key = {}
        token_index = defaultdict(set)  # token -> event id (representative token set)
        events = []
        rel_of = {}
        sim_of = {}

        for i in order:
            f = facts[i]
            ev = None
            rel = None
            sim = 1.0

            # 1. exact canonical URL
            ev = exact_url.get(f.url_key)
            if ev is not None:
                rel = REL_EXACT_URL
            else:
                # 2. identical normalised headline (within window)
                e0 = title_key.get(f.title_key)
                if e0 is not None and self._within_window(f, events[e0].rep_facts):
                    ev, rel = e0, REL_SAME_HEADLINE

            if rel is None:
                # 3. candidate blocking via token inverted index
                candidates = set()
                for tok in f.content_tokens:
                    candidates |= token_index.get(tok, set())
                ev, rel, sim = self._pick_candidate(f, candidates, events, facts)

            if rel is None:
                event_id = self._event_id(f)
                events.append(_Event(event_id=event_id, member_indices=[i], rep_facts=f, last_seen=f.published))
                ev = len(events) - 1
                rel = REL_NEW
            else:
                events[ev].member_indices.append(i)
                if f.published and (events[ev].last_seen is None or f.published > events[ev].last_seen):
                    events[ev].last_seen = f.published
                if self._better_rep(f, events[ev].rep_facts):
                    events[ev].rep_facts = f

            rel_of[i] = rel
            sim_of[i] = sim
            exact_url[f.url_key] = ev
            title_key[f.title_key] = ev
            for tok in f.content_tokens:
                token_index[tok].add(ev)

        return self._finalize(events, facts, rel_of, sim_of, warnings)

    # ------------------------------------------------------------------
    # similarity + event decision
    # ------------------------------------------------------------------

    def _pick_candidate(self, f, candidate_ids, events, facts):
        best = None
        ranked = [
            (len(f.content_tokens & events[ev].rep_facts.content_tokens), ev)
            for ev in candidate_ids
        ]
        ranked.sort(key=lambda x: -x[0])
        for _, ev in ranked[: self.max_candidates]:
            rep = events[ev].rep_facts
            if not self._within_window(f, rep):
                continue
            title_sim = self._jaccard(f.title_tokens, rep.title_tokens)
            content_sim = self._jaccard(f.content_tokens, rep.content_tokens)
            rel = self._relation(f, rep, title_sim, content_sim)
            if rel is None:
                continue
            score = max(title_sim, content_sim) if rel == REL_NEAR_DUPLICATE else content_sim
            if best is None or score > best[0]:
                best = (score, ev, rel, max(title_sim, content_sim))
        if best is None:
            return None, None, 0.0
        return best[1], best[2], best[3]

    def _relation(self, a, b, title_sim, content_sim):
        if title_sim >= self.echo_title or content_sim >= self.echo_content:
            return REL_NEAR_DUPLICATE
        if self._same_event(a, b, title_sim, content_sim):
            return REL_SAME_EVENT
        return None

    def _same_event(self, a, b, title_sim, content_sim):
        if self._number_conflict(a, b):
            return False
        if self._location_conflict(a, b):
            return False
        if self._action_conflict(a, b):
            return False
        if self._category_conflict(a, b):
            return False

        se = a.entities & b.entities
        st = a.event_terms & b.event_terms
        sa = a.actions & b.actions
        corr = self._corroborated(a, b)
        distinctive = self._distinctive_shared(a, b)

        if not se:
            if st and corr and content_sim >= self.event_content:
                return True
            return False
        if corr and distinctive >= 3:
            return True
        if st:
            if len(se) >= 2 or content_sim >= 0.4:
                return True
            return False
        if sa and content_sim >= self.event_content:
            return True
        if content_sim >= self.strong_content and distinctive >= 2:
            return True
        return False

    # ------------------------------------------------------------------
    # vetoes
    # ------------------------------------------------------------------

    def _within_window(self, a, b):
        pa, pb = a.published, b.published
        if pa is None or pb is None:
            return True
        hours = self.major_window_hours if (a.event_terms & b.event_terms) else self.window_hours
        return abs((pa - pb).total_seconds()) <= hours * 3600

    def _number_conflict(self, a, b):
        by_unit_a = defaultdict(set)
        by_unit_b = defaultdict(set)
        for v, u in a.key_numbers:
            if u:
                by_unit_a[u].add(v)
        for v, u in b.key_numbers:
            if u:
                by_unit_b[u].add(v)
        for u, vals in by_unit_a.items():
            if u in by_unit_b and not (vals & by_unit_b[u]):
                return True
        return False

    def _corroborated(self, a, b):
        if a.key_numbers & b.key_numbers:
            return True
        bare_a = {v for v, u in a.key_numbers if not u}
        bare_b = {v for v, u in b.key_numbers if not u}
        return bool(bare_a & bare_b)

    def _distinctive_shared(self, a, b):
        shared = (a.content_tokens & b.content_tokens)
        return len(shared - (a.entities | b.entities))

    def _location_conflict(self, a, b):
        sa, sb = a.states, b.states
        return bool(sa and sb and not (sa & sb))

    def _action_conflict(self, a, b):
        return bool(a.actions and b.actions and not (a.actions & b.actions))

    def _category_conflict(self, a, b):
        return bool(a.category and b.category and a.category != b.category)

    # ------------------------------------------------------------------
    # fact extraction
    # ------------------------------------------------------------------

    def _facts(self, article, category, states):
        title = clean_title(article.title) or ""
        summary = clean_title(article.summary or "") or ""
        text = f"{title} {summary}"

        url_key = canonicalize_url(article.url or article.canonical_url or "") or ""
        if not url_key and article.canonical_url:
            url_key = canonicalize_url(article.canonical_url) or ""

        title_tokens = self._tokens(title)
        content_tokens = self._tokens(text)

        entities, states_found = self._entities(text)
        if states:
            states_found = {str(s).strip().lower() for s in states if s and str(s).strip()}

        key_numbers = self._numbers(text)
        event_terms = self._match_terms(text, self._significance_regex)
        actions = self._match_actions(text)
        attribution_text = f"{text} {article.author or ''}"
        wire_group, wire_kind, _attrib = self._wire(attribution_text)

        source_id = article.source_id
        tier = article.tier if article.tier is not None else (self._source_tier.get(source_id) or 4)
        role = article.source_role or self._source_role.get(source_id) or "journalism"
        if wire_group is None:
            wire_group = self._source_group.get(source_id) or source_id
            wire_kind = KIND_OFFICIAL if role == "official-primary" else KIND_INDEPENDENT

        return _Facts(
            url_key=url_key,
            title_key=self._title_key(title),
            title_tokens=title_tokens,
            content_tokens=content_tokens,
            entities=entities,
            key_numbers=key_numbers,
            event_terms=event_terms,
            actions=actions,
            states=states_found,
            category=(category or "").lower() or None,
            wire_group=wire_group,
            wire_kind=wire_kind,
            source_id=source_id,
            tier=int(tier) if tier else 4,
            role=role,
            published=article.published,
        )

    def _entities(self, text):
        fp_spans = []
        for rx in self._fp_regexes:
            fp_spans.extend(m.span() for m in rx.finditer(text))

        def blocked(start, end):
            return any(start < fe and end > fs for fs, fe in fp_spans)

        entities = set()
        states = set()
        if self._entity_regex is not None:
            for m in self._entity_regex.finditer(text):
                if blocked(m.start(), m.end()):
                    continue
                raw = m.group(0).strip().lower()
                canon = self.aliases.get(raw, raw)
                entities.add(canon)
                if canon in self.state_or_ut_terms:
                    states.add(canon)
                elif canon in self.city_to_state:
                    states.add(self.city_to_state[canon])
        return frozenset(entities), frozenset(states)

    def _numbers(self, text):
        units = list(self.number_units)
        unit_pat = r"(?:%s)" % "|".join(re.escape(u) for u in units)
        out = set()
        # number + unit
        for m in re.finditer(
            r"(\d[\d]*(?:\.\d+)?)\s*(" + unit_pat + r")",
            text.lower(),
        ):
            value = self._norm_number(m.group(1))
            unit = self._norm_unit(m.group(2))
            out.add((value, unit))
        # bare numbers
        for m in re.finditer(r"\b(\d[\d]*(?:\.\d+)?)\b", text):
            value = self._norm_number(m.group(1))
            if value >= 100:
                out.add((value, ""))
        return frozenset(out)

    def _match_actions(self, text):
        if self._action_regex is None:
            return frozenset()
        actions = set()
        for m in self._action_regex.finditer(text):
            actions.add(self.event_actions.get(m.group(0).strip().lower()))
        return frozenset(a for a in actions if a)

    def _match_terms(self, text, regex):
        if regex is None:
            return frozenset()
        return frozenset(m.group(0).strip().lower() for m in regex.finditer(text))

    def _wire(self, text):
        attribution = None
        if self._wire_regex is not None:
            for m in self._wire_regex.finditer(text):
                attribution = self._wire_keywords.get(m.group(0).strip().lower())
                if attribution:
                    break
        if attribution:
            return attribution, KIND_WIRE, attribution
        return None, None, None

    # ------------------------------------------------------------------
    # tokenisation / normalisation
    # ------------------------------------------------------------------

    _TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'.-]*")

    def _tokens(self, text):
        out = set()
        for w in self._TOKEN_RE.findall((text or "").lower()):
            w = w.strip("'.-")
            if not w:
                continue
            if w.isdigit():
                w = str(self._norm_number(w))
            if len(w) < self.min_token_len and not (w.isdigit() or _is_float(w)):
                continue
            if w in self.stopwords:
                continue
            out.add(w)
        return frozenset(out)

    @staticmethod
    def _title_key(title):
        t = (title or "").lower()
        t = re.sub(r"[^a-z0-9]+", "", t)
        return t

    @staticmethod
    def _norm_number(raw):
        cleaned = _NUM_CLEAN.sub("", str(raw).strip())
        try:
            f = float(cleaned)
        except ValueError:
            return raw
        return int(f) if f == int(f) else f

    @staticmethod
    def _norm_unit(unit):
        u = unit.strip().lower()
        return {
            "%": "percent", "per cent": "percent",
            "bps": "basis points",
            "bn": "billion", "mn": "million", "cr": "crore", "crs": "crore",
            "km/h": "kmph", "tons": "tonnes",
        }.get(u, u)

    # ------------------------------------------------------------------
    # misc helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _jaccard(a, b):
        if not a or not b:
            return 0.0
        inter = len(a & b)
        return inter / len(a | b)

    def _better_rep(self, f, cur):
        if (cur.published or _EPOCH) == (f.published or _EPOCH):
            return f.tier < cur.tier
        return (f.published or _EPOCH) < (cur.published or _EPOCH)

    @staticmethod
    def _event_id(f):
        seed = f"{f.title_key}|{','.join(sorted(f.entities))}"
        return hashlib.sha256(seed.encode()).hexdigest()[:24]

    def _curated_institutions(self):
        return [
            "parliament", "lok sabha", "rajya sabha", "supreme court", "supreme court of india",
            "chief justice", "chief justice of india", "election commission", "election commission of india",
            "reserve bank", "reserve bank of india", "rbi", "sebi", "trai", "niti aayog", "npci",
            "centre", "central government", "union government", "government of india", "cabinet",
            "union cabinet", "cbi", "enforcement directorate", "nia", "isro", "drdo", "imd",
            "ndma", "ndrf", "icmr", "pib", "nta", "upsc", "cbse", "ugc", "cag", "cvc", "ncert",
            "indian railways", "railways",
        ]

    @staticmethod
    def _derive_role(source):
        if source.get("discovery"):
            return "discovery"
        if source.get("primary") and source.get("type") == "official":
            return "official-primary"
        stype = source.get("type")
        return stype if stype in {"official", "journalism", "specialist", "international", "discovery"} else "journalism"

    @staticmethod
    def _compile(terms):
        if not terms:
            return None
        ordered = sorted(set(terms), key=len, reverse=True)
        pattern = r"(?<!\w)(?:%s)(?!\w)" % "|".join(re.escape(t) for t in ordered)
        return re.compile(pattern, re.IGNORECASE)

    # ------------------------------------------------------------------
    # result building
    # ------------------------------------------------------------------

    def _finalize(self, events, facts, rel_of, sim_of, warnings):
        groups = []
        decisions = {}
        for ev in events:
            members = ev.member_indices
            epoch_key = lambda i: (facts[i].published or _EPOCH)
            rep = min(members, key=lambda i: (facts[i].tier, epoch_key(i), facts[i].source_id))
            first = min(members, key=lambda i: (epoch_key(i), i))

            rep_f = facts[rep]
            source_groups = OrderedDict()
            for i in members:
                f = facts[i]
                entry = source_groups.setdefault(
                    f.wire_group, {"kind": f.wire_kind, "source_ids": set(), "count": 0}
                )
                entry["source_ids"].add(f.source_id)
                entry["count"] += 1

            independent = {
                facts[i].wire_group for i in members if rel_of[i] in FIRST_HAND
            }
            independent_count = len(independent)
            unique_group_kind = None
            if independent:
                sample = next(iter(independent))
                unique_group_kind = source_groups.get(sample, {}).get("kind")

            is_wire_echo = (
                len(members) >= 2
                and independent_count == 1
                and unique_group_kind in (KIND_WIRE, KIND_OFFICIAL)
            )

            if is_wire_echo:
                confidence = "low"
            elif independent_count >= 2:
                confidence = "high"
            elif independent_count == 1:
                confidence = "medium"
            else:
                confidence = "low"

            event_time = min((facts[i].published for i in members if facts[i].published), default=None)

            reasons = [
                f"{len(members)} article(s), {independent_count} independent source group(s)",
                f"wire group: {rep_f.wire_group} ({rep_f.wire_kind})",
            ]
            if is_wire_echo:
                reasons.append(
                    f"wire/echo: single {unique_group_kind} group repeated across {len(members)} article(s)"
                )
            reasons.append(f"representative: {rep_f.source_id} (tier {rep_f.tier})")

            groups.append(
                EventGroup(
                    event_id=ev.event_id,
                    member_indices=members,
                    representative_index=rep,
                    category=rep_f.category,
                    states=sorted({s for i in members for s in facts[i].states}),
                    entities=sorted({e for i in members for e in facts[i].entities}),
                    event_time=event_time,
                    primary_source=rep_f.source_id,
                    first_source=facts[first].source_id,
                    wire_group=rep_f.wire_group,
                    wire_kind=rep_f.wire_kind,
                    source_groups={
                        k: {"kind": v["kind"], "source_ids": sorted(v["source_ids"]), "count": v["count"]}
                        for k, v in source_groups.items()
                    },
                    independent_source_groups=independent_count,
                    is_wire_echo=is_wire_echo,
                    confidence=confidence,
                    reasons=reasons,
                )
            )

            for i in members:
                decisions[i] = ArticleDecision(
                    index=i,
                    event_id=ev.event_id,
                    relationship=rel_of[i],
                    same_source=facts[i].source_id == rep_f.source_id,
                    similarity=round(sim_of.get(i, 1.0), 3),
                    is_representative=(i == rep),
                )

        return DedupResult(events=groups, decisions=decisions, warnings=warnings)


def _is_float(w):
    try:
        float(w)
        return True
    except ValueError:
        return False
