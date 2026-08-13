"""Source-grounded Telegram summarizer.

Deterministic, conservative summarization for every important
story.  A 2-5 sentence summary is composed from facts extracted
from the available source material:

    1. clean    - the existing editorial cleaning/filtering
                  pipeline (boilerplate, filler, headline
                  paraphrase, dedup, conflict) runs first;
    2. facts    - what happened, who/what is involved, where,
                  when, important numbers, consequences are
                  extracted from each surviving sentence;
3. compose  - fact-important sentences are selected
                   (2-5), article text is the primary source
                  when article extraction is available, and
                  only conservative fact-preserving rewrites
                  (trailing attribution fronting) are applied;
    4. verify   - every fact token and every sentence tail
                  must exist verbatim in the source text;
    5. quality  - completeness, artifacts, truncation,
                  headline repetition and duplication checks.

Nothing is ever invented: the summary contains only facts that
exist in the source, phrased in the source's own words.  A
story with fewer than two genuinely useful source-supported
sentences is rejected.
"""
import re

from src.telegram_briefing import (
    ENTITY_ALIASES,
    KNOWN_ENTITIES,
    LOCATION_SET,
    STOPWORDS,
    WORD_RE,
    _actions,
    _content_tokens,
    _entity_words,
    _normalized,
    _numbers_of,
    _stem_lite,
    _temporal_of,
    is_filler,
    is_headline_paraphrase,
    is_near_duplicate,
    strip_boilerplate,
)

_KEEP_CAPITAL = (
    KNOWN_ENTITIES
    | LOCATION_SET
    | set(ENTITY_ALIASES)
    | {
        "i", "january", "february", "march", "april", "june",
        "july", "august", "september", "october", "november",
        "december", "monday", "tuesday", "wednesday",
        "thursday", "friday", "saturday", "sunday",
    }
)

# ---------------------------------------------------------
# Config defaults
# ---------------------------------------------------------

MIN_SENTENCES = 2
MAX_SENTENCES = 5
TIER1_MAX = 5
ARTICLE_ITEM_SUFFIX = ":article"

# ---------------------------------------------------------
# Fact extraction
# ---------------------------------------------------------

REPORTING_VERBS = {
    "said", "says", "told", "report", "reported", "reports",
    "warn", "warned", "warns", "added", "adds", "noted",
    "notes", "confirmed", "confirms", "stated", "states",
    "announced", "announces", "declared", "declares",
    "stressed", "claimed", "claims", "suggested", "suggests",
    "revealed", "reveals", "explained", "explains",
}

SPEAKER_MAX_WORDS = 4

# Trailing attribution: "Three people were injured, police
# said." or "... , according to officials."
ATTRIBUTION_TRAIL_RE = re.compile(
    r",\s*(?:"
    r"according to\s+"
    r"([A-Za-z][\w\u2019'-]*(?:\s+[A-Za-z][\w\u2019'-]*){0,3})"
    r"|"
    r"([A-Za-z][\w\u2019'-]*(?:\s+[A-Za-z][\w\u2019'-]*){0,3})"
    r"\s+("
    + "|".join(REPORTING_VERBS)
    + r")"
    r")\s*[.!?]?\s*$",
    re.IGNORECASE,
)

# Impact units: consequences stated by the source.
IMPACT_UNITS = {
    "killed", "injured", "dead", "deaths", "fatalities",
    "displaced", "evacuated", "evacuees", "missing",
    "homes", "residents", "families", "troops", "soldiers",
    "hostages", "officers", "people",
}

IMPACT_RE = re.compile(
    r"(?:(?:more than|at least|nearly|about|around|over|"
    r"up to|almost|roughly)\s+)?"
    r"\d[\d,]*(?:\.\d+)?\s*"
    r"\b(?:"
    + "|".join(
        sorted(IMPACT_UNITS, key=len, reverse=True)
    )
    + r")\b",
    re.IGNORECASE,
)

QUANTIFIER_STRIP_RE = re.compile(
    r"^(?:more than|at least|nearly|about|around|over|"
    r"up to|almost|roughly)\s+",
    re.IGNORECASE,
)

LOCATION_RE = re.compile(
    r"\b(?:in|across|near|around|at|from|toward|towards)\s+"
    r"([A-Z][A-Za-z\u2019'-]*(?:\s+(?:"
    r"[A-Z][A-Za-z\u2019'-]*|(?:city|town|state|province|"
    r"county|island|prefecture|region|capital|coast|valley|"
    r"bay|river|village|district|area|park|north|south|east|"
    r"west|central|northern|southern|eastern|western|"
    r"national|metropolitan)\b"
    r")){0,2})"
)


def extract_facts(text):
    """Structured facts in a sentence: numbers, named entities,
    locations, temporal references, attribution and impact.

    Returns a dict of fact sets; never invents anything, every
    fact is literal text from `text`."""
    text = str(text or "")
    numbers = {
        n for n in _numbers_of(text)
    }
    entities = {
        e for e in _entity_words(text)
    }
    locations = set()
    for match in LOCATION_RE.finditer(text):
        location = match.group(1).strip()
        if len(location.split()) <= 4:
            locations.add(location.lower())
    temporal = set(_temporal_of(text))
    attribution = None
    if ATTRIBUTION_TRAIL_RE.search(text):
        attribution = "trailing"
    elif re.match(
        r"^[A-Za-z][\w\u2019'-]*(?:\s+[A-Za-z][\w\u2019'-]*){0,3}"
        r"\s+(?:(?:have|has|had|are|is|were|was|am)\s+)?"
        r"(?:"
        + "|".join(REPORTING_VERBS)
        + r")\b",
        text,
        re.IGNORECASE,
    ):
        attribution = "leading"
    impact = [
        match.group(0)
        for match in IMPACT_RE.finditer(text)
    ]
    return {
        "numbers": numbers,
        "entities": entities,
        "locations": locations,
        "temporal": temporal,
        "attribution": attribution,
        "impact": impact,
    }


def sentence_fact_score(facts):
    """Importance score of a sentence's facts.

    Consequences (impact numbers) weigh most, then plain
    numbers, locations, named entities, temporal references
    and attribution.  Scores are capped per category so long
    sentences cannot game the ranking."""
    score = 0.0
    score += 3.0 * min(2, len(facts["impact"]))
    score += 1.5 * min(2, len(facts["numbers"]))
    score += 1.0 * min(2, len(facts["locations"]))
    score += 0.75 * min(2, len(facts["entities"]))
    score += 0.5 if facts["temporal"] else 0
    score += 0.5 if facts["attribution"] else 0
    return score


def _has_fact(facts):
    return bool(
        facts["impact"]
        or facts["numbers"]
        or facts["locations"]
        or facts["entities"]
        or facts["temporal"]
        or facts["attribution"]
    )


# ---------------------------------------------------------
# Quote-boundary sentence re-splitting
#
# The formatter's split_sentences cannot see a boundary trapped
# inside a closing quote: "...food." The IMF said ...  -> the
# period sits inside the quote, so ". followed by a space is not
# a split point. The post-quote sentence (the IMF attribution +
# its statistic) fuses with the quoted fragment and, when it is
# the only sentence of the story, the whole story is rejected for
# having fewer than two sentences: the IMF statistic is lost.
#
# This splitter runs in the summarizer's own sentence-splitting
# stage: it peels a closing-quote boundary (".. punctuation,
# closing quote, whitespace, Capitalized word") into its own row
# so the post-quote sentence is selected, composed, verified and
# quality-checked as an independent fact. Continuous quoted
# sentences - where no closing quote precedes the boundary - are
# never touched, so quoted dialogue is preserved verbatim.
# ---------------------------------------------------------

# Terminal punctuation, a closing quote char, then whitespace.
# The lookbehind anchors on the two characters immediately before
# the whitespace; only the period-inside-quote gap matches.
_CLOSED_QUOTE_BOUNDARY_RE = re.compile(
    r"(?<=[.!?][\'\"\u2018\u2019\u201c\u201d])\s+"
)


def _split_quote_boundaries(rows, headline=None):
    """Split rows that fuse a quoted sentence with the sentence
    that follows its closing quote, e.g.

        "...food." The International Monetary Fund said ...

    Each fused row becomes one row per real sentence, the
    provenance fields (source, item_id and any others) copied to
    every derived row. Rows without a trapped closing-quote
    boundary are passed through unchanged.

    A split fragment that is a pure headline paraphrase (a lone
    quoted statement that restates the headline and adds no fact,
    e.g. the "We should reduce food." part of an IMF quote) is
    dropped: the headline is not repeated in the summary. The
    fact-bearing sentence after the quote is always kept."""
    if not rows:
        return rows
    out = []
    for row in rows:
        text = row.get("text") or ""
        if not _CLOSED_QUOTE_BOUNDARY_RE.search(text):
            out.append(row)
            continue
        parts = [
            p for p in _CLOSED_QUOTE_BOUNDARY_RE.split(text)
            if p.strip()
        ]
        if len(parts) <= 1:
            out.append(row)
            continue
        for part in parts:
            if (
                headline
                and is_headline_paraphrase(part, headline)
            ):
                continue
            new_row = dict(row)
            new_row["text"] = part
            out.append(new_row)
    return out


# ---------------------------------------------------------
# Live-blog relevance filtering
#
# Multi-topic live-blog / feed summaries often append unrelated
# stories ("Meanwhile, the IMF said ... 15 percent"; "Separately,
# a solar eclipse ...") after the lead. Such items carry no
# fact connected to the headline and must not enter the summary.
# Legitimate same-event context (a quote, a consequence, a
# shared place) is never dropped: a row is removed only when it
# has a STRONG off-topic signal AND no topical link to the
# headline at all.
# ---------------------------------------------------------

# Discourse markers that introduce a new sub-topic / separate
# story in a live-blog feed.
_TOPICSHIFT_PREFIX_RE = re.compile(
    r"^\s*(?:\"|\u201c|\u2018[^A-Za-z])*"  # optional leading quote
    r"(?:meanwhile|separately|however|in contrast|on the other "
    r"hand|elsewhere|additionally|also|furthermore|moreover|in "
    r"other news|in other developments|besides|as for|turning to|"
    r"on another note|by the way|in the meantime|on a different "
    r"note|shifting focus|in unrelated news|away from)|[^\w]",
    re.IGNORECASE,
)


def _capitalized_runs(text):
    """Multi-word proper-name runs (two-plus consecutive
    capitalized tokens) in the text, lowercased."""
    words = WORD_RE.findall(str(text or ""))
    runs = []
    i = 0
    while i < len(words):
        if words[i][:1].isupper():
            run = [words[i]]
            j = i + 1
            while j < len(words) and words[j][:1].isupper():
                run.append(words[j])
                j += 1
            if len(run) >= 2:
                runs.append(" ".join(run).lower())
            i = j
        else:
            i += 1
    return runs


def _off_topic_proper_entity(text, headline):
    """True when the text introduces a multi-word proper name
    that is not a location and is not mentioned in the headline.

    "International Monetary Fund" next to a wildfire headline is
    off-topic; "US, Canada and Mexico" (a shared location set) is
    not, and neither is "Bald Range wildfire" (links via the
    'wildfire' action)."""
    headline_lower = (headline or "").lower()
    headline_entities = set(_entity_words(headline))
    known = set(KNOWN_ENTITIES) | set(ENTITY_ALIASES)
    for run in _capitalized_runs(text):
        if run in headline_lower:
            continue
        if run in known:
            continue
        words = run.split()
        if any(w in LOCATION_SET for w in words) or any(
            w in headline_entities for w in words
        ):
            continue
        return True
    return False


def _acronyms(text):
    """Lowercased acronym tokens of the text: standalone all-caps
    words ("EC", "BLO", "PIL") plus the initials of a capitalized
    run ("Election Commission" -> "ec"), ignoring stopwords.

    Used to link a row that spells an entity by its abbreviation
    to a headline that spells it in full ("The EC said ..." links
    to a headline about the "Election Commission")."""
    out = set()
    for run in _capitalized_runs(text):
        words = [w for w in run.split() if w not in STOPWORDS]
        if len(words) >= 2:
            out.add("".join(w[0] for w in words))
    for word in WORD_RE.findall(str(text)):
        if (
            len(word) >= 2
            and word.isupper()
            and word.isalpha()
        ):
            out.add(word.lower())
    return out


def _row_links_to_headline(text, headline):
    """Whether the row shares any fact-bearing link with the
    headline: a named entity, a place, an action word, a content
    token (stem-equal or substring), an acronym of a headline
    entity, or an impact/consequence fact."""
    if not text or not headline:
        return False
    if _acronyms(text) & _acronyms(headline):
        return True
    he = set(_entity_words(headline))
    re_ = set(_entity_words(text))
    if re_ & he:
        return True
    if (re_ & LOCATION_SET) & (he & LOCATION_SET):
        return True
    if _actions(text) & _actions(headline):
        return True
    headline_stems = {_stem_lite(t) for t in _content_tokens(headline)}
    for token in _content_tokens(text):
        stem = _stem_lite(token)
        if stem in headline_stems:
            return True
        if len(stem) >= 4:
            for hs in headline_stems:
                if len(hs) >= 4 and (stem in hs or hs in stem):
                    return True
    if IMPACT_RE.search(text):
        return True
    return False


def _drop_unrelated_rows(rows, headline):
    """Filter out live-blog paragraphs that are clearly about a
    different story than the headline.

    A row is dropped only when it has a strong off-topic signal
    (a topic-shift marker or an unrelated multi-word proper name)
    AND no topical link to the headline. Same-event context that
    simply lacks a lexical overlap with the headline - a quote, a
    weather detail, a local place - is left untouched."""
    kept = []
    for row in rows:
        text = row.get("text") or ""
        if not text.strip():
            continue
        strong_signal = bool(_TOPICSHIFT_PREFIX_RE.match(text)) or (
            _off_topic_proper_entity(text, headline)
        )
        if strong_signal and not _row_links_to_headline(text, headline):
            continue
        kept.append(row)
    return kept


# ---------------------------------------------------------
# Conservative composition (fact-preserving rewrites only)
# ---------------------------------------------------------


def _lower_body_start(word):
    """Lowercase a body's first word after attribution
    fronting, unless it is a proper noun: acronyms, known
    entities/places/aliases, month/day names and possessives
    ("Max's") keep their capital."""
    if len(word) >= 2 and word.isupper():
        return word
    base = word.rstrip("\u2019'")
    if base.lower() in _KEEP_CAPITAL:
        return word
    if base.endswith("\u2019s") or base.endswith("'s"):
        return word
    return word[0].lower() + word[1:]


def front_attribution(sentence):
    """Move a trailing reporting attribution to the front.

    "Three people were injured, police said." becomes
    "Police said three people were injured." and
    "..., according to officials." becomes
    "According to officials, ...".

    Only applied when the trailing attribution is the final
    comma-separated clause of the sentence, the body is a
    complete clause of its own, and the body neither ends
    with terminal punctuation (it would be a second sentence)
    nor begins with a quotation.  Never rewrites facts: the
    speaker, the reporting verb and the body are verbatim.
    """
    if not sentence:
        return sentence
    match = ATTRIBUTION_TRAIL_RE.search(sentence)
    if not match:
        return sentence
    according_to = match.group(1) is not None
    speaker = (
        match.group(1)
        if according_to
        else match.group(2)
    )
    verb = match.group(3) or ""
    body = sentence[: match.start()].rstrip()
    body = re.sub(r"\s*[,.;:]+$", "", body)
    if len(body) < 6:
        return sentence
    if body[-1:] in ".!?":
        return sentence
    if body[:1] in "\"'\"\u201c\u2018\u00ab\u00bb":
        return sentence
    if speaker.split() and any(
        char.isdigit() for char in speaker
    ):
        return sentence
    if len(speaker.split()) > SPEAKER_MAX_WORDS:
        return sentence
    body_words = body.split()
    body_words[0] = _lower_body_start(body_words[0])
    body = " ".join(body_words)
    if according_to:
        front = "According to " + speaker + ","
        composed = front + " " + body
    else:
        front = (
            speaker[0].upper() + speaker[1:]
            if speaker[0].islower()
            else speaker
        ) + " " + verb
        composed = front + " " + body
    if not composed.endswith((".", "!", "?")):
        composed += "."
    return composed


# ---------------------------------------------------------
# Verification against the source
# ---------------------------------------------------------


# Fronted attribution, the composed form: "Police said ..."
# or "According to officials, ...".  For the tail check the
# body is what must mirror the source.
FRONT_ATTRIBUTION_RE = re.compile(
    r"^(?:"
    r"according to\s+"
    r"([A-Za-z][\w\u2019'-]*(?:\s+[A-Za-z][\w\u2019'-]*){0,3})"
    r",\s*"
    r"|"
    r"([A-Za-z][\w\u2019'-]*(?:\s+[A-Za-z][\w\u2019'-]*){0,3})"
    r"\s+("
    + "|".join(REPORTING_VERBS)
    + r")\s+"
    r")",
    re.IGNORECASE,
)


def _tail_substring(sentence, source_text):
    """The sentence tail (up to 40 chars) must exist verbatim
    in the source: catches truncation artifacts and rewrites
    that no longer mirror the source.  A fronted attribution
    is stripped first so short bodies verify: the attribution
    is itself verbatim from the source, only moved."""
    sentence = re.sub(r"\s+", " ", str(sentence or "")).strip()
    source = re.sub(r"\s+", " ", str(source_text or "")).strip()
    if not sentence:
        return True
    match = FRONT_ATTRIBUTION_RE.match(sentence)
    if match:
        sentence = sentence[match.end():]
    tail = sentence[-40:]
    return tail.lower() in source.lower()


def verify_row(row, source_text):
    """Check every fact token of a composed sentence against
    the source text.  Returns (verified, problems)."""
    text = row.get("text") or ""
    problems = []
    source_text = str(source_text or "")

    facts = extract_facts(text)

    missing_numbers = facts["numbers"] - _numbers_of(source_text)
    if missing_numbers:
        problems.append(
            "unsupported number(s): "
            + ", ".join(sorted(missing_numbers))
        )

    missing_entities = facts["entities"] - _entity_words(
        source_text
    )
    if missing_entities:
        problems.append(
            "unsupported name(s): "
            + ", ".join(sorted(missing_entities))
        )

    missing_temporal = facts["temporal"] - _temporal_of(
        source_text
    )
    if missing_temporal:
        problems.append(
            "unsupported time reference(s): "
            + ", ".join(sorted(missing_temporal))
        )

    for location in facts["locations"]:
        if location not in source_text.lower():
            problems.append(
                "unsupported location: " + location
            )

    if not _tail_substring(text, source_text):
        problems.append("sentence tail not in source")

    return (not problems, problems)


# ---------------------------------------------------------
# Final quality gate
# ---------------------------------------------------------

HTML_ENTITY_RE = re.compile(r"&[a-zA-Z#][a-zA-Z0-9#]*;")

# Function words that a well-formed sentence never ends on.
DANGLING_LAST_WORDS = {
    "and", "or", "but", "because", "while", "during", "after",
    "before", "with", "without", "from", "to", "of", "in",
    "on", "at", "for", "as", "than", "that", "which", "who",
    "where", "when", "an", "a", "the", "if", "then", "so",
}


def _balanced_quotes(text):
    """Quotes balance as matched open/close pairs.

    Handles straight quotes ('"') and curly pairs
    (\u201c/\u201d).  A single \u201cWe ...\u201d pair is
    balanced even though each character occurs once; a lone
    opening or closing quote of either family is not.

    Exactly one unpaired quote is tolerated as a quote-fragment
    edge: sentence splitting of a multi-sentence quoted span
    ("...30 minutes. There was no impact...") leaves one
    sentence carrying only the opening quote and the next only
    the closing quote.  Both are legitimate fragments of one
    quotation, so a single dangling quote passes while multiple
    dangling quotes still fail."""
    straight = 0
    open_stack = 0
    stray_closes = 0
    for ch in text:
        if ch == '"':
            straight = 1 - straight
        elif ch == "\u201c":
            open_stack += 1
        elif ch == "\u201d":
            if open_stack:
                open_stack -= 1
            else:
                stray_closes += 1
    unpaired = straight + open_stack + stray_closes
    return unpaired <= 1


def _dangling_ending(text):
    """True when the sentence ends with a dangling function
    word ("in the region" is fine; a sentence that truly ends
    "in the" is a cut)."""
    lowered = text.lower()
    words = re.findall(
        r"[a-z0-9\u2019'-]+", lowered
    )
    if not words:
        return False
    return words[-1] in DANGLING_LAST_WORDS


def quality_check_sentence(text, headline):
    """Problems for one composed sentence.  Empty list means
    the sentence passes every check."""
    problems = []
    text = str(text or "").strip()

    if not text:
        return ["empty sentence"]

    if not re.search(r"[.!?\u2026][\"'\u201d\u2019)\]]*$", text):
        problems.append("sentence is not complete")

    if _dangling_ending(text):
        problems.append("dangling ending")

    if HTML_ENTITY_RE.search(text):
        problems.append("html entity artifact")

    if strip_boilerplate(text) != text:
        problems.append("boilerplate fragment")

    if is_filler(text):
        problems.append("filler sentence")

    if is_headline_paraphrase(text, headline):
        problems.append("headline repetition")

    if not _balanced_quotes(text):
        problems.append("unbalanced quotation marks")

    first = text[0]
    if (
        first.islower()
        and text[:1] not in "\"'(\u2018\u201c\u201d\u00ab\u00bb\u2026"
    ):
        problems.append("sentence starts lowercase")

    return problems


def quality_check_summary(headline, rows):
    """(ok, problems) for the whole composed summary.

    Problems reference the offending row index; rows are
    never mutated here."""
    problems = []
    if not rows:
        return (False, ["empty summary"])
    if len(rows) < MIN_SENTENCES:
        problems.append("fewer than 2 useful sentences")
    if len(rows) > MAX_SENTENCES:
        problems.append("more than 5 sentences")
    if not (headline or "").strip():
        problems.append("missing headline")

    seen_keys = set()
    for index, row in enumerate(rows):
        text = row.get("text") or ""
        for problem in quality_check_sentence(text, headline):
            problems.append("sentence %d: %s" % (index + 1, problem))
        key = _normalized(text)
        if key in seen_keys:
            problems.append(
                "sentence %d: duplicate of an earlier sentence"
                % (index + 1)
            )
        seen_keys.add(key)
        if any(
            is_near_duplicate(
                {"text": text},
                {"text": other.get("text")},
            )
            for other in rows[:index]
        ):
            problems.append(
                "sentence %d: near-duplicate of an earlier "
                "sentence" % (index + 1)
            )

    return (not problems, problems)


# ---------------------------------------------------------
# Summary composition
# ---------------------------------------------------------


def _is_article_row(row, article_item_ids):
    item_id = row.get("item_id") or ""
    return item_id in article_item_ids or item_id.endswith(
        ARTICLE_ITEM_SUFFIX
    )


# Consequence verbs adjacent to an impact phrase, so
# "12 people were killed" and "12 people were injured" never
# count as the same consequence.
CONSEQUENCE_VERBS = {
    "killed", "injured", "hurt", "dead", "deaths",
    "fatalities", "displaced", "evacuated", "missing",
    "forced", "affected", "destroyed", "damaged",
}

QUANTIFIER_STRIP_RE = re.compile(
    r"^(?:more than|at least|nearly|about|around|over|"
    r"up to|almost|roughly)\s+",
    re.IGNORECASE,
)


def _impact_signature(text):
    """(impact phrase, consequence verb) pairs in a sentence.

    The phrase is the matched number+unit; the verb is the
    consequence word adjacent after the phrase ("evacuated"
    in "20,000 residents were evacuated"), or None."""
    out = set()
    for match in IMPACT_RE.finditer(text or ""):
        phrase = QUANTIFIER_STRIP_RE.sub(
            "", match.group(0)
        ).strip()
        verb = None
        after = text[match.end() : match.end() + 60]
        for candidate in CONSEQUENCE_VERBS:
            if re.search(r"\b" + candidate + r"\b", after):
                verb = candidate
                break
        out.add((phrase, verb))
    return out


def _fact_signature(row):
    """The consequential facts of a row: canonical numbers and
    normalized impact (phrase + consequence verb).  Two rows
    sharing both numbers and the same consequence report the
    same fact."""
    facts = extract_facts(row.get("text"))
    return (
        frozenset(facts["numbers"]),
        frozenset(_impact_signature(row.get("text"))),
    )


def _covers_signature(selected_rows, row):
    numbers, impact = _fact_signature(row)
    if not numbers or not impact:
        return False
    for selected in selected_rows:
        sel_numbers, sel_impact = _fact_signature(selected)
        if (
            numbers & sel_numbers
            and impact & sel_impact
        ):
            return True
    return False


def _select_pool(rows, tier1_max, cap):
    """Select up to `cap` rows from one source pool.

    Exact duplicates (same normalized text) are dropped.
    Fact-bearing sentences are ranked by fact score first
    (up to tier1_max), then the remaining informative
    sentences fill the pool in source order.  Returns rows in
    selection order."""
    seen = set()
    unique = []
    for row in rows:
        key = _normalized(row.get("text"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    accepted = []
    tier1 = []
    tier2 = []
    for row in unique:
        if _covers_signature(accepted, row):
            continue
        facts = extract_facts(row.get("text"))
        if _has_fact(facts):
            tier1.append(
                (sentence_fact_score(facts), row)
            )
        else:
            tier2.append(row)
    tier1.sort(key=lambda pair: pair[0], reverse=True)
    tier1 = [row for _, row in tier1]
    for row in tier1:
        if len(accepted) >= min(tier1_max, cap):
            break
        if _covers_signature(accepted, row):
            continue
        accepted.append(row)
    for row in tier2:
        if len(accepted) >= cap:
            break
        if _covers_signature(accepted, row):
            continue
        accepted.append(row)
    return accepted


def select_fact_rows(rows, article_item_ids, cfg=None):
    """Select the 2-5 fact-important sentences of the summary.

    Article-provenanced rows are the primary source and come
    first; the RSS rows fill any remaining slots.  Within
    each pool, fact-bearing sentences (impact, numbers,
    locations, entities, time, attribution) are ranked by
    fact score first, then the remaining informative
    sentences fill the pool in source order.  An RSS row that
    reports the same consequence (same numbers AND same
    impact unit) as an article row is dropped: the fact is
    already covered by the primary source."""
    tier1_max = int(
        (cfg or {}).get("tier1_max", TIER1_MAX)
    )
    max_sentences = int(
        (cfg or {}).get("max_sentences", MAX_SENTENCES)
    )

    article_rows = [
        r for r in rows if _is_article_row(r, article_item_ids)
    ]
    other_rows = [
        r for r in rows if not _is_article_row(r, article_item_ids)
    ]

    selected = _select_pool(
        article_rows,
        tier1_max,
        max_sentences,
    )

    if len(selected) < max_sentences:
        other_rows = [
            r
            for r in other_rows
            if not _covers_signature(selected, r)
        ]
        selected.extend(
            _select_pool(
                other_rows,
                tier1_max,
                max_sentences - len(selected),
            )
        )

    return selected[:max_sentences]


def compose_summary(rows, article_item_ids, cfg=None):
    """Apply conservative rewrites to selected rows.

    Only attribution fronting is performed; every factual
    token remains verbatim from the source.  Returns rows
    with the composed "text"."""
    composed = []
    for row in rows:
        composed_row = dict(row)
        composed_row["text"] = front_attribution(row.get("text"))
        composed_row["composed"] = (
            composed_row["text"] != row.get("text")
        )
        composed.append(composed_row)
    return composed


def verify_summary(rows, source_text):
    """(kept_rows, problems): drop rows that do not verify
    against the source; every problem is recorded."""
    kept = []
    problems = []
    for index, row in enumerate(rows):
        verified, row_problems = verify_row(row, source_text)
        if verified:
            kept.append(row)
        else:
            problems.append(
                {
                    "index": index + 1,
                    "text": (row.get("text") or "")[:80],
                    "problems": row_problems,
                }
            )
    return kept, problems


def apply_quality_gate(headline, rows):
    """(kept_rows, problems): drop rows that fail the quality
    gate.  Returns the surviving rows and every problem."""
    kept = []
    problems = []
    for index, row in enumerate(rows):
        sentence_problems = quality_check_sentence(
            row.get("text"), headline
        )
        if sentence_problems:
            problems.append(
                {
                    "index": index + 1,
                    "text": (row.get("text") or "")[:80],
                    "problems": sentence_problems,
                }
            )
        else:
            kept.append(row)
    return kept, problems


def summarize_rows(
    rows,
    source_text,
    headline,
    article_item_ids=None,
    cfg=None,
):
    """Compose, verify and quality-check the 2-5 sentence
    summary for one story.

    Returns (rows, stats) where stats records every rejection
    and problem.  Returns (None, stats) when fewer than two
    genuinely useful source-supported sentences survive: the
    story must be rejected, never padded or invented."""
    cfg = cfg or {}
    min_sentences = int(
        cfg.get("min_sentences", MIN_SENTENCES)
    )
    max_sentences = int(
        cfg.get("max_sentences", MAX_SENTENCES)
    )
    article_item_ids = article_item_ids or set()

    stats = {
        "selected": 0,
        "composed": 0,
        "verify_problems": [],
        "quality_problems": [],
        "rejected": None,
    }

    if not rows:
        stats["rejected"] = "insufficient_information"
        return None, stats

    # 1. Quote-boundary sentence splitting: peel a sentence that
    #    follows a closing quote (". " The IMF said ...) into its
    #    own row so its fact survives independently. See
    #    _split_quote_boundaries for why this lives here.
    rows = _split_quote_boundaries(rows, headline)

    # 2. Live-blog relevance: drop paragraphs that are clearly a
    #    separate story (topic-shift marker or an unrelated proper
    #   name) and share no fact with the headline.
    rows = _drop_unrelated_rows(rows, headline)

    selected = select_fact_rows(rows, article_item_ids, cfg)

    if len(selected) < min_sentences:
        stats["selected"] = len(selected)
        stats["rejected"] = "insufficient_information"
        return None, stats

    stats["selected"] = len(selected)

    composed = compose_summary(selected, article_item_ids, cfg)
    stats["composed"] = sum(
        1 for row in composed if row.get("composed")
    )

    kept, verify_problems = verify_summary(
        composed, source_text
    )
    stats["verify_problems"] = verify_problems

    if len(kept) < min_sentences:
        stats["rejected"] = "verification"
        return None, stats

    kept, quality_problems = apply_quality_gate(
        headline, kept
    )
    stats["quality_problems"] = quality_problems

    if len(kept) < min_sentences:
        stats["rejected"] = "quality"
        return None, stats

    if len(kept) > max_sentences:
        kept = kept[:max_sentences]

    return kept, stats
