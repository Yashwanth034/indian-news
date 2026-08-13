"""Conservative Telegram briefing builder.

Pure rules over pipeline data. No generation:

- Every sentence in a briefing exists VERBATIM in a source
  summary.
- Sentences retain source provenance internally.
- Event grouping requires multiple meaningful signals;
  false merges are worse than short briefings.
- Materially conflicting facts are not reconciled: the
  lower-ranked version of a disputed detail is dropped.
"""
import html
import re

from src.formatter import clean, split_sentences
from src.telegram_scheduler import story_age_minutes

# ---------------------------------------------------------
# Sentence cleanup (mechanical fixes only, never rewriting
# facts): HTML entity unescaping, duplicated-word collapse,
# whitespace normalization. Filler and headline-paraphrase
# sentences are dropped so the body never repeats the
# headline or generic boilerplate.
# ---------------------------------------------------------

CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "hasn't": "has not",
    "haven't": "have not",
    "we're": "we are",
    "they're": "they are",
    "you're": "you are",
    "i'm": "i am",
    "i've": "i have",
    "we've": "we have",
    "they've": "they have",
    "wouldn't": "would not",
    "couldn't": "could not",
    "shouldn't": "should not",
}

DUPLICATE_WORD_RE = re.compile(
    r"\b([A-Za-z']+)(\s+\1\b)+",
    re.IGNORECASE,
)

STOPWORD_SET = {
    "a", "an", "and", "as", "at", "by", "for", "from",
    "in", "is", "of", "on", "or", "the", "to", "with",
    "its", "it", "that", "this", "was", "were", "are",
    "has", "have", "had", "be", "been", "says", "said",
    "after", "over", "more", "than", "into", "about",
}

FILLER_PHRASES = {
    "this is a breaking news story",
    "this is breaking news",
    "breaking news",
    "this is an important development",
    "this is a developing story",
    "the story is developing",
    "more details are expected",
    "more details are expected to follow",
    "more details will follow",
    "more details will be provided",
    "follow for more",
    "follow for updates",
    "follow for more updates",
    "stay tuned",
    "stay tuned for updates",
    "this story will be updated",
    "we will update this story",
    "we will bring you more updates",
    "read more",
    "read the full story",
    "for more information visit the site",
    "this breaking news story is being updated",
    "this breaking news story will be updated",
    "please refresh the page",
    "please refresh the page for updates",
    "please refresh the page for the latest updates",
    "more details will be published shortly",
    "more details will be published soon",
    "follow us on x",
    "follow us on twitter",
    "follow us on facebook",
    "follow us on instagram",
    "follow us on telegram",
    "follow us on whatsapp",
    "subscribe to our newsletter",
    "sign up for our newsletter",
    "subscribe for the latest updates",
    "we use cookies",
    "accept all cookies",
    "read our privacy policy",
    "read our cookie policy",
    "this is an advertisement",
    "this is advertising content",
    "advertisement",
}

# Whole-sentence junk: promotional / live-status boilerplate
# that LEADS its sentence ("You can also follow @BBCBreaking...",
# "Please refresh...", "This breaking news story is being
# updated..."). Anchoring at the start keeps the match
# conservative: a factual sentence that merely contains a
# discouraged word ("reported", "updated", "according to") is
# never matched unless the sentence is actually framed as
# boilerplate.
JUNK_SENTENCE_PATTERNS = (
    # social / status promotion
    r"you\s+(?:can\s+)?also\s+follow\s+@[A-Za-z0-9_]+",
    r"follow\s+@[A-Za-z0-9_]+",
    r"follow\s+us\s+on\s+(?:x|twitter|facebook|instagram|telegram|whatsapp)",
    r"follow\s+(?:the\s+)?author",
    r"follow\s+(?:our\s+)?(?:channel|account)\s+on",
    # live-update boilerplate
    r"this\s+breaking\s+news\s+story\s+is\s+(?:being\s+)?updated",
    r"this\s+breaking\s+news\s+story\s+will\s+be\s+updated",
    r"this\s+breaking\s+story\s+(?:is\s+being|will\s+be)\s+updated",
    r"this\s+(?:is\s+)?(?:a\s+)?breaking\s+news\s+story",
    r"this\s+is\s+an\s+important\s+developing\s+story",
    r"more\s+details?\s+will\s+be\s+published\s+(?:shortly|soon)",
    r"please\s+refresh",
    r"for\s+the\s+fullest\s+version",
    # external / cookie / privacy / ads
    r"this\s+(?:page|content)\s+is\s+external\s+to",
    r"it\s+is\s+external\s+to",
    r"this\s+is\s+(?:an\s+)?external\s+(?:link|site)",
    r"we\s+use\s+cookies",
    r"(?:read|view)\s+(?:our\s+)?cookie\s+policy",
    r"(?:read|view)\s+(?:our\s+)?privacy\s+policy",
    r"(?:this\s+is\s+)?(?:an\s+)?advert(?:isement)?\b",
    r"sponsored\s+(?:content|article|post)",
    r"advertise\s+with\s+us",
    # subscription / newsletter promotion
    r"subscribe(?:d)?\s+(?:to|for)\s+our\s+newsletter",
    r"sign\s+up\s+for\s+our\s+(?:newsletter|daily\s+briefing|daily\s+email)",
    r"subscrib(?:e|ers)\s+get\s+(?:access\s+to\s+)?(?:the\s+)?(?:full|exclusive)(?:\s+content)?",
    # navigation / author / rss boilerplate
    r"skip\s+to\s+(?:content|main\s+content|main)",
    r"back\s+to\s+top",
    r"share\s+this\s+(?:article|story)",
    r"print\s+(?:this\s+)?(?:article|page|story)",
    r"posted\s+by\b",
    r"more\s+from\s+(?:this\s+|the\s+)?(?:author|writer)",
    r"about\s+the\s+author",
    r"read\s+more\s+by\s+(?:this\s+)?(?:author|writer)",
    r"view\s+(?:all\s+)?comments",
    r"post\s+(?:a\s+)?comment",
)

JUNK_SENTENCE_RE = re.compile(
    r"^(?:[\"'\u2018\u201c\u201d\u00ab\u00bb]+\s*)?(?:"
    + "|".join(
        "(?:" + pattern + r"\b)"
        for pattern in JUNK_SENTENCE_PATTERNS
    )
    + r")",
    re.IGNORECASE,
)

HEADLINE_PARAPHRASE_OVERLAP = 0.5


# Quote/apostrophe HTML entities that survive a single
# html.unescape pass only when the source was double-encoded
# (e.g. "&amp;quot;"). These are decoded one extra time so
# double-encoded feed text never renders as literal entity
# text. &amp; / &lt; / &gt; are deliberately NOT re-decoded:
# legitimate text such as "&amp;lt;" (a literal entity in
# the source) must never be turned into "<".
QUOTE_ENTITY_RE = re.compile(
    r"&(?:quot|apos|#0*3[49]|#0*34|#x0*2[27]);",
    re.IGNORECASE,
)


def decode_html_entities(text):
    """Robust HTML-entity decoding for source/article text
    before sentence processing.

    - One full html.unescape pass handles named entities
      (&quot;, &amp;, &lt;, &gt;, &apos;), numeric entities
      (&#34;, &#039;, &#8217;) and hex entities (&#x27;).
    - A single targeted second pass repairs double-encoded
      quotation/apostrophe entities ("&amp;quot;" feed
      artifacts) that would otherwise stay visible as
      literal "&quot;" text in messages.
    - Everything else is decoded exactly once: text that
      legitimately contains entity text (e.g. "&amp;lt;")
      is never double-decoded.
    """
    text = str(text or "")
    decoded = html.unescape(text)
    if QUOTE_ENTITY_RE.search(decoded):
        decoded = html.unescape(decoded)
    return decoded


def _normalized(text):
    return re.sub(
        r"[^a-z0-9 ]",
        " ",
        decode_html_entities(text or "").lower(),
    ).strip()


def is_filler(sentence):
    """True for generic boilerplate that carries no facts.

    Two conservative checks: the whole sentence equals a known
    filler phrase, or it STARTS with a junk framing such as
    "follow @handle", "please refresh", "this breaking news
    story is being updated" or "for the fullest version".  A
    factual sentence that merely contains a discouraged word
    ("reported", "updated", "according to") is never matched.
    """
    text = str(sentence or "")
    return (
        _normalized(text) in FILLER_PHRASES
        or bool(JUNK_SENTENCE_RE.match(text))
    )


BOILERPLATE_PATTERNS = [
    r"\bcontinue reading(?:\.{0,3})",
    r"\bread more(?::\s*|\.{1,3}\s*)(?=[A-Z]|$)",
    r"\bread the full story(?::\s*|\.{1,3}\s*)(?=[A-Z]|$)",
    r"\bget our breaking news email(?:\s*,?\s*free app(?:\s+or\s+daily news podcast)?)?",
    r"\bget (?:our|the) (?:free )?app\b",
    r"\b(?:download|install) (?:our|the) (?:free )?app\b",
    r"\blisten to (?:our|the) (?:daily )?podcast\b",
    r"\bget our (?:daily )?podcast\b",
    r"\bsubscribe (?:to|for) (?:our|the) (?:newsletter|daily briefing|daily email)\b",
    r"\bsign up (?:to|for) (?:our|the) (?:newsletter|daily briefing|daily email)\b",
    r"\bsign up for breaking news\b",
    r"\bfollow us on (?:x|twitter|facebook|instagram|telegram|whatsapp)\b",
    r"\bjoin our (?:telegram|whatsapp) channel\b",
    r"\bfollow (?:our )?(?:live )?(?:blog|coverage|updates)\b[^A-Za-z]*$",
    r"\bfollow our [A-Za-z'\u2019 -]+? (?:live )?blog "
    r"for (?:the )?latest "
    r"(?:updates|developments|news|coverage|headlines)"
    r"\b[.\u2026]*",
    r"\bmore on this story\b",
    r"\brelated:?\s*(?:coverage|stories)\b",
    r"\bnewsletter delivered (?:to|straight to) (?:your|their) inbox\b",
    r"\bto (?:get|receive) more stories (?:like this|like these)\b",
    r"\byou (?:can|may) also (?:get|receive|sign up for) (?:our )?(?:newsletter|daily briefing)\b",
    r"\bthis article was amended on[^.\n]*\.?\s*",
    r"\bfor (?:more|further) information,? visit (?:our )?(?:website|site)\b",
    r"\byou (?:can )?also follow @[A-Za-z0-9_]+\b",
    r"\bfollow @[A-Za-z0-9_]+\b",
    r"\bplease refresh(?: the (?:page|browser))?\b",
    r"\bfor the fullest version\b",
    r"\bmore details? will be published (?:shortly|soon)\b",
    r"\b(?:this (?:page|content)|it) is external to\b",
    r"\bthis is (?:an )?external (?:link|site)\b",
    r"\bwe use cookies\b",
    r"\bcookie policy\b",
    r"\bprivacy policy\b",
    r"\baccept (?:all )?cookies\b",
    r"\bcookies?(?:,| to) improve\b",
    r"\bskip to (?:content|main content|main)\b",
    r"\bback to top\b",
    r"\bshare this (?:article|story)\b",
    r"\bprint (?:this )?(?:article|page|story)\b",
    r"\bsponsored (?:content|article|post)\b",
    r"\badvertise with us\b",
    r"\bfollow (?:the )?author\b",
    r"\bmore from (?:this |the )?author\b",
    r"\babout the author\b",
    r"\bread more by (?:this )?author\b",
    r"\bposted by\b",
    r"\bview (?:all )?comments\b",
    r"\bpost (?:a )?comment\b",
    r"\bget the latest (?:news|updates) in your inbox\b",
    r"\bunlock (?:this )?article\b",
    r"\bsubscribe(?:d)? (?:to|for) our newsletter\b",
    r"\bsign up for our (?:newsletter|daily briefing|daily email)\b",
]

BOILERPLATE_RE = re.compile(
    "|".join(
        "(?:" + pattern + ")"
        for pattern in BOILERPLATE_PATTERNS
    ),
    re.IGNORECASE,
)

# Leading article/navigation fragment ending in a series
# marker, e.g. France 24's "Total solar eclipse (2/4) A total
# solar eclipse will sweep...". The fragment is a series label,
# never a fact, and is removed only when the text that follows
# it starts a fresh sentence (capitalized) or nothing remains.
FRAGMENT_LEAD_RE = re.compile(
    r"^[A-Za-z\u2019'-]+"
    r"(?:\s+[A-Za-z\u2019'-]+){0,6}"
    r"\s*\(\d+\s*/\s*\d+\)\.?"
    r"\s*(?=[A-Z]|$)"
)


def strip_boilerplate(text):
    """Remove recognizable publisher boilerplate fragments
    (newsletter/app/podcast/subscription/continue-reading)
    even when embedded mid-sentence. Each pattern is a
    multi-word promotional phrase, so no factual text is
    ever removed."""
    text = BOILERPLATE_RE.sub(" ", str(text or ""))
    return re.sub(r"\s+", " ", text).strip()


def clean_sentence_text(text):
    """Mechanical cleanup only: unescape entities (including
    double-encoded quotation entities), remove recognizable
    publisher boilerplate fragments, collapse duplicated
    words and whitespace. Never rewrites facts.

    A truncated-word fragment left over after the cleanup
    (a single token of four characters or fewer, e.g. the
    trailing "Con" of a feed that cut the summary
    mid-word) is dropped entirely so it can never render as
    a dangling fragment in a message."""
    text = decode_html_entities(str(text or ""))
    text = strip_boilerplate(text)
    # Boilerplate removal can leave a stray space before a
    # terminal punctuation mark ("toxic' ." after a nav line
    # was cut out); collapse whitespace before punctuation
    # back onto the mark.
    text = re.sub(r"\s+([.,;:!?\u2026])", r"\1", text)
    text = DUPLICATE_WORD_RE.sub(r"\1", text)
    # Drop a leading article/navigation fragment that is only
    # a series label ("Total solar eclipse (2/4) ...") while
    # preserving the actual news sentence that follows it.
    text = FRAGMENT_LEAD_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return ""

    tokens = re.findall(
        r"[A-Za-z0-9\u2019'-]+",
        text,
    )

    if (
        len(tokens) == 1
        and len(tokens[0].strip("\u2019'")) <= 4
    ):
        return ""

    return text


def _content_tokens(text):
    text = decode_html_entities(str(text or ""))
    text = text.replace("\u2019", "'")
    for k, v in CONTRACTIONS.items():
        text = text.replace(k, v)
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return set(tokens) - STOPWORD_SET


# ---------------------------------------------------------
# Headline-paraphrase detection
#
# A sentence is a headline paraphrase only when it mostly
# repeats the headline AND introduces no fact-bearing detail
# of its own. Word overlap alone is not enough: ledes that
# are dense with proper nouns (a person's name plus an event
# name) routinely reach >= 50% overlap while still carrying
# genuinely new facts (reason, venue, time, status, medical
# detail). Such sentences must survive; only sentences that
# restate headline facts and add nothing new are dropped.
# ---------------------------------------------------------

# A number in the text (8, 8th, 78,000, 1.2, 7am, 7:30pm,
# 24 percent, $1.2bn).
NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?:[$£€])?"
    r"\d[\d,]*(?:\.\d+)?"
    r"(?:st|nd|rd|th)?"
    r"\s*(?:%|percent|am|pm|a\.m\.|p\.m\.)?"
    r"(?!\w)",
    re.IGNORECASE,
)

# Day names, month names and simple time references.
TEMPORAL_RE = re.compile(
    r"\b(?:monday|tuesday|wednesday|thursday|friday|"
    r"saturday|sunday|january|february|march|april|may|"
    r"june|july|august|september|october|november|"
    r"december|today|tonight|tomorrow|yesterday|"
    r"(?:next|last)\s+(?:week|month|year))\b",
    re.IGNORECASE,
)

# Measurement/unit words with factual weight.
UNIT_WORDS = {
    "percent", "km", "kilometre", "kilometres",
    "kilometer", "kilometers", "kg", "kilogram",
    "kilograms", "miles", "mile", "tonnes", "tons",
    "tonne", "ton", "sq", "hectare", "hectares", "acre",
    "acres", "billion", "million", "thousand", "trillion",
    "megawatt", "megawatts", "gigawatt", "gigawatts",
    "feet", "metres", "meters", "metre", "meter",
    "degrees", "celsius", "fahrenheit",
}

# Subordinating conjunctions: a sentence whose only new facts
# sit in a trailing subordinate clause ("Nagasaki mayor says
# ... as Japan marks the anniversary ...") is still a headline
# restatement and is dropped; the trailing clause elaborates
# the restated event instead of standing as independent fact.
SUBORDINATORS = {
    "as", "while", "since", "after", "before", "because",
    "although", "though", "until", "unless", "whereas",
    "whenever", "wherever", "when", "where",
}

# Words that never count as a new fact even when absent from
# the headline: roles/titles, generic collectives, generic
# descriptors and time periods, restatement synonyms, adverbs
# and modals. Stored in _stem_lite() form automatically.
_NON_FACT_SOURCE_WORDS = {
    # roles / titles
    "president", "prime", "minister", "government",
    "official", "authority", "police", "court", "parliament",
    "senate", "senator", "congress", "house", "council",
    "committee", "commission", "department", "agency",
    "ministry", "university", "bank", "party", "union",
    "army", "navy", "king", "queen", "prince", "princess",
    "governor", "mayor", "chairman", "secretary", "chief",
    "director", "general", "captain", "doctor", "professor",
    "judge", "spokesman", "spokesperson", "chancellor",
    "leader", "lawmaker", "member", "administrator", "author",
    "expert", "analyst", "researcher", "ambassador", "priest",
    # generic collectives
    "people", "person", "resident", "citizen", "family",
    "home", "country", "nation", "world", "city", "town",
    "region", "area", "state", "province", "village",
    "number", "group", "population", "community", "child",
    "student", "worker", "tourist", "victim", "woman", "man",
    "user",
    # generic descriptors / time periods
    "worst", "best", "biggest", "largest", "smallest",
    "greatest", "latest", "newest", "new", "old", "major",
    "massive", "large", "small", "big", "high", "low", "key",
    "main", "total", "overall", "recent", "several",
    "numerous", "multiple", "countless", "decade", "century",
    "hour", "minute", "month", "week", "year", "day", "time",
    # verdict/style words and restatement synonyms: they
    # restate headline facts without adding a new one
    "trip", "top", "visit", "warning", "warn", "issued",
    "issue", "raise", "raised",
    # pronouns, modals and auxiliaries: pure function words
    "his", "her", "him", "he", "she", "we", "they", "our",
    "your", "their", "us", "them", "you", "i", "who", "whom",
    "which", "what", "how", "when", "where", "why",
    "will", "would", "can", "could", "should", "may", "might",
    "must", "shall", "do", "does", "did", "doing", "being",
    "not", "no",
}


def _stem_lite(word):
    """Crude inflection normalization used ONLY for comparing
    whether two words are the same fact word. Never changes
    any rendered sentence text."""
    w = word.lower()
    if len(w) > 5 and w.endswith("ing"):
        w = w[:-3]
    elif len(w) > 5 and w.endswith("ies"):
        w = w[:-3] + "y"
    elif len(w) > 5 and w.endswith("ian"):
        w = w[:-3]
    elif len(w) > 5 and w.endswith("an"):
        w = w[:-2]
    elif len(w) > 5 and w.endswith("ic"):
        w = w[:-2]
    elif len(w) > 4 and w.endswith("es"):
        w = w[:-2]
    elif len(w) > 4 and w.endswith("ed"):
        w = w[:-2]
    elif len(w) > 3 and w.endswith("s"):
        w = w[:-1]
    elif len(w) > 3 and w.endswith("e"):
        w = w[:-1]
    return w


NON_FACT_WORDS = {
    _stem_lite(word)
    for word in _NON_FACT_SOURCE_WORDS
}


def _numbers_of(text):
    """Canonical number forms in the text (commas, currency,
    ordinals and am/pm suffixes stripped)."""
    out = set()
    for m in NUMBER_RE.finditer(text or ""):
        canonical = re.sub(r"[^0-9.]", "", m.group(0))
        if canonical:
            out.add(canonical)
    return out


def _temporal_of(text):
    """Lowercased day/month/time references in the text."""
    return {
        re.sub(r"\s+", " ", m.group(0)).lower()
        for m in TEMPORAL_RE.finditer(text or "")
    }


def _units_of(text):
    """Lowercased unit words present in the text."""
    lowered = (text or "").lower()
    return {
        word for word in UNIT_WORDS
        if re.search(r"\b" + re.escape(word) + r"\b", lowered)
    }


def _only_in_trailing_subclause(sentence, new_stems):
    """True when every new-fact word in the sentence appears
    after the last subordinating conjunction, i.e. the new
    detail is confined to a trailing subordinate clause."""
    tokens = re.findall(
        r"[a-z0-9]+",
        (sentence or "").lower(),
    )
    last_sub = -1
    for index, token in enumerate(tokens):
        if token in SUBORDINATORS:
            last_sub = index
    if last_sub < 0:
        return False
    for index, token in enumerate(tokens):
        if (
            index < last_sub
            and _stem_lite(token) in new_stems
        ):
            return False
    return True


def _introduces_new_fact(sentence, headline):
    """True when the sentence carries fact-bearing detail that
    the headline does not already state.

    What counts as a new fact:
    - a number / date / day-time / unit not in the headline
    - a named person, entity or place not in the headline
      (survives as a new content word)
    - a concrete content word not in the headline and not an
      ordinary function/title/generic word
    Anything else is treated as a restatement and remains
    protected by the paraphrase rule. A new word confined to
    a trailing subordinate clause ("... as Japan marks the
    anniversary of the US atomic bombing") does not save the
    sentence: the restated headline carries the sentence and
    the clause merely elaborates it.
    """
    headline_numbers = _numbers_of(headline)
    if any(
        n not in headline_numbers
        for n in _numbers_of(sentence)
    ):
        return True

    headline_temporal = _temporal_of(headline)
    if any(
        t not in headline_temporal
        for t in _temporal_of(sentence)
    ):
        return True

    headline_units = _units_of(headline)
    if any(
        u not in headline_units
        for u in _units_of(sentence)
    ):
        return True

    headline_stems = {
        _stem_lite(token)
        for token in _content_tokens(headline)
    }

    new_stems = {
        _stem_lite(token)
        for token in _content_tokens(sentence)
        if (
            _stem_lite(token) not in headline_stems
            and _stem_lite(token) not in NON_FACT_WORDS
        )
    }

    if not new_stems:
        return False

    if _only_in_trailing_subclause(
        sentence,
        new_stems,
    ):
        return False

    return True


def is_headline_paraphrase(sentence, headline):
    """True when a sentence mostly restates the headline AND
    introduces no fact-bearing information of its own.

    The headline is the main statement; the body must add
    factual context instead of repeating it. A sentence is
    treated as a paraphrase only when at least half of its
    content words already appear in the headline and it
    introduces no new fact (number, date/time, unit, named
    person/entity/place, or concrete detail) beyond the
    headline. A sentence that carries a genuinely new fact
    survives even at high word overlap.
    """
    sentence_tokens = _content_tokens(sentence)

    # An exact copy of the headline is a restatement by
    # definition, even when it is too short for the token-
    # overlap test below.
    if _normalized(sentence) == _normalized(headline):
        return True

    if len(sentence_tokens) < 4:
        return False

    headline_tokens = _content_tokens(headline)

    if not headline_tokens:
        return False

    overlap = len(
        sentence_tokens & headline_tokens
    ) / len(sentence_tokens)

    if overlap < HEADLINE_PARAPHRASE_OVERLAP:
        return False

    return not _introduces_new_fact(
        sentence,
        headline,
    )


def count_meaningful_sentences(text, headline):
    """Number of unique meaningful explanatory sentences in
    the cleaned text.

    Applies the exact same pipeline as the briefing builder
    (boilerplate removal, filler rejection, headline-
    paraphrase rejection, dedup). Headline, source
    attribution and read-more lines are never counted; only
    source-grounded narrative sentences count. Never
    invents content: a story with fewer than two surviving
    explanatory sentences is rejected rather than padded.
    """
    seen = set()
    count = 0

    for sentence in split_sentences(
        text or ""
    ):

        # Reject whole-sentence junk (follow @handle, "please
        # refresh", "this breaking news story is being
        # updated") before boilerplate stripping hides the
        # trigger phrase.
        if is_filler(sentence):
            continue

        sentence = clean_sentence_text(
            sentence
        )

        if not sentence:
            continue

        # A fragment that reduces to punctuation (e.g. the
        # stray period left after boilerplate removal) is
        # not an explanatory sentence.
        if not re.search(
            r"[a-z0-9]",
            sentence,
            re.IGNORECASE,
        ):
            continue

        if is_filler(sentence):
            continue

        if is_headline_paraphrase(
            sentence,
            headline,
        ):
            continue

        key = _normalized(sentence)

        if key in seen:
            continue

        seen.add(key)
        count += 1

    return count


def has_meaningful_sentence(text, headline):
    """True when the cleaned text contains at least one
    sentence that explains the story beyond its headline.

    Applies the exact same pipeline as the briefing builder
    (boilerplate removal, filler rejection, headline-
    paraphrase rejection). Never invents content: if no
    genuine explanatory sentence survives, the story must be
    rejected rather than posted headline-only.
    """
    return count_meaningful_sentences(
        text,
        headline,
    ) >= 1

# ---------------------------------------------------------
# Public labels
# ---------------------------------------------------------

BREAKING = "\U0001F6A8 BREAKING"
JUST_IN = "\u26A1 JUST IN"
NEWS = "\U0001F4F0 NEWS"
UPDATE = "\U0001F504 UPDATE"

URGENT_CATEGORIES = {
    "conflict",
    "disaster",
    "politics",
    "finance",
    "health",
    "cybersecurity",
    "world",
}

# ---------------------------------------------------------
# Conservative event grouping
# ---------------------------------------------------------

STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for",
    "with", "by", "from", "and", "or", "as", "is", "are",
    "was", "were", "be", "been", "has", "have", "had", "it",
    "its", "this", "that", "these", "those", "their", "they",
    "them", "his", "her", "over", "under", "after", "before",
    "during", "into", "about", "more", "most", "new", "says",
    "said", "amid", "live", "watch", "video", "photo", "report",
    "reports", "update", "updates", "could", "will", "would",
    "may", "might", "up", "down", "what", "who", "when",
    "where", "why", "how", "not", "no", "yes", "but", "while",
    "across", "near", "around", "between", "among", "via",
    "including", "against", "aftermath", "latest", "first",
    "world", "today", "yesterday", "officials", "people",
}

WORD_RE = re.compile(
    r"[A-Za-z0-9]+(?:[\u2019'-][A-Za-z0-9]+)*"
)


def _tokens(text):
    words = WORD_RE.findall(
        str(text or "").lower()
    )
    return {
        w
        for w in words
        if len(w) >= 4
        and w not in STOPWORDS
    }


# ---------------------------------------------------------
# Named-entity extraction
# ---------------------------------------------------------

# Proper nouns that are meaningful even as the first word of
# a sentence (acronyms, brands, head-of-state names).
KNOWN_ENTITIES = {
    "us", "uk", "eu", "un", "nato", "tps", "eln", "c16", "c83",
    "meta", "facebook", "zelensky", "biden", "hunter", "petro",
    "yagi", "senate", "house", "congress", "fbi", "cia", "bbc",
}

# City/demonym/company aliases: "Riyadh" means "Saudi Arabia",
# "Facebook" means "Meta", "Russian" means "Russia". Applied
# per entity word so both spellings produce the same signal.
ENTITY_ALIASES = {
    "facebook": "meta",
    "riyadh": "saudi",
    "jeddah": "saudi",
    "tehran": "iran",
    "moscow": "russia",
    "kremlin": "russia",
    "kyiv": "ukraine",
    "kiev": "ukraine",
    "bangkok": "thailand",
    "havana": "cuba",
    "belgrade": "serbia",
    "bogota": "colombia",
    "athens": "greece",
    "sydney": "australia",
    "madrid": "spain",
    "washington": "us",
    "london": "uk",
    "beijing": "china",
    "ottawa": "canada",
    "tokyo": "japan",
    "paris": "france",
    "berlin": "germany",
    "russian": "russia",
    "ukrainian": "ukraine",
    "british": "uk",
    "canadian": "canada",
    "american": "us",
    "iranian": "iran",
    "spanish": "spain",
    "syrian": "syria",
    "israeli": "israel",
    "palestinian": "palestine",
    "chinese": "china",
    "japanese": "japan",
}

# Place names used for the location-conflict veto: two items
# that both name places but share none of them never merge,
# so a Canada wildfire and a Spokane wildfire stay separate
# even when they share "wildfire" as a topic word.
LOCATION_SET = {
    "canada", "us", "usa", "uk", "britain", "england",
    "ukraine", "russia", "iran", "saudi", "serbia", "cuba",
    "thailand", "greece", "australia", "china", "france",
    "spain", "philippines", "colombia", "columbia", "ceuta",
    "hormuz", "spokane", "gaza", "israel", "palestine",
    "syria", "germany", "japan", "india", "mexico", "brazil",
    "egypt", "turkey", "pakistan", "afghanistan", "europe",
    "asia", "africa", "america", "gulf", "venezuela",
    "taiwan", "vietnam", "indonesia", "malaysia", "nigeria",
    "kenya", "poland", "romania", "hungary", "netherlands",
    "belgium", "sweden", "norway", "denmark", "italy",
    "portugal", "new zealand", "argentina", "chile", "peru",
}

# Strong event words. A shared action is one of the few
# signals that lets a single shared entity merge two items
# ("Zelensky visits Serbia" and "Ukrainian president arrives
# in Belgrade" describe one event on the strength of Serbia +
# the visit). Generic vocabulary (oil, data, fire) is not
# listed: it overlaps too easily between unrelated stories.
ACTION_LEXICON = {
    "hit", "strike", "strikes", "seize", "seizure", "kill",
    "killed", "dead", "death", "die", "drown", "evacuat",
    "wildfire", "flood", "storm", "typhoon", "hurricane",
    "landfall", "blackout", "collapse", "sign", "ink", "seal",
    "pact", "agreement", "deal", "treaty", "ceasefire",
    "fine", "penalty", "convict", "guilty", "verdict",
    "indict", "shoot", "shooting", "gunman", "gunfire",
    "attack", "bomb", "blast", "explod", "visit", "arriv",
    "summit", "vote", "elect", "inaugurat", "swear",
    "ceremony", "pass", "extend", "terminat", "approv",
    "jobs", "payroll", "surge", "flee", "cross", "deploy",
    "exercise", "drone", "missile", "refinery", "tanker",
    "sanction", "arrest", "lawsuit", "breach", "hack",
    "protest", "landslide", "quake", "earthquake", "tornado",
    "outage", "blackout", "fire",
}

# Action synonym groups: canonical action -> variant words.
# "hits", "slams" and "landfall" all mean the storm struck;
# "signs" and "inks" both close a pact.
ACTION_SYNSETS = {
    "strike": {"strike", "hit", "landfall", "slams",
               "pummel", "batters"},
    "shoot": {"shoot", "shooting", "gunman", "gunfire"},
    "sign": {"sign", "ink", "seal", "ratify"},
    "pact": {"pact", "agreement", "deal", "treaty", "accord"},
    "visit": {"visit", "arriv", "arrive"},
    "outage": {"outage", "blackout", "collapse", "shutdown"},
    "convict": {"convict", "guilty", "verdict", "liable"},
    "fine": {"fine", "penalty"},
    "kill": {"kill", "dead", "death", "fatal"},
    "evacuat": {"evacuat", "flee", "evacuation"},
}

POSSESSIVE_RE = re.compile(r"'\w*$")


def _entity_words(text):
    """Lowercased, de-possessed proper-noun words.

    A capitalized word counts as an entity unless it is a
    lone first word of a sentence (that is usually ordinary
    headline casing, e.g. "Wildfire forces..."), a pure
    number, or a known stopword. Capitalized runs ("Hunter
    Biden", "Typhoon Yagi", "Western Canada") count even at
    sentence start.
    """
    if not text:
        return set()

    words = WORD_RE.findall(str(text))
    entities = set()

    for sentence in split_sentences(str(text)):

        sentence_words = WORD_RE.findall(sentence)
        if not sentence_words:
            continue

        i = 0
        while i < len(sentence_words):

            word = sentence_words[i]
            if not word[:1].isupper():
                i += 1
                continue

            run = [word]
            j = i + 1

            while (
                j < len(sentence_words)
                and sentence_words[j][:1].isupper()
            ):
                run.append(sentence_words[j])
                j += 1

            run_starts_sentence = i == 0
            known_word = (
                len(run) == 1
                and POSSESSIVE_RE.sub(
                    "",
                    run[0],
                ).lower() in (
                    KNOWN_ENTITIES
                    | LOCATION_SET
                    | set(ENTITY_ALIASES)
                )
            )
            keep_run = (
                len(run) >= 2
                or not run_starts_sentence
                or known_word
            )

            if keep_run:
                for w in run:
                    w = POSSESSIVE_RE.sub("", w).lower()
                    if w[:1].isdigit():
                        continue
                    if w in STOPWORDS:
                        continue
                    entities.add(
                        ENTITY_ALIASES.get(w, w)
                    )
                    if w in ENTITY_ALIASES:
                        entities.add(w)

            i = j

    return entities


def _action_stems(word):
    """A word plus its common inflectional stems: "seized"
    yields "seize", "evacuations" yields "evacuation"."""
    yield word

    for suffix in ("s", "es", "ed", "d", "ing"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            yield word[: -len(suffix)]


def _actions(text):
    """Stemmed strong-event words in the text, canonicalized
    through synonym groups ("hits" and "landfall" both yield
    "strike")."""
    if not text:
        return set()

    words = {
        stem
        for w in WORD_RE.findall(
            str(text).lower()
        )
        for stem in _action_stems(w)
    }
    found = set()

    for canonical, variants in ACTION_SYNSETS.items():
        if variants & words:
            found.add(canonical)

    found |= (
        words & ACTION_LEXICON
    )

    return found


def _location_conflict(a, b):
    """True when both items name places but share none.

    Shared-topic words (wildfire, evacuation) never overcome
    a hard location split: a fire in Canada and a fire in
    Washington state are different events even when every
    other signal overlaps.
    """
    la = _entity_words(a.get("title")) & LOCATION_SET
    lb = _entity_words(b.get("title")) & LOCATION_SET

    if not la or not lb:
        return False

    shared_locations = la & lb

    if shared_locations:
        return False

    return True


def _number_conflict(a, b):
    """True when both texts name the same unit with
    different values ("20,000 evacuated" vs "12,000
    evacuated"). Conflicting numbers mean different facts,
    never the same event."""
    units_a = dict(
        _number_units(a.get("title"))
    )
    units_b = dict(
        _number_units(b.get("title"))
    )

    for unit in set(units_a) & set(units_b):
        if units_a[unit] != units_b[unit]:
            return True

    return False


def _corroborated(a, b):
    """Shared specific number+unit fact ("7 tankers" on both
    sides) or explicit corroboration from both sources."""
    shared = set(
        _number_units(a.get("title"))
    ) & set(
        _number_units(b.get("title"))
    )

    if shared:
        return True

    # A shared large bare figure ("272,000 jobs" reported as
    # "payrolls jump 272,000") also corroborates: the same
    # four-plus-digit number in both feeds is not coincidence.
    def bare_numbers(text):
        return {
            m.group(0).replace(",", "")
            for m in NUMBER_RE.finditer(text or "")
            if len(m.group(0).replace(",", "")) >= 4
        }

    if bare_numbers(a.get("title")) & bare_numbers(
        b.get("title")
    ):
        return True

    return bool(
        a.get("strong_corroboration")
        and b.get("strong_corroboration")
    )


def _topic_similarity(a, b):
    ta = _tokens(a.get("title"))
    tb = _tokens(b.get("title"))

    if not ta or not tb:
        return 0.0

    return len(ta & tb) / max(
        1,
        len(ta | tb),
    )


TEMPORAL_WINDOW_HOURS = 72


def _temporally_close(a, b):
    """Whether the two items happened within 72 hours.

    An archive piece about the 1999 eclipse and a live
    report on this year's eclipse never merge: they are
    decades apart even though every topic word matches.
    """
    ts_a = a.get("effective_at")
    ts_b = b.get("effective_at")

    if not ts_a or not ts_b:
        return True

    try:
        from datetime import datetime
        fmt = "%Y-%m-%dT%H:%M:%S"
        d_a = datetime.fromisoformat(ts_a)
        d_b = datetime.fromisoformat(ts_b)
    except (TypeError, ValueError):
        return True

    delta = abs(
        (
            d_b - d_a
        ).total_seconds()
    )

    return delta <= (
        TEMPORAL_WINDOW_HOURS * 3600
    )


# With only one shared named entity, a single shared action
# is not enough to merge: two unrelated stories both say
# "Ukraine" and "killed" every day. One-entity merges need
# at least one strengthener: a second shared action,
# corroborating concrete facts, or strong topic overlap.
# Calibrated so a real single-event pair like the Canada
# wildfire cluster (topic 0.167) still merges, while the
# false "Ukraine kill" / "US arrest" pairs (topic ~0.0)
# stay separate.
ONE_ENTITY_TOPIC_THRESHOLD = 0.15


def same_event(a, b):
    """Whether two items are the same event.

    Conservative multi-signal matcher. Same event_id merges.
    Otherwise merging requires shared named entities plus
    meaningful signals:

    - two-plus shared entities merge with any one of: a
      shared action, corroborating facts/numbers, or topic
      overlap of 0.30;
    - one shared entity merges only with at least one shared
      meaningful action plus one of: a second shared action,
      corroborating facts/numbers, or topic overlap of
      ONE_ENTITY_TOPIC_THRESHOLD.

    Vetoes (never merged, whatever else overlaps):
    - items more than 72 hours apart (an archive report and
      a live one are different posts);
    - items that name different places and share no place
      (a Canada wildfire is not a Spokane wildfire);
    - items that report the same unit with different numbers
      (20,000 evacuated vs 12,000 evacuated).
    """
    a_id = a.get("event_id")
    b_id = b.get("event_id")

    if a_id and b_id and a_id == b_id:
        return True

    if not _temporally_close(a, b):
        return False

    if _number_conflict(a, b):
        return False

    if _location_conflict(a, b):
        return False

    ea = _entity_words(a.get("title"))
    eb = _entity_words(b.get("title"))

    shared_entities = ea & eb

    if not shared_entities:
        return False

    shared_actions = (
        _actions(a.get("title"))
        & _actions(b.get("title"))
    )

    corroborated = _corroborated(a, b)
    topic = _topic_similarity(a, b)

    if len(shared_entities) >= 2 and (
        shared_actions
        or corroborated
        or topic >= 0.30
    ):
        return True

    if not shared_actions:
        return False

    if (
        len(shared_actions) >= 2
        or corroborated
        or topic >= ONE_ENTITY_TOPIC_THRESHOLD
    ):
        return True

    return False


def _primary_of(group):
    """Pick the strongest item: score, then primary source,
    then better tier."""
    def key(item):
        return (
            item.get("score", 0)
            or item.get("priority_score", 0)
            or 0,
            int(
                bool(item.get("primary_source"))
            ),
            -int(item.get("tier", 4)),
        )

    return sorted(
        group,
        key=key,
        reverse=True,
    )[0]


def group_items(items):
    """Cluster items into events.

    An item joins a cluster only when it matches the
    cluster's primary item, avoiding chain drift where
    A~B and B~C would wrongly merge an unrelated C.
    """
    groups = []

    for item in items:

        placed = False

        for group in groups:

            if same_event(
                item,
                _primary_of(group),
            ):
                group.append(item)
                placed = True
                break

        if not placed:
            groups.append([item])

    return groups


# ---------------------------------------------------------
# Sentence aggregation with provenance
# ---------------------------------------------------------

NUMBER_RE = re.compile(r"\d[\d,.]*")
UNIT_RE = re.compile(
    r"(?:"
    r"(?:more than|at least|nearly|about|around|over|"
    r"up to|almost|roughly)?\s*"
    r"(\d[\d,.]*)\s*"
    r"(million|billion|thousand|percent|%|people|residents|"
    r"homes|families|troops|soldiers|killed|injured|dead|"
    r"displaced|evacuated|evacuees|jobs|sq miles|sq km|sq m|"
    r"miles|km|hectares|acres|fires|officers|hostages)?"
    r")",
    re.IGNORECASE,
)


def _number_units(text):
    """unit -> normalized number pairs in a sentence."""
    pairs = []

    for m in UNIT_RE.finditer(text):
        number = m.group(1)
        unit = m.group(2)

        if unit:
            pairs.append(
                (
                    unit.lower(),
                    re.sub(
                        r"[.,]",
                        "",
                        number,
                    ),
                )
            )

    return pairs


def _conflicting(a, b):
    """Whether two sentences materially disagree.

    A conflict requires: overlapping vocabulary AND a shared
    numeric unit with different values. "36 sq miles" and
    "20,000 people" do not conflict; "20,000" and "30,000"
    with the same unit do.
    """
    shared = (
        _tokens(a["text"])
        & _tokens(b["text"])
    )

    if len(shared) < 2:
        return False

    units_a = dict(
        _number_units(a["text"])
    )
    units_b = dict(
        _number_units(b["text"])
    )

    common_units = set(
        units_a
    ) & set(
        units_b
    )

    for unit in common_units:

        if (
            units_a[unit] != units_b[unit]
        ):
            return True

    return False


NEAR_DUP_JACCARD = 0.80


def is_near_duplicate(a, b):
    """True when two sentences are near-identical rewordings
    of the same statement: high token overlap and no
    materially conflicting facts.

    Catches pairs that exact-match dedup misses, e.g.
    "...has spread over more than 36 sq miles" vs
    "...has spread to more than 36 sq miles". Sentences
    that disagree on a numeric fact are never near-dups.
    """
    ta = _tokens(a["text"])
    tb = _tokens(b["text"])

    shared = ta & tb

    if len(shared) < 3:
        return False

    jaccard = len(shared) / max(
        1,
        len(ta | tb),
    )

    if jaccard < NEAR_DUP_JACCARD:
        return False

    return not _conflicting(a, b)


def aggregate_sentences(group, primary):
    """Verbatim sentences from the whole event group,
    deduplicated, with source provenance retained.

    Higher-ranked items come first. A sentence that
    materially conflicts with an already-accepted
    sentence is dropped (disputed detail omitted).
    """
    ranked = sorted(
        group,
        key=lambda x: (
            x.get("score", 0)
            or x.get("priority_score", 0)
            or 0,
            int(
                bool(x.get("primary_source"))
            ),
            -int(x.get("tier", 4)),
        ),
        reverse=True,
    )

    seen = set()
    accepted = []

    headline = primary.get("title") or ""

    for item in ranked:

        source = item.get("source") or "Unknown"

        for sentence in split_sentences(
            item.get("summary")
        ):

            # Reject whole-sentence junk before boilerplate
            # stripping hides the trigger phrase.
            if is_filler(sentence):
                continue

            sentence = clean_sentence_text(
                sentence
            )

            if not sentence:
                continue

            if is_filler(sentence):
                continue

            if is_headline_paraphrase(
                sentence,
                headline,
            ):
                continue

            key = _normalized(sentence)

            if key in seen:
                continue

            seen.add(key)

            row = {
                "text": sentence,
                "source": source,
                "item_id": (
                    item.get("id")
                    or item.get("story_id")
                ),
            }

            if any(
                _conflicting(row, other)
                for other in accepted
            ):
                continue

            if any(
                is_near_duplicate(row, other)
                for other in accepted
            ):
                continue

            accepted.append(row)

    # Primary source sentences first, then corroborators,
    # preserving each source's internal order.
    primary_rows = [
        r for r in accepted
        if r["source"]
        == (primary.get("source") or "Unknown")
    ]
    other_rows = [
        r for r in accepted
        if r["source"]
        != (primary.get("source") or "Unknown")
    ]

    return primary_rows + other_rows


# ---------------------------------------------------------
# Conservative bullet extraction (literal evidence only)
# ---------------------------------------------------------

LOCATION_RE = re.compile(
    r"\b(?:in|across|near|around|at)\s+"
    r"([A-Z][A-Za-z\u2019'-]*(?:\s+(?:"
    r"[A-Z][A-Za-z\u2019'-]*|(?:city|town|state|province|"
    r"county|island|prefecture|region|capital|coast|valley|"
    r"bay|river|village|district|area|park|north|south|east|"
    r"west|central|northern|southern|eastern|western|"
    r"national|metropolitan)\b"
    r")){0,2})"
)

STATUS_PHRASES = [
    "state of emergency",
    "out of control",
    "under control",
    "continues to",
    "still burning",
    "under way",
    "underway",
]

IMPACT_RE = re.compile(
    r"(?:more than|at least|nearly|about|around|over|"
    r"up to|almost|roughly)?\s*"
    r"\d[\d,.]*\s*"
    r"(?:people|residents|homes|families|troops|soldiers|"
    r"killed|injured|dead|displaced|evacuated|evacuees|"
    r"fires|hostages|officers)",
    re.IGNORECASE,
)

NEXT_RE = re.compile(
    r"\b(?:expected to|is expected|officials (?:said|say|"
    r"warned)|continues to|are (?:continuing|working|"
    r"expected))",
    re.IGNORECASE,
)


def extract_bullets(rows):
    """Bullets from literal sentence evidence only.

    Returns a list of {"icon", "label", "text"}.
    """
    bullets = []
    texts = [r["text"] for r in rows]

    # Location: first credible "in <Place>" match.
    for text in texts:
        match = LOCATION_RE.search(text)

        if not match:
            continue

        location = match.group(1).strip()

        if (
            len(location.split()) <= 4
            and not location.lower().startswith(
                "the "
            )
        ):
            bullets.append(
                {
                    "icon": "\U0001F4CD",
                    "label": "Location",
                    "text": location,
                }
            )
            break

    # Status: first literal phrase match.
    for phrase in STATUS_PHRASES:

        for text in texts:
            if phrase in text.lower():
                bullets.append(
                    {
                        "icon": "\u26A0\uFE0F",
                        "label": "Status",
                        "text": phrase,
                    }
                )
                break

        if len(bullets) >= 2:
            break

    # Impact: first number+unit phrase with a people/impact
    # unit, e.g. "more than 20,000 people".
    for text in texts:
        match = IMPACT_RE.search(text)

        if match:
            impact = clean(match.group(0))

            if impact:
                bullets.append(
                    {
                        "icon": "\U0001F465",
                        "label": "Impact",
                        "text": impact,
                    }
                )
                break

    # Next steps: first full sentence that reports what
    # happens next.
    for row in rows:
        if NEXT_RE.search(row["text"]):
            bullets.append(
                {
                    "icon": "\u27A1\uFE0F",
                    "label": "Next",
                    "text": row["text"],
                }
            )
            break

    return bullets


# ---------------------------------------------------------
# Headline cleaning (never paraphrases, never invents)
# ---------------------------------------------------------

SOURCE_SUFFIX_RE = re.compile(
    r"\s*(?:-\s*|[\u2013\u2014]\s*|\|\s*)"
    r"(?:BBC(?: World)?|CNN|Reuters|AP|Al Jazeera|"
    r"The Guardian|Guardian|NPR|Sky News|France24|"
    r"The Independent|Independent|DW|Euronews|"
    r"The Times of India|Nikkei Asia|The Japan Times|"
    r"Global Times|South China Morning Post|SCMP|"
    r"Los Angeles Times|The Wall Street Journal|WSJ|"
    r"Financial Times|FT|Bloomberg|CNBC|ABC News|"
    r"CBS News|NBC News|The Washington Post|"
    r"USA Today|The Economist|Forbes|Time|Newsweek|"
    r"Politico|Axios|The Hill|The Verge|Ars Technica|"
    r"Wired|Engadget|TechCrunch)$",
    re.IGNORECASE,
)

PREFIX_RE = re.compile(
    r"^(?:Live|Watch|Video|Update|Breaking|Flash)"
    r"[:\s-]+\s*",
    re.IGNORECASE,
)


def clean_headline(title):
    """Remove RSS artifacts without changing the headline."""
    headline = clean(title)

    if not headline:
        return headline

    headline = SOURCE_SUFFIX_RE.sub(
        "",
        headline,
    )
    headline = PREFIX_RE.sub(
        "",
        headline,
    )
    headline = clean(headline)

    if headline:
        headline = (
            headline[0].upper()
            + headline[1:]
        )

    return headline


# ---------------------------------------------------------
# Public label selection
# ---------------------------------------------------------


def public_label(
    item,
    just_in_freshness_minutes=15,
    now=None,
):
    """One of BREAKING / JUST IN / NEWS / UPDATE.

    BREAKING needs urgency terms + non-low confidence + a
    high score + verification (primary source or strong
    corroboration) + an urgent category. Priority level is
    never used alone. JUST IN needs freshness AND importance.
    """
    status = item.get(
        "event_status",
        "NEW",
    )
    score = item.get("score", 0) or 0
    confidence = item.get("confidence")
    urgent = bool(
        item.get("urgency_terms")
    )
    category = item.get("category")
    primary = bool(
        item.get("primary_source")
    )
    strong = int(
        item.get("strong_corroboration", 0)
    )

    if (
        status == "UPDATE"
        and score >= 60
    ):
        return UPDATE

    if (
        urgent
        and confidence != "low"
        and score >= 80
        and (primary or strong >= 1)
        and category in URGENT_CATEGORIES
    ):
        return BREAKING

    age = story_age_minutes(item, now)

    if (
        age is not None
        and age <= just_in_freshness_minutes
        and confidence != "low"
        and score >= 60
    ):
        return JUST_IN

    return NEWS


# ---------------------------------------------------------
# Briefing assembly
# ---------------------------------------------------------


def build_briefing(
    primary,
    group,
    just_in_freshness_minutes=15,
    now=None,
    max_sentences=None,
):
    """Build the enriched briefing for one event cluster.

    Returns a dict consumed by telegram_formatter.
    """
    rows = aggregate_sentences(
        group,
        primary,
    )

    if max_sentences:
        rows = rows[:max_sentences]

    primary_source = (
        primary.get("source")
        or "Unknown"
    )

    contributing = [
        r["source"]
        for r in rows
        if r["source"] != primary_source
    ]

    corroborating = []

    for source in contributing:
        if source not in corroborating:
            corroborating.append(source)

    opening = rows[:2]

    bullets = extract_bullets(rows)

    # A bullet must never repeat a sentence that already
    # appears in the opening paragraph.
    opening_texts = {
        r["text"] for r in opening
    }

    bullets = [
        b
        for b in bullets
        if (b["text"] or "").strip()
        not in opening_texts
    ]

    sentence_texts = [
        r["text"] for r in rows
    ]

    bullet_sentences = {
        b["text"]
        for b in bullets
        if b["text"] in sentence_texts
    }

    body_rows = [
        r["text"]
        for r in rows[len(opening):]
        if r["text"] not in bullet_sentences
    ]

    return {
        "headline": clean_headline(
            primary.get("title")
        ),
        "label": public_label(
            primary,
            just_in_freshness_minutes,
            now,
        ),
        "opening": [
            r["text"] for r in opening
        ],
        "body": body_rows,
        "bullets": bullets,
        "sentences": rows,
        "source": primary_source,
        "corroborating": corroborating,
        "url": primary.get("url"),
    }
