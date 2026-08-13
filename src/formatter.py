import re

POST_LIMIT = 270


def clean(text):
    """Normalize whitespace without destroying useful content."""
    return re.sub(r"\s+", " ", text or "").strip()


def clean_sentence(text):
    """Clean text and ensure it ends as a complete sentence.

    A sentence that already ends with terminal punctuation
    behind a closing quote or bracket ("...burn.") is left
    unchanged; appending another period would produce the
    malformed '".' artifact.
    """
    text = clean(text)

    if not text:
        return ""

    # A trailing single lowercase letter ("...a Qatar Airways
    # plane t") is a feed-truncated word fragment, never a
    # complete sentence ending; drop it so it can never
    # render as a dangling fragment.
    text = re.sub(r"\s+[a-z]\.?$", "", text)

    core = text.rstrip(
        "'\"\u2018\u2019\u201c\u201d)]}"
    )

    if core.endswith((".", "!", "?")):
        return text

    return text + "."


# ---------------------------------------------------------------------------
# Missing-punctuation sentence-boundary repair
#
# Some feeds (notably The Guardian) drop the period between
# the standfirst and the article lede, producing run-together
# text like:
#
#     "...moves north and further inland More than a million
#      people were moved to safety..."
#
# Repair is purely mechanical: only the whitespace between
# the two sentences is replaced with ". ". No words are
# added, removed or rewritten - a boundary is split only
# when there is strong evidence that a new sentence starts
# there.
# ---------------------------------------------------------------------------

# Strong sentence-initial words. A capitalized instance of
# one of these immediately after a lowercase word is almost
# always a dropped sentence boundary (mid-sentence instances
# are written lowercase). Proper-name components such as
# "New", "United", "North", "Little", "White" are excluded:
# "visited New York" must never be split.
SENTENCE_STARTERS = {
    "the", "a", "an", "it", "this", "that", "these", "those",
    "there", "they", "he", "she", "we", "you", "i",
    "some", "many", "most", "much", "more", "several", "few",
    "both",
    "each", "every", "either", "neither", "all",
    "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "dozens", "hundreds", "thousands",
    "millions",
    "but", "yet", "however", "meanwhile", "moreover",
    "furthermore", "also", "then", "now", "afterwards",
    "subsequently", "instead",
    "in", "on", "at", "after", "before", "during", "within",
    "without", "despite", "although", "though", "because",
    "since", "when", "while", "where", "why", "how", "what",
    "which", "who", "whose", "until", "throughout",
    "officials", "authorities", "police", "firefighters",
    "soldiers", "troops", "residents", "witnesses", "officers",
}

# Words that cannot end a sentence: prepositions,
# conjunctions, articles, subordinators and auxiliaries. A
# boundary candidate whose previous word is one of these is
# rejected (e.g. "in Morelia", "such as The Guardian",
# "with More than...", "and The...").
TRAILING_REJECT = {
    "a", "an", "the", "and", "or", "nor", "but", "so", "yet",
    "for", "of", "in", "on", "at", "to", "from", "with", "by",
    "about", "into", "onto", "upon", "over", "under", "through",
    "across", "along", "among", "between", "within", "without",
    "behind", "before", "after", "above", "below", "beneath",
    "beside", "besides", "beyond", "during", "except", "inside",
    "outside", "per", "plus", "since", "throughout", "till",
    "toward", "towards", "underneath", "unlike", "until", "up",
    "via", "as", "if", "that", "than", "when", "while",
    "because", "although", "though", "unless", "whereas",
    "whether", "is", "are", "was", "were", "be", "been",
    "being", "has", "have", "had", "do", "does", "did", "will",
    "would", "shall", "should", "can", "could", "may", "might",
    "must", "am", "not", "too", "also", "either", "neither",
    "all", "both", "each", "few", "many", "most", "several",
    "some", "such", "this", "these", "those", "any", "other",
    "another", "more", "much", "only", "own", "same", "what",
    "which", "who", "whom", "whose", "how", "why", "where",
    "whenever", "wherever", "s", "re", "ve", "d", "ll",
}

# Reporting verbs that make "<Name> said ..." a strong
# sentence-start signal for a capitalized proper name.
REPORTING_VERBS = {
    "said", "say", "says", "told", "added", "warned", "warn",
    "confirmed", "announced", "stated", "claimed", "argued",
    "noted", "stressed", "declared", "emphasized", "insisted",
    "revealed", "acknowledged", "admitted", "explained",
    "commented", "reported", "suggested", "urged", "promised",
    "indicated", "mentioned", "estimated", "recalled",
    "remarked", "responded", "replied", "concluded", "observed",
    "asked", "added", "boasted", "conceded", "disclosed",
}

# Auxiliary verbs that can also follow a proper-name sentence
# start ("Anthony Albanese has bowed", "John Smith will
# announce").  Together with REPORTING_VERBS they form the
# strong follow word set for a first+last name boundary.
NAME_FOLLOWING_VERBS = REPORTING_VERBS | {
    "has", "have", "had", "will", "would",
    "shall", "should", "could", "might",
}

# Object-taking verbs: a name directly after one of these is
# the verb's object ("Police said John Smith has been
# arrested", "A new poll shows John Smith has a lead"), never
# a new sentence.  Covers reporting verbs plus common
# transitive forms.
OBJECT_TAKING_VERBS = REPORTING_VERBS | {
    "show", "shows", "showed", "shown",
    "find", "finds", "found",
    "believe", "believes", "believed",
    "expect", "expects", "expected",
    "fear", "fears", "feared",
    "hope", "hopes", "hoped",
    "deny", "denies", "denied",
    "confirm", "confirms", "confirmed",
    "claim", "claims", "claimed",
    "reveal", "reveals", "revealed",
    "admit", "admits", "admitted",
    "note", "notes", "noted",
    "state", "states", "stated",
    "report", "reports", "reported",
    "add", "adds", "added",
    "indicate", "indicates", "indicated",
    "estimate", "estimates", "estimated",
}

# Titles/roles that legitimately precede a capitalized name
# ("mayor John Smith said") and therefore are NOT boundaries.
NAME_PREFIXES = {
    "mr", "mrs", "ms", "miss", "dr", "professor", "sir",
    "madam", "lord", "lady", "president", "prime", "minister",
    "vice", "premier", "chancellor", "governor", "mayor",
    "senator", "representative", "congressman", "congresswoman",
    "ambassador", "secretary", "chairman", "chairwoman",
    "chief", "executive", "director", "manager", "editor",
    "author", "lawyer", "attorney", "judge", "justice",
    "officer", "detective", "captain", "commander", "colonel",
    "general", "major", "lieutenant", "sergeant", "admiral",
    "marshal", "inspector", "superintendent", "constable",
    "king", "queen", "prince", "princess", "duke", "duchess",
    "emperor", "empress", "pope", "bishop", "archbishop",
    "cardinal", "imam", "ayatollah", "sheikh", "rabbi",
    "spokesman", "spokeswoman", "spokesperson", "reporter",
    "journalist", "analyst", "economist", "official",
    "founder", "co-founder", "chair", "leader", "head",
    "activist", "survivor", "witness", "teenager", "woman",
    "man", "boy", "girl", "student", "teacher", "doctor",
    "player", "coach", "star", "singer", "actor", "actress",
    "ceo", "cfo", "coo", "cto", "vp", "pm",
    "hurricane", "typhoon", "cyclone", "storm", "tornado",
    "blizzard", "monsoon",
}

# Known proper names that are never person names and therefore
# never mark a sentence start ("El Nino could push global
# temperatures", "La Nina has brought drought").
NON_PERSON_NAME_PAIRS = {
    "el nino",
    "la nina",
}

# Person-name + following-verb pattern: a first+last name
# pair immediately followed by a reporting verb or auxiliary
# ("John Smith said", "Anthony Albanese has bowed"). A single
# capitalized word plus verb is NOT strong enough evidence.
_NAME_VERB_RE = re.compile(
    r"^[A-Z][A-Za-z\u2019'-]*"
    r"\s+[A-Z][A-Za-z\u2019'-]*"
    r"\s+(?:"
    + "|".join(sorted(NAME_FOLLOWING_VERBS))
    + r")\b"
)

# Quotation marks (ASCII and curly). Used to detect whether a
# candidate boundary lies inside an open quotation.
_DOUBLE_QUOTE_CHARS = frozenset(
    ['"', "\u201c", "\u201d"]
)


def _inside_open_quote(text, position):
    """True when position lies inside an unclosed
    double-quoted span (ASCII or curly quotes)."""
    return (
        sum(
            1 for ch in text[:position]
            if ch in _DOUBLE_QUOTE_CHARS
        )
        % 2
        == 1
    )


_MIN_BOUNDARY_WORDS = 4


# Common abbreviations whose period must never be read as a
# sentence boundary: titles ("Dr.", "Mr."), number markers
# ("No. 266"), months/days, address units and Latin
# abbreviations.  Lowercased tokens; matched only when the
# token is capitalized in the text, so a genuine sentence
# ending in a lowercase word ("He said no.") still splits.
_ABBREVIATION_TOKENS = frozenset({
    # titles
    "dr", "mr", "mrs", "ms", "prof", "professor", "sr", "jr",
    "rev", "hon", "gen", "col", "maj", "capt", "lt", "sgt",
    "cpl", "rep", "sen", "gov", "pres", "supt", "insp", "fr",
    # number / address markers
    "no", "nos", "apt", "ave", "blvd", "bldg", "dept", "st",
    "mt", "ft", "rr",
    # months and days (never a sentence end when capitalized)
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep",
    "sept", "oct", "nov", "dec",
    "mon", "tue", "wed", "thu", "thur", "thurs", "fri", "sat",
    "sun",
})

# Latin / unit abbreviations that are never real words, so a
# lowercase instance is also a non-boundary ("vs.", "etc.",
# "approx.").
_LOWERCASE_ABBREVIATIONS = frozenset({
    "etc", "vs", "approx", "est", "fig",
})

# Dotted multi-letter runs that must never be split: "U.S.",
# "U.K.", "e.g.", "i.e.", "a.m.", "p.m.", "Ph.D."
_DOTTED_ABBREV_RE = re.compile(
    r"(?:[A-Za-z]\.){2,}$"
)


def _is_abbreviation_period(text, index):
    """True when the period at `index` ends an abbreviation, so
    the whitespace after it is not a sentence boundary.

    Handles dotted runs ("U.S.", "a.m."), single-letter
    initials ("J. Smith") and known abbreviation tokens
    ("Dr.", "No.", "Jan.").  A capitalized token is required
    for the known-abbreviation set so a real sentence ending in
    a lowercase word ("He said no.") still splits."""
    if text[index] != ".":
        return False
    if _DOTTED_ABBREV_RE.search(text[: index + 1]):
        return True
    word = re.search(r"([A-Za-z]+)$", text[:index])
    if not word:
        return False
    token = word.group(1)
    if len(token) == 1 and token.isupper():
        return True
    if token.lower() in _LOWERCASE_ABBREVIATIONS:
        return True
    return token[:1].isupper() and token.lower() in (
        _ABBREVIATION_TOKENS
    )


def _words_before(text, segment_start, position):
    """Count words in text from segment_start to position
    (exclusive)."""
    return len(
        re.findall(
            r"[A-Za-z0-9\u2019'-]+",
            text[segment_start:position],
        )
    )


def _repair_candidate(text, position, upper_word):
    """True when the whitespace at position is a dropped
    sentence boundary with strong evidence."""
    lower = upper_word.lower()

    # 1. The capitalized word must be a strong sentence
    #    starter...
    if lower not in SENTENCE_STARTERS:
        # ...or a capitalized first+last name directly
        # followed by a reporting verb
        # ("John Smith said ...", "Zawtar al-Gharbiya").
        if not _NAME_VERB_RE.match(text[position:]):
            return False

    # 2. The previous word must be able to end a sentence.
    words_before = re.findall(
        r"[A-Za-z0-9\u2019'-]+",
        text[:position],
    )
    if not words_before:
        return False
    prev_lower = words_before[-1].lower()
    if prev_lower in TRAILING_REJECT:
        return False

    if lower not in SENTENCE_STARTERS:
        # Proper-name boundary: the name must not follow a
        # title/role ("mayor John Smith said" is not a
        # boundary).
        if prev_lower in NAME_PREFIXES:
            return False
        # A name directly after an object-taking verb is the
        # verb's object ("Police said John Smith has been
        # arrested", "A new poll shows John Smith has a
        # lead"), not a new sentence.
        if prev_lower in OBJECT_TAKING_VERBS:
            return False
        # A name directly after another capitalized word is
        # part of a multi-word name ("Andres Manuel Lopez
        # Obrador"), not a new sentence.
        if words_before[-1][0].isupper():
            return False
        # A name preceded by an article-plus-modifier is the
        # head of the noun phrase ("A strengthening El Nino
        # could push..."), not a new sentence.
        if (
            len(words_before) >= 2
            and words_before[-2].lower() in {"a", "an", "the"}
        ):
            return False
        # Known non-person proper names that are never
        # sentence starts ("El Nino could push", "La Nina
        # has cooled").
        head = re.match(
            r"([A-Za-z\u2019'-]+)\s+([A-Za-z\u2019'-]+)",
            text[position:],
        )
        if head and (
            head.group(1).lower() + " " + head.group(2).lower()
        ) in NON_PERSON_NAME_PAIRS:
            return False

    return True


def repair_sentence_boundaries(text):
    """Split obvious run-together sentences without touching
    any words.

    Replaces the whitespace before a capitalized word with
    ". " only when the capitalized word is a strong
    sentence-initial word (or a proper name followed by a
    reporting verb), the previous word can end a sentence,
    and the preceding text is a complete clause. All other
    lowercase-to-capital transitions are left untouched.
    """
    text = clean(text)

    if not text:
        return text

    out = []
    last = 0
    scan = 0
    length = len(text)

    while scan < length:
        match = re.search(
            r"\s+([A-Z][A-Za-z\u2019'-]*)",
            text[scan:],
        )

        if not match:
            break

        start = scan + match.start()
        word_end = scan + match.end()
        word = match.group(1)

        if start == 0 or not text[start - 1].isalnum():
            scan = word_end
            continue

        # Never insert a boundary inside a quoted span:
        # "the last time I bought red meat" inside a quotation
        # is one continuous quoted sentence, not a boundary.
        if _inside_open_quote(text, start + 1):
            scan = word_end
            continue

        if not _repair_candidate(
            text,
            start + 1,
            word,
        ):
            scan = word_end
            continue

        # Strong evidence: split here. The words on both
        # sides are preserved exactly; only the whitespace
        # becomes ". ".
        if _words_before(
            text, last, start + 1
        ) < _MIN_BOUNDARY_WORDS:
            scan = word_end
            continue

        out.append(text[last:start])
        out.append(". ")
        out.append(word)
        last = word_end
        scan = word_end

    out.append(text[last:])

    return "".join(out)


def split_sentences(text):
    """
    Split normal prose into complete sentences.

    Missing-punctuation boundaries (feeds that drop the
    period between sentences) are repaired first, then the
    text is split on existing terminal punctuation.  A period
    that ends a known abbreviation ("Dr.", "No. 266", "a.m.",
    "U.S.") is never treated as a sentence boundary. If the
    source summary has poor punctuation, this function still
    returns usable text rather than failing.
    """
    text = repair_sentence_boundaries(text)

    if not text:
        return []

    parts = []
    start = 0
    for match in re.finditer(r"[.!?]\s+", text):
        if _is_abbreviation_period(text, match.start()):
            continue
        parts.append(text[start : match.start() + 1])
        start = match.end()

    parts.append(text[start:])

    return [
        clean_sentence(part)
        for part in parts
        if clean(part)
    ]

