"""Telegram message formatting for news briefings.

Renders the enriched briefing produced by telegram_briefing in
the final WorldNews message format:

    [ONE LABEL]

    [Clear headline]

    [2-4 useful explanatory sentences]

    \U0001F4F0 Source: [source name]

    \U0001F517 Read the full report

Nothing else appears in the message: no channel header, no
bullets, no region/location/status sections, no corroboration
text, no decorative lines. Every sentence is verbatim source
evidence; nothing is invented.

Safety rules:
- The footer (source + link) and the headline are never
  dropped.
- A story with fewer than 2 genuinely useful explanatory
  sentences is not published (build_message returns None).
- Truncation removes whole sentences, never cutting a
  sentence in the middle.
- Character budget is counted the way Telegram does
  (CJK/emoji count double).
"""
import re

from src.formatter import clean
from src.telegram_briefing import (
    _normalized,
    clean_headline,
    clean_sentence_text,
    count_meaningful_sentences,
    decode_html_entities,
    is_filler,
    is_headline_paraphrase,
    is_near_duplicate,
)

CJK_RE = re.compile(
    r"[\u3000-\u303f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff"
    r"\uf900-\ufaff\uff00-\uffef]"
)

EMOJI_RE = re.compile(
    "["
    "\U0001f300-\U0001f5ff"
    "\U0001f600-\U0001f64f"
    "\U0001f680-\U0001f6ff"
    "\U0001f7e0-\U0001f7ff"
    "\U0001f900-\U0001f9ff"
    "\u2600-\u26ff"
    "\u2700-\u27bf"
    "]"
)

HTML_ENTITY_RE = re.compile(r"&[a-zA-Z#0-9]+;")

# The final message carries 2-4 useful explanatory sentences:
# never fewer than two (below that the story is not
# published) and never more than four (extra sentences are
# dropped, never padded).
MIN_EXPLANATORY_SENTENCES = 2
MAX_EXPLANATORY_SENTENCES = 4


def telegram_visible_len(text):
    """Characters Telegram counts toward the 4096 limit."""
    text = text or ""
    text = HTML_ENTITY_RE.sub(
        "x",
        text
    )
    double = len(
        CJK_RE.findall(text)
    ) + len(
        EMOJI_RE.findall(text)
    )
    return len(text) + double


def truncate_by_char_limit(
    text,
    limit,
    ellipsis="\u2026"
):
    """Truncate to a Telegram character budget.

    CJK-safe: slicing happens on the unified-string
    boundary, keeping the last glyph whole.
    """
    if telegram_visible_len(text) <= limit:
        return text

    if not text:
        return text

    if limit < 8:
        return text[:limit]

    budget = limit - 1

    char_total = 0
    cut = len(text)

    for i, ch in enumerate(text):
        char_total += 2 if (
            CJK_RE.match(ch)
            or EMOJI_RE.match(ch)
        ) else 1

        if char_total > budget:
            cut = i
            break

    return text[:cut].rstrip() + ellipsis


def escape_html(text):
    # Unescape source HTML entities first (including
    # double-encoded quotation entities) so literal
    # &amp; / &#039; / &lt; / &gt; / &quot; / &apos; never
    # reach Telegram, then escape exactly once. Raw text
    # like "can&#039;t" renders as "can't"; "&amp;" as "&".
    # Quotation marks are left as literal characters:
    # Telegram's HTML parser accepts them in text content,
    # and escaping them would leak raw "&quot;" artifacts
    # into the message. Ampersands, angle brackets and the
    # apostrophe require no special handling for quotes.
    return (
        decode_html_entities(str(text or ""))
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def escape_html_attr(text):
    """Escape for an HTML attribute value (e.g. the Read
    More href). Unlike escape_html, source entities are
    never decoded: the URL is never modified, only escaped
    so the attribute stays well-formed."""
    return (
        str(text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def sentence_safe_truncate(text, limit):
    """Cut at the last sentence boundary within limit.

    Falls back to a word boundary only when the text is a
    single oversized sentence. The fallback never cuts a
    word in half.
    """
    if telegram_visible_len(text) <= limit:
        return text

    chunks = text.split(". ")

    if len(chunks) <= 1:
        return truncate_by_char_limit(
            text,
            limit,
        )

    result = chunks[0]
    found = False

    for chunk in chunks[1:]:
        candidate = result + ". " + chunk

        if telegram_visible_len(candidate) <= limit:
            result = candidate
            found = True
        else:
            break

    if telegram_visible_len(result) <= limit:
        return result

    if not found and len(chunks) == 2:
        return truncate_by_char_limit(
            text,
            limit,
        )

    return result


def render_source(source):
    return (
        "\U0001F4F0 Source: "
        + escape_html(source or "Unknown")
    )


def render_read_more(url):
    if not url:
        return None

    return (
        "\U0001F517 <a href=\""
        + escape_html_attr(url)
        + "\">Read the full report</a>"
    )


def _collect_explanatory_sentences(
    texts,
    headline,
    max_sentences=MAX_EXPLANATORY_SENTENCES,
):
    """Filter source sentences into the final body.

    Applies the exact same editorial filters as the briefing
    pipeline (boilerplate removal, filler rejection,
    headline-paraphrase rejection, dedup, near-duplicate
    collapse) so the body never repeats the headline,
    generic boilerplate, or an earlier sentence. Sentences
    are kept in source order and capped at max_sentences:
    a story simply carries fewer sentences when it has less
    information, never padding to a fixed length.
    """
    kept = []
    seen_keys = set()

    for text in texts or []:

        # Reject whole-sentence junk (follow @handle, "please
        # refresh", "this breaking news story is being
        # updated") before boilerplate stripping hides the
        # trigger phrase.
        if is_filler(text):
            continue

        sentence = clean_sentence_text(text)

        if not sentence:
            continue

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

        if key in seen_keys:
            continue

        if any(
            is_near_duplicate(
                {"text": sentence},
                {"text": kept_sentence},
            )
            for kept_sentence in kept
        ):
            continue

        seen_keys.add(key)
        kept.append(sentence)

        if len(kept) >= max_sentences:
            break

    return kept


def _render_final_message(
    label,
    headline,
    sentences,
    source,
    url,
    max_chars,
):
    """Render the final message from prepared parts.

    Parts are joined with blank lines; nothing decorative is
    added. The label, headline and footer are never dropped.
    When the body exceeds the budget, whole sentences are
    removed from the end (down to the minimum of two) before
    any mid-sentence cut is attempted.
    """
    parts = []

    if label:
        parts.append(escape_html(label))

    if headline:
        parts.append(
            "<b>" + escape_html(headline) + "</b>"
        )

    paragraph_index = len(parts)

    paragraph = " ".join(sentences)

    parts.append(escape_html(paragraph))
    parts.append(render_source(source))

    read_more = render_read_more(url)

    if read_more:
        parts.append(read_more)

    body = "\n\n".join(parts)

    while (
        telegram_visible_len(body) > max_chars
        and len(sentences) > MIN_EXPLANATORY_SENTENCES
    ):
        sentences.pop()
        parts[paragraph_index] = escape_html(
            " ".join(sentences)
        )
        body = "\n\n".join(parts)

    if telegram_visible_len(body) > max_chars:
        parts[paragraph_index] = sentence_safe_truncate(
            parts[paragraph_index],
            max_chars,
        )
        body = "\n\n".join(parts)

    return body


def build_briefing_message(item, max_chars):
    """Render the enriched briefing in the final format.

    The body comes from the briefing's full evidence rows
    (briefing["sentences"]), which retain every verbatim
    sentence that survived the pipeline filters - including
    sentences the old format diverted into bullets - so no
    genuine fact is lost. Region lines, bullets, and
    corroboration sections are never rendered.
    """
    briefing = item.get("briefing") or {}

    label = (
        item.get("public_label")
        or item.get("label")
    )

    headline = clean(
        item.get("headline")
        or clean_headline(item.get("title"))
    )

    source = (
        briefing.get("source")
        or item.get("source")
    )

    url = (
        briefing.get("url")
        or item.get("url")
    )

    rows = briefing.get("sentences") or []

    if rows:
        row_texts = [
            row.get("text")
            for row in rows
            if row and row.get("text")
        ]
    else:
        row_texts = (
            (briefing.get("opening") or [])
            + (briefing.get("body") or [])
        )

    sentences = _collect_explanatory_sentences(
        row_texts,
        headline,
    )

    if len(sentences) < MIN_EXPLANATORY_SENTENCES:
        return None

    return _render_final_message(
        label,
        headline,
        sentences,
        source,
        url,
        max_chars,
    )


def build_fallback_message(item, max_chars):
    """Fallback for items without an enriched briefing:
    headline + verbatim summary sentences + footer, in the
    same final format."""
    label = (
        item.get("public_label")
        or item.get("label")
    )

    headline = clean(
        item.get("headline")
        or clean_headline(item.get("title"))
    )

    source = item.get("source")
    url = item.get("url")

    from src.formatter import split_sentences

    sentences = _collect_explanatory_sentences(
        split_sentences(item.get("summary")),
        headline,
    )

    if len(sentences) < MIN_EXPLANATORY_SENTENCES:
        return None

    return _render_final_message(
        label,
        headline,
        sentences,
        source,
        url,
        max_chars,
    )


def build_message(item, cfg, now=None):
    """Build an HTML telegram message from a queue item.

    Returns None if the item has no usable content or fewer
    than two genuinely useful explanatory sentences.
    """
    target = int(
        cfg.get(
            "target_message_chars",
            1500
        )
    )

    max_chars = int(
        cfg.get(
            "max_message_chars",
            3000
        )
    )

    if max_chars < target:
        max_chars = target

    title = clean(
        item.get("headline")
        or item.get("title")
    )

    if not title and not (
        item.get("briefing")
        or {}
    ).get("sentences"):
        return None

    briefing = item.get("briefing") or {}

    # Minimum-content gate: a story is publishable only when
    # it contains at least TWO meaningful explanatory
    # sentences. Counting is done on the underlying
    # narrative/source-grounded sentences (the briefing
    # rows), never on visual formatting rows: a fact that
    # the old format placed into a bullet still counts
    # exactly once. Headline, source attribution and Read
    # More never count. The existing empty-message
    # protection (no usable content at all) remains in
    # place.
    if briefing:
        narrative = "\n\n".join(
            (r.get("text") or "")
            for r in (briefing.get("sentences") or [])
        )
    else:
        narrative = "\n\n".join(
            [
                item.get("summary")
            ]
        )

    if count_meaningful_sentences(
        narrative,
        title,
    ) < MIN_EXPLANATORY_SENTENCES:
        return None

    if item.get("briefing"):
        body = build_briefing_message(
            item,
            max_chars,
        )
    else:
        body = build_fallback_message(
            item,
            max_chars,
        )

    if body is None:
        return None

    return {
        "text": body,
        "parse_mode": "HTML",
    }
