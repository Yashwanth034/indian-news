"""Normalization/validation layer for collected articles.

Takes raw Articles from ingestion and produces clean, consistent records
for the later relevance/classification stages. Rejects unusable records
with explicit reasons and emits non-fatal warnings for recoverable issues.
"""
import html as html_module
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.ingest.builder import VALID_ROLES
from src.ingest.normalize import canonicalize_url
from src.models.article import Article

MAX_FUTURE_SKEW = timedelta(hours=24)
MAX_CATEGORY_HINTS = 10

SOURCE_ID_PATTERN = re.compile(r"^[a-z0-9-]+$")
HTML_TAG_RE = re.compile(r"<[^>]+>")

# Common " | SiteName" / " - SiteName" title suffixes seen in Indian feeds.
SITE_SUFFIXES = (
    "Mint", "The Hindu", "Times of India", "India Today", "Hindustan Times",
    "Business Standard", "The Indian Express", "The New Indian Express",
    "NDTV", "Deccan Herald", "The Economic Times", "Economic Times",
    "Moneycontrol", "The Print", "ThePrint", "MediaNama", "Entrackr",
    "Bharat Shakti", "Indian Defence Review", "Live Law", "Bar and Bench",
    "Reuters", "BBC News", "BBC News India",
)

SUFFIX_RE = re.compile(
    r"\s*(?:[-|]\s*)(?:%s)\s*$" % "|".join(re.escape(s) for s in SITE_SUFFIXES),
    re.IGNORECASE,
)

LANGUAGE_ALIASES = {
    "english": "en", "eng": "en", "en-in": "en", "en-us": "en", "en-gb": "en",
    "hindi": "hi", "hind": "hi", "bengali": "bn", "tamil": "ta", "telugu": "te",
    "marathi": "mr", "kannada": "kn", "gujarati": "gu", "malayalam": "ml",
}


@dataclass
class ValidationResult:
    article: Optional[Article] = None
    rejected: bool = False
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def normalize_article(
    article: Article,
    *,
    now: Optional[datetime] = None,
    require_english: bool = False,
) -> ValidationResult:
    """Validate and normalize an Article into a clean record."""
    now = now or datetime.now(timezone.utc)
    result = ValidationResult()

    reasons = []
    warnings = []

    # --- source identity ---
    source_id = (article.source_id or "").strip()
    if not source_id or not SOURCE_ID_PATTERN.match(source_id):
        reasons.append(f"source_id invalid: {article.source_id!r}")
    source_name = (article.source_name or "").strip()
    if not source_name:
        reasons.append("source_name empty")
    role = (article.source_role or "").strip()
    if role not in VALID_ROLES:
        reasons.append(f"source_role invalid: {article.source_role!r}")
    tier = _coerce_tier(article.tier, warnings)
    if tier is None:
        reasons.append(f"tier invalid: {article.tier!r}")

    # --- URL / canonical URL ---
    url = canonicalize_url(article.url)
    if not url:
        reasons.append(f"url invalid: {article.url!r}")
    canonical = canonicalize_url(article.canonical_url) if article.canonical_url else url
    if article.canonical_url and not canonical:
        warnings.append(f"canonical_url invalid, falling back to url: {article.canonical_url!r}")
        canonical = url

    # --- title ---
    title = clean_title(article.title)
    if not title:
        reasons.append("title empty")

    # --- language ---
    language = normalize_language(article.language)
    if require_english and language != "en":
        reasons.append(f"language not english: {language!r}")

    if reasons:
        result.rejected = True
        result.reasons = reasons
        result.warnings = warnings
        return result

    # --- text fields ---
    summary = clean_summary(article.summary)
    author = clean_author(article.author)

    # --- dates ---
    published = _normalize_datetime(article.published)
    if published and published > now + MAX_FUTURE_SKEW:
        warnings.append(f"published date implausibly in the future: {published.isoformat()}")
        published = None
    updated = _normalize_datetime(article.updated)
    if updated and published and updated < published:
        warnings.append(f"updated before published, dropping updated: {updated.isoformat()}")
        updated = None

    # --- category hints ---
    hints = normalize_category_hints(article.category_hints)

    result.article = Article(
        source_id=source_id,
        source_name=source_name,
        tier=tier,
        source_role=role,
        url=url,
        canonical_url=canonical,
        title=title,
        summary=summary,
        published=published,
        updated=updated,
        author=author,
        category_hints=hints,
        language=language,
        raw=article.raw,
        fetched_at=article.fetched_at,
    )
    result.warnings = warnings
    return result


# --- field cleaners -------------------------------------------------------


def clean_title(title) -> Optional[str]:
    """Collapse whitespace, strip site suffixes, unescape HTML, trim punctuation."""
    if title is None:
        return None
    text = html_module.unescape(str(title))
    text = _collapse(text)
    text = SUFFIX_RE.sub("", text).strip()
    text = re.sub(r"[\s:|\-–—]+$", "", text).strip()
    return text or None


def clean_summary(summary) -> Optional[str]:
    """Strip HTML, collapse whitespace, unescape entities."""
    if summary is None:
        return None
    text = html_module.unescape(str(summary))
    text = HTML_TAG_RE.sub(" ", text)
    text = _collapse(text)
    return text or None


def clean_author(author) -> Optional[str]:
    """Normalize author name: strip 'By ', collapse whitespace."""
    if author is None:
        return None
    text = _collapse(author)
    text = re.sub(r"^by\s+", "", text, flags=re.IGNORECASE).strip()
    return text or None


def normalize_language(language) -> str:
    """Map language codes/words to a canonical lowercase ISO code."""
    if not language:
        return "en"
    text = str(language).strip().lower()
    base = text.split("-")[0]
    return LANGUAGE_ALIASES.get(text) or LANGUAGE_ALIASES.get(base) or base


def normalize_category_hints(hints) -> list[str]:
    """Lowercase, strip, dedupe (preserving order), cap length."""
    seen = set()
    out = []
    for hint in hints or []:
        if not isinstance(hint, str):
            continue
        text = _collapse(hint).lower()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out[:MAX_CATEGORY_HINTS]


# --- helpers --------------------------------------------------------------


def _coerce_tier(tier, warnings) -> Optional[int]:
    if isinstance(tier, bool):
        return None
    try:
        value = int(tier)
    except (TypeError, ValueError):
        return None
    if value < 1:
        return None
    if not isinstance(tier, int):
        warnings.append(f"tier coerced from {tier!r} to {value}")
    return value


def _normalize_datetime(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    return None


def _collapse(value) -> str:
    return " ".join(str(value).split())
