"""Article extraction for thin important Telegram stories.

Extractive-only enrichment: for an important story whose RSS
briefing carries fewer than two useful sentences, fetch the
original article, extract verbatim text with trafilatura, and
let the existing briefing pipeline filter it (filler, headline
paraphrase, duplication, conflict).  Results are cached per
story so each story is fetched at most once per TTL window.

Safety properties (never relaxed):
  - http/https only, no cookies, no credentials, no JS
  - private/loopback/link-local/ULA IPs are rejected, and every
    redirect hop is re-validated (DNS + allowlist + scheme)
  - bounded size (2 MB), bounded time (10 s connect / 20 s
    read / 30 s total), at most 3 redirects
  - domain allowlist derived from config/sources.json
    ``allow_domains`` (the single source of truth for allowed
    article domains)
  - robots.txt is honored via urllib.robotparser
  - paywalls and WAF-blocked pages are never bypassed
  - a failing story never fails the pipeline

Port of the proven WorldNews article_extractor with India
config/source adaptations (see module docstring notes in code).
"""
import ipaddress
import json
import re
import socket
import sqlite3
import threading
import time
import urllib.error
import urllib.request
import urllib.robotparser
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse

import requests
import trafilatura

from courlan import extract_domain
from src.formatter import split_sentences
from src.telegram_briefing import (
    JUST_IN,
    UPDATE,
    clean_sentence_text,
    is_filler,
    is_headline_paraphrase,
)

USER_AGENT = (
    "IndiaNewsBot/1.0 "
    "(briefing enrichment; news RSS pipeline; contact: local)"
)

# Failure statuses stored with the shorter (negative) cache TTL.
NEGATIVE_STATUSES = {
    "http_error",
    "blocked",
    "non_article",
    "paywall",
    "no_text",
    "timeout",
    "too_large",
    "not_html",
    "network_error",
}

OK_STATUS = "ok"

PAYWALL_MARKERS = [
    "to continue reading",
    "subscribe to read",
    "subscriber-only",
    "subscribers only",
    "sign in to continue",
    "subscription required",
    "to read the full",
    "become a subscriber",
    "unlock this article",
]

# Article domains live on a sibling domain of the source's own
# feed domain.  India sources list their full allow_domains in
# config/sources.json (e.g. bbc-india lists bbc.co.uk AND
# bbci.co.uk), so the alias map below is a defensive net only
# and never widens the allowlist beyond the configured domains.
DOMAIN_ALIASES = {
    "bbci.co.uk": "bbc.co.uk",
}

# Aggregator hosts whose URLs are wrappers that only redirect
# to arbitrary publishers; never worth fetching.
HOST_BLOCKLIST = {
    "news.google.com",
}


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Config shape helpers (India config bundle vs flat shape)
# ---------------------------------------------------------------------------

def _section(cfg, name):
    """Read a pipeline config section.

    India loads the validated bundle from config_loader.get_config(),
    which nests pipeline settings under ``cfg["config"]``.  A flat
    (deserialized config.json) shape is also accepted so the bundled
    helpers stay usable from any caller.
    """
    bundle = cfg.get("config")
    if isinstance(bundle, dict) and name in bundle:
        return bundle[name] or {}
    return cfg.get(name) or {}


def _sources(cfg):
    """The source list for allowlist derivation.

    India keeps sources in config/sources.json (``cfg["sources"]``
    key in the bundle, a list under ``sources["sources"]``).  Falls
    back to a plain list or the WorldNews-style ``feeds`` key.
    """
    src = cfg.get("sources")
    if isinstance(src, dict):
        return src.get("sources", []) or []
    if isinstance(src, list):
        return src
    return cfg.get("feeds", []) or []


# ---------------------------------------------------------------------------
# URL classification
# ---------------------------------------------------------------------------

def non_article_url(url, non_article_segments):
    """Whether the URL points to a video / liveblog / listing
    page instead of a plain article (segment-based, so e.g.
    "/news/liveblog/..." is rejected while "/news/world" is not).
    """
    if not url:
        return True
    parsed = urlparse(url)
    if (parsed.hostname or "").lower() in HOST_BLOCKLIST:
        return True
    path = parsed.path or ""
    segments = {
        s.lower()
        for s in path.split("/")
        if s and s != "news"
    }
    blocked = {
        s.strip().lower()
        for s in (non_article_segments or [])
        if isinstance(s, str) and s.strip()
    }
    return bool(segments & blocked)


def _registered_domain(url):
    if not url:
        return None
    try:
        return extract_domain(url)
    except Exception:
        netloc = urlparse(url).netloc.lower()
        if not netloc:
            return None
        parts = netloc.split(":")[0].split(".")
        return ".".join(parts[-2:]) if len(parts) >= 2 else netloc


def feed_domain_allowlist(feeds):
    """Registrable domains of all configured feed URLs, e.g.
    feeds.bbci.co.uk -> bbci.co.uk (plus aliased article
    domains like bbc.co.uk).  None/empty input yields an
    empty allowlist (no fetches allowed)."""
    domains = set()
    for feed in (feeds or []):
        url = (feed or {}).get("url")
        domain = _registered_domain(url)
        if domain:
            domains.add(domain)
            alias = DOMAIN_ALIASES.get(domain)
            if alias:
                domains.add(alias)
    return domains


def source_domain_allowlist(sources):
    """Registrable article domains allowed for fetching.

    India: config/sources.json is the single source of truth and
    each source's ``allow_domains`` drives the allowlist.  Both
    the exact configured value and its registrable form are
    added, so "timesofindia.indiatimes.com" (a subdomain) and
    "indiatimes.com" both map to the registrable "indiatimes.com"
    used by the URL check.

    WorldNews-style feed entries (a URL but no ``allow_domains``)
    contribute their feed URL's registrable domain plus aliases,
    preserving the proven behavior for that shape.
    """
    domains = set()
    for source in (sources or []):
        allowed = (source or {}).get("allow_domains")
        if allowed:
            for entry in allowed:
                if not entry:
                    continue
                entry = str(entry).strip().lower()
                domains.add(entry)
                registered = _registered_domain("https://" + entry)
                if registered and registered != entry:
                    domains.add(registered)
                alias = DOMAIN_ALIASES.get(registered or entry)
                if alias:
                    domains.add(alias)
            continue
        url = (source or {}).get("url")
        domain = _registered_domain(url)
        if domain:
            domains.add(domain)
            alias = DOMAIN_ALIASES.get(domain)
            if alias:
                domains.add(alias)
    return domains


def domain_allowed(url, allowlist):
    if not url:
        return False
    domain = _registered_domain(url)
    if not domain or not allowlist:
        return False
    return domain in allowlist


def _public_ip(hostname):
    """First public IPv4/IPv6 address for the hostname, or None
    when the hostname resolves only to private/loopback/etc."""
    try:
        infos = socket.getaddrinfo(
            hostname, None, type=socket.SOCK_STREAM
        )
    except OSError:
        return None
    for info in infos:
        try:
            ip = ipaddress.ip_address(
                info[4][0].split("%")[0]
            )
        except ValueError:
            continue
        if ip.is_private or ip.is_loopback:
            continue
        if ip.is_link_local or ip.is_unspecified:
            continue
        if ip.is_multicast or ip.is_reserved:
            continue
        if ip.version == 4:
            if (
                ip in ipaddress.ip_network("100.64.0.0/10")
                or ip in ipaddress.ip_network("169.254.0.0/16")
            ):
                continue
        elif ip.is_site_local:
            continue
        return str(ip)
    return None


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

class _PaceLock:
    """Same-domain pacing (at least min_interval between two
    requests to the same host).  The global per-run fetch
    budget lives in enrich_thin_stories."""

    def __init__(self, min_interval):
        self._lock = threading.Lock()
        self._last = {}
        self._min_interval = float(min_interval or 0)

    def wait_for(self, hostname, now_func=time.monotonic):
        with self._lock:
            last = self._last.get(hostname)
            if last is not None:
                remaining = (
                    last + self._min_interval - now_func()
                )
                if remaining > 0:
                    time.sleep(remaining)
            self._last[hostname] = now_func()
            return True


class _RobotCache:
    """Fetch + cache robots.txt parsers per host.

    robots.txt is fetched with the project ``USER_AGENT`` (not
    urllib's default ``Python-urllib/x.x``, which many sources' WAFs
    reject), then parsed with ``urllib.robotparser``.  The fail-safe
    policy matches urllib.robotparser's ``read()``: 401/403 denies
    everything; other 4xx allows everything; a fetch/parse failure
    denies everything (never bypass robots.txt).
    """

    def __init__(self, timeout=10):
        self._cache = {}
        self._timeout = timeout

    def _fetch(self, hostname):
        parser = urllib.robotparser.RobotFileParser()
        parser.set_url(f"https://{hostname}/robots.txt")
        req = urllib.request.Request(
            parser.url,
            headers={"User-Agent": USER_AGENT},
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = resp.read()
        except urllib.error.HTTPError as err:
            if err.code in (401, 403):
                parser.disallow_all = True
            elif err.code >= 400 and err.code < 500:
                parser.allow_all = True
            return parser
        except Exception:
            # Fail-safe: cannot read robots.txt -> do not fetch.
            return None
        parser.parse(body.decode("utf-8", "replace").splitlines())
        return parser

    def allowed(self, hostname, path):
        if hostname not in self._cache:
            self._cache[hostname] = self._fetch(hostname)
        parser = self._cache[hostname]
        if parser is None:
            return False
        return parser.can_fetch(USER_AGENT, path)


def _validate_url(url, allowlist):
    """Static URL checks before any network activity."""
    if not url:
        return False
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        return False
    if not domain_allowed(url, allowlist):
        return False
    if _public_ip(hostname) is None:
        return False
    return True


def _looks_like_html(head):
    head = (head or "").lstrip()[:2048].lower()
    if "<!doctype html" in head or "<html" in head:
        return True
    return "<" in head and not head.lstrip().startswith("{")


def _is_paywall(text, markers):
    if not text:
        return False
    lowered = text.lower()
    return any(m in lowered for m in markers)


def fetch_article(url, cfg, allowlist, robots=None, pace=None):
    """Fetch and extract one article.  Returns
    (status, payload) where payload carries text/title on
    success.  Never raises."""
    payload = {}
    try:
        if not url:
            return ("non_article", payload)
        if non_article_url(
            url, cfg.get("non_article_segments")
        ):
            return ("non_article", payload)
        if not _validate_url(url, allowlist):
            return ("blocked", payload)

        total_timeout = float(
            cfg.get("total_timeout_seconds", 30)
        )
        connect_timeout = float(
            cfg.get("connect_timeout_seconds", 10)
        )
        read_timeout = float(
            cfg.get("read_timeout_seconds", 20)
        )
        max_bytes = int(cfg.get("max_bytes", 2 * 1024 * 1024))
        max_redirects = int(cfg.get("max_redirects", 3))

        if robots is None:
            robots = _RobotCache()
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower()
        if not robots.allowed(hostname, parsed.path or "/"):
            return ("blocked", payload)

        deadline = time.monotonic() + total_timeout
        current = url
        redirect_count = 0

        while True:
            if redirect_count > max_redirects:
                return ("http_error", payload)
            if time.monotonic() > deadline:
                return ("timeout", payload)
            parsed = urlparse(current)
            if parsed.scheme not in ("http", "https"):
                return ("blocked", payload)
            hostname = (parsed.hostname or "").lower()
            if not _validate_url(current, allowlist):
                return ("blocked", payload)
            if not robots.allowed(hostname, parsed.path or "/"):
                return ("blocked", payload)
            if pace is not None:
                if not pace.wait_for(hostname):
                    return ("blocked", payload)

            remaining = max(
                deadline - time.monotonic(), 1
            )
            resp = requests.get(
                current,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": (
                        "text/html,application/xhtml+xml,"
                        "*/*;q=0.8"
                    ),
                },
                timeout=(
                    connect_timeout,
                    min(
                        read_timeout,
                        remaining,
                    ),
                ),
                allow_redirects=False,
                stream=True,
                cookies=None,
            )
            try:
                if resp.status_code in (301, 302, 303, 307, 308):
                    location = resp.headers.get("Location")
                    if not location:
                        return ("http_error", payload)
                    current = (
                        location
                        if location.startswith("http")
                        else urljoin(current, location)
                    )
                    redirect_count += 1
                    continue
                if resp.status_code != 200:
                    return ("http_error", payload)
                content_type = (
                    resp.headers.get("Content-Type") or ""
                ).lower()
                if content_type and (
                    "text/html" not in content_type
                    and "application/xhtml+xml" not in content_type
                ):
                    return ("not_html", payload)
                length_header = resp.headers.get(
                    "Content-Length"
                )
                if (
                    length_header
                    and int(length_header) > max_bytes
                ):
                    return ("too_large", payload)
                chunks = []
                size = 0
                for chunk in resp.iter_content(65536):
                    if time.monotonic() > deadline:
                        return ("timeout", payload)
                    size += len(chunk)
                    if size > max_bytes:
                        return ("too_large", payload)
                    chunks.append(chunk)
                body = b"".join(chunks)
                if not _looks_like_html(body[:2048].decode(
                    "utf-8", "ignore"
                )):
                    return ("not_html", payload)
            finally:
                resp.close()
            break

        html = body.decode("utf-8", "replace")
        text = trafilatura.extract(
            html,
            url=current,
            output_format="txt",
            include_comments=False,
            include_tables=False,
            include_links=False,
            deduplicate=True,
        )

        if not text or not text.strip():
            return ("no_text", payload)

        if len(text.strip()) < 200 or _is_paywall(
            text,
            cfg.get("paywall_markers") or PAYWALL_MARKERS,
        ):
            return ("paywall", payload)

        payload["text"] = text.strip()
        payload["title"] = None
        try:
            meta = trafilatura.extract_metadata(
                html, url=current
            )
            if meta is not None and getattr(meta, "title", None):
                payload["title"] = meta.title.strip()
        except Exception:
            pass
        payload["url"] = current
        return (OK_STATUS, payload)
    except requests.Timeout:
        return ("timeout", payload)
    except requests.ConnectionError:
        return ("network_error", payload)
    except Exception:
        return ("network_error", payload)


# ---------------------------------------------------------------------------
# Sentence extraction (verbatim only)
# ---------------------------------------------------------------------------

JUNK_LINE_MARKERS = [
    "recommended stories",
    "list of",
    "related topics",
    "more top stories",
    "get in touch",
    "- published",
    "you may also like",
    "read more",
    "advertisement",
    "sign up",
    "newsletter",
    "watch:",
    "video:",
]

_SENT_END_RE = re.compile(r"[.!?…][\"'\u201d\u2019)\]]*$")


def article_sentences(
    text, headline, max_sentences=12
):
    """Verbatim sentences from extracted text, cleaned and
    filler-filtered.

    The extracted text is one paragraph per line; lines that
    are not complete sentences (the headline, section
    headers, nav labels, topic tags, recommendation blocks)
    are skipped, and each remaining paragraph is split into
    its sentences.  Headline-paraphrase filtering is also
    applied here and again downstream by the briefing
    pipeline (aggregate_sentences)."""
    out = []
    seen = set()
    for line in (text or "").split("\n"):
        line = line.strip()
        if not line:
            continue
        lowered = line.lower()
        if any(
            marker in lowered
            for marker in JUNK_LINE_MARKERS
        ):
            continue
        if not _SENT_END_RE.search(line):
            continue
        for sentence in split_sentences(line):
            if is_filler(sentence):
                continue
            sentence = clean_sentence_text(sentence)
            if not sentence:
                continue
            if is_filler(sentence):
                continue
            if is_headline_paraphrase(
                sentence, headline
            ):
                continue
            key = re.sub(
                r"[^a-z0-9 ]", " ", sentence.lower()
            ).strip()
            if key in seen:
                continue
            seen.add(key)
            out.append(sentence)
            if max_sentences and len(out) >= max_sentences:
                return out
    return out


def count_useful(summary, title):
    """Number of useful explanatory sentences in a summary
    (the thinness measure used by the enrichment gate)."""
    seen = set()
    useful = 0
    for sentence in split_sentences(summary or ""):
        if is_filler(sentence):
            continue
        sentence = clean_sentence_text(sentence)
        if not sentence:
            continue
        if is_filler(sentence):
            continue
        if is_headline_paraphrase(sentence, title or ""):
            continue
        key = re.sub(
            r"[^a-z0-9 ]", " ", sentence.lower()
        ).strip()
        if key in seen:
            continue
        seen.add(key)
        useful += 1
    return useful


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class ArticleCache:
    """SQLite-backed cache of extraction results.

    Positive results are cached for cache_ttl_hours_ok hours,
    failures for cache_ttl_hours_error hours.  Raw HTML is
    never stored."""

    def __init__(self, db_path):
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS article_cache ("
            " story_id TEXT PRIMARY KEY,"
            " url TEXT NOT NULL,"
            " status TEXT NOT NULL,"
            " text TEXT,"
            " sentences_json TEXT,"
            " title TEXT,"
            " fetched_at TEXT NOT NULL,"
            " ttl_hours INTEGER NOT NULL)"
        )
        self._conn.commit()

    def get(self, story_id, now=None):
        now = now or datetime.now(timezone.utc)
        row = self._conn.execute(
            "SELECT story_id, url, status, text, "
            "sentences_json, title, fetched_at, ttl_hours "
            "FROM article_cache WHERE story_id=?",
            (story_id,),
        ).fetchone()
        if row is None:
            return None
        entry = {
            "story_id": row[0],
            "url": row[1],
            "status": row[2],
            "text": row[3],
            "sentences_json": row[4],
            "title": row[5],
            "fetched_at": row[6],
            "ttl_hours": row[7],
        }
        fetched = datetime.fromisoformat(entry["fetched_at"])
        if fetched.tzinfo is None:
            fetched = fetched.replace(tzinfo=timezone.utc)
        age = now - fetched
        if age.total_seconds() > float(entry["ttl_hours"]) * 3600:
            return None
        return entry

    def set(self, story_id, url, status, text=None,
            sentences=None, title=None, ttl_ok=48,
            ttl_error=24, now=None):
        now = now or datetime.now(timezone.utc)
        ttl = (
            ttl_ok if status == OK_STATUS else ttl_error
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO article_cache ("
            "story_id, url, status, text, sentences_json, "
            "title, fetched_at, ttl_hours) VALUES (?,?,?,?,?,"
            "?,?,?)",
            (
                story_id,
                url,
                status,
                text,
                json.dumps(sentences or [], ensure_ascii=False)
                if sentences is not None else None,
                title,
                now.isoformat(),
                ttl,
            ),
        )
        self._conn.commit()

    def close(self):
        self._conn.close()


def cache_sentences(entry):
    if not entry or not entry.get("sentences_json"):
        return []
    try:
        return json.loads(entry["sentences_json"]) or []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Mass-casualty signal
#
# A narrow, high-precision signal used only by the enrichment
# gate.  A thin HIGH-priority story scoring 60-64 may be fetched
# when it carries strong casualty evidence in a serious context.
#
# A single generic word ("death", "die", "bodies") is never
# sufficient.  The signal requires a mass-casualty phrase
# ("mass grave", "death toll", "killings"), a casualty count of
# two or more ("25 killed", "30 bodies", "dozens dead"), or
# plural victims with a casualty word ("people killed",
# "remains of children and women").
# ---------------------------------------------------------------------------

MASS_CASUALTY_PHRASES = re.compile(
    r"\b(?:mass\s+graves?|death\s+tolls?|massacre|killings)\b",
    re.IGNORECASE,
)

_CASUALTY_VICTIMS = (
    r"(?:people|civilians|children|women|men|villagers|residents|"
    r"soldiers|troops|officers|hostages|families|workers|miners|"
    r"pilots|passengers|students|staff)"
)

CASUALTY_COUNT_RE = re.compile(
    r"(?:(?:more\s+than|at\s+least|nearly|about|around|over|"
    r"up\s+to|almost|roughly|some)\s+)?"
    r"(\d[\d,]*|dozens|scores|hundreds|thousands|multiple|several)\s*"
    r"(?:" + _CASUALTY_VICTIMS + r"\s+)?"
    r"(killed|dead|died|deaths|fatalities|bodies|remains|casualties)\b",
    re.IGNORECASE,
)

PLURAL_CASUALTY_RE = re.compile(
    r"\b" + _CASUALTY_VICTIMS + r"\s+"
    r"(killed|dead|died|deaths|fatalities|bodies)\b",
    re.IGNORECASE,
)

BODIES_CONTEXT_RE = re.compile(
    r"\b(?:bodies|remains)\b"
    r"(?=[^.!?]{0,100}\b(?:grave|graves|exhumed|buried|unearthed|"
    r"recovered|found|children|women|men|civilians|villagers|"
    r"soldiers|troops|victims|killed|dead)\b)",
    re.IGNORECASE,
)

_NO_CASUALTY = re.compile(
    r"\b(?:no|zero|without|no\s+reported|no\s+known|denied|"
    r"no\s+confirmed)\s+"
    r"(?:bodies|remains|killings|fatalities|deaths|dead|killed|"
    r"casualties)\b",
    re.IGNORECASE,
)


def _casualty_count(value):
    """Smallest implied casualty count for a count token.
    Word counts map to conservative minimums; digits parse
    directly."""
    digits = re.sub(r"[^\d]", "", value or "")
    if digits:
        return int(digits)
    return {
        "several": 3,
        "multiple": 2,
        "dozens": 24,
        "scores": 40,
        "hundreds": 200,
        "thousands": 2000,
    }.get((value or "").lower(), 0)


def has_mass_casualty(text):
    """True when text carries strong, context-laden casualty
    evidence (a mass grave, a casualty count of two or more, or
    plural victims paired with a casualty word).  Explicit
    denials ("no fatalities", "zero bodies") never qualify, and
    a bare "death" or "die" is never enough."""
    text = text or ""
    if not text.strip():
        return False
    text = _NO_CASUALTY.sub(" ", text)
    if MASS_CASUALTY_PHRASES.search(text):
        return True
    for match in CASUALTY_COUNT_RE.finditer(text):
        if _casualty_count(match.group(1)) >= 2:
            return True
    if PLURAL_CASUALTY_RE.search(text):
        return True
    if BODIES_CONTEXT_RE.search(text):
        return True
    return False


# ---------------------------------------------------------------------------
# Enrichment driver
# ---------------------------------------------------------------------------

def enrich_thin_stories(
    candidates, cfg, now_dt, cache=None, fetcher=None
):
    """Fetch original articles for important thin stories and
    attach verbatim sentences as candidate["article_sentences"].
    The briefing pipeline then decides what actually renders.

    Groups and primaries are resolved exactly like
    build_telegram_stories, so sentences always land on the
    story that renders the post for the event.

    Returns (enriched_candidates, stats).  cfg is the full India
    config bundle (from config_loader.get_config()); reads the
    "article_extraction" and "telegram" sections plus the sources
    allowlist.  cache/fetcher are injectable for tests.
    """
    stats = {
        "enabled": bool(
            _section(cfg, "article_extraction")
            .get("enabled", True)
        ),
        "candidates": len(candidates),
        "eligible": 0,
        "thin": 0,
        "important": 0,
        "mass_casualty": 0,
        "non_article": 0,
        "domain_blocked": 0,
        "cache_hits": 0,
        "fetched": 0,
        "ok": 0,
        "http_error": 0,
        "blocked": 0,
        "paywall": 0,
        "no_text": 0,
        "timeout": 0,
        "too_large": 0,
        "not_html": 0,
        "network_error": 0,
        "expanded": 0,
        "not_expanded": 0,
        "budget_exhausted": 0,
    }
    art_cfg = _section(cfg, "article_extraction")
    if not art_cfg.get("enabled", True):
        return candidates, stats

    from src.telegram_briefing import (
        build_briefing,
        group_items,
        public_label,
        URGENT_CATEGORIES,
    )

    telegram_cfg = _section(cfg, "telegram")
    just_in_minutes = int(
        telegram_cfg.get("just_in_freshness_minutes", 15)
    )
    allowlist = source_domain_allowlist(_sources(cfg))

    # Resolve groups and primaries exactly like
    # build_telegram_stories, so the enriched candidate is
    # the story that actually renders the post.
    eligible = []
    for group in group_items(candidates):
        primary = sorted(
            group,
            key=lambda x: (
                x.get("score", 0)
                or x.get("priority_score", 0)
                or 0,
                int(bool(x.get("primary_source"))),
                -int(x.get("tier", 4)),
            ),
            reverse=True,
        )[0]

        # Importance gate: IMMEDIATE, JUST IN, score >= 65 or
        # an update.  Additionally, a thin HIGH-priority story
        # scoring 60-64 may qualify when it carries strong
        # mass-casualty evidence in a serious/conflict/disaster/
        # major-event category (the Sudan mass-grave case), as
        # long as the thinness gate below still holds.
        label = public_label(
            primary, just_in_minutes, now_dt
        )
        score = primary.get("score") or 0
        mass_casualty = (
            60 <= score <= 64
            and primary.get("priority_level") == "HIGH"
            and primary.get("category") in URGENT_CATEGORIES
            and has_mass_casualty(
                f"{primary.get('title', '')} "
                f"{primary.get('summary', '')}"
            )
        )
        important = (
            primary.get("priority_level") == "IMMEDIATE"
            or label == JUST_IN
            or score >= 65
            or primary.get("event_status") == "UPDATE"
            or mass_casualty
        )
        if not important:
            continue
        if mass_casualty:
            stats["mass_casualty"] += 1
        stats["important"] += 1

        # Thinness gate: fewer than two useful sentences in
        # the event briefing (the same content the message
        # renders, without any article expansion).
        briefing = build_briefing(
            primary,
            group,
            just_in_minutes,
            now_dt,
        )
        if len(briefing["sentences"]) >= 2:
            continue
        stats["thin"] += 1
        eligible.append((primary, group))

    fetcher = fetcher or fetch_article
    max_fetches = int(art_cfg.get("max_fetches_per_run", 15))
    pace = _PaceLock(
        art_cfg.get("min_domain_interval_seconds", 1),
    )
    robots = _RobotCache()

    expanded = []
    stats["eligible"] = len(eligible)
    fetched_count = 0

    def _enrich_one(cand):
        """Best-effort article enrichment for one event candidate.

        Attaches cand["article_sentences"] (>= 2 useful sentences)
        on success and returns True; otherwise records the failure
        and returns False.  Never raises.
        """
        nonlocal fetched_count
        url = cand.get("url")
        headline = cand.get("title") or ""
        if non_article_url(
            url, art_cfg.get("non_article_segments")
        ):
            stats["non_article"] += 1
            return False
        if not domain_allowed(url, allowlist):
            stats["domain_blocked"] += 1
            return False

        try:
            entry = cache.get(
                cand.get("story_id"), now=now_dt
            ) if cache else None
        except Exception:
            # Cache failure must never fail the pipeline; degrade
            # to a fetch exactly like a cache miss.
            entry = None
        if entry is not None:
            stats["cache_hits"] += 1
            if entry["status"] == OK_STATUS:
                sentences = cache_sentences(entry)
                if len(sentences) >= 2:
                    cand["article_sentences"] = sentences
                    stats["expanded"] += 1
                    expanded.append(cand)
                    return True
                stats["not_expanded"] += 1
                return False
            elif entry["status"] == "non_article":
                stats["non_article"] += 1
            else:
                stats[entry["status"]] = stats.get(
                    entry["status"], 0
                ) + 1
            return False

        if fetched_count >= max_fetches:
            stats["budget_exhausted"] += 1
            return False
        fetched_count += 1

        try:
            status, payload = fetcher(
                url, art_cfg, allowlist, robots=robots, pace=pace
            )
        except Exception:
            status, payload = ("network_error", {})
        stats["fetched"] += 1
        stats[status] = stats.get(status, 0) + 1

        sentences = []
        if status == OK_STATUS:
            sentences = article_sentences(
                payload.get("text", ""),
                headline,
                int(art_cfg.get("max_article_sentences", 12)),
            )
        try:
            if cache is not None:
                cache.set(
                    cand.get("story_id"),
                    url,
                    status,
                    text=payload.get("text") if status == OK_STATUS
                    else None,
                    sentences=sentences if status == OK_STATUS
                    else None,
                    title=payload.get("title") if status == OK_STATUS
                    else None,
                    ttl_ok=int(
                        art_cfg.get("cache_ttl_hours_ok", 48)
                    ),
                    ttl_error=int(
                        art_cfg.get("cache_ttl_hours_error", 24)
                    ),
                    now=now_dt,
                )
        except Exception:
            # A failing cache never fails the pipeline; the
            # enrichment result still applies for this run.
            pass

        if status == OK_STATUS and len(sentences) >= 2:
            cand["article_sentences"] = sentences
            stats["expanded"] += 1
            expanded.append(cand)
            return True
        stats["not_expanded"] += 1
        return False

    for primary, group in eligible:
        primary_sid = primary.get("story_id")

        if _enrich_one(primary):
            continue

        # The event's primary article could not supply >= 2 useful
        # sentences.  Try the event's other members in deterministic
        # group order (primary first already handled above) and attach
        # the first usable extraction to the primary, so the event is
        # still expandable when a co-member's article is fetchable
        # (e.g. the primary's source blocks bots).  The attribution
        # below and in build_telegram_stories keeps the event's post
        # tied to the primary while the article text is traced to its
        # real source.
        if fetched_count >= max_fetches:
            continue
        for member in group:
            if member.get("story_id") == primary_sid:
                continue
            if _enrich_one(member):
                primary["article_sentences"] = (
                    member["article_sentences"]
                )
                primary["enrichment_source"] = (
                    member.get("source")
                    or member.get("source_id")
                    or member.get("url")
                )
                break

    return candidates, stats