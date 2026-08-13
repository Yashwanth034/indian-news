"""Telegram media attachment helpers.

Finds ONE suitable image (preferred) or video for a story's
article page and downloads it for attachment to the Telegram
post. Everything in this module is best-effort: any failure
returns None and the caller publishes the text-only post.

Selection rules
- og:image / twitter:image / link[rel=image_src] are the
  trusted primary images and are tried first.
- Remaining <img> tags are ranked by size.
- Videos (og:video / <video>) are used only when no suitable
  image exists; an image is always preferred.
- Logos, icons, avatars, tracking pixels, ad banners, tiny
  thumbnails, sidebar/related images and other non-primary
  media are rejected via URL and container heuristics plus
  minimum size checks.

Robustness / policy
- Only public, page-linked media URLs are fetched with a
  normal User-Agent; access controls and paywalls are never
  bypassed.
- Downloads respect timeouts and a size cap. A failed,
  oversized, or wrong-type download simply means no media.
- Image dimensions are parsed from the file header
  (JPEG/PNG/GIF/WebP) without any image library.
"""
import re
from dataclasses import dataclass
from urllib.parse import urljoin

import requests
from lxml import html as lhtml

# Telegram caption limit (characters) for sendPhoto/sendVideo.
TELEGRAM_CAPTION_MAX = 1024

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# URL tokens that indicate a non-primary media resource.
URL_TOKEN_RE = re.compile(
    r"\b(logo|logos|favicon|pixel|spacer|placeholder|transparent|"
    r"sprite|avatar|watermark|track|tracking|tracker|analytics|banner|"
    r"advert|advertisements?|ads?|icons?|thumbnails?|thumbs?|badges?|"
    r"captcha|emoji|social|share|btn|button|1x1|blank|empty|newsletter)\b",
    re.IGNORECASE,
)

# Container class/id tokens that indicate chrome around the
# article (header/nav/ad/sidebar), not the story's own image.
CONTAINER_TOKEN_RE = re.compile(
    r"\b(logo|header|footer|nav|menu|sidebar|ad|ads|advert|banner|"
    r"avatar|icon|related|comment|comments|subscribe|promo|sponsor|"
    r"widget|share|author|byline|social|button|btn|cooki)\b",
    re.IGNORECASE,
)

DATA_SCHEMES = (
    "data:",
    "about:",
    "javascript:",
    "file:",
    "mailto:",
    "blob:",
)

_EXTENSIONS = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/gif": "gif",
    "image/webp": "webp",
    "image/avif": "avif",
    "image/bmp": "bmp",
    "image/tiff": "tiff",
    "video/mp4": "mp4",
    "video/webm": "webm",
    "video/quicktime": "mov",
    "video/x-msvideo": "avi",
    "video/x-matroska": "mkv",
}


@dataclass
class MediaCandidate:
    kind: str
    url: str
    source: str = ""
    width: int = None
    height: int = None
    container: str = ""


@dataclass
class MediaAttachment:
    kind: str
    data: bytes
    content_type: str
    filename: str


def _ua(cfg):
    return (
        cfg.get("user_agent")
        or DEFAULT_USER_AGENT
    )


def _clean_url(url, base):
    if not url:
        return None

    url = url.strip()

    if not url or url.lower().startswith(DATA_SCHEMES):
        return None

    return urljoin(base, url)


def _int_attr(value):
    if not value:
        return None

    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _best_srcset(srcset):
    best_url = None
    best_width = -1

    for part in (srcset or "").split(","):
        tokens = part.split()

        if not tokens:
            continue

        url = tokens[0]

        if not url:
            continue

        width = -1

        for token in tokens[1:]:
            match = re.match(r"^(\d+)w$", token)

            if match:
                width = int(match.group(1))

        if best_url is None:
            best_url = url
            best_width = width

        if width > best_width:
            best_url = url
            best_width = width

    return best_url


def _img_url(img, base):
    srcset = (
        img.get("srcset")
        or img.get("data-srcset")
    )

    if srcset:
        url = _best_srcset(srcset)

        if url:
            return _clean_url(url, base)

    for attr in (
        "src",
        "data-src",
        "data-original",
        "data-lazy-src",
        "data-lazy",
        "data-url",
        "data-image",
        "data-echo",
    ):
        value = img.get(attr)

        if value:
            return _clean_url(value, base)

    return None


def _container_tokens(el):
    tokens = []
    node = el.getparent()

    while node is not None:
        cls = (node.get("class") or "").lower()
        nid = (node.get("id") or "").lower()
        tokens.extend(
            re.findall(r"[a-z0-9_-]+", cls + " " + nid)
        )
        node = node.getparent()

        if len(tokens) > 100:
            break

    return " ".join(tokens)


def _is_rejected_url(url):
    if not url:
        return True

    low = url.lower()

    if low.startswith(DATA_SCHEMES):
        return True

    return bool(URL_TOKEN_RE.search(low))


def _is_rejected_container(container):
    return bool(
        container
        and CONTAINER_TOKEN_RE.search(container)
    )


def _is_suitable(candidate, cfg):
    if _is_rejected_url(candidate.url):
        return False

    if candidate.container and _is_rejected_container(
        candidate.container
    ):
        return False

    if candidate.kind == "image":
        min_width = int(
            cfg.get("image_min_width", 300)
        )
        min_height = int(
            cfg.get("image_min_height", 200)
        )

        if (
            candidate.width is not None
            and candidate.width < min_width
        ):
            return False

        if (
            candidate.height is not None
            and candidate.height < min_height
        ):
            return False

    return True


def extract_candidates(html_text, base_url):
    """Return ordered media candidates from article HTML.

    Ordering is by preference: og:image first, then
    link[rel=image_src], then <img> tags (later ranked by
    size), then videos. Videos are intentionally last so an
    image is preferred whenever one exists.
    """
    candidates = []

    try:
        doc = lhtml.fromstring(html_text)
    except Exception:
        return candidates

    for selector in (
        "og:image",
        "og:image:url",
        "og:image:secure_url",
        "twitter:image",
        "twitter:image:src",
    ):
        for el in doc.xpath(
            "//meta[@property=$s or @name=$s]",
            s=selector,
        ):
            url = _clean_url(
                el.get("content"),
                base_url,
            )

            if url:
                candidates.append(
                    MediaCandidate(
                        "image",
                        url,
                        source=selector,
                    )
                )

    for el in doc.xpath(
        "//link[translate(@rel, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', "
        "'abcdefghijklmnopqrstuvwxyz')='image_src']"
    ):
        url = _clean_url(el.get("href"), base_url)

        if url:
            candidates.append(
                MediaCandidate(
                    "image",
                    url,
                    source="image_src",
                )
            )

    for img in doc.xpath("//img"):
        url = _img_url(img, base_url)

        if not url:
            continue

        candidates.append(
            MediaCandidate(
                "image",
                url,
                source="img",
                width=_int_attr(img.get("width")),
                height=_int_attr(img.get("height")),
                container=_container_tokens(img),
            )
        )

    for selector in (
        "og:video",
        "og:video:url",
        "og:video:secure_url",
    ):
        for el in doc.xpath(
            "//meta[@property=$s]",
            s=selector,
        ):
            url = _clean_url(
                el.get("content"),
                base_url,
            )

            if url:
                candidates.append(
                    MediaCandidate(
                        "video",
                        url,
                        source=selector,
                    )
                )

    for video in doc.xpath("//video"):
        src = video.get("src")

        if src:
            url = _clean_url(src, base_url)

            if url:
                candidates.append(
                    MediaCandidate(
                        "video",
                        url,
                        source="video",
                    )
                )

        for source in video.xpath(".//source"):
            src = source.get("src")

            if src:
                url = _clean_url(src, base_url)

                if url:
                    candidates.append(
                        MediaCandidate(
                            "video",
                            url,
                            source="video source",
                        )
                    )

    return candidates


def select_candidate(candidates, cfg):
    """Pick the best suitable image, else the best video."""
    images = [
        c for c in candidates
        if c.kind == "image"
    ]
    videos = [
        c for c in candidates
        if c.kind == "video"
    ]

    imgs = [
        c for c in images
        if c.source == "img"
    ]

    imgs.sort(
        key=lambda c: -(
            (c.width or 0) * (c.height or 0)
        )
    )

    preferred = (
        [c for c in images if c.source != "img"]
        + imgs
        + videos
    )

    for candidate in preferred:
        if _is_suitable(candidate, cfg):
            return candidate

    return None


def fetch_article(url, cfg):
    """Fetch the article page HTML (best-effort)."""
    timeout = int(cfg.get("timeout_seconds", 12))
    max_bytes = int(cfg.get("max_html_bytes", 2097152))

    try:
        with requests.get(
            url,
            headers={"User-Agent": _ua(cfg)},
            timeout=(timeout, timeout),
            stream=True,
        ) as response:
            response.raise_for_status()

            ctype = (
                response.headers.get("Content-Type")
                or ""
            ).lower()

            if ctype and not (
                ctype.startswith("text/")
                or "html" in ctype
                or "xml" in ctype
                or ctype.startswith("application/")
            ):
                return None

            chunks = []
            size = 0

            for chunk in response.iter_content(65536):
                chunks.append(chunk)
                size += len(chunk)

                if size > max_bytes:
                    return None

            return b"".join(chunks).decode(
                "utf-8",
                "replace",
            )
    except Exception:
        return None


def _jpeg_dimensions(data):
    i = 2
    n = len(data)

    while i + 9 < n:
        if data[i] != 0xFF:
            i += 1
            continue

        marker = data[i + 1]

        if marker == 0x01 or 0xD0 <= marker <= 0xD7:
            i += 2
            continue

        length = (
            (data[i + 2] << 8)
            | data[i + 3]
        )

        if marker in (
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        ):
            height = (
                (data[i + 5] << 8)
                | data[i + 6]
            )
            width = (
                (data[i + 7] << 8)
                | data[i + 8]
            )
            return (width, height)

        if length < 2:
            return None

        i += 2 + length

    return None


def _webp_dimensions(data):
    if len(data) < 30:
        return None

    fourcc = data[12:16]

    if fourcc == b"VP8X":
        width = (
            int.from_bytes(
                data[24:27],
                "little",
            )
            + 1
        )
        height = (
            int.from_bytes(
                data[27:30],
                "little",
            )
            + 1
        )
        return (width, height)

    if fourcc == b"VP8L" and len(data) >= 25:
        bits = int.from_bytes(
            data[21:25],
            "little",
        )
        width = (bits & 0x3FFF) + 1
        height = ((bits >> 14) & 0x3FFF) + 1
        return (width, height)

    return None


def image_dimensions(data):
    """Parse (width, height) from an image header, or None.

    Supports PNG, GIF, JPEG and WebP without any image
    decoding library. Unknown formats return None (the
    caller then relies on content-type and size checks).
    """
    if not data:
        return None

    if (
        data.startswith(b"\x89PNG\r\n\x1a\n")
        and len(data) >= 24
    ):
        width = int.from_bytes(data[16:20], "big")
        height = int.from_bytes(data[20:24], "big")
        return (width, height)

    if (
        data[:6] in (b"GIF87a", b"GIF89a")
        and len(data) >= 10
    ):
        return (
            data[6] | (data[7] << 8),
            data[8] | (data[9] << 8),
        )

    if data[:2] == b"\xff\xd8":
        return _jpeg_dimensions(data)

    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return _webp_dimensions(data)

    return None


def _extension_for(content_type):
    return _EXTENSIONS.get(
        content_type,
        "jpg" if content_type.startswith("image/") else "mp4",
    )


def download_media(candidate, cfg):
    """Download and validate a media candidate.

    Returns a MediaAttachment, or None when the download
    fails, the content-type is wrong, the size cap is
    exceeded, or an image is below the minimum dimensions.
    """
    max_bytes = int(cfg.get("max_bytes", 10485760))
    timeout = int(cfg.get("timeout_seconds", 12))

    try:
        with requests.get(
            candidate.url,
            headers={"User-Agent": _ua(cfg)},
            timeout=(timeout, timeout),
            stream=True,
        ) as response:
            response.raise_for_status()

            ctype = (
                response.headers.get("Content-Type")
                or ""
            ).split(";")[0].strip().lower()

            is_image = ctype.startswith("image/")
            is_video = ctype.startswith("video/")

            if not is_image and not is_video:
                return None

            data = b""

            for chunk in response.iter_content(65536):
                data += chunk

                if len(data) > max_bytes:
                    return None
    except Exception:
        return None

    if not data:
        return None

    kind = "photo" if is_image else "video"

    if kind == "photo":
        dims = image_dimensions(data)

        if dims:
            min_width = int(
                cfg.get("image_min_width", 300)
            )
            min_height = int(
                cfg.get("image_min_height", 200)
            )

            if (
                dims[0] < min_width
                or dims[1] < min_height
            ):
                return None

    return MediaAttachment(
        kind,
        data,
        ctype,
        "telegram_media." + _extension_for(ctype),
    )


def build_media_attachment(url, cfg=None):
    """Best-effort full pipeline: fetch -> select -> download.

    Returns a MediaAttachment or None. Never raises for
    media reasons; callers treat None as text-only.
    """
    cfg = cfg or {}

    if not url or not cfg.get("enabled", True):
        return None

    html_text = fetch_article(url, cfg)

    if not html_text:
        return None

    candidates = extract_candidates(html_text, url)

    if not candidates:
        return None

    candidate = select_candidate(candidates, cfg)

    if candidate is None:
        return None

    return download_media(candidate, cfg)
