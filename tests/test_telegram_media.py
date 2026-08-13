"""Unit tests for the telegram media attachment helpers.

Run with:  .venv/bin/python -m pytest src/test_telegram_media.py -q
"""
import pytest

from src.telegram_media import (
    MediaAttachment,
    MediaCandidate,
    build_media_attachment,
    download_media,
    extract_candidates,
    fetch_article,
    image_dimensions,
    select_candidate,
)

MEDIA_CFG = {
    "enabled": True,
    "timeout_seconds": 5,
    "max_bytes": 1048576,
    "max_html_bytes": 1048576,
    "image_min_width": 300,
    "image_min_height": 200,
}


def png_bytes(width=800, height=500):
    return (
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\rIHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + b"\x08\x06\x00\x00\x00"
        + b"\x00\x00\x00\x00"
    )


def jpeg_bytes(width=800, height=500):
    return (
        b"\xff\xd8"
        + b"\xff\xc0\x00\x09\x08"
        + height.to_bytes(2, "big")
        + width.to_bytes(2, "big")
        + b"\x01\x01\x01\x00"
        + b"\xff\xd9"
    )


def gif_bytes(width=800, height=500):
    return (
        b"GIF89a"
        + width.to_bytes(2, "little")
        + height.to_bytes(2, "little")
        + b"\x00\x00\x00"
    )


def webp_bytes(width=800, height=500):
    return (
        b"RIFF"
        + (26).to_bytes(4, "little")
        + b"WEBP"
        + b"VP8X"
        + (10).to_bytes(4, "little")
        + b"\x00" * 4
        + (width - 1).to_bytes(3, "little")
        + (height - 1).to_bytes(3, "little")
        + b"\x00" * 20
    )


class FakeResponse:
    def __init__(
        self,
        body=b"",
        ctype="image/png",
        status=200,
        exc=None,
    ):
        self.headers = {"Content-Type": ctype}
        self.status_code = status
        self._exc = exc
        self._chunks = [
            body[i : i + 65536]
            for i in range(0, len(body), 65536)
        ]

    def raise_for_status(self):
        if self._exc:
            raise self._exc

    def iter_content(self, chunk_size):
        for chunk in self._chunks:
            yield chunk

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def patch_get(monkeypatch, responses):
    responses = list(responses)
    index = [0]

    def fake_get(url, headers=None, timeout=None, stream=None):
        response = responses[index[0]]
        index[0] += 1
        return response

    monkeypatch.setattr(
        "src.telegram_media.requests.get",
        fake_get,
    )


# ---------------------------------------------------------
# extraction
# ---------------------------------------------------------


def test_extract_og_image():
    html = (
        "<html><head><meta property='og:image' "
        "content='https://img.example.com/a.jpg'/></head>"
        "<body></body></html>"
    )
    cands = extract_candidates(html, "https://example.com/s")
    assert len(cands) == 1
    assert cands[0].kind == "image"
    assert cands[0].url == "https://img.example.com/a.jpg"


def test_extract_twitter_image_and_image_src():
    html = (
        "<html><head>"
        "<meta name='twitter:image' "
        "content='https://img.example.com/t.jpg'/>"
        "<link rel='image_src' "
        "href='https://img.example.com/i.jpg'/>"
        "</head></html>"
    )
    cands = extract_candidates(html, "https://example.com/s")
    urls = [c.url for c in cands]
    assert "https://img.example.com/t.jpg" in urls
    assert "https://img.example.com/i.jpg" in urls


def test_extract_img_srcset_picks_largest():
    html = (
        "<html><body><img srcset='"
        "https://img.example.com/small.jpg 400w, "
        "https://img.example.com/big.jpg 1200w'/></body></html>"
    )
    cands = extract_candidates(html, "https://example.com/s")
    imgs = [c for c in cands if c.source == "img"]
    assert imgs
    assert imgs[0].url == "https://img.example.com/big.jpg"


def test_extract_img_lazy_data_src():
    html = (
        "<html><body><img data-src="
        "'https://img.example.com/lazy.png'/></body></html>"
    )
    cands = extract_candidates(html, "https://example.com/s")
    assert any(
        c.url == "https://img.example.com/lazy.png"
        for c in cands
    )


def test_relative_url_resolved():
    html = (
        "<html><head><meta property='og:image' "
        "content='/img/photo.jpg'/></head></html>"
    )
    cands = extract_candidates(html, "https://example.com/news/a")
    assert cands[0].url == (
        "https://example.com/img/photo.jpg"
    )


def test_extract_video_and_select_video_fallback():
    html = (
        "<html><head><meta property='og:video' "
        "content='https://cdn.example.com/v.mp4'/></head>"
        "<body></body></html>"
    )
    cands = extract_candidates(html, "https://example.com/s")
    chosen = select_candidate(cands, MEDIA_CFG)
    assert chosen is not None
    assert chosen.kind == "video"
    assert chosen.url == "https://cdn.example.com/v.mp4"


def test_select_prefers_image_over_video():
    html = (
        "<html><head>"
        "<meta property='og:image' "
        "content='https://img.example.com/a.jpg'/>"
        "<meta property='og:video' "
        "content='https://cdn.example.com/v.mp4'/>"
        "</head></html>"
    )
    cands = extract_candidates(html, "https://example.com/s")
    chosen = select_candidate(cands, MEDIA_CFG)
    assert chosen.kind == "image"
    assert chosen.url == "https://img.example.com/a.jpg"


def test_select_prefers_suitable_img_over_rejected_og():
    html = (
        "<html><head><meta property='og:image' "
        "content='https://img.example.com/logo.png'/></head>"
        "<body><img src='https://img.example.com/photo.jpg' "
        "width='800' height='500'/></body></html>"
    )
    cands = extract_candidates(html, "https://example.com/s")
    chosen = select_candidate(cands, MEDIA_CFG)
    assert chosen is not None
    assert chosen.url == "https://img.example.com/photo.jpg"


def test_no_media_empty_html():
    cands = extract_candidates(
        "<html><body>no media here</body></html>",
        "https://example.com/s",
    )
    assert cands == []
    assert select_candidate(cands, MEDIA_CFG) is None


def test_reject_logo_url():
    assert select_candidate(
        extract_candidates(
            "<html><body><img src="
            "'https://img.example.com/header-logo.png' "
            "width='800' height='500'/></body></html>",
            "https://example.com/s",
        ),
        MEDIA_CFG,
    ) is None


def test_reject_icon_avatar_and_tracking_urls():
    for url in (
        "https://img.example.com/favicon.ico",
        "https://img.example.com/avatar.jpg",
        "https://img.example.com/pixel.gif",
        "https://img.example.com/track.php?id=1",
        "https://img.example.com/ads/banner.jpg",
        "https://img.example.com/1x1.png",
    ):
        cands = extract_candidates(
            "<html><body><img src='"
            + url
            + "' width='800' height='500'/>"
            + "</body></html>",
            "https://example.com/s",
        )
        assert select_candidate(cands, MEDIA_CFG) is None, url


def test_reject_sidebar_container():
    html = (
        "<html><body><div class='sidebar'>"
        "<img src='https://img.example.com/promo.jpg' "
        "width='800' height='500'/></div></body></html>"
    )
    cands = extract_candidates(html, "https://example.com/s")
    assert select_candidate(cands, MEDIA_CFG) is None


def test_reject_author_byline_container():
    html = (
        "<html><body><div class='author byline'>"
        "<img src='https://img.example.com/reporter.jpg' "
        "width='800' height='500'/></div></body></html>"
    )
    cands = extract_candidates(html, "https://example.com/s")
    assert select_candidate(cands, MEDIA_CFG) is None


def test_reject_tiny_image_attrs():
    html = (
        "<html><body><img src='https://img.example.com/t.png' "
        "width='100' height='50'/></body></html>"
    )
    cands = extract_candidates(html, "https://example.com/s")
    assert select_candidate(cands, MEDIA_CFG) is None


def test_reject_data_uri():
    html = (
        "<html><body><img "
        "src='data:image/png;base64,AAAA'/></body></html>"
    )
    cands = extract_candidates(html, "https://example.com/s")
    assert all(
        not c.url.startswith("data:")
        for c in cands
    )


# ---------------------------------------------------------
# image dimensions
# ---------------------------------------------------------


def test_image_dimensions_png():
    assert image_dimensions(png_bytes(800, 500)) == (800, 500)


def test_image_dimensions_jpeg():
    assert image_dimensions(jpeg_bytes(640, 480)) == (640, 480)


def test_image_dimensions_gif():
    assert image_dimensions(gif_bytes(320, 240)) == (320, 240)


def test_image_dimensions_webp():
    assert image_dimensions(webp_bytes(1000, 700)) == (1000, 700)


def test_image_dimensions_unknown_returns_none():
    assert image_dimensions(b"not an image") is None
    assert image_dimensions(b"") is None


# ---------------------------------------------------------
# download / validation
# ---------------------------------------------------------


def test_download_success_png(monkeypatch):
    patch_get(
        monkeypatch,
        [
            FakeResponse(
                body=png_bytes(),
                ctype="image/png",
            )
        ],
    )
    cand = select_candidate(
        extract_candidates(
            "<img src='https://img.example.com/a.png' "
            "width='800' height='500'/>",
            "https://example.com/s",
        ),
        MEDIA_CFG,
    )
    att = download_media(cand, MEDIA_CFG)
    assert isinstance(att, MediaAttachment)
    assert att.kind == "photo"
    assert att.content_type == "image/png"
    assert att.filename == "telegram_media.png"
    assert att.data == png_bytes()


def test_download_video_success(monkeypatch):
    patch_get(
        monkeypatch,
        [
            FakeResponse(
                body=b"fakemp4",
                ctype="video/mp4",
            )
        ],
    )
    cand = MediaCandidate(
        "video",
        "https://cdn.example.com/v.mp4",
        source="og:video",
    )
    att = download_media(cand, MEDIA_CFG)
    assert att.kind == "video"
    assert att.filename == "telegram_media.mp4"


def test_download_rejects_non_media_content_type(monkeypatch):
    patch_get(
        monkeypatch,
        [
            FakeResponse(
                body=b"<html></html>",
                ctype="text/html",
            )
        ],
    )
    cand = MediaCandidate(
        "image",
        "https://img.example.com/a.png",
        source="og:image",
    )
    assert download_media(cand, MEDIA_CFG) is None


def test_download_rejects_too_large(monkeypatch):
    patch_get(
        monkeypatch,
        [
            FakeResponse(
                body=bytes(2000000),
                ctype="image/png",
            )
        ],
    )
    cand = MediaCandidate(
        "image",
        "https://img.example.com/huge.png",
        source="og:image",
    )
    assert download_media(cand, MEDIA_CFG) is None


def test_download_rejects_tiny_dimensions(monkeypatch):
    patch_get(
        monkeypatch,
        [
            FakeResponse(
                body=png_bytes(100, 50),
                ctype="image/png",
            )
        ],
    )
    cand = MediaCandidate(
        "image",
        "https://img.example.com/tiny.png",
        source="og:image",
    )
    assert download_media(cand, MEDIA_CFG) is None


def test_download_failure_returns_none(monkeypatch):
    patch_get(
        monkeypatch,
        [
            FakeResponse(
                exc=OSError("connection refused"),
            )
        ],
    )
    cand = MediaCandidate(
        "image",
        "https://img.example.com/a.png",
        source="og:image",
    )
    assert download_media(cand, MEDIA_CFG) is None


def test_download_empty_body_returns_none(monkeypatch):
    patch_get(
        monkeypatch,
        [
            FakeResponse(
                body=b"",
                ctype="image/png",
            )
        ],
    )
    cand = MediaCandidate(
        "image",
        "https://img.example.com/a.png",
        source="og:image",
    )
    assert download_media(cand, MEDIA_CFG) is None


# ---------------------------------------------------------
# full pipeline
# ---------------------------------------------------------


def test_build_media_attachment_success(monkeypatch):
    page = (
        "<html><head><meta property='og:image' "
        "content='https://img.example.com/a.png'/></head>"
        "<body>article</body></html>"
    )
    patch_get(
        monkeypatch,
        [
            FakeResponse(
                body=page.encode("utf-8"),
                ctype="text/html",
            ),
            FakeResponse(
                body=png_bytes(),
                ctype="image/png",
            ),
        ],
    )
    att = build_media_attachment(
        "https://example.com/news/1",
        MEDIA_CFG,
    )
    assert isinstance(att, MediaAttachment)
    assert att.kind == "photo"
    assert att.data == png_bytes()


def test_build_media_attachment_fetch_failure(monkeypatch):
    patch_get(
        monkeypatch,
        [
            FakeResponse(
                exc=OSError("network down"),
            )
        ],
    )
    assert (
        build_media_attachment(
            "https://example.com/news/1",
            MEDIA_CFG,
        )
        is None
    )


def test_build_media_attachment_no_candidates(monkeypatch):
    patch_get(
        monkeypatch,
        [
            FakeResponse(
                body=b"<html><body>no media</body></html>",
                ctype="text/html",
            )
        ],
    )
    assert (
        build_media_attachment(
            "https://example.com/news/1",
            MEDIA_CFG,
        )
        is None
    )


def test_build_media_attachment_unrelated_media(monkeypatch):
    page = (
        "<html><body><div class='sidebar'>"
        "<img src='https://img.example.com/logo.png' "
        "width='800' height='500'/></div>"
        "<img src='https://img.example.com/pixel.gif' "
        "width='800' height='500'/></body></html>"
    )
    patch_get(
        monkeypatch,
        [
            FakeResponse(
                body=page.encode("utf-8"),
                ctype="text/html",
            )
        ],
    )
    assert (
        build_media_attachment(
            "https://example.com/news/1",
            MEDIA_CFG,
        )
        is None
    )


def test_build_media_attachment_disabled(monkeypatch):
    cfg = dict(MEDIA_CFG)
    cfg["enabled"] = False

    def boom(url, headers=None, timeout=None, stream=None):
        raise AssertionError("must not fetch when disabled")

    monkeypatch.setattr(
        "src.telegram_media.requests.get",
        boom,
    )
    assert (
        build_media_attachment(
            "https://example.com/news/1",
            cfg,
        )
        is None
    )


def test_build_media_attachment_no_url():
    assert (
        build_media_attachment("", MEDIA_CFG)
        is None
    )
    assert (
        build_media_attachment(None, MEDIA_CFG)
        is None
    )
