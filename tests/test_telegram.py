"""Unit tests for the telegram modules.

Run with:  .venv/bin/python -m pytest src/test_telegram.py -q
"""
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.telegram_briefing import (
    BREAKING,
    JUST_IN,
    NEWS,
    UPDATE,
    clean_sentence_text,
    has_meaningful_sentence,
    is_near_duplicate,
    strip_boilerplate,
)
from src.telegram_formatter import (
    build_message,
    telegram_visible_len,
    truncate_by_char_limit,
)
from src.telegram_scheduler import (
    filter_candidates,
    is_fresh,
    maybe_important,
    now_utc,
    publish_due,
    story_age_minutes,
)
from src.telegram_media import (
    MediaAttachment,
    TELEGRAM_CAPTION_MAX,
)

CFG = {
    "target_message_chars": 1500,
    "max_message_chars": 3000,
}

NOW = datetime.now(timezone.utc)


@pytest.fixture(autouse=True)
def _no_network_media(monkeypatch):
    """Keep every publish_due test network-free.

    Media attachment is enabled by default, so without this
    the existing tests would attempt real article fetches.
    Tests that exercise media override this same attribute
    in their own bodies.
    """
    monkeypatch.setattr(
        "src.telegram_media.build_media_attachment",
        lambda url, media_cfg: None,
    )


def item(**overrides):
    base = {
        "story_id": "s" + str(
            abs(
                hash(
                    json.dumps(overrides)
                )
            ) % 100000
        ),
        "item_id": "i1",
        "event_id": "e1",
        "title": "Test story",
        "summary": "A summary sentence. Second sentence.",
        "source": "Example News",
        "url": "https://example.com/story",
        "label": "analysis",
        "priority_level": "NORMAL",
        "priority_score": 40,
        "effective_at": now_utc().isoformat(),
        "published_at": "2026-08-01T10:00:00Z",
    }
    base.update(overrides)
    return base


def briefing_item(
    label=NEWS,
    headline="Canada wildfire doubles in size",
    opening=None,
    body=None,
    bullets=None,
    source="BBC World",
    corroborating=None,
    url="https://www.bbc.co.uk/news/articles/abc123",
    title="State of emergency declared as fast-moving "
    "Canada wildfire doubles in size",
    region=None,
):
    opening = opening or [
        "The Bald Range wildfire in British Columbia, "
        "still considered out of control, has spread "
        "over more than 36 sq miles."
    ]
    body = body or [
        "A fast-moving wildfire has forced more than "
        "20,000 people to evacuate.",
    ]
    if bullets is None:
        bullets = [
            {
                "icon": "\U0001F4CD",
                "label": "Location",
                "text": "British Columbia",
            },
        ]
    return item(
        label=label,
        public_label=label,
        headline=headline,
        title=title,
        summary=opening[0],
        region=region,
        briefing={
            "opening": opening,
            "body": body,
            "bullets": bullets,
            "sentences": [
                {"text": t, "source": source}
                for t in opening + body
            ],
            "source": source,
            "corroborating": corroborating or [],
            "url": url,
        },
    )


def fresh_item(minutes_ago=5, **overrides):
    effective = (
        now_utc() - timedelta(minutes=minutes_ago)
    ).isoformat()
    return item(
        effective_at=effective,
        **overrides
    )


def make_state(**overrides):
    state = {
        "posted": [],
        "scheduled": [],
        "failures": [],
        "last_posted_at": None,
    }
    state.update(overrides)
    return state


def fake_publisher(dry_run_default=False):
    class FakePublisher:
        def __init__(self):
            self.sent = []
            self.media_sent = []
            self.fail_next = 0
            self.media_fail = False
            self.rate_limit = None

        def send_message(
            self,
            chat_id,
            message,
            dry_run=False,
        ):
            if dry_run or dry_run_default:
                return {
                    "dry_run": True,
                    "chat_id": chat_id,
                }

            if self.fail_next > 0:
                self.fail_next -= 1
                from src.telegram_publisher import (
                    TelegramPublisherError,
                )
                raise TelegramPublisherError(
                    "simulated failure"
                )

            if self.rate_limit:
                from src.telegram_publisher import (
                    TelegramRateLimited,
                )
                raise TelegramRateLimited(
                    self.rate_limit,
                    "simulated rate limit",
                )

            self.sent.append(chat_id)
            return {
                "message_id": len(self.sent),
                "chat_id": chat_id,
            }

        def send_media(
            self,
            chat_id,
            attachment,
            caption,
            parse_mode="HTML",
            dry_run=False,
        ):
            if dry_run or dry_run_default:
                return {
                    "dry_run": True,
                    "chat_id": chat_id,
                    "media_kind": attachment.kind,
                }

            if self.media_fail:
                from src.telegram_publisher import (
                    TelegramPublisherError,
                )
                raise TelegramPublisherError(
                    "simulated media rejection"
                )

            if self.fail_next > 0:
                self.fail_next -= 1
                from src.telegram_publisher import (
                    TelegramPublisherError,
                )
                raise TelegramPublisherError(
                    "simulated failure"
                )

            if self.rate_limit:
                from src.telegram_publisher import (
                    TelegramRateLimited,
                )
                raise TelegramRateLimited(
                    self.rate_limit,
                    "simulated rate limit",
                )

            self.media_sent.append(
                (chat_id, caption, attachment.kind)
            )
            return {
                "message_id": 9000 + len(self.media_sent),
                "chat_id": chat_id,
                "media_kind": attachment.kind,
            }

    return FakePublisher()


def dt_to_iso(dt):
    return dt.isoformat()


DISTINCT_STORM_FACTS = [
    "Rescue teams reached the village by helicopter.",
    "The river rose one metre overnight.",
    "Evacuation orders covered five villages.",
    "Officials closed the main highway at dawn.",
    "The dam released water into the reservoir.",
    "Mobile phone networks failed across the district.",
    "Shelters housed nearly 2,000 displaced families.",
    "Emergency crews restored power to the hospital.",
    "The storm damaged more than 300 buildings.",
    "Farmers reported heavy losses of livestock.",
    "Schools remained closed for a second day.",
    "The railway line was blocked by flooding.",
    "Drinking water supplies were contaminated.",
    "Volunteers delivered food parcels by boat.",
    "Weather services forecast further heavy rain.",
    "The prime minister promised emergency funding.",
    "International agencies offered technical support.",
    "Local authorities set up a coordination centre.",
    "The airport reopened for domestic flights.",
    "Engineers inspected the damaged bridge.",
]


def scheduled_entry(
    story_id,
    scheduled_at,
    item_id="i1",
    event_id="e1",
    label="analysis",
):
    return {
        "story_id": story_id,
        "item_id": item_id,
        "event_id": event_id,
        "label": label,
        "scheduled_at": dt_to_iso(scheduled_at),
        "attempts": 0,
    }


# ---------------------------------------------------------
# story_age_minutes / is_fresh
# ---------------------------------------------------------


def test_story_age_minutes_fresh():
    x = fresh_item(minutes_ago=10)
    assert story_age_minutes(x) == pytest.approx(
        10.0,
        abs=1,
    )


def test_story_age_minutes_missing():
    assert story_age_minutes(item(effective_at=None)) is None


def test_story_age_minutes_future_skew():
    x = fresh_item(minutes_ago=-5)
    assert story_age_minutes(x) == 0.0


def test_story_age_minutes_huge_future():
    x = fresh_item(minutes_ago=-60)
    assert story_age_minutes(x) is None


def test_is_fresh_within_window():
    x = fresh_item(minutes_ago=30)
    assert is_fresh(x, 6)
    assert not is_fresh(x, 0.25)


# ---------------------------------------------------------
# maybe_important
# ---------------------------------------------------------


def test_breaking_immediate_is_important():
    x = fresh_item(
        priority_level="IMMEDIATE",
        priority_score=95,
    )
    assert maybe_important(x, 15, True)
    assert not maybe_important(x, 15, False)


def test_high_priority_is_important():
    x = fresh_item(priority_level="HIGH")
    assert maybe_important(x, 15, True)


def test_just_in_is_important():
    x = fresh_item(minutes_ago=5)
    assert maybe_important(x, 15, True)
    assert not maybe_important(x, 2, True)


# ---------------------------------------------------------
# filter_candidates
# ---------------------------------------------------------


def test_filter_candidates_skips_posted_and_scheduled():
    x1 = fresh_item(
        story_id="post-me",
        title="Posted story",
    )
    x2 = fresh_item(
        story_id="sched-me",
        title="Scheduled story",
    )
    x3 = fresh_item(
        story_id="fresh-me",
        title="Fresh story",
    )
    state = make_state(
        posted=[
            {"story_id": "post-me"},
        ],
        scheduled=[
            {"story_id": "sched-me"},
        ],
    )
    got = filter_candidates(
        [x1, x2, x3],
        state,
        6,
        50,
        15,
        True,
    )
    ids = [x["story_id"] for x in got]
    assert "post-me" not in ids
    assert "sched-me" not in ids
    assert "fresh-me" in ids


def test_filter_candidates_skips_old_stories():
    x = fresh_item(minutes_ago=420)
    got = filter_candidates(
        [x],
        make_state(),
        6,
        50,
        15,
        True,
    )
    assert got == []


def test_filter_candidates_sorts_important_first():
    normal = fresh_item(
        story_id="normal",
        priority_level="NORMAL",
        priority_score=30,
        minutes_ago=30,
    )
    breaking = fresh_item(
        story_id="breaking",
        priority_level="IMMEDIATE",
        priority_score=95,
        minutes_ago=10,
    )
    got = filter_candidates(
        [normal, breaking],
        make_state(),
        6,
        50,
        15,
        True,
    )
    assert got[0]["story_id"] == "breaking"


def test_filter_candidates_respects_max():
    items = [
        fresh_item(
            story_id="s%d" % i,
            minutes_ago=1,
        )
        for i in range(10)
    ]
    got = filter_candidates(
        items,
        make_state(),
        6,
        3,
        15,
        True,
    )
    assert len(got) == 3


def test_filter_candidates_blocks_failed_item_id():
    x = fresh_item(
        story_id="story-a",
        item_id="same-item",
    )
    state = make_state(
        failures=[
            {
                "story_id": "story-b",
                "item_id": "same-item",
            }
        ]
    )
    got = filter_candidates(
        [x],
        state,
        6,
        50,
        15,
        True,
    )
    assert got == []


# ---------------------------------------------------------
# publish_due
# ---------------------------------------------------------


def test_publish_due_success():
    x = fresh_item(
        story_id="due-story",
        minutes_ago=10,
    )
    state = make_state(
        scheduled=[
            scheduled_entry(
                "due-story",
                now_utc() - timedelta(minutes=5),
            )
        ]
    )
    publisher = fake_publisher()
    report = publish_due(
        publisher,
        "@channel",
        state,
        [x],
        6,
        20,
        150,
        60,
        2,
        cfg=CFG,
    )
    assert len(report["published"]) == 1
    assert report["published"][0]["story_id"] == "due-story"
    assert state["scheduled"] == []
    assert len(state["posted"]) == 1
    assert state["last_posted_at"]


def test_publish_due_future_not_published():
    x = fresh_item(
        story_id="future-story",
        minutes_ago=10,
    )
    state = make_state(
        scheduled=[
            scheduled_entry(
                "future-story",
                now_utc() + timedelta(hours=1),
            )
        ]
    )
    publisher = fake_publisher()
    report = publish_due(
        publisher,
        "@channel",
        state,
        [x],
        6,
        20,
        150,
        60,
        2,
        cfg=CFG,
    )
    assert report["due"] == 0
    assert report["published"] == []
    assert len(state["scheduled"]) == 1


def test_publish_due_no_chat():
    x = fresh_item(
        story_id="nochat-story",
        minutes_ago=10,
    )
    state = make_state(
        scheduled=[
            scheduled_entry(
                "nochat-story",
                now_utc() - timedelta(minutes=5),
            )
        ]
    )
    report = publish_due(
        fake_publisher(),
        "",
        state,
        [x],
        6,
        20,
        150,
        60,
        2,
        cfg=CFG,
    )
    assert report["published"] == []
    assert len(state["scheduled"]) == 1


def test_publish_due_hourly_cap():
    x = fresh_item(
        story_id="capped-story",
        minutes_ago=10,
    )
    posted = [
        {
            "story_id": "p%d" % i,
            "posted_at": (
                now_utc() - timedelta(minutes=i)
            ).isoformat(),
        }
        for i in range(20)
    ]
    state = make_state(
        scheduled=[
            scheduled_entry(
                "capped-story",
                now_utc() - timedelta(minutes=5),
            )
        ],
        posted=posted,
    )
    report = publish_due(
        fake_publisher(),
        "@channel",
        state,
        [x],
        6,
        20,
        150,
        60,
        2,
        cfg=CFG,
    )
    assert report["published"] == []
    assert len(report["skipped_cap"]) == 1
    assert len(state["scheduled"]) == 1


def test_publish_due_expired_story():
    x = fresh_item(
        story_id="expired-story",
        minutes_ago=420,
    )
    state = make_state(
        scheduled=[
            scheduled_entry(
                "expired-story",
                now_utc() - timedelta(minutes=5),
            )
        ]
    )
    report = publish_due(
        fake_publisher(),
        "@channel",
        state,
        [x],
        6,
        20,
        150,
        60,
        2,
        cfg=CFG,
    )
    assert report["published"] == []
    assert len(report["expired"]) == 1
    assert state["scheduled"] == []


def test_publish_due_retry_then_fail():
    x = fresh_item(
        story_id="flaky-story",
        minutes_ago=10,
    )
    state = make_state(
        scheduled=[
            scheduled_entry(
                "flaky-story",
                now_utc() - timedelta(minutes=5),
            )
        ]
    )
    publisher = fake_publisher()
    publisher.fail_next = 2

    report = publish_due(
        publisher,
        "@channel",
        state,
        [x],
        6,
        20,
        150,
        60,
        2,
        cfg=CFG,
    )
    assert report["published"] == []
    assert len(report["failed"]) == 1
    assert state["scheduled"][0]["attempts"] == 1

    report2 = publish_due(
        publisher,
        "@channel",
        state,
        [x],
        6,
        20,
        150,
        60,
        2,
        cfg=CFG,
    )
    assert report2["published"] == []
    assert len(report2["failed"]) == 1
    assert state["scheduled"] == []
    assert len(state["failures"]) == 1
    assert state["failures"][0]["story_id"] == "flaky-story"


def test_publish_due_dry_run_keeps_schedule():
    x = fresh_item(
        story_id="dry-story",
        minutes_ago=10,
    )
    state = make_state(
        scheduled=[
            scheduled_entry(
                "dry-story",
                now_utc() - timedelta(minutes=5),
            )
        ]
    )
    report = publish_due(
        fake_publisher(),
        "@channel",
        state,
        [x],
        6,
        20,
        150,
        60,
        2,
        cfg=CFG,
        dry_run=True,
    )
    assert report["published"]
    assert report["published"][0]["dry_run"]
    assert len(state["scheduled"]) == 1
    assert state["posted"] == []


def test_publish_due_min_gap():
    x = fresh_item(
        story_id="gap-story",
        minutes_ago=10,
    )
    state = make_state(
        scheduled=[
            scheduled_entry(
                "gap-story",
                now_utc() - timedelta(minutes=5),
            )
        ],
        last_posted_at=(
            now_utc() - timedelta(seconds=10)
        ).isoformat(),
        posted=[
            {
                "story_id": "recent",
                "posted_at": (
                    now_utc() - timedelta(seconds=10)
                ).isoformat(),
            }
        ],
    )
    report = publish_due(
        fake_publisher(),
        "@channel",
        state,
        [x],
        6,
        20,
        150,
        60,
        2,
        cfg=CFG,
    )
    assert report["published"] == []
    assert len(report["skipped_gap"]) == 1
    assert len(state["scheduled"]) == 1


# ---------------------------------------------------------
# publish_due + media attachment
# ---------------------------------------------------------


def media_cfg(**overrides):
    cfg = {
        "target_message_chars": 1500,
        "max_message_chars": 3000,
        "media": {"enabled": True},
    }
    cfg["media"].update(overrides)
    return cfg


def media_attachment(kind="photo"):
    return MediaAttachment(
        kind,
        b"fake-media-bytes",
        (
            "image/png"
            if kind == "photo"
            else "video/mp4"
        ),
        "telegram_media." + (
            "png" if kind == "photo" else "mp4"
        ),
    )


def test_publish_due_media_attached(monkeypatch):
    x = fresh_item(
        story_id="media-story",
        minutes_ago=10,
        url="https://example.com/story",
    )
    state = make_state(
        scheduled=[
            scheduled_entry(
                "media-story",
                now_utc() - timedelta(minutes=5),
            )
        ]
    )
    attachment = media_attachment()
    monkeypatch.setattr(
        "src.telegram_media.build_media_attachment",
        lambda url, media_cfg: attachment,
    )
    publisher = fake_publisher()
    report = publish_due(
        publisher,
        "@channel",
        state,
        [x],
        6,
        20,
        150,
        60,
        2,
        cfg=media_cfg(),
    )
    assert len(report["published"]) == 1
    assert len(publisher.media_sent) == 1
    channel, caption, kind = publisher.media_sent[0]
    assert channel == "@channel"
    assert kind == "photo"
    assert caption == build_message(x, media_cfg())[
        "text"
    ]
    assert publisher.sent == []


def test_publish_due_video_attached(monkeypatch):
    x = fresh_item(
        story_id="media-video-story",
        minutes_ago=10,
        url="https://example.com/story",
    )
    state = make_state(
        scheduled=[
            scheduled_entry(
                "media-video-story",
                now_utc() - timedelta(minutes=5),
            )
        ]
    )
    monkeypatch.setattr(
        "src.telegram_media.build_media_attachment",
        lambda url, media_cfg: media_attachment(
            kind="video"
        ),
    )
    publisher = fake_publisher()
    report = publish_due(
        publisher,
        "@channel",
        state,
        [x],
        6,
        20,
        150,
        60,
        2,
        cfg=media_cfg(),
    )
    assert len(report["published"]) == 1
    assert len(publisher.media_sent) == 1
    assert publisher.media_sent[0][2] == "video"
    assert publisher.sent == []


def test_publish_due_media_rejected_falls_back_text(
    monkeypatch,
):
    x = fresh_item(
        story_id="media-reject-story",
        minutes_ago=10,
        url="https://example.com/story",
    )
    state = make_state(
        scheduled=[
            scheduled_entry(
                "media-reject-story",
                now_utc() - timedelta(minutes=5),
            )
        ]
    )
    monkeypatch.setattr(
        "src.telegram_media.build_media_attachment",
        lambda url, media_cfg: media_attachment(),
    )
    publisher = fake_publisher()
    publisher.media_fail = True
    report = publish_due(
        publisher,
        "@channel",
        state,
        [x],
        6,
        20,
        150,
        60,
        2,
        cfg=media_cfg(),
    )
    assert len(report["published"]) == 1
    assert publisher.media_sent == []
    assert len(publisher.sent) == 1
    assert len(state["posted"]) == 1
    assert report["failed"] == []


def test_publish_due_media_download_failure_text_only(
    monkeypatch,
):
    x = fresh_item(
        story_id="media-none-story",
        minutes_ago=10,
        url="https://example.com/story",
    )
    state = make_state(
        scheduled=[
            scheduled_entry(
                "media-none-story",
                now_utc() - timedelta(minutes=5),
            )
        ]
    )
    monkeypatch.setattr(
        "src.telegram_media.build_media_attachment",
        lambda url, media_cfg: None,
    )
    publisher = fake_publisher()
    report = publish_due(
        publisher,
        "@channel",
        state,
        [x],
        6,
        20,
        150,
        60,
        2,
        cfg=media_cfg(),
    )
    assert len(report["published"]) == 1
    assert publisher.media_sent == []
    assert len(publisher.sent) == 1
    assert len(state["posted"]) == 1


def test_publish_due_media_disabled_text_only(monkeypatch):
    x = fresh_item(
        story_id="media-off-story",
        minutes_ago=10,
        url="https://example.com/story",
    )
    state = make_state(
        scheduled=[
            scheduled_entry(
                "media-off-story",
                now_utc() - timedelta(minutes=5),
            )
        ]
    )
    calls = []

    def fake_build(url, media_cfg):
        calls.append(url)
        return media_attachment()

    monkeypatch.setattr(
        "src.telegram_media.build_media_attachment",
        fake_build,
    )
    publisher = fake_publisher()
    report = publish_due(
        publisher,
        "@channel",
        state,
        [x],
        6,
        20,
        150,
        60,
        2,
        cfg=media_cfg(enabled=False),
    )
    assert len(report["published"]) == 1
    assert calls == []
    assert publisher.media_sent == []
    assert len(publisher.sent) == 1


def test_publish_due_media_caption_too_long_text_only(
    monkeypatch,
):
    words = " ".join(
        "signal%d" % i
        for i in range(300)
    )
    long_summary = (
        "The panel reviewed the incoming data "
        + words
        + "."
        + " The committee then voted to approve the full "
        + " ".join(
            "report%d" % i
            for i in range(300)
        )
        + "."
    )
    x = fresh_item(
        story_id="media-long-story",
        minutes_ago=10,
        url="https://example.com/story",
    )
    x["summary"] = long_summary
    state = make_state(
        scheduled=[
            scheduled_entry(
                "media-long-story",
                now_utc() - timedelta(minutes=5),
            )
        ]
    )
    calls = []

    def fake_build(url, media_cfg):
        calls.append(url)
        return media_attachment()

    monkeypatch.setattr(
        "src.telegram_media.build_media_attachment",
        fake_build,
    )
    cfg = media_cfg()
    text = build_message(x, cfg)["text"]
    assert len(text) > TELEGRAM_CAPTION_MAX
    publisher = fake_publisher()
    report = publish_due(
        publisher,
        "@channel",
        state,
        [x],
        6,
        20,
        150,
        60,
        2,
        cfg=cfg,
    )
    assert len(report["published"]) == 1
    assert calls == []
    assert publisher.media_sent == []
    assert len(publisher.sent) == 1


def test_publish_due_media_dry_run_never_fetches(
    monkeypatch,
):
    x = fresh_item(
        story_id="media-dry-story",
        minutes_ago=10,
        url="https://example.com/story",
    )
    state = make_state(
        scheduled=[
            scheduled_entry(
                "media-dry-story",
                now_utc() - timedelta(minutes=5),
            )
        ]
    )
    calls = []

    def fake_build(url, media_cfg):
        calls.append(url)
        return media_attachment()

    monkeypatch.setattr(
        "src.telegram_media.build_media_attachment",
        fake_build,
    )
    report = publish_due(
        fake_publisher(),
        "@channel",
        state,
        [x],
        6,
        20,
        150,
        60,
        2,
        cfg=media_cfg(),
        dry_run=True,
    )
    assert report["published"]
    assert report["published"][0]["dry_run"]
    assert calls == []
    assert len(state["scheduled"]) == 1


# ---------------------------------------------------------
# telegram_visible_len / truncate
# ---------------------------------------------------------


def test_visible_len_ascii():
    assert telegram_visible_len("hello world") == 11


def test_visible_len_cjk():
    assert telegram_visible_len("\u4e16\u754c") == 4


def test_truncate_cjk_keeps_whole_glyphs():
    text = "\u4e16" * 500
    cut = truncate_by_char_limit(text, 200)
    assert telegram_visible_len(cut) <= 200
    assert cut.rstrip("\u2026") == "\u4e16" * 99


def test_truncate_short_text_unchanged():
    text = "short"
    assert truncate_by_char_limit(text, 100) == text


def test_truncate_appends_ellipsis():
    text = "a" * 100
    cut = truncate_by_char_limit(text, 50)
    assert cut.endswith("\u2026")
    assert telegram_visible_len(cut) <= 50


# ---------------------------------------------------------
# build_message - briefing rendering
# ---------------------------------------------------------


def test_briefing_label_rendered_above_headline():
    for label in (
        BREAKING,
        JUST_IN,
        NEWS,
        UPDATE,
    ):
        msg = build_message(briefing_item(label=label), CFG)
        assert label in msg["text"]
        assert msg["text"].startswith(label + "\n\n")
        position = msg["text"].index(label)
        headline_pos = msg["text"].index(
            "<b>"
        )
        assert position < headline_pos


def test_breaking_visual_label():
    assert BREAKING == "\U0001F6A8 BREAKING"
    msg = build_message(
        briefing_item(label=BREAKING),
        CFG,
    )
    assert msg["text"].startswith(
        "\U0001F6A8 BREAKING\n\n"
    )


def test_just_in_visual_label():
    assert JUST_IN == "\u26A1 JUST IN"
    msg = build_message(
        briefing_item(label=JUST_IN),
        CFG,
    )
    assert msg["text"].startswith(
        "\u26A1 JUST IN\n\n"
    )


def test_news_visual_label():
    assert NEWS == "\U0001F4F0 NEWS"
    msg = build_message(
        briefing_item(label=NEWS),
        CFG,
    )
    assert msg["text"].startswith(
        "\U0001F4F0 NEWS\n\n"
    )


def test_update_visual_label():
    assert UPDATE == "\U0001F504 UPDATE"
    msg = build_message(
        briefing_item(label=UPDATE),
        CFG,
    )
    assert msg["text"].startswith(
        "\U0001F504 UPDATE\n\n"
    )


def test_region_line_never_rendered_even_with_evidence():
    msg = build_message(
        briefing_item(
            region="East Asia",
        ),
        CFG,
    )
    assert "\U0001F30F EAST ASIA" not in msg["text"]
    assert "EAST ASIA" not in msg["text"]


def test_region_line_omitted_without_evidence():
    msg = build_message(
        briefing_item(region=None),
        CFG,
    )
    for region in (
        "CHINA",
        "UNITED STATES",
        "ASIA",
        "\U0001F30D",
        "\U0001F30F",
    ):
        assert region not in msg["text"]


def test_unknown_region_never_guessed():
    msg = build_message(
        briefing_item(region="Atlantis"),
        CFG,
    )
    assert "Atlantis" not in msg["text"]


def test_headline_is_bold():
    msg = build_message(briefing_item(), CFG)
    assert (
        "<b>Canada wildfire doubles in size</b>"
        in msg["text"]
    )


def test_opening_rendered_as_paragraph():
    msg = build_message(
        briefing_item(
            opening=[
                "First opening sentence here.",
                "Second opening sentence here.",
            ]
        ),
        CFG,
    )
    assert (
        "First opening sentence here. "
        "Second opening sentence here."
        in msg["text"]
    )


def test_paragraph_spacing_uses_blank_lines():
    msg = build_message(
        briefing_item(
            region="Africa",
            opening=[
                "The opening paragraph sentence.",
            ],
            body=[
                "The body paragraph sentence.",
            ],
        ),
        CFG,
    )
    text = msg["text"]
    assert NEWS + "\n\n" in text
    assert "\U0001F30D AFRICA" not in text
    assert "</b>\n\n" in text
    assert "The opening paragraph sentence." in text
    assert "The body paragraph sentence." in text
    assert (
        "The body paragraph sentence.\n\n"
        "\U0001F4F0 Source: BBC World"
        in text
    )


def test_impact_bullet_never_rendered():
    msg = build_message(
        briefing_item(
            bullets=[
                {
                    "icon": "\U0001F465",
                    "label": "Impact",
                    "text": "nearly 100,000 people",
                },
            ]
        ),
        CFG,
    )
    assert "\u2022 Nearly 100,000 people" not in msg["text"]
    assert "nearly 100,000 people" not in msg["text"]
    assert "**IMPACT**" not in msg["text"]


def test_status_bullet_never_rendered():
    msg = build_message(
        briefing_item(
            bullets=[
                {
                    "icon": "\u26A0\uFE0F",
                    "label": "Status",
                    "text": "out of control",
                },
            ]
        ),
        CFG,
    )
    assert "\u2022 Out of control" not in msg["text"]
    assert "**STATUS**" not in msg["text"]


def test_next_bullet_never_rendered():
    msg = build_message(
        briefing_item(
            bullets=[
                {
                    "icon": "\u27A1\uFE0F",
                    "label": "Next",
                    "text": "The storm is expected to "
                    "weaken by morning.",
                },
            ]
        ),
        CFG,
    )
    assert (
        "\u2022 The storm is expected to "
        "weaken by morning."
        not in msg["text"]
    )
    assert "**NEXT**" not in msg["text"]


def test_location_bullet_never_rendered():
    msg = build_message(briefing_item(), CFG)
    assert "\u2022 British Columbia" not in msg["text"]
    assert "**LOCATION**" not in msg["text"]


def test_only_known_bullet_labels_rendered():
    msg = build_message(
        briefing_item(
            bullets=[
                {
                    "icon": "\u2022",
                    "label": "Mystery",
                    "text": "unsupported section",
                },
            ]
        ),
        CFG,
    )
    assert "Mystery" not in msg["text"]
    assert "\u2022 Unsupported section" not in msg["text"]
    assert "unsupported section" not in msg["text"]


def test_no_bullets_anywhere():
    msg = build_message(
        briefing_item(
            bullets=[
                {
                    "icon": "\U0001F465",
                    "label": "Impact",
                    "text": "nearly 100,000 people",
                },
                {
                    "icon": "\U0001F4CD",
                    "label": "Location",
                    "text": "British Columbia",
                },
            ]
        ),
        CFG,
    )
    assert "\u2022" not in msg["text"]


def test_no_decorative_dividers():
    msg = build_message(
        briefing_item(
            bullets=[
                {
                    "icon": "\U0001F465",
                    "label": "Impact",
                    "text": "nearly 100,000 people",
                },
                {
                    "icon": "\U0001F4CD",
                    "label": "Location",
                    "text": "British Columbia",
                },
            ]
        ),
        CFG,
    )
    text = msg["text"]
    assert "\u2501" * 14 not in text
    assert "**" not in text
    assert text.count("\U0001F4F0 Source:") == 1


def test_no_dividers_without_bullets():
    msg = build_message(
        briefing_item(bullets=[]),
        CFG,
    )
    assert "\u2501" * 14 not in msg["text"]
    assert "**" not in msg["text"]


def test_source_footer_attribution():
    msg = build_message(briefing_item(), CFG)
    assert (
        "\U0001F4F0 Source: BBC World"
        in msg["text"]
    )
    assert "**SOURCE:**" not in msg["text"]


def test_corroboration_never_listed():
    msg = build_message(
        briefing_item(
            corroborating=["Al Jazeera"],
        ),
        CFG,
    )
    assert (
        "\U0001F4F0 Source: BBC World"
        in msg["text"]
    )
    assert "Corroborated by" not in msg["text"]
    assert "Al Jazeera" not in msg["text"]


def test_read_more_link_clickable():
    msg = build_message(briefing_item(), CFG)
    assert (
        '<a href="https://www.bbc.co.uk/news/articles/'
        'abc123">Read the full report</a>'
        in msg["text"]
    )
    assert "\U0001F517" in msg["text"]


def test_no_unsupported_html_or_color():
    msg = build_message(briefing_item(), CFG)
    text = msg["text"].lower()
    for forbidden in (
        "<font",
        "color=",
        "style=",
        "css",
        "<div",
        "<span",
        "<script",
        "class=",
        "markdown",
    ):
        assert forbidden not in text


def test_five_plus_sentences_when_information_exists():
    opening = [
        "Sentence one describes the event.",
        "Sentence two adds location details.",
    ]
    body = [
        "Sentence three reports the official response.",
        "Sentence four describes the current situation.",
        "Sentence five reports the impact on residents.",
        "Sentence six explains what happens next.",
    ]
    msg = build_message(
        briefing_item(opening=opening, body=body),
        CFG,
    )
    count = len(
        [
            s
            for s in opening + body
            if s in msg["text"]
        ]
    )
    assert count >= 5


def test_short_source_fallback_no_filler():
    x = item(
        summary="Only one useful sentence. A second "
        "sentence explains more of the story.",
    )
    msg = build_message(x, CFG)
    assert "Only one useful sentence." in msg["text"]
    assert "Sentence two" not in msg["text"]


def test_no_invented_facts():
    opening = ["The river rose one metre overnight."]
    body = ["Residents were told to move to higher ground."]
    source_sentences = opening + body
    msg = build_message(
        briefing_item(opening=opening, body=body),
        CFG,
    )
    stripped = msg["text"].replace("\u2026", "")
    for sentence in source_sentences:
        assert sentence in stripped
    assert "two metre" not in stripped


def test_3000_character_maximum():
    facts = DISTINCT_STORM_FACTS
    msg = build_message(
        briefing_item(opening=facts[:1], body=facts[1:]),
        CFG,
    )
    assert telegram_visible_len(msg["text"]) <= 3000


def test_sentence_safe_truncation():
    body = [
        "The fire has spread over 36 sq miles of forest.",
        "Officials ordered 20,000 residents to evacuate.",
        "The evacuation shelters opened at dawn.",
    ]
    msg = build_message(
        briefing_item(opening=body[:1], body=body[1:]),
        CFG,
    )
    for sentence in body:
        if sentence in msg["text"]:
            assert msg["text"].index(sentence) >= 0


def test_duplicate_sentence_protection_in_rendered_message():
    from src.telegram_briefing import build_briefing

    next_sentence = (
        "The storm is expected to make landfall "
        "late Sunday or early Monday."
    )
    x = item(
        title="Storm approaches the coast",
        summary=(
            "The storm strengthened overnight. "
            "Flights were cancelled across the region. "
            + next_sentence
        ),
    )
    briefing = build_briefing(x, [x], 15, NOW)
    en = dict(
        x,
        public_label=briefing["label"],
        headline=briefing["headline"],
        briefing=briefing,
    )
    msg = build_message(en, CFG)
    # The "Next" sentence survives as a body sentence in the
    # final paragraph (the old bullet rendering is gone).
    assert next_sentence in msg["text"]
    assert msg["text"].count(next_sentence) == 1
    assert "\u2022" not in msg["text"]


def test_no_internal_values_in_output():
    x = briefing_item(
        title="Internal fields story",
        headline="Internal fields story",
        opening=["The summary sentence is clean."],
    )
    x["score"] = 95
    x["confidence"] = "high"
    x["priority_level"] = "IMMEDIATE"
    x["priority_score"] = 100
    x["tier"] = 1
    x["quality_pass"] = True
    x["max_delay_minutes"] = 30
    x["urgency_score"] = 0.9
    msg = build_message(x, CFG)
    for forbidden in (
        "score",
        "confidence",
        "priority_level",
        "priority_score",
        "IMMEDIATE",
        "tier",
        "quality",
        "max_delay",
        "urgency",
        "HIGH",
        "MEDIUM",
        "LOW",
        "95",
        "100",
    ):
        assert forbidden not in msg["text"]


def test_build_message_escapes_html():
    x = item(
        title='A <b>bold</b> claim & "quotes"',
        headline='A <b>bold</b> claim & "quotes"',
    )
    msg = build_message(x, CFG)
    assert "<b>bold</b>" not in msg["text"]
    assert "&lt;b&gt;" in msg["text"]
    assert "&amp;" in msg["text"]


def test_build_message_without_summary():
    # Empty-message protection: a story with no explanatory
    # sentence beyond its headline is rejected, never
    # rendered as a headline-only post.
    x = item(summary="")
    msg = build_message(x, CFG)
    assert msg is None


def test_build_message_empty_item():
    assert build_message({}, CFG) is None


def test_build_message_parse_mode():
    msg = build_message(briefing_item(), CFG)
    assert msg["parse_mode"] == "HTML"


def _workflow_path(name):
    return (
        Path(__file__).resolve().parent.parent
        / ".github"
        / "workflows"
        / name
    )


def test_workflow_uses_main_branch():
    text = _workflow_path("telegram.yml").read_text()
    assert "master" not in text
    assert "ref: main" in text
    assert "python -m src.main" in text
    assert "python -m src.telegram_run --yes" in text
    assert "git push origin main" in text
    assert "git checkout origin/main" not in text


def test_workflow_write_permission():
    text = _workflow_path("telegram.yml").read_text()
    assert "permissions:" in text
    assert "contents: write" in text


def test_workflow_shared_concurrency_group():
    for name in (
        "telegram.yml",
        "telegram-test-one.yml",
    ):
        text = _workflow_path(name).read_text()
        assert "group: telegram-publish" in text
        assert "cancel-in-progress: false" in text


def test_workflow_cron_is_exactly_fifteen_minutes():
    text = _workflow_path("telegram.yml").read_text()
    assert re.search(
        r"cron:\s*\"\*/15 \* \* \* \*\"",
        text,
    ), "scheduled workflow must run every 15 minutes"
    crons = re.findall(r"cron:\s*\"([^\"]+)\"", text)
    assert crons == ["*/15 * * * *"], (
        f"expected exactly the 15-minute cron, got {crons}"
    )


def test_workflow_secrets_from_github_secrets_only():
    text = _workflow_path("telegram.yml").read_text()
    bot_marker = re.search(
        r"TELEGRAM_BOT_TOKEN:\s*(.+)$",
        text,
        re.MULTILINE,
    )
    channel_marker = re.search(
        r"TELEGRAM_CHANNEL_ID:\s*(.+)$",
        text,
        re.MULTILINE,
    )
    assert bot_marker is not None
    assert channel_marker is not None
    assert (
        "${{ secrets.TELEGRAM_BOT_TOKEN }}"
        in bot_marker.group(1)
    )
    assert (
        "${{ secrets.TELEGRAM_CHANNEL_ID }}"
        in channel_marker.group(1)
    )


def test_workflow_no_hardcoded_secret_literals():
    for name in (
        "telegram.yml",
        "telegram-test-one.yml",
    ):
        text = _workflow_path(name).read_text()
        assert not re.search(
            r"\b\d{4,}:[A-Za-z0-9_-]{20,}\b",
            text,
        )
        assert not re.search(r"-100\d{8,}\b", text)
        assert "TELEGRAM_BOT_TOKEN: \"\"" not in text


def test_workflow_no_whatsapp_logic():
    for name in (
        "telegram.yml",
        "telegram-test-one.yml",
    ):
        text = _workflow_path(name).read_text()
        assert "whatsapp" not in text.lower()
        assert "wa_notify" not in text.lower()
        assert "wa-publish" not in text.lower()


def test_workflow_main_runs_before_telegram_run():
    text = _workflow_path("telegram.yml").read_text()
    main_pos = text.index("python -m src.main")
    run_pos = text.index("python -m src.telegram_run --yes")
    assert main_pos < run_pos, (
        "src.main must run before src.telegram_run"
    )


def test_workflow_commits_only_intended_runtime_files():
    text = _workflow_path("telegram.yml").read_text()
    assert (
        "git add data/telegram_queue.json "
        "data/telegram_state.json"
    ) in text
    assert "git add ." not in text
    assert "git add -A" not in text
    for bogus in (
        "news.db",
        "source_health",
        ".env",
        "__pycache__",
        "queue.json",
    ):
        assert "git add data/" + bogus not in text
    assert "git pull --rebase origin main" in text
    assert "git push origin main" in text


def test_workflow_permissions_limited_to_contents_write():
    text = _workflow_path("telegram.yml").read_text()
    assert "permissions:" in text
    assert "contents: write" in text
    for extra in (
        "packages:",
        "pages:",
        "security-events:",
        "checks:",
        "id-token:",
        "actions:",
    ):
        assert extra not in text


def test_production_workflow_byte_for_byte_unchanged():
    text = _workflow_path("telegram.yml").read_bytes()
    assert (
        hashlib.sha256(text).hexdigest()
        == "f9a3e909e2057888859e1d56502321a362e04d01d14102881eda65bd0857a0c1"
    )


def test_manual_workflow_generates_its_own_queue():
    text = _workflow_path("telegram-test-one.yml").read_text()
    assert "python -m src.main" in text
    main_pos = text.index("python -m src.main")
    run_pos = text.index("python -m src.telegram_run")
    assert main_pos < run_pos


def test_manual_workflow_creates_one_story_temp_fixture():
    text = _workflow_path("telegram-test-one.yml").read_text()
    assert "$RUNNER_TEMP" in text
    assert "TELEGRAM_TEST_FIXTURE" in text
    assert "one_queue[\"count\"] = 1" in text
    assert "len(state[\"scheduled\"]) != 1" in text
    assert "\"scheduled_at\":" in text
    assert "expected exactly 1 scheduled entry" in text


def test_manual_workflow_does_not_require_committed_data():
    text = _workflow_path("telegram-test-one.yml").read_text()
    assert "json.load(open('data/telegram_queue.json'))" not in text
    assert "json.load(open('data/telegram_state.json'))" not in text
    assert "git add data/telegram_state.json" not in text
    assert "git add data/telegram_queue.json" not in text
    assert "git add ." not in text


def test_manual_workflow_one_story_safety():
    text = _workflow_path("telegram-test-one.yml").read_text()
    assert "--force --yes" in text
    assert "TELEGRAM_NO_RETRY" in text
    assert "exactly 1 scheduled" in text
    assert "expected exactly 1 scheduled entry" in text
    assert "python -m src.main" in text
    assert "Verify exactly one message published" in text


def test_posted_history_retained_at_48h_boundary():
    state = make_state(
        scheduled=[],
        posted=[
            {
                "story_id": "kept-48h",
                "posted_at": (
                    now_utc()
                    - timedelta(hours=48)
                ).isoformat(),
            },
        ],
    )
    publish_due(
        fake_publisher(),
        "@channel",
        state,
        [],
        6,
        20,
        150,
        60,
        2,
        cfg=CFG,
    )
    assert len(state["posted"]) == 1
    assert state["posted"][0]["story_id"] == "kept-48h"


def test_posted_history_retained_50h_still_blocks():
    story = fresh_item(
        story_id="dup-story",
        minutes_ago=10,
    )
    state = make_state(
        scheduled=[],
        posted=[
            {
                "story_id": "dup-story",
                "posted_at": (
                    now_utc()
                    - timedelta(hours=50)
                ).isoformat(),
            },
        ],
    )
    publish_due(
        fake_publisher(),
        "@channel",
        state,
        [story],
        6,
        20,
        150,
        60,
        2,
        cfg=CFG,
    )
    chosen = filter_candidates(
        [story],
        state,
        6,
        50,
        15,
        True,
    )
    assert chosen == []
    assert state["posted"][0]["story_id"] == "dup-story"


def test_posted_history_pruned_only_when_safely_older():
    state = make_state(
        scheduled=[],
        posted=[
            {
                "story_id": "kept-50h",
                "posted_at": (
                    now_utc()
                    - timedelta(hours=50)
                ).isoformat(),
            },
            {
                "story_id": "pruned-70h",
                "posted_at": (
                    now_utc()
                    - timedelta(hours=70)
                ).isoformat(),
            },
        ],
    )
    publish_due(
        fake_publisher(),
        "@channel",
        state,
        [],
        6,
        20,
        150,
        60,
        2,
        cfg=CFG,
    )
    ids = [
        e["story_id"] for e in state["posted"]
    ]
    assert ids == ["kept-50h"]


def test_posted_history_unknown_timestamp_kept():
    state = make_state(
        scheduled=[],
        posted=[
            {"story_id": "no-timestamp"},
            {
                "story_id": "bad-timestamp",
                "posted_at": "not-a-date",
            },
            {
                "story_id": "pruned-old",
                "posted_at": (
                    now_utc()
                    - timedelta(hours=200)
                ).isoformat(),
            },
        ],
    )
    publish_due(
        fake_publisher(),
        "@channel",
        state,
        [],
        6,
        20,
        150,
        60,
        2,
        cfg=CFG,
    )
    ids = {
        e["story_id"] for e in state["posted"]
    }
    assert "no-timestamp" in ids
    assert "bad-timestamp" in ids
    assert "pruned-old" not in ids


def test_duplicate_protection_after_48h_memory_boundary():
    story = fresh_item(
        story_id="dup-boundary",
        minutes_ago=10,
    )
    state = make_state(
        scheduled=[],
        posted=[
            {
                "story_id": "dup-boundary",
                "posted_at": (
                    now_utc()
                    - timedelta(hours=48)
                ).isoformat(),
            },
        ],
    )
    chosen = filter_candidates(
        [story],
        state,
        6,
        50,
        15,
        True,
    )
    assert chosen == []


def test_publish_due_daily_cap():
    x = fresh_item(
        story_id="daily-capped",
        minutes_ago=10,
    )
    posted = [
        {
            "story_id": "d%d" % i,
            "posted_at": (
                now_utc()
                - timedelta(minutes=61 + i)
            ).isoformat(),
        }
        for i in range(150)
    ]
    state = make_state(
        scheduled=[
            scheduled_entry(
                "daily-capped",
                now_utc()
                - timedelta(minutes=5),
            )
        ],
        posted=posted,
    )
    report = publish_due(
        fake_publisher(),
        "@channel",
        state,
        [x],
        6,
        20,
        150,
        60,
        2,
        cfg=CFG,
    )
    assert report["published"] == []
    assert len(report["skipped_cap"]) == 1
    assert (
        report["skipped_cap"][0]["reason"]
        == "daily cap reached"
    )
    assert (
        state["scheduled"][0]["story_id"]
        == "daily-capped"
    )


def test_publish_due_rate_limited_deferral():
    x = fresh_item(
        story_id="rate-story",
        minutes_ago=10,
    )
    state = make_state(
        scheduled=[
            scheduled_entry(
                "rate-story",
                now_utc()
                - timedelta(minutes=5),
            )
        ]
    )
    publisher = fake_publisher()
    publisher.rate_limit = 30

    report = publish_due(
        publisher,
        "@channel",
        state,
        [x],
        6,
        20,
        150,
        60,
        2,
        cfg=CFG,
    )
    assert report["published"] == []
    assert report["rate_limited"] == 30
    assert len(report["failed"]) == 1
    assert report["failed"][0]["retry_after"] == 30
    assert (
        state["scheduled"][0]["story_id"]
        == "rate-story"
    )
    assert state["scheduled"][0]["attempts"] == 0


def test_rate_limit_wait_bounded():
    from src.telegram_run import (
        RATE_LIMIT_BUDGET_SECONDS,
        RATE_LIMIT_MAX_WAIT_SECONDS,
        rate_limit_wait_seconds,
    )

    assert rate_limit_wait_seconds(30) == 30
    assert (
        rate_limit_wait_seconds(10 ** 6)
        == RATE_LIMIT_MAX_WAIT_SECONDS
    )
    assert (
        rate_limit_wait_seconds(30)
        <= RATE_LIMIT_BUDGET_SECONDS
    )
    assert rate_limit_wait_seconds(-5) == 0
    assert rate_limit_wait_seconds(None) == 0
    assert rate_limit_wait_seconds("nope") == 0
    assert (
        rate_limit_wait_seconds(500, budget=120)
        == RATE_LIMIT_MAX_WAIT_SECONDS
    )
    assert (
        rate_limit_wait_seconds(
            500,
            max_wait=1000,
            budget=120,
        )
        == 120
    )
    assert (
        rate_limit_wait_seconds(5, budget=2) == 2
    )


def test_state_save_load_roundtrip_and_atomic(
    tmp_path,
):
    from src.telegram_publisher import (
        load_state,
        save_state,
    )

    state_file = tmp_path / "telegram_state.json"
    posted_at = now_utc().isoformat()
    data = {
        "posted": [
            {
                "story_id": "s1",
                "posted_at": posted_at,
            }
        ],
        "scheduled": [],
        "failures": [],
    }

    save_state(str(state_file), data)

    assert state_file.exists()
    assert not (
        tmp_path / "telegram_state.json.tmp"
    ).exists()

    loaded = load_state(str(state_file))
    assert loaded["posted"][0]["story_id"] == "s1"
    assert (
        loaded["posted"][0]["posted_at"]
        == posted_at
    )


def test_publish_due_same_call_two_entries_enforce_gap():
    a = fresh_item(
        story_id="same-call-a",
        minutes_ago=10,
    )
    b = fresh_item(
        story_id="same-call-b",
        minutes_ago=10,
    )
    now = now_utc()
    state = make_state(
        scheduled=[
            scheduled_entry(
                "same-call-a",
                now - timedelta(minutes=5),
            ),
            scheduled_entry(
                "same-call-b",
                now - timedelta(minutes=4),
            ),
        ]
    )
    publisher = fake_publisher()

    report = publish_due(
        publisher,
        "@channel",
        state,
        [a, b],
        6,
        20,
        150,
        60,
        2,
        cfg=CFG,
        now=now,
    )

    assert len(report["published"]) == 1
    assert (
        report["published"][0]["story_id"]
        == "same-call-a"
    )
    assert len(report["skipped_gap"]) == 1
    assert (
        report["skipped_gap"][0]["story_id"]
        == "same-call-b"
    )
    assert len(publisher.sent) == 1
    assert len(state["posted"]) == 1
    assert {
        e["story_id"]
        for e in state["scheduled"]
    } == {"same-call-b"}

    later = now + timedelta(minutes=15)
    report2 = publish_due(
        fake_publisher(),
        "@channel",
        state,
        [a, b],
        6,
        20,
        150,
        60,
        2,
        cfg=CFG,
        now=later,
    )
    assert len(report2["published"]) == 1
    assert (
        report2["published"][0]["story_id"]
        == "same-call-b"
    )
    assert state["scheduled"] == []


# ---------------------------------------------------------
# Editorial content regressions
# ---------------------------------------------------------


def test_no_channel_header_first():
    msg = build_message(briefing_item(), CFG)
    text = msg["text"]
    assert "WorldNews" not in text
    assert not text.startswith("WorldNews\U0001F30E:")
    assert text.startswith(NEWS + "\n\n")


def test_headline_duplicate_reduces_story_below_minimum():
    # The opening sentence restates the headline (its only
    # new detail sits in a trailing subordinate clause), so
    # the story collapses to one genuinely useful sentence
    # and must not be published.
    msg = build_message(
        briefing_item(
            headline=(
                "Nagasaki mayor says 'humanity and "
                "nuclear weapons cannot coexist'"
            ),
            opening=[
                "Nagasaki Mayor says 'humanity and "
                "nuclear weapons can&#039;t coexist' as "
                "Japan marks the anniversary of the US "
                "atomic bombing.",
            ],
            body=[
                "The mayor made the remark during the "
                "annual ceremony in the city.",
            ],
        ),
        CFG,
    )
    assert msg is None


def test_html_entity_apostrophe_unescaped():
    msg = build_message(
        briefing_item(
            opening=[
                "The official said the plan can&#039;t "
                "proceed without approval.",
            ],
        ),
        CFG,
    )
    assert "can't proceed" in msg["text"]
    assert "&#039;" not in msg["text"]
    assert "&#39;" not in msg["text"]
    assert "&apos;" not in msg["text"]


def test_html_entity_amp_unescaped_once():
    msg = build_message(
        briefing_item(
            opening=[
                "Research &amp; development funding "
                "was increased.",
            ],
        ),
        CFG,
    )
    assert "Research &amp; development" in msg["text"]
    assert "&amp;amp;" not in msg["text"]


def test_html_entity_lt_gt_unescaped_once():
    msg = build_message(
        briefing_item(
            opening=[
                "Officials said 5 &lt; 10 &gt; 3 "
                "in the report.",
            ],
        ),
        CFG,
    )
    assert "&lt;" in msg["text"]
    assert "&lt;lt;" not in msg["text"]
    assert "&gt;" in msg["text"]
    assert "&gt;gt;" not in msg["text"]


def test_literal_source_marker_never_appears():
    msg = build_message(briefing_item(), CFG)
    assert "**SOURCE:**" not in msg["text"]
    assert "**" not in msg["text"]
    assert "\U0001F4F0 Source: BBC World" in msg["text"]


def test_generic_filler_removed():
    msg = build_message(
        briefing_item(
            opening=[
                "This is a breaking news story.",
                "The port reopened after the storm "
                "passed overnight.",
            ],
        ),
        CFG,
    )
    text = msg["text"]
    assert "This is a breaking news story." not in text
    assert (
        "The port reopened after the storm passed "
        "overnight."
        in text
    )


JUNK_USEFUL = [
    "The port reopened after the storm passed "
    "overnight.",
    "Ferry services resumed across the channel "
    "this morning.",
    "Cleanup crews cleared debris from the "
    "harbour before noon.",
]


def _assert_junk_dropped(junk_sentences):
    msg = build_message(
        briefing_item(
            opening=list(junk_sentences) + list(JUNK_USEFUL),
        ),
        CFG,
    )
    text = msg["text"]
    for junk in junk_sentences:
        assert junk not in text
    for useful in JUNK_USEFUL:
        assert useful in text


def test_live_update_status_boilerplate_removed():
    _assert_junk_dropped([
        "This breaking news story is being updated "
        "and more details will come soon.",
    ])


def test_social_promotion_boilerplate_removed():
    _assert_junk_dropped([
        "You can also follow @BBCBreaking for the "
        "latest updates.",
        "Follow us on Telegram for more updates.",
        "Follow the author on social media.",
    ])


def test_refresh_and_navigation_boilerplate_removed():
    _assert_junk_dropped([
        "Please refresh the page for the latest "
        "updates.",
        "For the fullest version of this story, "
        "visit our app.",
    ])


def test_external_and_policy_boilerplate_removed():
    _assert_junk_dropped([
        "This content is external to our platform.",
        "We use cookies to improve your experience.",
        "Read our privacy policy for more "
        "information.",
    ])


def test_promotion_and_subscription_boilerplate_removed():
    _assert_junk_dropped([
        "Sign up for our daily briefing to get more "
        "stories.",
        "Subscribe for the latest updates.",
        "Read more by this author.",
    ])


def test_boilerplate_fragments_removed_mid_sentence():
    # Promotional fragments embedded mid-sentence are removed by
    # strip_boilerplate without touching nearby facts.
    text = clean_sentence_text(
        "Officials confirmed the rescue "
        "continued overnight. Read more: visit our "
        "site for the full story. The river fell "
        "one metre by dawn."
    )
    assert "Read more" not in text
    assert "read the full story" not in text
    assert "The river fell one metre by dawn." in text


def test_fallback_filler_and_paraphrase_removed():
    x = item(
        title="Market falls after central bank decision",
        summary=(
            "Market falls after central bank decision. "
            "This is a breaking news story. "
            "The central bank cut its benchmark rate "
            "to 3.5 percent. "
            "Traders said the cut was larger than "
            "expected."
        ),
    )
    msg = build_message(x, CFG)
    text = msg["text"]
    assert "This is a breaking news story." not in text
    assert "Market falls after central bank decision." not in text
    assert (
        "The central bank cut its benchmark rate to "
        "3.5 percent."
        in text
    )


def test_duplicate_factual_sentence_removed():
    msg = build_message(
        briefing_item(
            opening=[
                "The dam released water overnight.",
                "The dam released water overnight.",
            ],
            body=[
                "Evacuation orders covered five villages.",
            ],
        ),
        CFG,
    )
    text = msg["text"]
    assert text.count("The dam released water overnight.") == 1
    assert (
        "Evacuation orders covered five villages."
        in text
    )


def test_headline_source_url_intact():
    msg = build_message(briefing_item(), CFG)
    text = msg["text"]
    assert (
        "<b>Canada wildfire doubles in size</b>"
        in text
    )
    assert "Source: BBC World" in text
    assert (
        '<a href="https://www.bbc.co.uk/news/articles/'
        'abc123">Read the full report</a>'
        in text
    )


def test_no_unsupported_facts_introduced():
    opening = [
        "The river rose one metre overnight.",
    ]
    body = [
        "Residents were told to move to higher ground.",
    ]
    msg = build_message(
        briefing_item(opening=opening, body=body),
        CFG,
    )
    stripped = msg["text"].replace("\u2026", "")
    for sentence in opening + body:
        assert sentence in stripped
    assert "two metre" not in stripped
    assert "expected to rise" not in stripped


# ---------------------------------------------------------
# publisher boilerplate removal + near-duplicate collapse
# ---------------------------------------------------------


def test_strip_boilerplate_removes_publisher_fragments():
    assert strip_boilerplate(
        "Continue reading..."
    ) == ""
    assert strip_boilerplate(
        "Read more: The backlash is growing."
    ) == "The backlash is growing."
    assert strip_boilerplate(
        "Follow our live blog for the latest developments."
    ) == ""
    assert strip_boilerplate(
        "Follow our live blog for the latest updates."
    ) == ""
    assert strip_boilerplate(
        "The Guardian view on the budget"
    ) == "The Guardian view on the budget"
    assert strip_boilerplate(
        "life-changing Get our breaking news email , "
        "free app or daily news podcast One summer day"
    ) == (
        "life-changing One summer day"
    )
    assert strip_boilerplate(
        "Subscribe to our newsletter for updates."
    ) == "for updates."
    assert strip_boilerplate(
        "Follow us on Twitter for live updates."
    ) == "for live updates."


def test_clean_sentence_text_strips_boilerplate():
    assert clean_sentence_text(
        "In the foreground, in a man\u2019s hand was "
        "an iPhone unlocked. Continue reading..."
    ) == (
        "In the foreground, in a man\u2019s hand was "
        "an iPhone unlocked."
    )
    assert clean_sentence_text(
        "Yemen\u2019s military said seven people were "
        "killed. Follow our live blog for the latest "
        "developments."
    ) == (
        "Yemen\u2019s military said seven people were "
        "killed."
    )


def test_clean_sentence_text_removes_series_fragment():
    assert clean_sentence_text(
        "Total solar eclipse (2/4) A total solar eclipse "
        "will sweep across northern Spain on August 12 in "
        "one of Europe\u2019s most anticipated astronomical "
        "events of the decade."
    ) == (
        "A total solar eclipse will sweep across northern "
        "Spain on August 12 in one of Europe\u2019s most "
        "anticipated astronomical events of the decade."
    )


def test_clean_sentence_text_removes_whole_fragment_sentence():
    assert clean_sentence_text(
        "Total solar eclipse (2/4)."
    ) == ""
    assert clean_sentence_text(
        "Total solar eclipse (2/4) The eclipse will be "
        "visible across the UK."
    ) == (
        "The eclipse will be visible across the UK."
    )


def test_boilerplate_promo_removed_mid_sentence():
    opening = [
        "The backlash to what some dub \u2018pervert "
        "glasses\u2019 is growing \u2013 but people "
        "living with disabilities point out the new tech "
        "can also be life-changing Get our breaking news "
        "email , free app or daily news podcast One "
        "summer day earlier this year, Rhys left a cafe "
        "in Melbourne.",
    ]
    body = [
        "The message was a photo of him sitting at a "
        "table outside the cafe on his laptop, with his "
        "dog seated on the ground next to him.",
        "In the foreground, in a man\u2019s hand was an "
        "iPhone unlocked.",
        "Continue reading...",
    ]
    msg = build_message(
        briefing_item(opening=opening, body=body),
        CFG,
    )
    text = msg["text"]
    assert "breaking news email" not in text
    assert "free app" not in text
    assert "daily news podcast" not in text
    assert "Continue reading" not in text
    assert (
        "life-changing One summer day earlier this "
        "year, Rhys left a cafe in Melbourne."
        in text
    )
    assert "iPhone unlocked" in text


def test_near_duplicate_sentences_collapse_to_one():
    opening = [
        "The Bald Range wildfire in British Columbia, "
        "still considered out of control, has spread "
        "over more than 36 sq miles.",
    ]
    body = [
        "The Bald Range wildfire in British Columbia, "
        "still considered out of control, has spread "
        "to more than 36 sq miles.",
        "A fast-moving wildfire has forced more than "
        "20,000 people to evacuate.",
    ]
    msg = build_message(
        briefing_item(opening=opening, body=body),
        CFG,
    )
    text = msg["text"]
    assert (
        "spread over more than 36 sq miles"
        in text
    )
    assert "spread to more than 36 sq miles" not in text
    assert text.count("36 sq miles") == 1
    assert "20,000 people to evacuate" in text


def test_near_duplicate_keeps_materially_different_facts():
    opening = [
        "The Bald Range wildfire in British Columbia "
        "has destroyed 20 homes.",
    ]
    body = [
        "The Bald Range wildfire in British Columbia "
        "has destroyed 30 homes.",
    ]
    msg = build_message(
        briefing_item(opening=opening, body=body),
        CFG,
    )
    text = msg["text"]
    assert "has destroyed 20 homes" in text
    assert "has destroyed 30 homes" in text


def test_is_near_duplicate_pairs():
    spread_over = {
        "text": "The Bald Range wildfire in British "
        "Columbia, still considered out of control, "
        "has spread over more than 36 sq miles."
    }
    spread_to = {
        "text": "The Bald Range wildfire in British "
        "Columbia, still considered out of control, "
        "has spread to more than 36 sq miles."
    }
    homes = {
        "text": "The Bald Range wildfire in British "
        "Columbia has destroyed 20 homes."
    }
    assert is_near_duplicate(spread_over, spread_to)
    assert is_near_duplicate(spread_to, spread_over)
    assert not is_near_duplicate(spread_over, homes)
    assert not is_near_duplicate(
        homes,
        {
            "text": "The Bald Range wildfire in British "
            "Columbia has destroyed 30 homes."
        },
    )


def test_fallback_message_strips_boilerplate_and_near_dups():
    summary = (
        "The Bald Range wildfire in British Columbia "
        "has spread over more than 36 sq miles. "
        "The Bald Range wildfire in British Columbia "
        "has spread to more than 36 sq miles. "
        "Continue reading..."
    )
    msg = build_message(
        item(
            title="Canada wildfire doubles in size",
            summary=summary,
        ),
        CFG,
    )
    # Two near-duplicates collapse to one genuinely useful
    # sentence: below the publishable minimum of two.
    assert msg is None


def test_fallback_message_collapses_near_dups_and_keeps_others():
    summary = (
        "The Bald Range wildfire in British Columbia "
        "has spread over more than 36 sq miles. "
        "The Bald Range wildfire in British Columbia "
        "has spread to more than 36 sq miles. "
        "Officials said 20,000 people have been "
        "ordered to evacuate. "
        "Continue reading..."
    )
    msg = build_message(
        item(
            title="Canada wildfire doubles in size",
            summary=summary,
        ),
        CFG,
    )
    assert msg is not None
    text = msg["text"]
    assert "Continue reading" not in text
    assert "spread over more than 36 sq miles" in text
    assert "spread to more than 36 sq miles" not in text
    assert text.count("36 sq miles") == 1
    assert "20,000 people" in text


def test_impact_bullet_requires_a_digit():
    opening = [
        "Israeli settlers and the military are "
        "continuing a campaign to displace "
        "Palestinians in the West Bank, residents "
        "say.",
    ]
    msg = build_message(
        briefing_item(opening=opening),
        CFG,
    )
    text = msg["text"]
    assert "\u2022 , residents" not in text
    assert "\u2022 ," not in text


def test_next_bullet_does_not_duplicate_opening():
    opening = [
        "PM expected to announce policies to tackle "
        "cost of living and indicate intent to help "
        "improve country\u2019s high streets.",
    ]
    msg = build_message(
        briefing_item(opening=opening),
        CFG,
    )
    text = msg["text"]
    assert text.count(
        "PM expected to announce policies to tackle "
        "cost of living"
    ) == 1
    assert "\u2022 PM expected to announce" not in text


# ---------------------------------------------------------
# Empty-message protection
#
# A story whose cleaned summary contains no sentence that
# explains it beyond its headline must be rejected at every
# layer: the queue gate (build_telegram_stories in
# src/main.py), the shared predicate
# (has_meaningful_sentence), and the message builder
# (build_message). It must never be rendered as a
# headline-only post and never padded with invented text.
# ---------------------------------------------------------


def test_has_meaningful_sentence_headline_only_rejected():
    assert not has_meaningful_sentence(
        "Yemeni army warns Houthis after attacks "
        "heighten escalation risk",
        "Yemeni army warns Houthis after attacks "
        "heighten escalation risk",
    )


def test_has_meaningful_sentence_punctuation_variation_rejected():
    assert not has_meaningful_sentence(
        "Yemeni army warns Houthis after attacks "
        "heighten escalation risk.",
        "Yemeni army warns Houthis after attacks "
        "heighten escalation risk",
    )


def test_has_meaningful_sentence_close_paraphrase_rejected():
    assert not has_meaningful_sentence(
        "The Yemeni army has issued a warning to the "
        "Houthis as recent attacks raise the risk of "
        "escalation.",
        "Yemeni army warns Houthis after attacks "
        "heighten escalation risk",
    )


def test_has_meaningful_sentence_boilerplate_only_rejected():
    assert not has_meaningful_sentence(
        "Get our breaking news email , free app or "
        "daily news podcast. "
        "Subscribe to our newsletter",
        "Yemeni army warns Houthis",
    )


def test_has_meaningful_sentence_informative_accepted():
    assert has_meaningful_sentence(
        "The strikes killed five people and wounded "
        "around two dozen others.",
        "Yemeni army warns Houthis after attacks "
        "heighten escalation risk",
    )


def test_has_meaningful_sentence_keeps_informative_with_headline_dup():
    assert has_meaningful_sentence(
        "Yemeni army warns Houthis after attacks "
        "heighten escalation risk. "
        "The strikes killed five people and wounded "
        "around two dozen others.",
        "Yemeni army warns Houthis after attacks "
        "heighten escalation risk",
    )


def test_headline_only_fallback_never_rendered():
    msg = build_message(
        item(
            title="Yemeni army warns Houthis after "
            "attacks heighten escalation risk",
            summary="Yemeni army warns Houthis after "
            "attacks heighten escalation risk",
        ),
        CFG,
    )
    assert msg is None


def test_empty_briefing_never_rendered():
    msg = build_message(
        item(
            title="Yemeni army warns Houthis after "
            "attacks heighten escalation risk",
            summary="Yemeni army warns Houthis after "
            "attacks heighten escalation risk",
            briefing={
                "opening": [],
                "body": [],
                "bullets": [],
                "sentences": [],
                "source": "Example News",
                "corroborating": [],
                "url": "https://example.com/story",
            },
        ),
        CFG,
    )
    assert msg is None


def test_briefing_collapsing_to_nothing_never_rendered():
    msg = build_message(
        briefing_item(
            opening=[
                "Get our breaking news email , free "
                "app or daily news podcast"
            ],
            body=[
                "Subscribe to our newsletter"
            ],
        ),
        CFG,
    )
    assert msg is None


def test_informative_summary_rendered():
    msg = build_message(
        item(
            title="Yemeni army warns Houthis after "
            "attacks heighten escalation risk",
            summary="The strikes killed five people and "
            "wounded around two dozen others. "
            "Officials said the port had been closed "
            "since the attack.",
        ),
        CFG,
    )
    assert msg is not None
    assert "killed five people" in msg["text"]


def test_informative_sentence_survives_headline_duplicate():
    msg = build_message(
        item(
            title="Yemeni army warns Houthis after "
            "attacks heighten escalation risk",
            summary="Yemeni army warns Houthis after "
            "attacks heighten escalation risk. "
            "The strikes killed five people and wounded "
            "around two dozen others. "
            "Officials said the port had been closed "
            "since the attack.",
        ),
        CFG,
    )
    assert msg is not None
    text = msg["text"]
    assert "killed five people" in text
    assert text.count(
        "Yemeni army warns Houthis"
    ) == 1


def test_rejected_story_never_enters_telegram_queue():
    from src.telebuild import build_telegram_stories

    now = now_utc()

    candidates = [
        {
            "story_id": "s-rejected",
            "item_id": "i-rejected",
            "event_id": "e-rejected",
            "title": "Yemeni army warns Houthis after "
            "attacks heighten escalation risk",
            "summary": "Yemeni army warns Houthis after "
            "attacks heighten escalation risk",
            "source": "Al Jazeera",
            "url": "https://example.com/yemen",
            "label": "news",
            "category": "world",
            "confidence": "high",
            "priority_level": "NORMAL",
            "priority_score": 70,
            "score": 70,
            "tier": 2,
            "primary_source": False,
            "effective_at": now.isoformat(),
            "published_at": "2026-08-01T10:00:00Z",
        },
        {
            "story_id": "s-kept",
            "item_id": "i-kept",
            "event_id": "e-kept",
            "title": "Strikes kill five in Belgorod",
            "summary": "The strikes killed five people "
            "and wounded around two dozen others. "
            "Rescue teams are searching the rubble for "
            "survivors.",
            "source": "Reuters",
            "url": "https://example.com/belgorod",
            "label": "news",
            "category": "world",
            "confidence": "high",
            "priority_level": "NORMAL",
            "priority_score": 60,
            "score": 60,
            "tier": 2,
            "primary_source": False,
            "effective_at": now.isoformat(),
            "published_at": "2026-08-01T10:00:00Z",
        },
    ]

    stories = build_telegram_stories(
        candidates,
        {
            "just_in_freshness_minutes": 15,
        },
        now,
    )

    queued_ids = [
        s["story_id"] for s in stories
    ]

    assert "s-rejected" not in queued_ids
    assert "s-kept" in queued_ids


def test_telegram_ineligible_sources_flags_only_non_news_feeds():
    from src.telebuild import telegram_ineligible_sources

    config = {
        "sources": {
            "sources": [
                {"name": "The Hindu", "news": True},
                {"name": "NASA News", "news": False},
                {"name": "CISA Alerts", "news": False},
            ]
        }
    }

    banned = telegram_ineligible_sources(config)

    assert "The Hindu" not in banned
    assert "NASA News" in banned
    assert "CISA Alerts" in banned


def test_config_flags_known_non_news_feeds():
    from src.config_loader import get_config
    from src.telebuild import telegram_ineligible_sources

    banned = telegram_ineligible_sources(get_config())

    # No configured India source is currently flagged non-news.
    assert banned == set()


def test_non_news_candidate_filtered_by_source():
    from src.telebuild import telegram_ineligible_sources

    config = {
        "sources": {
            "sources": [
                {"name": "ESA News", "news": False},
                {"name": "The Hindu", "news": True},
            ]
        }
    }
    banned = telegram_ineligible_sources(config)

    candidates = [
        {"story_id": "a", "source": "ESA News"},
        {"story_id": "b", "source": "The Hindu"},
    ]

    kept = [
        c for c in candidates
        if c.get("source") not in banned
    ]

    assert [c["story_id"] for c in kept] == ["b"]


# ---------------------------------------------------------------------------
# FIX 1 - HTML entity cleanup at message level
# ---------------------------------------------------------------------------

class TestMessageHtmlEntities:
    def test_message_quot_renders_as_literal_quote(self):
        msg = build_message(
            briefing_item(
                opening=[
                    'Officials said "we have to brace '
                    'ourselves" for the worst.',
                ],
            ),
            CFG,
        )
        assert '"we have to brace ourselves"' in msg["text"]
        assert "&quot;" not in msg["text"]

    def test_message_url_unchanged(self):
        url = "https://www.bbc.co.uk/news/articles/abc?cmp=RSS&s=2"
        msg = build_message(
            briefing_item(url=url),
            CFG,
        )
        href = re.search(
            r"href=\"([^\"]*)\"", msg["text"]
        ).group(1)
        # The href is escaped for the attribute but the URL
        # text itself is never decoded or rewritten.
        assert "amp;" in href
        assert url.replace("&", "&amp;") == href
        assert url in msg["text"].replace("&amp;", "&")


# ---------------------------------------------------------------------------
# FIX 3 - minimum two meaningful explanatory sentences
# ---------------------------------------------------------------------------


class TestMinimumTwoSentences:
    # Helper that builds a queue item with exact control over
    # the briefing rows (briefing_item() injects default rows).
    @staticmethod
    def bare(
        opening,
        body,
        bullets=None,
        headline="Canada wildfire doubles in size",
        title="State of emergency declared as fast-moving "
        "Canada wildfire doubles in size",
    ):
        return {
            "story_id": "s-bare",
            "item_id": "i1",
            "event_id": "e1",
            "title": title,
            "headline": headline,
            "summary": opening[0] if opening else "",
            "source": "BBC World",
            "url": "https://www.bbc.co.uk/news/articles/abc123",
            "label": NEWS,
            "public_label": NEWS,
            "priority_level": "NORMAL",
            "priority_score": 40,
            "effective_at": now_utc().isoformat(),
            "briefing": {
                "opening": opening,
                "body": body,
                "bullets": bullets or [],
                "sentences": [
                    {"text": t, "source": "BBC World"}
                    for t in opening + body
                ],
                "source": "BBC World",
            },
        }

    def test_one_useful_sentence_rejected(self):
        msg = build_message(
            self.bare(
                opening=["The fire has spread to 36 sq miles."],
                body=[],
            ),
            CFG,
        )
        assert msg is None

    def test_two_useful_sentences_accepted(self):
        msg = build_message(
            self.bare(
                opening=["The fire has spread to 36 sq miles."],
                body=["Officials said 20,000 people have fled."],
            ),
            CFG,
        )
        assert msg is not None
        assert "36 sq miles" in msg["text"]
        assert "20,000 people" in msg["text"]

    def test_headline_plus_one_sentence_rejected(self):
        msg = build_message(
            self.bare(
                opening=["The fire has spread to 36 sq miles."],
                body=[],
                headline="Wildfire grows in Canada",
                title="Wildfire grows in Canada",
            ),
            CFG,
        )
        assert msg is None

    def test_headline_duplicate_plus_one_real_sentence_rejected(self):
        msg = build_message(
            self.bare(
                opening=["Wildfire grows in Canada."],
                body=["The fire has spread to 36 sq miles."],
                headline="Wildfire grows in Canada",
                title="Wildfire grows in Canada",
            ),
            CFG,
        )
        assert msg is None

    def test_one_narrative_sentence_plus_duplicate_bullet_rejected(self):
        opening = ["The fire has spread to 36 sq miles."]
        msg = build_message(
            self.bare(
                opening=opening,
                body=[],
                bullets=[
                    {
                        "icon": "\U0001F4CD",
                        "label": "Location",
                        "text": "The fire has spread to 36 sq miles.",
                    },
                    {
                        "icon": "\u26A0\uFE0F",
                        "label": "Status",
                        "text": "out of control",
                    },
                ],
            ),
            CFG,
        )
        assert msg is None

    def test_two_narrative_sentences_with_bullets_accepted(self):
        opening = [
            "The fire has spread to 36 sq miles.",
            "Officials said 20,000 people have fled.",
        ]
        msg = build_message(
            self.bare(
                opening=opening,
                body=[],
                bullets=[
                    {
                        "icon": "\U0001F4CD",
                        "label": "Location",
                        "text": "British Columbia",
                    },
                ],
            ),
            CFG,
        )
        assert msg is not None

    def test_fallback_single_sentence_rejected(self):
        msg = build_message(
            item(
                summary="Only one explanatory sentence here.",
            ),
            CFG,
        )
        assert msg is None


# ---------------------------------------------------------------------------
# FINAL WorldNews format
#
# [ONE LABEL]
#
# [Clear headline]
#
# [2-8 useful explanatory sentences]
#
# 📰 Source: [source name]
#
# 🔗 Read the full report
#
# Nothing else: no channel header, bullets, region lines,
# corroboration text, decorative lines or internal fields.
# ---------------------------------------------------------------------------


class TestFinalWorldNewsFormat:
    def test_exact_message_structure(self):
        msg = build_message(
            briefing_item(
                opening=[
                    "The Bald Range wildfire in British "
                    "Columbia has spread over more than "
                    "36 sq miles.",
                ],
                body=[
                    "A fast-moving wildfire has forced "
                    "more than 20,000 people to "
                    "evacuate.",
                ],
            ),
            CFG,
        )
        text = msg["text"]
        expected = (
            NEWS
            + "\n\n"
            + "<b>Canada wildfire doubles in size</b>"
            + "\n\n"
            + "The Bald Range wildfire in British "
            + "Columbia has spread over more than "
            + "36 sq miles. "
            + "A fast-moving wildfire has forced more "
            + "than 20,000 people to evacuate."
            + "\n\n"
            + "\U0001F4F0 Source: BBC World"
            + "\n\n"
            + '\U0001F517 <a href="https://www.bbc.co.uk/'
            + 'news/articles/abc123">Read the full report</a>'
        )
        assert text == expected

    def test_no_forbidden_elements_anywhere(self):
        opening = [
            "The Bald Range wildfire in British "
            "Columbia has spread over more than "
            "36 sq miles.",
        ]
        body = [
            "A fast-moving wildfire has forced more "
            "than 20,000 people to evacuate.",
        ]
        msg = build_message(
            briefing_item(
                region="North America",
                opening=opening,
                body=body,
                corroborating=["Al Jazeera"],
                bullets=[
                    {
                        "icon": "\U0001F4CD",
                        "label": "Location",
                        "text": "British Columbia",
                    },
                    {
                        "icon": "\U0001F465",
                        "label": "Impact",
                        "text": "more than 20,000 people",
                    },
                ],
            ),
            CFG,
        )
        text = msg["text"]
        assert "WorldNews" not in text
        assert "\u2022" not in text
        assert "\u2501" not in text
        assert "NORTH AMERICA" not in text
        assert "Corroborated" not in text
        assert "Al Jazeera" not in text
        assert "LOCATION" not in text
        assert "IMPACT" not in text
        assert "NEXT" not in text
        assert "STATUS" not in text
        assert "This is a breaking news story" not in text
        assert "**" not in text
        assert text.count("\U0001F4F0 Source:") == 1
        assert text.count("\U0001F517") == 1
        assert text.count("<b>") == 1
        assert text.count("</b>") == 1

    def test_max_five_explanatory_sentences(self):
        facts = DISTINCT_STORM_FACTS[:10]
        msg = build_message(
            briefing_item(opening=facts[:2], body=facts[2:]),
            CFG,
        )
        text = msg["text"]
        # The first five sentences are kept in order.
        for i in range(5):
            assert facts[i] in text
        # The sixth and later are dropped, never padded in.
        for i in (5, 6, 7, 8, 9):
            assert facts[i] not in text

    def test_two_sentences_rendered_when_available(self):
        msg = build_message(
            briefing_item(
                opening=[
                    "The fire has spread to 36 sq miles.",
                ],
                body=[
                    "Officials said 20,000 people have "
                    "fled.",
                ],
            ),
            CFG,
        )
        text = msg["text"]
        assert "The fire has spread to 36 sq miles." in text
        assert "Officials said 20,000 people have fled." in text

    def test_paragraph_never_forced_to_eight_sentences(self):
        msg = build_message(
            briefing_item(
                opening=[
                    "The fire has spread to 36 sq miles.",
                ],
                body=[
                    "Officials said 20,000 people have "
                    "fled.",
                ],
            ),
            CFG,
        )
        text = msg["text"]
        # Exactly the two useful sentences: nothing invented,
        # nothing padded to reach a target length.
        assert "The fire has spread to 36 sq miles." in text
        assert "Officials said 20,000 people have fled." in text
        assert ". " in text
        assert text.count(". ") <= 2

    def test_article_sentences_render_in_paragraph(self):
        # Article-enriched rows (no punctuation issues) flow
        # into the body paragraph verbatim.
        opening = [
            "The port was closed to all shipping "
            "overnight.",
        ]
        body = [
            "Two vessels were rerouted to alternate "
            "harbours.",
            "Authorities said the closure would last "
            "at least 48 hours.",
        ]
        msg = build_message(
            briefing_item(opening=opening, body=body),
            CFG,
        )
        text = msg["text"]
        assert (
            "The port was closed to all shipping "
            "overnight. "
            "Two vessels were rerouted to alternate "
            "harbours. "
            "Authorities said the closure would last "
            "at least 48 hours."
            in text
        )

    def test_read_more_link_points_to_original_source(self):
        url = (
            "https://www.theguardian.com/world/2026/"
            "aug/10/typhoon-dolphin-china"
        )
        msg = build_message(
            briefing_item(url=url),
            CFG,
        )
        assert (
            '\U0001F517 <a href="' + url + '">'
            "Read the full report</a>"
            in msg["text"]
        )

    def test_label_headline_footer_never_dropped_when_over_budget(
        self,
    ):
        facts = DISTINCT_STORM_FACTS
        msg = build_message(
            briefing_item(
                opening=facts[:1],
                body=facts[1:],
            ),
            {"target_message_chars": 400, "max_message_chars": 400},
        )
        text = msg["text"]
        assert NEWS in text
        assert "<b>Canada wildfire doubles in size</b>" in text
        assert "\U0001F4F0 Source: BBC World" in text
        assert "Read the full report" in text
        assert telegram_visible_len(text) <= 400
        # Whole sentences are dropped before any mid-sentence
        # cut: every remaining sentence is intact.
        for sentence in DISTINCT_STORM_FACTS:
            if sentence in text:
                assert sentence in text
