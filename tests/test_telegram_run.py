"""Unit tests for the telegram publishing runner (src.telegram_run).

Covers CLI flags, dry-run safety, publish gating, state
tracking, retries, 429 handling and media integration without
any network access. Every test writes its queue/state/config
into a pytest tmp_path so the repo's real data/ is untouched.
"""
import builtins
import json
from datetime import timedelta

import pytest

from src.telegram_publisher import (
    TelegramPublisherError,
    TelegramRateLimited,
)

from src.telegram_run import (
    RATE_LIMIT_BUDGET_SECONDS,
    RATE_LIMIT_MAX_WAIT_SECONDS,
    main,
    rate_limit_wait_seconds,
)


def make_item(**overrides):
    import src.telegram_run as run_mod

    now = run_mod.now_utc()
    base = {
        "story_id": "story-1",
        "item_id": "item-1",
        "event_id": "event-1",
        "title": "Flood alerts issued across the region",
        "headline": "Flood alerts issued across the region",
        "summary": (
            "Heavy overnight rain pushed rivers over their "
            "banks. Officials advised residents to move to "
            "higher ground."
        ),
        "source": "News Agency",
        "url": "https://example.com/story",
        "label": "news",
        "public_label": "\U0001F4F0 NEWS",
        "priority_level": "NORMAL",
        "priority_score": 60,
        "effective_at": (
            now - timedelta(minutes=5)
        ).isoformat(),
        "published_at": "2026-08-01T10:00:00Z",
    }
    base.update(overrides)
    return base


def write_config(tmp_path, **telegram_overrides):
    telegram = {
        "channel_id": "@test-channel",
        "telegram_state_file": str(
            tmp_path / "telegram_state.json"
        ),
        "telegram_queue_file": str(
            tmp_path / "telegram_queue.json"
        ),
        "freshness_hours": 6,
        "posted_retention_hours": 48,
        "max_candidates": 50,
        "max_posts_per_hour": 20,
        "max_posts_per_day": 150,
        "min_gap_seconds": 0,
        "normal_delay_seconds": [0, 0],
        "important_delay_seconds": [0, 0],
        "breaking_immediate": True,
        "just_in_freshness_minutes": 15,
        "target_message_chars": 1500,
        "max_message_chars": 3000,
        "max_attempts_per_story": 2,
        "media": {"enabled": False},
    }
    telegram.update(telegram_overrides)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"telegram": telegram}),
        encoding="utf-8",
    )
    return str(config_path)


def write_queue(tmp_path, items):
    queue_path = tmp_path / "telegram_queue.json"
    queue_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-08-01T10:00:00Z",
                "count": len(items),
                "stories": items,
            }
        ),
        encoding="utf-8",
    )
    return str(queue_path)


def load_state(tmp_path):
    path = tmp_path / "telegram_state.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


class FakeTelegramPublisher:
    def __init__(
        self,
        token="test-token",
        fails=0,
        rate_limit=None,
    ):
        self.enabled = bool(token)
        self.sent = []
        self.media = []
        self.fails = fails
        self.rate_limit = rate_limit

    def send_message(self, chat_id, message, dry_run=False):
        if dry_run:
            return {
                "dry_run": True,
                "chat_id": chat_id,
            }
        if self.fails > 0:
            self.fails -= 1
            raise TelegramPublisherError(
                "simulated failure"
            )
        if self.rate_limit:
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
        if dry_run:
            return {
                "dry_run": True,
                "chat_id": chat_id,
                "media_kind": attachment.kind,
            }
        if self.fails > 0:
            self.fails -= 1
            raise TelegramPublisherError(
                "simulated media failure"
            )
        if self.rate_limit:
            raise TelegramRateLimited(
                self.rate_limit,
                "simulated rate limit",
            )
        self.media.append((chat_id, caption))
        return {
            "message_id": 9000 + len(self.media),
            "chat_id": chat_id,
        }

    def get_me(self):
        return {"username": "fake-bot"}


@pytest.fixture
def fake_publisher_factory(monkeypatch):
    def factory(**kwargs):
        publisher = FakeTelegramPublisher(**kwargs)
        monkeypatch.setattr(
            "src.telegram_run.TelegramPublisher",
            lambda: publisher,
        )
        return publisher

    return factory


@pytest.fixture(autouse=True)
def _no_network_media(monkeypatch):
    monkeypatch.setattr(
        "src.telegram_media.build_media_attachment",
        lambda url, media_cfg: None,
    )
    monkeypatch.setattr(
        "src.telegram_run.time.sleep",
        lambda seconds: None,
    )


# ---------------------------------------------------------
# rate_limit_wait_seconds policy
# ---------------------------------------------------------


def test_rate_limit_wait_seconds_clamped_policy():
    assert rate_limit_wait_seconds(30) == 30
    assert (
        rate_limit_wait_seconds(10 ** 6)
        == RATE_LIMIT_MAX_WAIT_SECONDS
    )
    assert (
        rate_limit_wait_seconds(30)
        <= RATE_LIMIT_BUDGET_SECONDS
    )
    assert rate_limit_wait_seconds(-1) == 0
    assert rate_limit_wait_seconds(None) == 0
    assert rate_limit_wait_seconds("nonsense") == 0
    assert (
        rate_limit_wait_seconds(1000, budget=120)
        == RATE_LIMIT_MAX_WAIT_SECONDS
    )
    assert (
        rate_limit_wait_seconds(1000, max_wait=10)
        == 10
    )
    assert rate_limit_wait_seconds(5, budget=2) == 2


# ---------------------------------------------------------
# CLI flag parsing
# ---------------------------------------------------------


def test_parse_args_defaults():
    from src.telegram_run import parse_args

    args = parse_args([])
    assert args.config is None
    assert not args.force
    assert not args.dry_run
    assert not args.dry_run_publish
    assert not args.yes
    assert not args.no_live


def test_parse_args_all_flags():
    from src.telegram_run import parse_args

    args = parse_args(
        [
            "--config",
            "x.json",
            "--force",
            "--dry-run",
            "--dry-run-publish",
            "--yes",
            "--no-live",
        ]
    )
    assert args.config == "x.json"
    assert args.force
    assert args.dry_run
    assert args.dry_run_publish
    assert args.yes
    assert args.no_live


# ---------------------------------------------------------
# dry-run safety
# ---------------------------------------------------------


def test_dry_run_reads_queue_touches_nothing(
    tmp_path,
    capsys,
):
    config_path = write_config(tmp_path)
    queue_path = write_queue(
        tmp_path,
        [make_item()],
    )
    assert queue_path

    code = main(["--config", config_path, "--dry-run"])

    assert code == 0
    assert load_state(tmp_path) is None
    assert capsys.readouterr().out.count(
        "candidate"
    ) >= 1


def test_dry_run_publish_writes_no_state(
    tmp_path,
    fake_publisher_factory,
):
    fake_publisher_factory()
    config_path = write_config(tmp_path)
    write_queue(tmp_path, [make_item()])

    code = main(
        [
            "--config",
            config_path,
            "--dry-run-publish",
        ]
    )

    assert code == 0
    assert load_state(tmp_path) is None


def test_missing_queue_fatal(tmp_path):
    config_path = write_config(tmp_path)

    code = main(["--config", config_path])

    assert code == 2


# ---------------------------------------------------------
# publish gating
# ---------------------------------------------------------


def test_no_live_mode_records_schedule_state(
    tmp_path,
):
    config_path = write_config(tmp_path)
    write_queue(tmp_path, [make_item()])

    code = main(
        ["--config", config_path, "--no-live"]
    )

    assert code == 0
    state = load_state(tmp_path)
    assert state is not None
    assert len(state["scheduled"]) == 1
    assert state["scheduled"][0]["story_id"] == "story-1"
    assert state["posted"] == []


def test_publish_blocked_without_env_or_force(
    tmp_path,
    fake_publisher_factory,
):
    publisher = fake_publisher_factory()
    config_path = write_config(tmp_path)
    write_queue(tmp_path, [make_item()])

    code = main(["--config", config_path])

    assert code == 0
    assert publisher.sent == []
    assert publisher.media == []
    state = load_state(tmp_path)
    assert len(state["scheduled"]) == 1


def test_publish_env_value_zero_is_not_allowed(
    tmp_path,
    fake_publisher_factory,
    monkeypatch,
):
    publisher = fake_publisher_factory()
    monkeypatch.setenv("TELEGRAM_PUBLISH", "0")
    config_path = write_config(tmp_path)
    write_queue(tmp_path, [make_item()])

    code = main(
        ["--config", config_path]
    )

    assert code == 0
    assert publisher.sent == []
    assert publisher.media == []
    state = load_state(tmp_path)
    assert len(state["scheduled"]) == 1


def test_force_publish_missing_channel_fatal(
    tmp_path,
    fake_publisher_factory,
):
    publisher = fake_publisher_factory()
    config_path = write_config(
        tmp_path,
        channel_id="",
    )
    write_queue(tmp_path, [make_item()])

    code = main(
        ["--config", config_path, "--force", "--yes"]
    )

    assert code == 2
    assert publisher.sent == []
    assert load_state(tmp_path) is None


def test_force_publish_requires_confirmation(
    tmp_path,
    fake_publisher_factory,
    monkeypatch,
):
    publisher = fake_publisher_factory()
    config_path = write_config(tmp_path)
    write_queue(tmp_path, [make_item()])

    def refuse(*args, **kwargs):
        raise EOFError("no tty")

    monkeypatch.setattr(
        builtins,
        "input",
        refuse,
    )

    code = main(
        ["--config", config_path, "--force"]
    )

    assert code == 2
    assert publisher.sent == []


def test_force_publish_with_yes_sends_and_tracks(
    tmp_path,
    fake_publisher_factory,
):
    publisher = fake_publisher_factory()
    config_path = write_config(tmp_path)
    write_queue(tmp_path, [make_item()])

    code = main(
        [
            "--config",
            config_path,
            "--force",
            "--yes",
        ]
    )

    assert code == 0
    assert publisher.sent == ["@test-channel"]
    assert publisher.media == []
    state = load_state(tmp_path)
    assert state["scheduled"] == []
    assert len(state["posted"]) == 1
    assert state["posted"][0]["story_id"] == "story-1"
    assert state["posted"][0]["message_id"]
    assert state["last_posted_at"]


# ---------------------------------------------------------
# retries
# ---------------------------------------------------------


def test_failure_increments_attempts_and_exits_1(
    tmp_path,
    fake_publisher_factory,
):
    fake_publisher_factory(fails=2)
    config_path = write_config(tmp_path)
    write_queue(tmp_path, [make_item()])

    code = main(
        [
            "--config",
            config_path,
            "--force",
            "--yes",
        ]
    )

    assert code == 1
    state = load_state(tmp_path)
    assert len(state["scheduled"]) == 1
    assert state["scheduled"][0]["attempts"] == 1
    assert state["posted"] == []
    assert state["failures"] == []


def test_second_run_reaches_failure_history(
    tmp_path,
    fake_publisher_factory,
):
    publisher = fake_publisher_factory(fails=2)
    config_path = write_config(tmp_path)
    write_queue(tmp_path, [make_item()])

    code = main(
        [
            "--config",
            config_path,
            "--force",
            "--yes",
        ]
    )
    assert code == 1

    code = main(
        [
            "--config",
            config_path,
            "--force",
            "--yes",
        ]
    )

    assert code == 1
    state = load_state(tmp_path)
    assert state["scheduled"] == []
    assert len(state["failures"]) == 1
    assert state["failures"][0]["story_id"] == "story-1"
    assert state["failures"][0]["attempts"] == 2
    assert state["posted"] == []


# ---------------------------------------------------------
# 429 rate limiting
# ---------------------------------------------------------


def test_rate_limited_defers_and_exits_0(
    tmp_path,
    fake_publisher_factory,
):
    fake_publisher_factory(rate_limit=30)
    config_path = write_config(tmp_path)
    write_queue(tmp_path, [make_item()])

    code = main(
        [
            "--config",
            config_path,
            "--force",
            "--yes",
        ]
    )

    assert code == 0
    state = load_state(tmp_path)
    assert len(state["scheduled"]) == 1
    assert state["posted"] == []


# ---------------------------------------------------------
# media integration
# ---------------------------------------------------------


def test_media_attached_when_enabled(
    tmp_path,
    fake_publisher_factory,
    monkeypatch,
):
    from src.telegram_media import MediaAttachment

    publisher = fake_publisher_factory()

    def fake_media(url, media_cfg):
        return MediaAttachment(
            "photo",
            b"fake-bytes",
            "image/png",
            "telegram_media.png",
        )

    monkeypatch.setattr(
        "src.telegram_media.build_media_attachment",
        fake_media,
    )

    config_path = write_config(
        tmp_path,
        media={
            "enabled": True,
            "timeout_seconds": 12,
            "max_bytes": 10485760,
            "max_html_bytes": 2097152,
            "image_min_width": 300,
            "image_min_height": 200,
            "max_caption_chars": 1024,
        },
    )
    write_queue(tmp_path, [make_item()])

    code = main(
        [
            "--config",
            config_path,
            "--force",
            "--yes",
        ]
    )

    assert code == 0
    assert publisher.sent == []
    assert len(publisher.media) == 1
    assert publisher.media[0][0] == "@test-channel"
    state = load_state(tmp_path)
    assert len(state["posted"]) == 1