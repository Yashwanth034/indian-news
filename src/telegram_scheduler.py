"""Scheduling and throttling for telegram publishing.

Pure policy + state. Does not import requests at module
level, so it is safe to unit-test with plain dictionaries.
The optional media-attachment hook performs its network I/O
through a lazy import inside publish_due.
"""
from datetime import datetime, timezone

# Posted-story retention: a posted story_id must stay in
# the history at least as long as the engine's story
# memory (48h) so a re-emitted story_id can never slip
# through duplicate protection. A safety margin on top
# covers posting lag and the engine's run cadence, so an
# entry is pruned only when it is safely older than the
# longest period in which the engine could still re-emit
# the same story_id.
POSTED_RETENTION_HOURS = 48
POSTED_RETENTION_MARGIN_HOURS = 6


def now_utc():
    return datetime.now(timezone.utc)


def parse_dt(value, default=None):
    """Parse an ISO timestamp. Returns None if unparseable."""
    if not value:
        return default

    if isinstance(value, datetime):
        return value

    try:
        parsed = datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        )
    except ValueError:
        return default

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed


def story_age_minutes(item, now=None):
    """Minutes since the story's effective time.

    Allows up to 10 minutes of feed clock skew. Returns
    None when the timestamp is missing/unparseable.
    """
    if now is None:
        now = now_utc()

    effective = parse_dt(
        item.get("effective_at")
    )

    if effective is None:
        return None

    age = (
        now - effective
    ).total_seconds() / 60.0

    if age < -10:
        return None

    return max(0.0, age)


def is_fresh(item, freshness_hours, now=None):
    """Fresh = age is within [0, freshness_hours]."""
    age = story_age_minutes(item, now)

    if age is None:
        return False

    return 0 <= age <= freshness_hours * 60


def maybe_important(
    item,
    just_in_freshness_minutes,
    breaking_immediate,
    now=None,
):
    """Whether the item is a just-in or breaking story."""
    if item.get("priority_level") == "IMMEDIATE":
        if breaking_immediate:
            return True
        return False

    if item.get("priority_level") in (
        "HIGH",
        "EXCLUSIVE",
    ):
        return True

    age = story_age_minutes(item, now)

    if (
        age is not None
        and age <= just_in_freshness_minutes
    ):
        return True

    return False


def filter_candidates(
    items,
    state,
    freshness_hours,
    max_candidates,
    just_in_freshness_minutes,
    breaking_immediate,
    now=None,
):
    """Pick Telegram candidates from a run's queue.

    Excludes already-posted story ids, active schedule
    entries, and the failure history. A failure counts only
    for the same source item id, never for the whole story.
    """
    if now is None:
        now = now_utc()

    posted_ids = set(
        entry.get("story_id")
        for entry in state.get("posted", [])
    )

    scheduled_ids = set(
        entry.get("story_id")
        for entry in state.get("scheduled", [])
    )

    scheduled_item_ids = set(
        entry.get("item_id")
        for entry in state.get("scheduled", [])
    )

    failed_item_ids = set(
        entry.get("item_id")
        for entry in state.get("failures", [])
    )

    candidates = []

    for item in items:

        story_id = item.get("story_id")
        item_id = item.get("item_id")

        if story_id in posted_ids:
            continue

        if story_id in scheduled_ids:
            continue

        if item_id and item_id in scheduled_item_ids:
            continue

        if item_id and item_id in failed_item_ids:
            continue

        if not is_fresh(
            item,
            freshness_hours,
            now,
        ):
            continue

        candidates.append(item)

    candidates.sort(
        key=lambda x: (
            not maybe_important(
                x,
                just_in_freshness_minutes,
                breaking_immediate,
                now,
            ),
            -(
                x.get("priority_score")
                or x.get("priority", 0)
                or 0
            ),
            story_age_minutes(x, now) or 0,
        )
    )

    return candidates[:max_candidates]


def publish_due(
    publisher,
    chat_id,
    state,
    items,
    freshness_hours,
    max_posts_per_hour,
    max_posts_per_day,
    min_gap_seconds,
    max_attempts_per_story,
    cfg=None,
    now=None,
    dry_run=False,
):
    """Publish everything due, respecting all throttles.

    `items` is the current run's queue so scheduled stories
    can be resolved to full content. Returns a report dict.
    """
    if now is None:
        now = now_utc()

    report = {
        "due": 0,
        "published": [],
        "skipped_cap": [],
        "skipped_gap": [],
        "expired": [],
        "rate_limited": None,
        "failed": [],
        "errors": [],
    }

    items_by_id = {}

    for item in items:
        story_id = item.get("story_id")
        if story_id:
            items_by_id[story_id] = item

    due = []

    for entry in state.get("scheduled", []):

        when = parse_dt(
            entry.get("scheduled_at")
        )

        if when is None:
            due.append(entry)
            continue

        if when <= now:
            due.append(entry)

    due.sort(
        key=lambda e: (
            str(e.get("scheduled_at", "")),
            str(e.get("story_id", "")),
        )
    )

    report["due"] = len(due)

    if state.get("posted"):

        retention_hours = float(
            cfg.get(
                "posted_retention_hours",
                POSTED_RETENTION_HOURS,
            )
        ) if cfg else POSTED_RETENTION_HOURS

        cutoff = now.timestamp() - (
            retention_hours
            + POSTED_RETENTION_MARGIN_HOURS
        ) * 3600

        kept = []

        for entry in state["posted"]:

            posted_dt = parse_dt(
                entry.get("posted_at")
            )

            if (
                posted_dt is None
                or posted_dt.timestamp() >= cutoff
            ):
                kept.append(entry)

        state["posted"] = (
            sorted(
                kept,
                key=lambda e: str(
                    e.get("posted_at", "")
                ),
                reverse=True,
            )
        )

    if not due:
        return report

    if not chat_id:
        report["errors"].append(
            "publishing skipped: no telegram channel "
            "configured"
        )
        return report

    last_posted_at = state.get("last_posted_at")
    last_posted_dt = parse_dt(last_posted_at)

    posted_hourly = 0
    posted_daily = 0

    now_ts = now.timestamp()

    for entry in state.get("posted", []):

        posted_dt = parse_dt(
            entry.get("posted_at")
        )

        if posted_dt is None:
            continue

        posted_ts = posted_dt.timestamp()

        if now_ts - posted_ts <= 3600:
            posted_hourly += 1

        if now_ts - posted_ts <= 86400:
            posted_daily += 1

    for entry in due:

        story_id = entry.get("story_id")
        item = items_by_id.get(story_id)

        if item is None:

            expired_dt = parse_dt(
                entry.get("scheduled_at")
            ) or now

            if now_ts - expired_dt.timestamp() > 43200:

                state["scheduled"].remove(entry)
                report["expired"].append(
                    {
                        "story_id": story_id,
                        "reason": "item missing from queue "
                        "and schedule too old",
                    }
                )
                continue

            report["expired"].append(
                {
                    "story_id": story_id,
                    "reason": "item missing from current "
                    "queue",
                }
            )
            continue

        age = story_age_minutes(item, now)

        if (
            age is not None
            and age > freshness_hours * 60
        ):

            state["scheduled"].remove(entry)
            report["expired"].append(
                {
                    "story_id": story_id,
                    "reason": "story no longer fresh",
                }
            )
            continue

        if (
            entry.get("attempts", 0) > 0
            or last_posted_dt is not None
        ) and last_posted_dt is not None:

            gap = (
                now_ts - last_posted_dt.timestamp()
            )

            if gap < min_gap_seconds:
                report["skipped_gap"].append(
                    {
                        "story_id": story_id,
                        "retry_after_seconds": int(
                            min_gap_seconds - gap
                        ),
                    }
                )
                continue

        if posted_hourly >= max_posts_per_hour:
            report["skipped_cap"].append(
                {
                    "story_id": story_id,
                    "reason": "hourly cap reached",
                }
            )
            continue

        if posted_daily >= max_posts_per_day:
            report["skipped_cap"].append(
                {
                    "story_id": story_id,
                    "reason": "daily cap reached",
                }
            )
            continue

        message = None

        try:
            from src.telegram_formatter import build_message
            message = build_message(
                item,
                cfg or {},
                now,
            )
        except Exception:
            report["errors"].append(
                {
                    "story_id": story_id,
                    "error": "message build failed",
                }
            )
            continue

        if message is None:

            state["scheduled"].remove(entry)
            report["expired"].append(
                {
                    "story_id": story_id,
                    "reason": "no usable content",
                }
            )
            continue

        media = None
        media_cfg = (cfg or {}).get("media") or {}

        # Best-effort media attachment: fetch, select and
        # download one image (preferred) or video for the
        # story. Any failure here means None and the post is
        # sent text-only, exactly as before.
        if (
            media_cfg.get("enabled", True)
            and not dry_run
            and (message.get("text") or "")
        ):

            try:
                from src.telegram_media import (
                    TELEGRAM_CAPTION_MAX,
                    build_media_attachment,
                )

                max_caption = int(
                    media_cfg.get(
                        "max_caption_chars",
                        TELEGRAM_CAPTION_MAX,
                    )
                )

                if len(message["text"]) <= max_caption:
                    media = build_media_attachment(
                        item.get("url"),
                        media_cfg,
                    )
            except Exception:
                media = None

        try:
            if media is not None:
                try:
                    result = publisher.send_media(
                        chat_id,
                        media,
                        caption=message["text"],
                        parse_mode=message.get(
                            "parse_mode",
                            "HTML",
                        ),
                        dry_run=dry_run,
                    )
                except Exception as exc:
                    from src.telegram_publisher import (
                        TelegramRateLimited,
                    )

                    if isinstance(
                        exc,
                        TelegramRateLimited,
                    ):
                        raise

                    # Telegram rejected the media (wrong
                    # format, oversize, etc.): never let
                    # that block the story, resend as a
                    # text-only post with the exact same
                    # text.
                    result = publisher.send_message(
                        chat_id,
                        message,
                        dry_run=dry_run,
                    )
            else:
                result = publisher.send_message(
                    chat_id,
                    message,
                    dry_run=dry_run,
                )
        except Exception as exc:

            from src.telegram_publisher import (
                TelegramRateLimited,
                TelegramPublisherError,
            )

            if isinstance(exc, TelegramRateLimited):
                report["rate_limited"] = (
                    exc.retry_after
                )
                report["failed"].append(
                    {
                        "story_id": story_id,
                        "error": str(exc),
                        "retry_after": (
                            exc.retry_after
                        ),
                    }
                )
                break

            entry["attempts"] = (
                entry.get("attempts", 0) + 1
            )

            report["failed"].append(
                {
                    "story_id": story_id,
                    "attempts": entry["attempts"],
                    "error": str(exc),
                }
            )

            if (
                entry["attempts"]
                >= max_attempts_per_story
            ):

                state["scheduled"].remove(entry)
                state.setdefault(
                    "failures",
                    []
                ).append(
                    {
                        "story_id": story_id,
                        "item_id": entry.get(
                            "item_id"
                        ),
                        "attempts": entry["attempts"],
                        "failed_at": now.isoformat(),
                    }
                )

            continue

        if result.get("dry_run"):

            report["published"].append(
                {
                    "story_id": story_id,
                    "event_id": entry.get(
                        "event_id"
                    ),
                    "label": entry.get("label"),
                    "dry_run": True,
                    "scheduled_at": entry.get(
                        "scheduled_at"
                    ),
                }
            )
            continue

        posted_at = now.isoformat()

        state["posted"].append(
            {
                "story_id": story_id,
                "event_id": entry.get("event_id"),
                "item_id": entry.get("item_id"),
                "label": entry.get("label"),
                "message_id": result.get(
                    "message_id"
                ),
                "posted_at": posted_at,
                "scheduled_at": entry.get(
                    "scheduled_at"
                ),
            }
        )

        state["last_posted_at"] = posted_at

        # Enforce the minimum gap at the actual publish
        # point: refresh the reference immediately so the
        # next due entry in this same call is blocked
        # unless >= min_gap_seconds have elapsed since the
        # previous successful post.
        last_posted_dt = parse_dt(posted_at)
        posted_hourly += 1
        posted_daily += 1

        state["scheduled"].remove(entry)

        report["published"].append(
            {
                "story_id": story_id,
                "event_id": entry.get("event_id"),
                "message_id": result.get(
                    "message_id"
                ),
                "posted_at": posted_at,
            }
        )

    return report
