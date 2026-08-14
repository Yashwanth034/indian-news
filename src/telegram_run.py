"""Telegram publishing runner (India).

Connects the queue written by ``src.telebuild`` to the ported
scheduler and publisher:

    data/telegram_queue.json
      -> read stories
      -> filter_candidates (freshness, posted/scheduled/failure state)
      -> schedule (per-priority delays)
      -> publish_due (throttles, media, retries, rate-limit)
      -> Telegram channel
      -> data/telegram_state.json

EXIT CODES
    0  run complete (nothing sent, or everything sent)
    1  run complete but some messages failed permanently
    2  fatal error (config, queue, or API access)

DRY-RUN MODE
    --dry-run            read queue + report candidate decisions,
                         send nothing, touch nothing
    --dry-run-publish    simulate scheduling + publishing against a
                         copy of the schedule, send nothing, write
                         no state

Real publishing only happens with --force, or when
TELEGRAM_PUBLISH=1 is set. The token comes from
TELEGRAM_BOT_TOKEN (never from files).

Environment:
    TELEGRAM_BOT_TOKEN    bot token (required to send)
    TELEGRAM_CHANNEL_ID   channel override (else config channel_id)
    TELEGRAM_PUBLISH      set to "1" to allow real publishing
    TELEGRAM_NO_RETRY     set to "1" for one-shot (no retries)
"""
import argparse
import copy
import datetime
import json
import os
import random
import sys
import time
from pathlib import Path

from src.config_loader import ROOT, get_config
from src.telegram_publisher import (
    TelegramPublisher,
    TelegramPublisherError,
    load_state,
    save_state,
    sleep_until,
)
from src.telegram_scheduler import (
    filter_candidates,
    now_utc,
    parse_dt,
    publish_due,
)

# Bounds for HTTP 429 handling: a single wait never
# exceeds RATE_LIMIT_MAX_WAIT_SECONDS and the whole run
# never sleeps more than RATE_LIMIT_BUDGET_SECONDS, so
# retry_after is respected where practical without ever
# producing an uncontrolled wait or an infinite loop.
RATE_LIMIT_MAX_WAIT_SECONDS = 90
RATE_LIMIT_BUDGET_SECONDS = 180

DEFAULT_CONFIG = ROOT / "config" / "config.json"


def rate_limit_wait_seconds(
    retry_after,
    max_wait=RATE_LIMIT_MAX_WAIT_SECONDS,
    budget=RATE_LIMIT_BUDGET_SECONDS,
):
    """Bounded wait for a Telegram 429 retry_after value."""
    try:
        wait = int(retry_after)
    except (TypeError, ValueError):
        return 0

    if wait < 0:
        wait = 0

    return min(wait, max_wait, budget)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        default=None,
        help="path to a config JSON (default: India config)",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="send messages even without TELEGRAM_PUBLISH=1",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only show which stories would be scheduled",
    )

    parser.add_argument(
        "--dry-run-publish",
        action="store_true",
        help="simulate publishing against the schedule, send nothing",
    )

    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip confirmation when force publishing",
    )

    parser.add_argument(
        "--no-live",
        action="store_true",
        help="force no-live mode (no delay, no send, "
        "still records dry-run schedule state)",
    )

    return parser.parse_args(argv)


def report_line(prefix, text):
    print("{:<11} {}".format(prefix, text))


def load_config(config_path):
    """Load config: an explicit JSON file, else the India bundle.

    Returns (full_config, telegram_section).  Accepts either the
    India bundle shape (``{"config": {"telegram": {...}}})`` or a
    flat ``{"telegram": {...}}`` file for testing/overrides.
    """
    if config_path:
        with open(config_path, encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = get_config()

    if (
        isinstance(data.get("config"), dict)
        and data["config"].get("telegram") is not None
    ):
        full = data["config"]
    else:
        full = data

    return full, (full.get("telegram") or {})


def _resolve_path(rel, fallback):
    path = Path(rel or fallback)
    if not path.is_absolute():
        path = ROOT / path
    return path


def main(argv=None):
    args = parse_args(argv)

    try:
        full_cfg, telegram_cfg = load_config(args.config)
    except (OSError, ValueError) as exc:
        print("fatal: cannot read config:", exc)
        return 2

    state_file = _resolve_path(
        telegram_cfg.get(
            "telegram_state_file",
            "data/telegram_state.json",
        ),
        "data/telegram_state.json",
    )

    queue_file = _resolve_path(
        telegram_cfg.get(
            "telegram_queue_file",
            "data/telegram_queue.json",
        ),
        "data/telegram_queue.json",
    )

    state_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        with open(queue_file, encoding="utf-8") as f:
            queue = json.load(f)
    except (OSError, ValueError) as exc:
        print("fatal: cannot read queue file:", exc)
        return 2

    items = queue.get("stories", [])
    now = now_utc()

    publisher = TelegramPublisher()

    if not publisher.enabled:
        print(
            "warning: TELEGRAM_BOT_TOKEN is not set; "
            "running in no-live mode"
        )

    state = load_state(str(state_file))
    state.setdefault("posted", [])
    state.setdefault("scheduled", [])
    state.setdefault("failures", [])
    state.setdefault("last_posted_at", None)

    max_attempts = int(
        telegram_cfg.get(
            "max_attempts_per_story",
            2,
        )
    )

    if os.environ.get("TELEGRAM_NO_RETRY") == "1":
        max_attempts = 1
        print(
            "note: TELEGRAM_NO_RETRY=1 - one-shot mode, "
            "retries disabled"
        )

    candidates = filter_candidates(
        items,
        state,
        float(
            telegram_cfg.get(
                "freshness_hours",
                6,
            )
        ),
        int(
            telegram_cfg.get(
                "max_candidates",
                50,
            )
        ),
        int(
            telegram_cfg.get(
                "just_in_freshness_minutes",
                15,
            )
        ),
        bool(
            telegram_cfg.get(
                "breaking_immediate",
                True,
            )
        ),
        now,
    )

    print()
    print("=== telegram candidate selection ===")
    print("queue stories :", len(items))
    print("candidates    :", len(candidates))

    for item in candidates:
        report_line(
            "candidate",
            "{} | {} | prio={} | fresh={}m".format(
                item.get("story_id", "?")[:8],
                (item.get("title") or "?")[:60],
                item.get("priority_score", 0),
                int(
                    (
                        (
                            now
                            - parse_dt(
                                item.get(
                                    "effective_at"
                                )
                            )
                        ).total_seconds()
                        / 60
                    )
                ),
            ),
        )

    if args.dry_run:
        print()
        print(
            "=== dry-run: nothing sent, nothing touched ==="
        )
        return 0

    scheduled_ids = {
        e.get("story_id")
        for e in state.get("scheduled", [])
    }

    new_scheduled = []

    for item in candidates:

        story_id = item.get("story_id")

        if story_id in scheduled_ids:
            continue

        is_breaking = (
            item.get("priority_level")
            == "IMMEDIATE"
        )

        if (
            is_breaking
            and telegram_cfg.get(
                "breaking_immediate",
                True,
            )
        ):
            delay = 0
        elif is_breaking:
            delay = random.randint(
                *[
                    int(x)
                    for x in telegram_cfg.get(
                        "important_delay_seconds",
                        [60, 240],
                    )
                ]
            )
        else:

            item_age = (
                (
                    now
                    - parse_dt(
                        item.get("effective_at")
                    )
                ).total_seconds()
            )

            if (
                0
                <= item_age
                <= telegram_cfg.get(
                    "just_in_freshness_minutes",
                    15,
                )
                * 60
            ):

                delay = random.randint(
                    *[
                        int(x)
                        for x in telegram_cfg.get(
                            "important_delay_seconds",
                            [60, 240],
                        )
                    ]
                )
            else:
                delay = random.randint(
                    *[
                        int(x)
                        for x in telegram_cfg.get(
                            "normal_delay_seconds",
                            [120, 420],
                        )
                    ]
                )

        scheduled_at = now_utc().replace(
            microsecond=0
        )

        scheduled_at = (
            scheduled_at
            + datetime.timedelta(seconds=delay)
        )

        label = (
            item.get("public_label")
            or item.get("label")
        )

        new_scheduled.append(
            {
                "story_id": story_id,
                "event_id": item.get("event_id"),
                "item_id": (
                    item.get("item_id")
                    or item.get("story_id")
                ),
                "label": label,
                "scheduled_at": scheduled_at.isoformat(),
                "attempts": 0,
            }
        )

    print()
    print(
        "=== schedule === "
        "({} new, {} total)".format(
            len(new_scheduled),
            len(state.get("scheduled", []))
            + len(new_scheduled),
        )
    )

    if args.dry_run_publish:
        return _simulate_publish(
            copy.deepcopy(state),
            new_scheduled,
            items,
            args,
            telegram_cfg,
            publisher,
        )

    if new_scheduled:
        state["scheduled"] = (
            state.get("scheduled", [])
            + new_scheduled
        )

    # Chronological schedule ordering (same key publish_due uses):
    # the run loop sleeps to the FIRST scheduled entry, so the list
    # must be sorted by scheduled_at or the loop can sleep past
    # earlier-due stories.  Sort unconditionally so unsorted state
    # persisted by older versions is repaired automatically.
    state["scheduled"] = sorted(
        state.get("scheduled", []),
        key=lambda e: (
            str(e.get("scheduled_at", "")),
            str(e.get("story_id", "")),
        ),
    )

    for entry in state.get("scheduled", []):
        report_line(
            "scheduled",
            "{} | {} | attempts={}".format(
                entry.get("story_id", "?")[:8],
                entry.get("scheduled_at", "?"),
                entry.get("attempts", 0),
            ),
        )

    if args.no_live:
        save_state(str(state_file), state)
        print()
        print(
            "=== no-live: schedule state saved, nothing "
            "sent ==="
        )
        return 0

    publish_allowed = (
        os.environ.get("TELEGRAM_PUBLISH") == "1"
        or args.force
    )

    if not publish_allowed:
        save_state(str(state_file), state)
        print()
        print(
            "=== no-live (TELEGRAM_PUBLISH unset, no "
            "--force): schedule state saved, nothing sent "
            "==="
        )
        return 0

    channel_id = (
        str(
            telegram_cfg.get(
                "channel_id",
                "",
            )
        ).strip()
        or os.environ.get(
            "TELEGRAM_CHANNEL_ID",
            "",
        ).strip()
    )

    if not channel_id:
        print()
        print(
            "fatal: force publishing requested but no "
            "channel_id in config and no TELEGRAM_CHANNEL_ID"
        )
        return 2

    if not args.yes:
        try:
            input(
                "CONFIRM: send telegram messages to "
                + channel_id
                + "? [y/N] "
            )
            print()
        except EOFError:
            print("no tty; refusing without --yes")
            return 2

    try:
        me = publisher.get_me()
        print(
            "=== bot ok:",
            me.get("username", "?"),
            "===",
        )
    except TelegramPublisherError as exc:
        print("fatal: bot check failed:", exc)
        return 2

    totals = {
        "due": 0,
        "published": [],
        "skipped_cap": [],
        "skipped_gap": [],
        "expired": [],
        "missing": [],
        "failed": [],
        "rate_limited": None,
    }

    wait_budget = RATE_LIMIT_BUDGET_SECONDS

    for entry in list(
        state.get("scheduled", [])
    ):

        scheduled_at = entry.get("scheduled_at")

        if not scheduled_at:
            continue

        when = parse_dt(scheduled_at)

        if when is None:
            continue

        remaining = (
            when - now_utc()
        ).total_seconds()

        if remaining > 0:
            print()
            print(
                "=== waiting",
                int(remaining),
                "seconds for",
                entry.get("story_id", "?")[:8],
                "===",
            )
            sleep_until(when.timestamp())

        report = publish_due(
            publisher,
            channel_id,
            state,
            items,
            float(
                telegram_cfg.get(
                    "freshness_hours",
                    6,
                )
            ),
            int(
                telegram_cfg.get(
                    "max_posts_per_hour",
                    20,
                )
            ),
            int(
                telegram_cfg.get(
                    "max_posts_per_day",
                    150,
                )
            ),
            int(
                telegram_cfg.get(
                    "min_gap_seconds",
                    60,
                )
            ),
            max_attempts,
            cfg=telegram_cfg,
            now=now_utc(),
        )

        save_state(str(state_file), state)

        totals["due"] += report["due"]
        totals["published"].extend(
            report["published"]
        )
        totals["skipped_cap"].extend(
            report["skipped_cap"]
        )
        totals["skipped_gap"].extend(
            report["skipped_gap"]
        )
        totals["expired"].extend(
            report["expired"]
        )
        totals["missing"].extend(
            report["missing"]
        )
        totals["failed"].extend(
            report["failed"]
        )

        if (
            report["rate_limited"]
            and totals["rate_limited"] is None
        ):
            totals["rate_limited"] = (
                report["rate_limited"]
            )

        if report["rate_limited"] and wait_budget > 0:

            wait = rate_limit_wait_seconds(
                report["rate_limited"],
                budget=wait_budget,
            )

            if wait > 0:

                print()
                print(
                    "=== rate limited: waiting",
                    wait,
                    "seconds before retrying ===",
                )

                time.sleep(wait)
                wait_budget -= wait

        if report["published"]:
            for p in report["published"]:
                report_line(
                    "published",
                    "{} | message_id={}".format(
                        p.get("story_id", "?")[:8],
                        p.get("message_id", "dry"),
                    ),
                )

        for f in report["failed"]:
            report_line(
                "failed",
                "{} | attempts={} | {}".format(
                    f.get("story_id", "?")[:8],
                    f.get("attempts", 1),
                    f.get("error", "?"),
                ),
            )

        if not report["due"]:
            break

    print()
    print("=== publish summary ===")
    print("due            :", totals["due"])
    print(
        "published      :",
        len(totals["published"]),
    )
    print(
        "skipped (caps) :",
        len(totals["skipped_cap"]),
    )
    print(
        "skipped (gap)  :",
        len(totals["skipped_gap"]),
    )
    print(
        "expired        :",
        len(totals["expired"]),
    )
    print(
        "missing        :",
        len(totals["missing"]),
    )
    print(
        "failed         :",
        len(totals["failed"]),
    )

    if totals["rate_limited"]:
        print(
            "rate limited   : retry after",
            totals["rate_limited"],
            "seconds",
        )

    hard_failures = [
        f
        for f in totals["failed"]
        if not f.get("retry_after")
    ]

    print()

    if totals["published"]:
        print("telegram publishing complete")

    return 1 if hard_failures else 0


def _simulate_publish(
    state,
    new_scheduled,
    items,
    args,
    telegram_cfg,
    publisher,
):
    """Simulate scheduling + publishing against a copy of state.

    Never calls Telegram, never downloads media, never writes
    state.  The freshly-computed ``new_scheduled`` entries (which
    carry the same per-priority decisions as a real run) are
    merged into a deep copy with their scheduled_at stamped to
    now, so the preview reports every fresh candidate as due and
    shows what publishing would do without any network call.
    """
    scheduled_ids = {
        e.get("story_id")
        for e in state.get("scheduled", [])
    }

    for entry in new_scheduled:
        if entry.get("story_id") in scheduled_ids:
            continue
        sim_entry = dict(entry)
        sim_entry["scheduled_at"] = now_utc().replace(
            microsecond=0
        ).isoformat()
        state["scheduled"].append(sim_entry)

    report = publish_due(
        TelegramPublisher(),
        "@dry-run",
        state,
        items,
        float(
            telegram_cfg.get(
                "freshness_hours",
                6,
            )
        ),
        int(
            telegram_cfg.get(
                "max_posts_per_hour",
                20,
            )
        ),
        int(
            telegram_cfg.get(
                "max_posts_per_day",
                150,
            )
        ),
        int(
            telegram_cfg.get(
                "min_gap_seconds",
                60,
            )
        ),
        1,
        cfg=telegram_cfg,
        now=now_utc(),
        dry_run=True,
    )

    print()
    print(
        "=== dry-run-publish: nothing sent, no state "
        "written ==="
    )

    for p in report["published"]:
        report_line(
            "would_publish",
            "{} | message_id=dry".format(
                p.get("story_id", "?")[:8],
            ),
        )

    for f in report["failed"]:
        report_line(
            "would_fail",
            "{} | {}".format(
                f.get("story_id", "?")[:8],
                f.get("error", "?"),
            ),
        )

    print()
    print("=== simulated summary ===")
    print("due            :", report["due"])
    print(
        "would publish  :",
        len(report["published"]),
    )
    print(
        "expired        :",
        len(report["expired"]),
    )
    print(
        "missing        :",
        len(report["missing"]),
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())