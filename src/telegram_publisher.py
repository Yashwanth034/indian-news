"""Telegram bot publisher using the requests library only.

Zero additional dependencies: the pipeline already
installs `requests`.
"""
import json
import os
import time

import requests

TELEGRAM_API = "https://api.telegram.org/bot{token}/"


class TelegramPublisherError(Exception):
    pass


class TelegramRateLimited(TelegramPublisherError):
    def __init__(self, retry_after, message):
        super().__init__(message)
        self.retry_after = retry_after


class TelegramPublisher:

    def __init__(self, token=None, timeout=15):
        self.token = token or os.environ.get(
            "TELEGRAM_BOT_TOKEN",
            ""
        ).strip()
        self.timeout = timeout
        self.enabled = bool(self.token)

    def api_url(self, method):
        return TELEGRAM_API.format(
            token=self.token
        ) + method

    def _post(self, method, payload, files=None):
        if not self.enabled:
            raise TelegramPublisherError(
                "publishing disabled: TELEGRAM_BOT_TOKEN "
                "is not configured"
            )

        if files is not None:
            response = requests.post(
                self.api_url(method),
                data=payload,
                files=files,
                timeout=self.timeout,
            )
        else:
            response = requests.post(
                self.api_url(method),
                json=payload,
                timeout=self.timeout,
            )

        if response.status_code == 429:

            retry_after = 30

            try:
                retry_after = int(
                    response.json()
                    .get("parameters", {})
                    .get("retry_after", 30)
                )
            except Exception:
                pass

            raise TelegramRateLimited(
                retry_after,
                "rate limited for "
                + str(retry_after)
                + "s",
            )

        try:
            data = response.json()
        except Exception:
            raise TelegramPublisherError(
                "telegram returned non-JSON "
                + str(response.status_code)
            )

        if not data.get("ok"):
            raise TelegramPublisherError(
                str(data.get("description"))
                or "telegram api error"
            )

        return data

    def send_message(
        self,
        chat_id,
        message,
        dry_run=False,
    ):
        """Send one message. Returns the API result dict.

        dry_run=True returns a dict without any network
        call, so a real run never mistakes a rehearsal for
        a published message.
        """
        if dry_run or not self.enabled:
            return {
                "dry_run": True,
                "chat_id": chat_id,
                "message": message,
            }

        data = self._post(
            "sendMessage",
            {
                "chat_id": chat_id,
                "text": message["text"],
                "parse_mode": message.get(
                    "parse_mode",
                    "HTML",
                ),
                "disable_web_page_preview": True,
            },
        )

        result = data.get("result", {})

        return {
            "message_id": result.get(
                "message_id"
            ),
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
        """Send one message with a single media attachment.

        `attachment` must expose kind ("photo"/"video"),
        data (bytes), filename and content_type. The caption
        carries the exact same text as a text-only post.

        dry_run=True returns a dict without any network
        call, so a real run never mistakes a rehearsal for
        a published message.
        """
        if dry_run or not self.enabled:
            return {
                "dry_run": True,
                "chat_id": chat_id,
                "media_kind": attachment.kind,
            }

        method = (
            "sendPhoto"
            if attachment.kind == "photo"
            else "sendVideo"
        )

        field = (
            "photo"
            if attachment.kind == "photo"
            else "video"
        )

        data = self._post(
            method,
            {
                "chat_id": chat_id,
                "caption": caption,
                "parse_mode": parse_mode,
            },
            files={
                field: (
                    attachment.filename,
                    attachment.data,
                    attachment.content_type,
                )
            },
        )

        result = data.get("result", {})

        return {
            "message_id": result.get(
                "message_id"
            ),
            "chat_id": chat_id,
        }

    def get_me(self):
        """Validate the token. Raises on failure."""
        data = self._post(
            "getMe",
            {}
        )
        return data.get("result", {})


def sleep_until(timestamp):
    delay = timestamp - time.time()

    if delay > 0:
        time.sleep(delay)


def retry_delay(attempt, base=5, max_delay=120):
    """Exponential backoff capped at max_delay seconds."""
    delay = min(
        base * (2 ** attempt),
        max_delay
    )
    return delay


def load_state(state_file, default=None):
    """Load JSON state, tolerating a missing/corrupt file."""
    if default is None:
        default = {}

    if not state_file:
        return default

    try:
        with open(
            state_file,
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return default

    if not isinstance(data, dict):
        return default

    return data


def save_state(state_file, data):
    try:
        tmp_file = str(state_file) + ".tmp"
        with open(
            tmp_file,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                data,
                f,
                indent=2,
                ensure_ascii=False,
            )
        os.replace(tmp_file, state_file)
    except OSError as exc:
        raise TelegramPublisherError(
            "cannot write state file: "
            + str(exc)
        )
