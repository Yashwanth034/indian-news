"""Source health and error reporting.

Persists per-source status to a JSON file so we can see working sources,
failed sources, malformed responses, timeout/error counts and last success.
"""
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class SourceHealth:
    source_id: str
    status: str = "unknown"  # unknown | ok | error
    last_success: str | None = None
    last_fetch: str | None = None
    consecutive_failures: int = 0
    total_errors: int = 0
    timeouts: int = 0
    malformed: int = 0
    items_found: int = 0
    last_error: str | None = None

    def to_dict(self):
        return asdict(self)


class HealthStore:
    """Load/save per-source health state from/to a JSON file."""

    def __init__(self, path):
        self.path = Path(path)
        self._health: dict[str, SourceHealth] = {}

    def load(self):
        if not self.path.exists():
            self._health = {}
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self._health = {}
            return
        self._health = {
            sid: SourceHealth(**h) for sid, h in data.items() if isinstance(h, dict)
        }

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {sid: h.to_dict() for sid, h in self._health.items()}
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def get(self, source_id: str) -> SourceHealth | None:
        return self._health.get(source_id)

    def record_success(self, source_id: str, *, items_found: int, fetched_at: str):
        h = self._health.setdefault(source_id, SourceHealth(source_id))
        h.status = "ok"
        h.last_success = fetched_at
        h.last_fetch = fetched_at
        h.consecutive_failures = 0
        h.items_found = items_found
        h.last_error = None

    def record_failure(self, source_id: str, *, error: str, timeout: bool = False, malformed: bool = False):
        h = self._health.setdefault(source_id, SourceHealth(source_id))
        h.status = "error"
        h.total_errors += 1
        h.consecutive_failures += 1
        h.last_error = error
        if timeout:
            h.timeouts += 1
        if malformed:
            h.malformed += 1

    def as_dict(self) -> dict:
        return {sid: h.to_dict() for sid, h in self._health.items()}
