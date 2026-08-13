"""Tests for the source health store."""
import pytest

from src.ingest.health import HealthStore


def test_health_record_failure_variants(tmp_path):
    store = HealthStore(tmp_path / "health.json")
    store.record_failure("s1", error="parse failed", malformed=True)
    store.record_failure("s1", error="timeout", timeout=True)
    h = store.get("s1")
    assert h.status == "error"
    assert h.total_errors == 2
    assert h.malformed == 1
    assert h.timeouts == 1
    assert h.consecutive_failures == 2


def test_health_success_resets_failure_counts(tmp_path):
    store = HealthStore(tmp_path / "health.json")
    store.record_failure("s1", error="boom")
    store.record_success("s1", items_found=5, fetched_at="2026-08-11T12:00:00Z")
    h = store.get("s1")
    assert h.status == "ok"
    assert h.consecutive_failures == 0
    assert h.items_found == 5
    assert h.last_success == "2026-08-11T12:00:00Z"


def test_health_load_missing_file_is_empty(tmp_path):
    store = HealthStore(tmp_path / "nope.json")
    store.load()
    assert store.as_dict() == {}


def test_health_all_reports_current_sources(tmp_path):
    store = HealthStore(tmp_path / "health.json")
    store.record_success("a", items_found=1, fetched_at="2026-08-11T12:00:00Z")
    store.record_failure("b", error="boom")
    data = store.as_dict()
    assert set(data.keys()) == {"a", "b"}
    assert data["a"]["status"] == "ok"
    assert data["b"]["status"] == "error"
