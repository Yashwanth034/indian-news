"""India news pipeline entry point.

Collects fresh articles from every enabled source, runs them through the
full NewsPipeline (relevance, classification, geography, dedupe,
editorial, priority, candidate gate), and builds the Telegram queue file
(``data/telegram_queue.json``).  Publishing is a separate step
(``python -m src.telegram_run``) so collection can be re-run without
re-sending anything.

Run as ``python -m src.main`` from the repository root.
"""
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from src.config_loader import get_config
from src.ingest.collect import collect_sources
from src.ingest.health import HealthStore
from src.pipeline.integration import NewsPipeline
from src.telebuild import build_telegram_queue

ROOT = Path(__file__).resolve().parent.parent


def _exit(msg: str, code: int = 1, *, exc: Exception | None = None) -> None:
    print(f"main: {msg}")
    if exc is not None:
        traceback.print_exc()
    sys.exit(code)


def main() -> int:
    try:
        bundle = get_config()
    except Exception as exc:
        _exit(f"fatal: cannot load India config: {exc}", exc=exc)

    cfg = bundle["config"]
    sources = bundle["sources"]["sources"]

    health_path = ROOT / (cfg.get("source_health_file") or "data/source_health.json")
    try:
        health_path.parent.mkdir(parents=True, exist_ok=True)
        health = HealthStore(health_path)
    except Exception as exc:
        print(f"main: warning: cannot open health store {health_path}: {exc}")
        health = None

    enabled = [s for s in sources if s.get("enabled")]
    print(
        f"main: collecting from {len(enabled)}/{len(sources)} enabled sources"
    )

    try:
        report = collect_sources(
            sources,
            health=health,
        )
    except Exception as exc:
        _exit(f"fatal: collection failed: {exc}", exc=exc)

    articles = report.articles
    print(
        f"main: collected {len(articles)} articles, "
        f"{len(report.ok_sources)} ok, {len(report.failed_sources)} failed"
    )
    for source_id in report.failed_sources:
        print(f"main:   failed source {source_id}: {report.status(source_id)}")

    now = datetime.now(timezone.utc)

    try:
        result = NewsPipeline(bundle).run(articles, now=now)
    except Exception as exc:
        _exit(f"fatal: pipeline run failed: {exc}", exc=exc)

    print(
        "main: pipeline "
        f"collected={result.collected} "
        f"normalized={result.normalized} "
        f"relevant={result.relevant} "
        f"events={result.events} "
        f"candidates={len(result.candidates)} "
        f"queued={len(result.queue)}"
    )

    try:
        stories, stats = build_telegram_queue(result, bundle, now_dt=now)
    except Exception as exc:
        _exit(f"fatal: telegram queue build failed: {exc}", exc=exc)

    print(f"main: telegram queue written with {len(stories)} stories")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())