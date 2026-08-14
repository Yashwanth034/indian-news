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


# Article-extraction statuses that count as enrichment errors.  These
# mirror the status keys the extractor records per candidate; they are
# surfaced so a zero-story run can be diagnosed without digging into
# data/telegram_queue.json.
_ENRICHMENT_ERROR_STATUSES = (
    "http_error",
    "network_error",
    "timeout",
    "too_large",
    "not_html",
    "paywall",
    "no_text",
    "blocked",
    "domain_blocked",
    "non_article",
    "budget_exhausted",
)


def _telegram_observability_lines(stats: dict) -> list[str]:
    """Render build_telegram_queue stats as diagnostic log lines.

    Only reads the stats already collected by build_telegram_queue /
    build_telegram_stories -- no second statistics system.
    """
    lines: list[str] = []

    flt = stats.get("filter") or {}
    lines.append(
        f"telegram candidates received={flt.get('candidates', 0)}"
    )
    lines.append(
        "telegram freshness "
        f"fresh={flt.get('fresh', 0)} "
        f"stale={flt.get('stale', 0)} "
        f"no_effective_at={flt.get('no_effective_at', 0)} "
        f"non_news={flt.get('non_news_filtered', 0)} "
        f"kept={flt.get('kept', 0)}"
    )

    art = stats.get("article_extraction") or {}
    if art.get("error"):
        lines.append(f"telegram enrichment error={art['error']}")
    else:
        errors = {
            status: int(art.get(status, 0))
            for status in _ENRICHMENT_ERROR_STATUSES
        }
        error_total = sum(errors.values())
        lines.append(
            "telegram enrichment "
            f"eligible={art.get('eligible', 0)} "
            f"expanded={art.get('expanded', 0)} "
            f"cache_hits={art.get('cache_hits', 0)} "
            f"fetched={art.get('fetched', 0)} "
            f"errors={error_total}"
        )
        error_detail = " ".join(
            f"{status}={count}"
            for status, count in errors.items()
            if count
        )
        if error_detail:
            lines.append(f"telegram enrichment errors: {error_detail}")

    summ = stats.get("summarization") or {}
    lines.append(
        "telegram summarization "
        f"considered={summ.get('stories_considered', 0)} "
        f"summarized={summ.get('summarized', 0)} "
        f"article_source={summ.get('article_source', 0)} "
        f"rss_source={summ.get('rss_source', 0)} "
        f"rejected_insufficient={summ.get('rejected_insufficient', 0)} "
        f"rejected_verification={summ.get('rejected_verification', 0)} "
        f"rejected_quality={summ.get('rejected_quality', 0)}"
    )
    for problem in summ.get("problems") or []:
        reasons = ", ".join(problem.get("problems") or [])
        text = (problem.get("text") or "").strip()
        lines.append(
            "telegram summarization problem "
            f"story={problem.get('story', '?')} "
            f"stage={problem.get('stage', '?')} "
            f"text={text!r} "
            f"reasons={reasons!r}"
        )

    return lines


def _pipeline_observability_lines(result) -> list[str]:
    """Render candidate-gate outcomes as diagnostic log lines.

    Counts candidates by final status (queued / held / rejected / filler)
    and breaks the rejected set down by reason so a run that produced
    nothing is self-explanatory without opening data/telegram_queue.json.

    Tolerates thin fake results (SimpleNamespace) used in tests: it only
    reads ``candidates`` / ``queue`` and per-candidate ``status`` /
    ``reasons`` / ``priority`` / ``editorial`` via getattr.
    """
    candidates = list(getattr(result, "candidates", None) or [])
    queue = list(getattr(result, "queue", None) or [])

    held = [c for c in candidates if getattr(c, "status", "") == "held"]
    rejected = [c for c in candidates if getattr(c, "status", "") == "rejected"]
    filler = [c for c in candidates if getattr(c, "status", "") == "filler"]

    lines: list[str] = [
        f"pipeline candidates={len(candidates)} queued={len(queue)} "
        f"held={len(held)} rejected={len(rejected)} filler={len(filler)}"
    ]

    if held:
        lines.append(f"pipeline held: priority_below_high={len(held)}")

    if rejected:
        buckets = {"no_category": 0, "editorial_reject": 0, "geo_gate": 0,
                   "priority_blocked": 0}
        for c in rejected:
            reasons = " ".join(getattr(c, "reasons", None) or [])
            editorial = getattr(c, "editorial", None)
            priority = getattr(c, "priority", None)
            if "no category classified" in reasons:
                buckets["no_category"] += 1
            elif getattr(editorial, "decision", None) == "reject":
                buckets["editorial_reject"] += 1
            elif "state not identifiable" in reasons:
                buckets["geo_gate"] += 1
            elif getattr(priority, "blocked", False):
                buckets["priority_blocked"] += 1
        detail = " ".join(f"{k}={v}" for k, v in buckets.items() if v)
        lines.append(f"pipeline rejected: {detail}")

    if filler:
        lines.append(f"pipeline filler: editorial_filler={len(filler)}")

    return lines


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
    for line in _pipeline_observability_lines(result):
        print(f"main: {line}")

    try:
        stories, stats = build_telegram_queue(result, bundle, now_dt=now)
    except Exception as exc:
        _exit(f"fatal: telegram queue build failed: {exc}", exc=exc)

    for line in _telegram_observability_lines(stats):
        print(f"main: {line}")

    print(f"main: telegram queue written with {len(stories)} stories")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())