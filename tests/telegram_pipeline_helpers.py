"""Shared test helper: pipeline-level telegram story builder.

Re-exports the production queue builder (``src.telebuild``) so
tests exercise the same code the orchestrator uses.  Groups
candidates into events, enriches the primary with article text
when present, builds the source-grounded briefing, and composes
one story per event.
"""
from src.telebuild import (
    build_telegram_stories,
)


def _build_telegram_stories(candidates, telegram_cfg, now_dt,
                            summarization_stats=None):
    """Test-local alias of the production builder."""
    return build_telegram_stories(
        candidates,
        telegram_cfg,
        now_dt,
        summarization_stats=summarization_stats,
    )
