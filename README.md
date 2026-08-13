# India News Telegram

Automated collection and Telegram publishing of important, reliable India news.
Phase 1 is India-wide only. Phase 2 (future) will add state-level news.

The system is a fresh, independently maintainable implementation that reuses
proven patterns from the [WorldNews Telegram](https://github.com/Yashwanth034/world-news-telegram)
project where appropriate (source-grounded summarization, event grouping,
Telegram scheduling/publishing, article extraction). It is not a copy: the
India-specific editorial, relevance, geo-scope, echo/corroboration and source
ingestion layers are designed for this project.

## Editorial goal

Important, useful and reliable India news only. Not a dump.

Excluded: clickbait, gossip, celebrity/Bollywood, astrology, rumours,
unverified social-media claims, low-value minor incidents, duplicates,
opinion-as-fact, routine non-newsworthy government announcements, repetitive
cricket/sports updates, and state-only/local stories during Phase 1.

Allowed: major national events, state-level events with clear national
significance, and international events affecting India.

## Categories

18 categories are defined in `config/categories.json` and are configurable
without changing pipeline code.

## Sources

Source configuration lives in `config/sources.json`. Every source declares its
own best ingestion method (RSS, official API, public endpoint, or public HTML
listing) rather than assuming RSS. All candidate endpoints are marked
`enabled: false` / `verified: false` until confirmed during Phase 1A.

## Layout

```
config/
  config.json          pipeline tuning (scores, thresholds, telegram)
  sources.json         source metadata
  categories.json      18 categories
  editorial.json       editorial gate patterns (clickbait/gossip/routine/...)
  india_entities.json  entity aliases, locations, ministry/agency terms
  india_geo.json       geo-scope config (national/state/local gate)
  schemas/             JSON Schema for every config file
src/
  config_loader.py     load + validate all config files (Phase 0)
tests/
  test_*.py            Phase 0 config sanity tests
data/                  runtime state (gitignored except queue/state)
```

## Environment

Python 3.12.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest tests -q
```

Secrets (Telegram bot token, channel id) are supplied at run time via
environment variables / GitHub Actions secrets and never stored in the repo.

## Status

- [x] Phase 0 — project scaffold, configuration structure, schemas, basic tests
- [ ] Phase 1A — ingestion, relevance, categorization, geo scope, dedup/echo,
      event memory, importance, editorial gates, queue generation
- [ ] Phase 1B — article enrichment, briefing, summarization, Telegram stack
- [ ] Phase 1C — GitHub Actions automation, live one-message test, tuning
- [ ] Phase 2 — state-level news (not started)
