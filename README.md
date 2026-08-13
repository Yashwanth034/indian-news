# India News 

Automated collection, triage and Telegram publishing of important, reliable India news.

The system is a fresh, independently maintainable implementation. It reuses proven
patterns from the [WorldNews Telegram](https://github.com/Yashwanth034/world-news-telegram)
project where appropriate — source-grounded summarization, event grouping, Telegram
scheduling/publishing, and article extraction. It is not a copy: the India-specific
editorial, relevance, geo-scope, echo/corroboration and source-ingestion layers are
designed for this project.

## Editorial goal

Important, useful and reliable India news only — not a dump.

**Excluded:** clickbait, gossip, celebrity/Bollywood, astrology, rumours, unverified
social-media claims, low-value minor incidents, duplicates, opinion-as-fact, routine
non-newsworthy government announcements, repetitive cricket/sports updates, and
state-only/local stories during Phase 1.

**Allowed:** major national events, state-level events with clear national significance,
and international events affecting India.

## How it works

Each run collects fresh articles, then pushes them through a staged pipeline:

```
collect → normalize → relevance → classify → geo-scope
        → dedupe/event-grouping → editorial gate → priority → queue
        → article enrichment → briefing → summarization → schedule → publish
```

Important stories (HIGH / URGENT / IMMEDIATE) always enter the queue; routine and
low-value content is filtered out. Deduplication groups the same event across sources
into a single story, and the scheduler publishes stories with pacing and volume caps
(hourly and daily ceilings) so bursts of genuine news are covered without spam.

Telegram publishing runs automatically from GitHub Actions on a schedule; the queue
and state files are persisted between runs to avoid re-posting.

## Layout

```
config/
  config.json          pipeline tuning (scores, thresholds, telegram limits)
  sources.json         source metadata
  categories.json      category definitions
  editorial.json       editorial gate patterns (clickbait/gossip/routine/...)
  india_entities.json  entity aliases, locations, ministry/agency terms
  india_geo.json       geo-scope config (national/state/local gate)
  schemas/             JSON Schema for every config file
src/
  main.py              pipeline entry point (collect → queue)
  ingest/              collection, fetch, parsing, health
  pipeline/            normalize, relevance, classify, geo, dedupe,
                       editorial, priority, integration
  telebuild.py         telegram candidate gate + story building
  telegram_briefing.py briefing construction
  telegram_summarizer.py extractive 2-5 sentence summarization
  telegram_scheduler.py publishing policy, pacing and caps
  telegram_run.py      schedule + publish runner
  telegram_formatter.py / telegram_media.py / telegram_publisher.py
  config_loader.py     load + validate all config files
.github/workflows/
  telegram.yml         scheduled collection + publishing
tests/
  test_*.py            unit + integration tests
data/                  runtime state (gitignored except queue/state)
```

## Requirements

Python 3.12.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
```

## Test

```bash
.venv/bin/python -m pytest tests -q
```

## Run locally

Collect fresh news and build the Telegram queue:

```bash
.venv/bin/python -m src.main
```

Schedule and publish (requires the Telegram secrets below):

```bash
.venv/bin/python -m src.telegram_run --yes
```

Useful flags on `telegram_run`:

- `--dry-run` — show which stories would be scheduled, send nothing.
- `--dry-run-publish` — simulate publishing against the schedule, send nothing.
- `--no-live` — save schedule state, send nothing.

## Secrets

Telegram credentials are supplied at run time via environment variables / GitHub
Actions secrets and are never stored in the repo:

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHANNEL_ID`

Publishing requires `TELEGRAM_PUBLISH=1` (or `--force`).

## Status

- [x] Phase 0 — scaffold, configuration, schemas, validation
- [x] Phase 1A — ingestion, relevance, categorization, geo-scope, dedup/echo,
      event memory, editorial gates, priority, queue generation
- [x] Phase 1B — article enrichment, briefing, summarization, Telegram stack
- [x] Phase 1C — GitHub Actions automation, live testing, tuning
- [ ] Phase 2 — state-level news (not started)
