"""Pipeline-level tests: article-enriched candidates through
build_telegram_stories and the rendered Telegram message.

Run with:  .venv/bin/python -m pytest src/test_article_enrichment.py -q
"""
from datetime import datetime, timedelta, timezone

import pytest

from tests.telegram_pipeline_helpers import build_telegram_stories
from src.telegram_formatter import build_message

NOW = datetime(2026, 8, 9, 7, 3, 10, tzinfo=timezone.utc)

CFG = {
    "just_in_freshness_minutes": 15,
    "max_briefing_sentences": 10,
    "target_message_chars": 1500,
    "max_message_chars": 3000,
}

HEADLINE = (
    "State of emergency declared as fast-moving Canada "
    "wildfire doubles in size"
)


def candidate(**overrides):
    base = {
        "story_id": "story-1",
        "id": "story-1",
        "event_id": "event-1",
        "title": HEADLINE,
        "summary": (
            "The Bald Range wildfire in British Columbia, "
            "still considered out of control, has spread over "
            "more than 36 sq miles."
        ),
        "url": "https://www.bbc.co.uk/news/articles/c1234",
        "source": "BBC World",
        "category": "disaster",
        "score": 70,
        "confidence": "medium",
        "priority_level": "HIGH",
        "event_status": "NEW",
        "primary_source": False,
        "tier": 2,
        "urgency_terms": ["wildfire", "state of emergency"],
        "effective_at": (
            NOW - timedelta(minutes=5)
        ).isoformat(),
    }
    base.update(overrides)
    return base


ARTICLE = [
    "Officials in British Columbia said the situation "
    "remained extremely dangerous.",
    "More than 20,000 residents have been forced from "
    "their homes so far.",
    "Fire crews are working to protect the town of "
    "Ashcroft from the advancing flames.",
    "The provincial government has opened three "
    "emergency shelters for displaced families.",
]


def build_one(cands):
    stories = build_telegram_stories(cands, CFG, NOW)
    return stories[0]


class TestArticleEnrichmentPipeline:
    def test_article_sentences_enter_briefing(self):
        story = build_one(
            [candidate(article_sentences=list(ARTICLE))]
        )
        rows = story["briefing"]["sentences"]
        texts = {
            r["text"] for r in rows
        }
        # The key consequence fact leads the concise summary.
        assert any(
            s.startswith("More than 20,000 residents")
            for s in texts
        )
        # Conciseness: the summary never balloons to the full
        # article; secondary detail ("Fire crews are working")
        # is dropped in favour of the core facts.
        assert 2 <= len(rows) <= 3

    def test_article_sentences_capped_at_ten(self):
        body = list(ARTICLE) + [
            "Additional sentence number %d here." % i
            for i in range(1, 30)
        ]
        story = build_one(
            [candidate(article_sentences=body)]
        )
        assert len(story["briefing"]["sentences"]) <= 10

    def test_no_article_sentences_briefing_unchanged(self):
        # Without article extraction the RSS summary is the
        # source; with article extraction the article rows
        # become the primary source and come first.
        rss = (
            "Officials said 20,000 residents have been forced "
            "from their homes so far. "
            "Fire crews are working through the night to "
            "contain the blaze."
        )
        plain = build_one([candidate(summary=rss)])
        with_article = build_one(
            [
                candidate(
                    summary=rss,
                    article_sentences=list(ARTICLE),
                )
            ]
        )
        assert plain["briefing"]["source"] == "BBC World"
        assert with_article["briefing"]["source"] == "BBC World"
        assert len(with_article["briefing"]["sentences"]) >= len(
            plain["briefing"]["sentences"]
        )
        # Article facts lead the concise summary when available.
        article_texts = [
            r["text"] for r in with_article["briefing"]["sentences"]
        ]
        assert any(
            t.startswith("Officials in British Columbia said")
            for t in article_texts
        )
        assert any(
            "20,000 residents" in t for t in article_texts
        )
        # The summary stays concise; an RSS row that merely
        # restates an already-covered article fact is dropped.
        assert not any(
            "20,000 residents" in t
            and t.startswith("Officials said")
            for t in article_texts
        )
        assert not any(
            t.startswith("Fire crews are working through")
            for t in article_texts
        )

    def test_headline_paraphrase_article_sentences_dropped(self):
        story = build_one(
            [
                candidate(
                    article_sentences=[
                        "The fast-moving Canada wildfire "
                        "doubles in size.",
                        "Officials confirmed the fire had "
                        "spread across the province.",
                    ]
                )
            ]
        )
        texts = [
            r["text"]
            for r in story["briefing"]["sentences"]
        ]
        assert not any(
            "doubles in size" in t for t in texts
        )
        assert any(
            "spread across the province" in t for t in texts
        )

    def test_filler_article_sentences_dropped(self):
        story = build_one(
            [
                candidate(
                    article_sentences=[
                        "This is a developing story.",
                        "Officials said the fire was still "
                        "growing this morning.",
                    ]
                )
            ]
        )
        texts = [
            r["text"]
            for r in story["briefing"]["sentences"]
        ]
        assert not any(
            "developing story" in t for t in texts
        )

    def test_duplicate_article_sentence_dropped(self):
        story = build_one(
            [
                candidate(
                    article_sentences=[
                        ARTICLE[0],
                        ARTICLE[0],
                        ARTICLE[1],
                    ]
                )
            ]
        )
        texts = [
            r["text"]
            for r in story["briefing"]["sentences"]
        ]
        assert texts.count(ARTICLE[0]) == 1

    def test_headline_only_story_still_rejected(self):
        cands = [
            candidate(
                summary="A fast-moving Canada wildfire "
                "doubles in size.",
            )
        ]
        stories = build_telegram_stories(cands, CFG, NOW)
        assert stories == []

    def test_article_sentences_rescue_headline_only_story(self):
        cands = [
            candidate(
                summary="A fast-moving Canada wildfire "
                "doubles in size.",
                article_sentences=list(ARTICLE),
            )
        ]
        stories = build_telegram_stories(cands, CFG, NOW)
        assert len(stories) == 1

    def test_provenance_attributed_to_primary_source(self):
        story = build_one(
            [candidate(article_sentences=list(ARTICLE))]
        )
        sources = {
            r["source"] for r in story["briefing"]["sentences"]
        }
        assert sources == {"BBC World"}

    def test_no_user_visible_expansion_marker(self):
        story = build_one(
            [candidate(article_sentences=list(ARTICLE))]
        )
        msg = build_message(story, CFG, NOW)
        lowered = (msg.get("text") or "").lower()
        assert "expanded" not in lowered
        assert "supplemented" not in lowered
        assert "article text" not in lowered

    def test_default_cap_unchanged_without_article(self):
        # Without an article the RSS summary is the source and
        # the composed summary stays concise: it targets two
        # sentences and keeps the key consequence, dropping the
        # secondary detail that filled the old higher cap.
        cfg = dict(CFG)
        cfg["max_briefing_sentences"] = 10
        cands = [
            candidate(
                summary=(
                    "Officials said the fire had grown. "
                    "Residents were told to evacuate. "
                    "The town of Ashcroft was threatened. "
                    "Three shelters have opened. "
                    "Crews arrived overnight. "
                    "Power lines were cut. "
                    "Roads were closed. "
                    "Drones surveyed the area. "
                    "Tankers dropped water. "
                    "Helicopters joined the effort. "
                    "Schools were closed on Friday. "
                    "A curfew was imposed."
                )
            )
        ]
        story = build_telegram_stories(cands, cfg, NOW)[0]
        rows = story["briefing"]["sentences"]
        assert 2 <= len(rows) <= 5
        texts = [r["text"] for r in rows]
        # The key consequence (evacuation) survives the concise
        # selection; the summary does not balloon to the full
        # RSS summary.
        assert any("evacuate" in t for t in texts)
        assert len(rows) <= 3


# ---------------------------------------------------------------------------
# FIX 3 - minimum two meaningful explanatory sentences (pipeline level)
# ---------------------------------------------------------------------------


class TestMinimumTwoSentencesPipeline:
    def test_article_one_sentence_rejected_without_rss(self):
        # Article extraction yields a single sentence and the
        # RSS summary is headline-only: the story has no
        # explanatory content at all and is dropped.
        cands = [
            candidate(
                summary=(
                    "A fast-moving Canada wildfire doubles "
                    "in size."
                ),
                article_sentences=[
                    "Officials said the fire remained "
                    "dangerous.",
                ],
            )
        ]
        stories = build_telegram_stories(cands, CFG, NOW)
        assert stories == []

    def test_article_one_sentence_still_rejected_with_one_rss_sentence(self):
        # One useful RSS sentence is not enough: a single
        # article sentence is never attached to the briefing
        # (article enrichment requires at least two), so the
        # story stays below the two-sentence gate and is
        # rejected at the pipeline - it never enters the
        # telegram queue.
        cands = [
            candidate(
                summary=(
                    "A fast-moving Canada wildfire doubles "
                    "in size. Officials said 20,000 residents "
                    "have been forced from their homes."
                ),
                article_sentences=[
                    "Officials said the fire remained "
                    "dangerous.",
                ],
            )
        ]
        stories = build_telegram_stories(cands, CFG, NOW)
        assert stories == []

    def test_article_accepted_when_rss_provides_second_useful_sentence(self):
        cands = [
            candidate(
                summary=(
                    "A fast-moving Canada wildfire doubles "
                    "in size. Officials said 20,000 residents "
                    "have been forced from their homes. "
                    "Crews are working through the night to "
                    "contain the blaze."
                ),
                article_sentences=[
                    "Officials said the fire remained "
                    "dangerous.",
                ],
            )
        ]
        stories = build_telegram_stories(cands, CFG, NOW)
        assert len(stories) == 1
        msg = build_message(stories[0], CFG, NOW)
        assert msg is not None

    def test_article_two_sentences_accepted(self):
        cands = [
            candidate(
                summary=(
                    "A fast-moving Canada wildfire doubles "
                    "in size."
                ),
                article_sentences=[
                    "Officials said the fire remained "
                    "dangerous.",
                    "Crews are protecting the town of "
                    "Ashcroft from the flames.",
                ],
            )
        ]
        stories = build_telegram_stories(cands, CFG, NOW)
        assert len(stories) == 1
        msg = build_message(stories[0], CFG, NOW)
        assert msg is not None

    def test_maximum_five_meaningful_sentences_preserved(self):
        cfg = dict(CFG)
        cfg["max_briefing_sentences"] = 10
        article = [
            "The Bald Range wildfire has grown to cover 36 "
            "square miles.",
            "Officials said 20,000 residents have been "
            "evacuated so far.",
            "Crews are dropping water on the blaze from "
            "helicopters.",
            "The fire started near the town of Ashcroft on "
            "Sunday.",
            "Winds of up to 40 mph are pushing the flames "
            "northeast.",
            "Authorities have declared a state of emergency "
            "in the region.",
            "Smoke is drifting across the border into "
            "Alberta.",
            "Local shelters are reporting they are at "
            "capacity.",
            "Evacuation orders now cover eight communities.",
            "Temperatures are forecast to stay above 30 "
            "degrees this week.",
            "Police are patrolling evacuated neighbourhoods "
            "to deter looters.",
            "Power lines have been cut to prevent new "
            "ignitions.",
            "Rail services through the valley have been "
            "suspended.",
            "The province has asked the military for "
            "assistance.",
            "Air quality readings in Kamloops reached "
            "hazardous levels.",
            "Wildlife officers are moving animals away from "
            "the fire zone.",
            "A donation fund has been set up for displaced "
            "families.",
        ]
        cands = [
            candidate(
                summary=(
                    "A fast-moving Canada wildfire doubles "
                    "in size."
                ),
                article_sentences=article,
            )
        ]
        stories = build_telegram_stories(cands, cfg, NOW)
        assert len(stories) == 1
        rows = stories[0]["briefing"]["sentences"]
        assert 2 <= len(rows) <= 5
        # Every retained row must carry real text; the
        # headline-only RSS sentence never enters the list.
        assert all(r["text"] for r in rows)
        assert any("evacuated" in r["text"] for r in rows)
        assert any("square miles" in r["text"] for r in rows)
        msg = build_message(stories[0], cfg, NOW)
        assert msg is not None
