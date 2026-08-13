"""Regression tests for the source-grounded summarizer.

Covers: 2-8 sentence summaries, source-supported
paraphrasing, preservation of numbers/names/dates,
attribution and uncertainty, conflicting facts, no invented
information, duplicate fact removal, insufficient source
content rejection, truncated/junk removal, and unchanged
final output format.

Run with:  .venv/bin/python -m pytest src/test_telegram_summarizer.py -q
"""
from datetime import datetime, timedelta, timezone

from src.telegram_formatter import build_message
from src.telegram_summarizer import (
    front_attribution,
    quality_check_sentence,
    summarize_rows,
    verify_row,
)
from tests.telegram_pipeline_helpers import build_telegram_stories

NOW = datetime(2026, 8, 10, 8, 0, 0, tzinfo=timezone.utc)

CFG = {
    "just_in_freshness_minutes": 15,
    "max_briefing_sentences": 10,
    "target_message_chars": 1500,
    "max_message_chars": 3000,
    "summarization": {
        "min_sentences": 2,
        "max_sentences": 8,
        "tier1_max": 6,
    },
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


def story(rows):
    """A queue-style story whose briefing carries `rows`."""
    return {
        "story_id": "story-1",
        "title": HEADLINE,
        "headline": HEADLINE,
        "source": "BBC World",
        "url": "https://www.bbc.co.uk/news/articles/c1234",
        "briefing": {"sentences": rows},
    }


def row(text, item_id="story-1"):
    return {"text": text, "source": "BBC World", "item_id": item_id}


def summarize(texts, source=None, headline=HEADLINE):
    """Helper: run the full summarize_rows pipeline."""
    rows = [
        row(t) for t in texts
    ]
    return summarize_rows(
        rows,
        source or " ".join(texts),
        headline,
        cfg=CFG["summarization"],
    )


# ---------------------------------------------------------------------------
# 1. Clear 2-8 sentence summaries
# ---------------------------------------------------------------------------


class TestSummaryLength:
    def test_rich_source_yields_between_two_and_eight(self):
        texts = [
            "More than 20,000 residents have been forced from "
            "their homes.",
            "The fire started near the town of Ashcroft on "
            "Sunday.",
            "Winds of up to 40 mph are pushing the flames "
            "northeast.",
            "Temperatures are forecast to stay above 30 "
            "degrees this week.",
            "Evacuation orders now cover eight communities.",
            "Air quality readings in Kamloops reached "
            "hazardous levels.",
            "Police are patrolling evacuated neighbourhoods "
            "to deter looters.",
            "Power lines have been cut to prevent new "
            "ignitions.",
            "Rail services through the valley have been "
            "suspended.",
            "The province has asked the military for "
            "assistance.",
        ]
        kept, stats = summarize(texts)
        assert kept is not None
        assert 2 <= len(kept) <= 8
        assert stats["rejected"] is None

    def test_never_padded_beyond_source(self):
        texts = [
            "More than 20,000 residents have been forced from "
            "their homes.",
            "The fire started near the town of Ashcroft on "
            "Sunday.",
        ]
        kept, _ = summarize(texts)
        assert len(kept) == 2

    def test_pipeline_story_same_limits(self):
        cands = [
            candidate(
                summary=(
                    "The Bald Range wildfire has grown to "
                    "cover 36 square miles. "
                    "Officials said 20,000 residents have been "
                    "evacuated so far. "
                    "Crews are dropping water on the blaze "
                    "from helicopters. "
                    "The fire started near the town of "
                    "Ashcroft on Sunday."
                )
            )
        ]
        stories = build_telegram_stories(cands, CFG, NOW)
        assert len(stories) == 1
        rows = stories[0]["briefing"]["sentences"]
        assert 2 <= len(rows) <= 8


# ---------------------------------------------------------------------------
# 2. Source-supported paraphrasing
# ---------------------------------------------------------------------------


class TestParaphrasing:
    def test_trailing_attribution_fronted(self):
        assert front_attribution(
            "Three people were injured, police said."
        ) == "Police said three people were injured."

    def test_according_to_fronted(self):
        assert front_attribution(
            "The road was closed, according to officials."
        ) == "According to officials, the road was closed."

    def test_facts_survive_rewrite_verbatim(self):
        # The source carries both the attributed paragraph form
        # and the independent lead sentence; the composed
        # attribution-fronted sentences must verify against it.
        sentences = [
            "Three people were injured, police said.",
            "The road was closed, according to officials.",
        ]
        source = (
            "Police said three people were injured. "
            "According to officials, the road was closed. "
            + " ".join(sentences)
        )
        kept, _ = summarize(
            sentences,
            source=source,
            headline="Crash closes highway",
        )
        texts = [r["text"] for r in kept]
        assert "Police said three people were injured." in texts
        assert (
            "According to officials, the road was closed."
            in texts
        )

    def test_unsafe_trailing_attribution_untouched(self):
        # Body ends with terminal punctuation: two sentences,
        # never fused.
        assert front_attribution(
            "He was seen fleeing. Police said."
        ) == "He was seen fleeing. Police said."
        # Quoted body is never spliced.
        assert front_attribution(
            '"We are proud", the mayor said.'
        ) == '"We are proud", the mayor said.'


# ---------------------------------------------------------------------------
# 3. Preservation of numbers, names, dates
# ---------------------------------------------------------------------------


class TestFactPreservation:
    def test_numbers_names_dates_preserved(self):
        texts = [
            "More than 20,000 residents have been forced from "
            "their homes.",
            "The fire started near the town of Ashcroft on "
            "Sunday.",
            "Winds of up to 40 mph are pushing the flames "
            "northeast.",
        ]
        kept, _ = summarize(texts)
        joined = " ".join(r["text"] for r in kept)
        assert "20,000" in joined
        assert "Ashcroft" in joined
        assert "Sunday" in joined
        assert "40 mph" in joined

    def test_numbers_never_altered(self):
        # 36 vs 63: the exact figure must survive untouched.
        text = "The fire has spread over more than 36 sq miles."
        kept, _ = summarize(
            [text, "The fire started near Ashcroft on Sunday."],
            source=" ".join(
                [text, "The fire started near Ashcroft on Sunday."]
            ),
            headline="Fire grows in the valley",
        )
        assert any("36 sq miles" in r["text"] for r in kept)


# ---------------------------------------------------------------------------
# 4. Attribution and uncertainty
# ---------------------------------------------------------------------------


class TestAttribution:
    def test_attribution_not_confirmed(self):
        text = "Police said the crash killed two people."
        kept, _ = summarize(
            [text, "The road was closed, according to officials."],
            source=(
                text
                + " The road was closed. "
                "The road was closed, according to officials."
            ),
            headline="Crash reported on motorway",
        )
        joined = " ".join(r["text"] for r in kept)
        assert "said" in joined
        # No confirmation wording is invented.
        assert "confirmed" not in joined

    def test_uncertainty_preserved(self):
        text = "Officials said the number could rise."
        kept, _ = summarize(
            [text, "Residents were told to stay indoors."],
            source=" ".join(
                [text, "Residents were told to stay indoors."]
            ),
            headline="Officials assess storm damage",
        )
        assert any("could" in r["text"] for r in kept)


# ---------------------------------------------------------------------------
# 5. Conflicting facts
# ---------------------------------------------------------------------------


class TestConflicts:
    def test_conflicting_numbers_never_merged(self):
        # The article is the primary source: its 12,000 figure
        # wins and the conflicting RSS 20,000 is dropped, so
        # the summary never merges the two claims.
        cands = [
            candidate(
                summary=(
                    "A fast-moving Canada wildfire doubles "
                    "in size. "
                    "Officials said 20,000 residents were "
                    "evacuated."
                ),
                article_sentences=[
                    "Officials said 12,000 residents were "
                    "evacuated.",
                    "The fire has grown to cover 36 square "
                    "miles.",
                ],
            )
        ]
        stories = build_telegram_stories(cands, CFG, NOW)
        assert len(stories) == 1
        joined = " ".join(
            r["text"]
            for r in stories[0]["briefing"]["sentences"]
        )
        # Only one figure survives; the two are never combined.
        assert ("20,000" in joined) != ("12,000" in joined)
        assert "12,000" in joined

    def test_article_primary_source_wins_conflict(self):
        cands = [
            candidate(
                summary=(
                    "A fast-moving Canada wildfire doubles "
                    "in size. "
                    "Officials said 20,000 residents were "
                    "evacuated."
                ),
                article_sentences=[
                    "Officials said 12,000 residents were "
                    "evacuated.",
                    "The fire has grown to cover 36 square "
                    "miles.",
                ],
            )
        ]
        stories = build_telegram_stories(cands, CFG, NOW)
        joined = " ".join(
            r["text"]
            for r in stories[0]["briefing"]["sentences"]
        )
        assert "12,000" in joined
        assert "20,000" not in joined


# ---------------------------------------------------------------------------
# 6. No invented information
# ---------------------------------------------------------------------------


class TestNoInvention:
    def test_verification_blocks_unsupported_numbers(self):
        row = {
            "text": "The fire killed 4,000 people.",
            "item_id": "x",
        }
        ok, problems = verify_row(row, "The fire grew overnight.")
        assert not ok
        assert any("4000" in p for p in problems)

    def test_verification_blocks_unsupported_names(self):
        row = {
            "text": "Mayor Kamloops declared the emergency.",
            "item_id": "x",
        }
        ok, problems = verify_row(row, "The fire grew overnight.")
        assert not ok
        assert any("kamloops" in p for p in problems)

    def test_composed_summary_never_adds_facts(self):
        texts = [
            "The fire has grown to cover 36 square miles.",
            "Officials said 20,000 residents have been "
            "evacuated.",
        ]
        kept, _ = summarize(texts)
        from src.telegram_summarizer import extract_facts
        source_numbers = {"36", "20000"}
        for r in kept:
            facts = extract_facts(r["text"])
            assert facts["numbers"] <= source_numbers


# ---------------------------------------------------------------------------
# 7. Duplicate fact removal
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_identical_sentences_count_once(self):
        text = "More than 20,000 residents have been evacuated."
        kept, _ = summarize(
            [text, text, "Winds of up to 40 mph are pushing "
             "the flames northeast."]
        )
        assert len(kept) == 2
        joined = " ".join(r["text"] for r in kept)
        assert joined.count("20,000") == 1

    def test_cross_source_same_consequence_kept_once(self):
        # RSS and the article report the same evacuation
        # consequence; the article is the primary source and
        # also carries a distinct fact.  The consequence
        # appears exactly once in the final briefing.
        cands = [
            candidate(
                summary=(
                    "Officials said 20,000 residents were "
                    "evacuated."
                ),
                article_sentences=[
                    "More than 20,000 residents were evacuated.",
                    "The fire has grown to cover 36 square "
                    "miles.",
                ],
            )
        ]
        stories = build_telegram_stories(cands, CFG, NOW)
        assert len(stories) == 1
        joined = " ".join(
            r["text"]
            for r in stories[0]["briefing"]["sentences"]
        )
        assert joined.count("20,000") == 1
        assert "36 square miles" in joined

    def test_distinct_facts_both_kept(self):
        texts = [
            "20,000 residents were evacuated.",
            "40 mph winds are pushing the flames northeast.",
        ]
        kept, _ = summarize(texts)
        joined = " ".join(r["text"] for r in kept)
        assert "20,000" in joined
        assert "40 mph" in joined

    def test_same_unit_fact_counted_once(self):
        # Several sentences repeat the same factual claim
        # ("depth of 10 km"); it must appear exactly once even
        # though impact-signature dedup cannot see it.
        texts = [
            "The quake was recorded at a depth of 10 km.",
            "The tremor was recorded at a depth of 10 km.",
            "The earthquake struck at a depth of 10 km.",
            "Officials said damage was not reported.",
        ]
        kept, _ = summarize(texts)
        joined = " ".join(r["text"] for r in kept)
        assert joined.count("10 km") == 1
        assert "damage was not reported" in joined
        assert len(kept) == 2

    def test_distinct_numbers_same_unit_both_kept(self):
        # "10 km" and "15 km" are different claims and both stay.
        texts = [
            "The quake was recorded at a depth of 10 km.",
            "The aftershock was centred 15 km away.",
        ]
        kept, _ = summarize(texts)
        joined = " ".join(r["text"] for r in kept)
        assert "10 km" in joined
        assert "15 km" in joined


# ---------------------------------------------------------------------------
# 8. Insufficient source content -> rejected
# ---------------------------------------------------------------------------


class TestInsufficientContent:
    def test_single_sentence_rejected(self):
        kept, stats = summarize(
            ["The fire is growing near the town."]
        )
        assert kept is None
        assert stats["rejected"] == "insufficient_information"

    def test_headline_only_rejected(self):
        kept, stats = summarize(
            ["A fast-moving Canada wildfire doubles in size."]
        )
        assert kept is None

    def test_pipeline_never_queues_below_two(self):
        cands = [
            candidate(
                summary=(
                    "A fast-moving Canada wildfire doubles "
                    "in size. "
                    "Officials said the fire remains "
                    "dangerous."
                )
            )
        ]
        stats = {}
        stories = build_telegram_stories(
            cands, CFG, NOW, summarization_stats=stats
        )
        assert stories == []
        assert stats["rejected_insufficient"] == 1


# ---------------------------------------------------------------------------
# 9. Truncated / junk article content -> removed
# ---------------------------------------------------------------------------


class TestJunkRemoval:
    def test_truncated_tail_dropped(self):
        # The source says "their healthy 14-year-old son";
        # a summary sentence ending "...their healthy." is a
        # feed truncation and must be dropped by verification.
        source = (
            "The families were left in shock after the sudden "
            "death of their healthy 14-year-old son."
        )
        text = (
            "The families were left in shock after the sudden "
            "death of their healthy."
        )
        kept, stats = summarize(
            [text],
            source=source,
            headline="Families demand answers",
        )
        if kept is not None:
            assert "their healthy." not in " ".join(
                r["text"] for r in kept
            )
        else:
            assert stats["rejected"] in (
                "insufficient_information",
                "verification",
            )

    def test_nav_junk_line_rejected(self):
        kept, stats = summarize(
            ["Recommended stories for you."]
        )
        assert kept is None
        assert stats["rejected"] == "insufficient_information"

    def test_boilerplate_sentence_dropped(self):
        kept, stats = summarize(
            [
                "Sign up for our newsletter to get more "
                "stories like this.",
                "Officials said the fire remains dangerous.",
                "Evacuation orders now cover eight communities.",
            ]
        )
        assert kept is not None
        assert not any(
            "newsletter" in r["text"] for r in kept
        )

    def test_app_promotion_sentence_dropped(self):
        kept, stats = summarize(
            [
                "The agency also directed users to its BhooKamp "
                "app for more information on the earthquake.",
                "Officials said the quake struck at dawn.",
                "No damage was reported so far.",
            ]
        )
        assert kept is not None
        assert not any(
            "app" in r["text"] or "BhooKamp" in r["text"]
            for r in kept
        )

    def test_update_status_sentence_dropped(self):
        kept, stats = summarize(
            [
                "Further details, including any reports of "
                "damage, were awaited.",
                "The quake struck Leh at 6:05 am.",
                "Tremors were felt across the region.",
            ]
        )
        assert kept is not None
        assert not any(
            "awaited" in r["text"] for r in kept
        )

    def test_story_being_updated_sentence_dropped(self):
        kept, stats = summarize(
            [
                "This story is being updated.",
                "The quake struck Leh at 6:05 am.",
                "Officials said no damage was reported.",
            ]
        )
        assert kept is not None
        assert not any(
            "being updated" in r["text"] for r in kept
        )

    def test_quality_gate_flags_incomplete_sentence(self):
        problems = quality_check_sentence(
            "The fire grew rapidly",
            HEADLINE,
        )
        assert any("complete" in p for p in problems)


# ---------------------------------------------------------------------------
# 9b. Generic educational explainers -> never used while event facts exist
# ---------------------------------------------------------------------------


class TestGenericExplainers:
    def test_generic_comparison_not_used_when_event_facts_exist(self):
        # "Shallow earthquakes are generally more dangerous than
        # deep earthquakes." is a generic educational filler; with
        # two event-specific sentences available it must not be
        # appended to reach a third sentence.
        kept, stats = summarize(
            [
                "The 5.5-magnitude earthquake in Leh was "
                "recorded at a depth of 10 km, placing it in "
                "the shallow category.",
                "As of the latest available information, there "
                "were no immediate reports of casualties or "
                "major structural damage following Thursday "
                "morning's tremor.",
                "Shallow earthquakes are generally more "
                "dangerous than deep earthquakes.",
            ]
        )
        assert kept is not None
        joined = " ".join(r["text"] for r in kept)
        assert "generally more dangerous" not in joined
        assert "depth of 10 km" in joined
        assert "no immediate reports" in joined
        assert len(kept) == 2

    def test_tend_to_be_explainer_not_used(self):
        kept, stats = summarize(
            [
                "The quake struck Leh at 6:05 am IST.",
                "Some aftershocks tend to be weaker than the "
                "main quake.",
                "Officials said no damage was reported at the "
                "site.",
            ]
        )
        assert kept is not None
        joined = " ".join(r["text"] for r in kept)
        assert "tend to be" not in joined
        assert len(kept) == 2

    def test_typically_more_than_explainer_dropped(self):
        kept, stats = summarize(
            [
                "Deep quakes are typically less damaging than "
                "shallow ones.",
                "The quake struck Leh at 6:05 am IST.",
                "Officials said no damage was reported at the "
                "site.",
            ]
        )
        assert kept is not None
        joined = " ".join(r["text"] for r in kept)
        assert "typically less damaging" not in joined
        assert len(kept) == 2

    def test_explainer_allowed_only_to_reach_minimum(self):
        # With a single genuine event sentence, a generic
        # explainer pads exactly to the two-sentence minimum.
        kept, stats = summarize(
            [
                "The quake struck Leh at 6:05 am IST.",
                "Some aftershocks tend to be weaker than the "
                "main quake.",
            ]
        )
        assert kept is not None
        joined = " ".join(r["text"] for r in kept)
        assert "tend to be" in joined
        assert len(kept) == 2

    def test_generic_explainer_only_when_under_minimum(self):
        # A source that is only a generic explainer never builds a
        # padded summary from it alone.
        kept, stats = summarize(
            [
                "Shallow earthquakes are generally more "
                "dangerous than deep earthquakes.",
            ]
        )
        assert kept is None


# ---------------------------------------------------------------------------
# 10. Final output format unchanged
# ---------------------------------------------------------------------------


class TestFinalFormat:
    def test_message_structure_unchanged(self):
        cands = [
            candidate(
                effective_at=(
                    NOW - timedelta(minutes=60)
                ).isoformat(),
                summary=(
                    "The Bald Range wildfire has grown to "
                    "cover 36 square miles. "
                    "Officials said 20,000 residents have been "
                    "evacuated so far. "
                    "Crews are dropping water on the blaze "
                    "from helicopters."
                ),
            )
        ]
        stories = build_telegram_stories(cands, CFG, NOW)
        msg = build_message(stories[0], CFG, NOW)
        assert msg is not None
        assert msg["parse_mode"] == "HTML"
        text = msg["text"]
        lines = text.split("\n\n")
        # label, headline, summary, source, read-more
        assert len(lines) == 5
        assert lines[0] == "\U0001F4F0 NEWS"
        assert lines[1] == "<b>" + HEADLINE + "</b>"
        assert lines[3] == "\U0001F4F0 Source: BBC World"
        assert lines[4].startswith("\U0001F517 ")
        assert "Read the full report" in lines[4]
        assert "\u2b50" not in text
        assert "\U0001F4CD" not in text

    def test_two_sentence_message_renders(self):
        cands = [
            candidate(
                summary=(
                    "The Bald Range wildfire has grown to "
                    "cover 36 square miles. "
                    "Officials said 20,000 residents have been "
                    "evacuated so far."
                )
            )
        ]
        stories = build_telegram_stories(cands, CFG, NOW)
        msg = build_message(stories[0], CFG, NOW)
        assert msg is not None

    def test_summary_body_matches_briefing_rows(self):
        cands = [
            candidate(
                summary=(
                    "The Bald Range wildfire has grown to "
                    "cover 36 square miles. "
                    "Officials said 20,000 residents have been "
                    "evacuated so far."
                )
            )
        ]
        stories = build_telegram_stories(cands, CFG, NOW)
        rows = stories[0]["briefing"]["sentences"]
        msg = build_message(stories[0], CFG, NOW)
        assert msg is not None
        for r in rows:
            assert r["text"] in msg["text"]


# ---------------------------------------------------------------------------
# 11. Quote-boundary fact preservation
# ---------------------------------------------------------------------------


class TestQuoteBoundaryFactPreservation:
    """A sentence following a closing quote must be recognized as a
    separate sentence, and its facts must survive the whole pipeline.

    The formatter's split_sentences cannot see a period trapped
    inside a closing quote (". " The IMF said ...), so without this
    fix the IMF sentence fuses with the quoted fragment. When that
    fused row is the only sentence the story is rejected for having
    fewer than two sentences and the statistic is lost."""

    def test_post_quote_sentence_recognized_as_separate(self):
        rows = [
            {
                "text": (
                    '"We should reduce food." The International '
                    "Monetary Fund said today that subsidies must "
                    "fall by 15 percent."
                ),
                "source": "BBC",
                "item_id": "i",
            }
        ]
        kept, stats = summarize_rows(
            rows,
            '"We should reduce food." The International Monetary '
            "Fund said today that subsidies must fall by 15 "
            "percent.",
            "IMF warns on food subsidies",
            cfg=CFG["summarization"],
        )
        assert stats["rejected"] is None
        texts = [r["text"] for r in kept]
        # The post-quote sentence is its own fact-bearing row.
        assert any(
            t.startswith("The International Monetary Fund")
            for t in texts
        )
        # The statistic survives.
        assert any("15 percent" in t for t in texts)

    def test_single_fused_quote_row_no_longer_rejected(self):
        # The story's only row fuses the quote with the IMF sentence.
        # Before the fix this collapsed to a single sentence and the
        # story was rejected as <2 sentences, losing the statistic.
        text = (
            '"We should reduce food." The International Monetary '
            "Fund said today that subsidies must fall by 15 percent."
        )
        kept, stats = summarize_rows(
            [{"text": text, "source": "BBC", "item_id": "i"}],
            text,
            "IMF warns on food subsidies",
            cfg=CFG["summarization"],
        )
        assert stats["rejected"] is None
        joined = " ".join(r["text"] for r in kept)
        assert "15 percent" in joined
        assert "International Monetary Fund" in joined

    def test_curly_quote_boundary_split(self):
        # A curly closing quote works exactly like a straight one.
        text = (
            "\u201cWe should reduce food.\u201d The International "
            "Monetary Fund said today that subsidies must fall by "
            "15 percent."
        )
        kept, stats = summarize_rows(
            [{"text": text, "source": "BBC", "item_id": "i"}],
            text,
            "IMF warns on food subsidies",
            cfg=CFG["summarization"],
        )
        assert stats["rejected"] is None
        texts = [r["text"] for r in kept]
        assert any(
            t.startswith("The International Monetary Fund")
            for t in texts
        )
        assert any("15 percent" in t for t in texts)

    def test_continuous_quoted_sentence_not_split(self):
        # Two sentences inside ONE quote must stay a single sentence;
        # only a boundary after the closing quote is split.
        from src.telegram_summarizer import _split_quote_boundaries
        text = (
            '"I honestly can\u2019t remember the last time I bought '
            'red meat. We replaced it with chicken," she told DW.'
        )
        rows = _split_quote_boundaries(
            [{"text": text, "source": "BBC", "item_id": "i"}],
            "Iranians feel the cost of war",
        )
        assert len(rows) == 1
        assert "red meat. We replaced it with chicken" in rows[0]["text"]
        assert rows[0]["text"].endswith('she told DW.')

    def test_quote_split_only_at_closing_quote(self):
        # A comma-quote (no trapped period) is never split.
        from src.telegram_summarizer import _split_quote_boundaries
        text = '"We should reduce food," the IMF said.'
        rows = _split_quote_boundaries(
            [{"text": text, "source": "BBC", "item_id": "i"}],
            "IMF food advice",
        )
        assert len(rows) == 1
        assert rows[0]["text"].startswith('"We should reduce food,"')


# ---------------------------------------------------------------------------
# 12. Live-blog relevance filtering
# ---------------------------------------------------------------------------


class TestLiveBlogRelevance:
    """Unrelated stories appended by a live-blog or multi-topic
    source must not enter the summary; same-event context survives."""

    HEADLINE = "Canada wildfire doubles in size"

    LIVE_BLOG = (
        "The fire has grown to cover 36 square miles. "
        "An international effort is under way, with crews from "
        "the US, Canada and Mexico. "
        "Meanwhile, the International Monetary Fund warned that "
        "the country must cut spending by 15 percent. "
        "Separately, a total solar eclipse will cross the sky "
        "tonight, astronomers said."
    )

    def _summary(self, texts, headline=HEADLINE):
        from src.formatter import split_sentences
        rows = [
            {"text": s, "source": "BBC", "item_id": "i"}
            for s in split_sentences(self.LIVE_BLOG)
        ]
        return summarize_rows(
            rows,
            self.LIVE_BLOG,
            headline,
            cfg=CFG["summarization"],
        )

    def test_unrelated_liveblog_items_dropped(self):
        kept, stats = self._summary([self.LIVE_BLOG])
        joined = " ".join(r["text"] for r in (kept or []))
        assert stats["rejected"] is None
        assert "International Monetary Fund" not in joined
        assert "15 percent" not in joined
        assert "solar eclipse" not in joined

    def test_headline_event_context_kept(self):
        kept, _ = self._summary([self.LIVE_BLOG])
        joined = " ".join(r["text"] for r in (kept or []))
        assert "36 square miles" in joined
        # Crews from the US, Canada and Mexico are same-event context.
        assert "Canada" in joined

    def test_same_event_meanwhile_kept(self):
        from src.formatter import split_sentences
        text = (
            "At least four people, including a child, were killed "
            "in Russian drone and missile strikes on Kyiv and "
            "surrounding regions. "
            "Meanwhile, Ukraine struck another oil refinery in "
            "Russia, as Kyiv continues its campaign of long-range "
            "attacks on Russian energy facilities."
        )
        kept, stats = summarize_rows(
            [
                {"text": s, "source": "F24", "item_id": "i"}
                for s in split_sentences(text)
            ],
            text,
            "Russian attacks kill several in Ukraine",
            cfg=CFG["summarization"],
        )
        joined = " ".join(r["text"] for r in (kept or []))
        assert stats["rejected"] is None
        assert "oil refinery" in joined
        assert "Ukraine" in joined

    def test_relevant_unquoted_headline_link_kept(self):
        # Same-event context survives the relevance filter.
        kept, _ = self._summary([])
        joined = " ".join(r["text"] for r in (kept or []))
        assert "36 square miles" in joined
        assert "Canada" in joined


# ---------------------------------------------------------------------------
# 13. Abbreviation-aware splitting + quote fragments + acronym links
# ---------------------------------------------------------------------------


class TestAbbreviationAndQuoteFragmentRows:
    """Rows containing abbreviations or the two halves of a
    multi-sentence quote must survive verification and the
    quality gate instead of being dropped."""

    CFG = {"min_sentences": 2, "max_sentences": 8, "tier1_max": 6}

    def test_dr_abbreviation_row_kept(self):
        source = (
            "The court was hearing a PIL filed by Dr. Anurag "
            "Srivastava seeking a direction to declare cancer a "
            "notifiable disease nationwide. The plea said the "
            "disease burden is rising sharply."
        )
        from src.formatter import split_sentences
        rows = [
            {"text": s, "source": "BBC", "item_id": "i"}
            for s in split_sentences(source)
        ]
        kept, stats = summarize_rows(
            rows, source,
            "Declare cancer a notifiable disease nationwide: plea",
            cfg=self.CFG,
        )
        assert stats["rejected"] is None
        assert len(kept) == 2
        assert "Dr. Anurag Srivastava" in kept[0]["text"]

    def test_multi_sentence_quote_fragments_kept(self):
        # Splitting "30 minutes. There was no impact..." leaves one
        # row carrying only the opening quote and one only the
        # closing quote; both are legitimate fragments.
        source = (
            '"Airport teams responded promptly, clearing the '
            'water under 30 minutes. There was no impact on '
            'flight operations," the airport said. The move came '
            "after heavy rain lashed the city."
        )
        from src.formatter import split_sentences
        rows = [
            {"text": s, "source": "BBC", "item_id": "i"}
            for s in split_sentences(source)
        ]
        kept, stats = summarize_rows(
            rows, source,
            "Waterlogging reported at Noida airport after heavy rain",
            cfg=self.CFG,
        )
        assert stats["rejected"] is None
        joined = " ".join(r["text"] for r in kept)
        assert "30 minutes" in joined
        assert "no impact on flight operations" in joined

    def test_acronym_linked_context_row_kept(self):
        # "The EC said ..." must link to a headline about the
        # "Election Commission" instead of being dropped as an
        # unrelated proper-name introduction.
        source = (
            '"During the verification it was found that Mr '
            'Prakash Raj had shifted from the registered address '
            'at No. 266, Garden Apartment," the statement said. '
            "The EC said the action was taken after a complaint."
        )
        from src.formatter import split_sentences
        rows = [
            {"text": s, "source": "BBC", "item_id": "i"}
            for s in split_sentences(source)
        ]
        kept, stats = summarize_rows(
            rows, source,
            "Election Commission removes Prakash Raj from voters list",
            cfg=self.CFG,
        )
        assert stats["rejected"] is None
        joined = " ".join(r["text"] for r in kept)
        assert "No. 266" in joined
        assert "EC said the action" in joined


# ---------------------------------------------------------------------------
# 12. Conciseness: prefer exactly two sentences by default
# ---------------------------------------------------------------------------


class TestConciseness:
    def test_ordinary_story_prefers_exactly_two_sentences(self):
        # Default policy: a story with plenty of source content
        # is summarised in exactly two sentences, not filled to
        # the maximum available.
        texts = [
            "The 5.5-magnitude earthquake in Leh was recorded "
            "at a depth of 10 km.",
            "No immediate reports of casualties followed the "
            "tremor.",
            "Aftershocks were felt across the valley.",
            "Roads to the affected villages were blocked.",
        ]
        kept, stats = summarize(texts)
        assert stats["rejected"] is None
        assert len(kept) == 2

    def test_complex_story_can_use_three_when_needed(self):
        # A story carrying distinct essential facts - impact,
        # a numbered claim and a time reference - may grow to
        # three sentences so no key fact is omitted.
        texts = [
            "More than 20,000 residents have been forced from "
            "their homes.",
            "The fire started near the town of Ashcroft on "
            "Sunday.",
            "Winds of up to 40 mph are pushing the flames "
            "northeast.",
            "Crews are working through the night.",
        ]
        kept, stats = summarize(texts)
        assert stats["rejected"] is None
        assert len(kept) == 3
        joined = " ".join(r["text"] for r in kept)
        assert "20,000" in joined
        assert "40 mph" in joined

    def test_supplementary_rows_never_exceed_what_is_needed(self):
        # A story whose essential facts are already covered by
        # two sentences must not grow just because more source
        # rows are available.
        texts = [
            "The 5.5-magnitude earthquake in Leh was recorded "
            "at a depth of 10 km.",
            "No immediate reports of casualties followed the "
            "tremor.",
            "Shallow earthquakes are generally more dangerous "
            "than deep ones.",
            "Officials urged residents to stay alert.",
        ]
        kept, stats = summarize(texts)
        assert stats["rejected"] is None
        assert len(kept) == 2

    def test_supreme_court_ai_style_long_article_is_trimmed(self):
        # A long detailed article with many secondary points is
        # condensed to the two essential facts; unnecessary
        # detail is removed.
        texts = [
            "The Supreme Court on Thursday asked the union "
            "government to consider a petition flagging the lack "
            "of a comprehensive statutory regime on the use of "
            "high-risk AI in areas affecting food, health "
            "benefits, wages and pension.",
            "A three-judge Bench headed by the Chief Justice "
            "asked the government to consider the petition as a "
            "representation for taking comprehensive action.",
            "In his petition, the advocate questioned whether "
            "the government could employ AI, machine learning, "
            "automated decision-making systems, risk-scoring "
            "mechanisms and biometric surveillance tools without "
            "a specific statutory framework.",
            "The lawyer clarified that the constitutional vice "
            "pleaded was not that technology must be prohibited.",
            "The petition said he was only trying to "
            "constitutionalise innovation.",
        ]
        kept, stats = summarize(
            texts,
            headline=(
                "Supreme Court asks government to consider "
                "high-risk AI plea"
            ),
        )
        assert stats["rejected"] is None
        assert len(kept) == 2
        joined = " ".join(r["text"] for r in kept)
        assert "Supreme Court" in joined
        # Secondary elaboration is gone from the summary.
        assert "only trying to constitutionalise" not in joined

    def test_generic_explainers_never_expand_a_two_sentence_summary(self):
        texts = [
            "The quake struck Leh at 6:05 am.",
            "No immediate reports of damage followed.",
            "Shallow earthquakes tend to be more dangerous than "
            "deep earthquakes.",
        ]
        kept, stats = summarize(texts)
        assert stats["rejected"] is None
        assert len(kept) == 2
        joined = " ".join(r["text"] for r in kept)
        assert "tend to be" not in joined

    def test_ladakh_earthquake_stays_concise_at_two_sentences(self):
        # The earthquake example stays a tight two-sentence
        # summary: the generic explainer never pads it.
        texts = [
            "The 5.5-magnitude earthquake in Leh was recorded "
            "at a depth of 10 km.",
            "As of the latest available information there were "
            "no immediate reports of casualties or major "
            "structural damage following Thursday morning's "
            "tremor.",
            "Shallow earthquakes are generally more dangerous "
            "than deep earthquakes.",
        ]
        kept, stats = summarize(texts)
        assert stats["rejected"] is None
        assert len(kept) == 2
        joined = " ".join(r["text"] for r in kept)
        assert "10 km" in joined
        assert "no immediate reports of casualties" in joined
        assert "generally more dangerous" not in joined

