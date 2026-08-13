"""Regression tests for the formatter feed-text repair.

Economic Times and similar RSS descriptions occasionally
arrive with sentences concatenated together: dropped spaces,
dropped periods, injected numeric reference IDs (`.133197844“She
...`), and `Also Read: 'Title': ...` promo blocks woven into
the middle of the article.

Covers: dropped space after punctuation, closing-quote
binding, injected ID removal, dropped periods before capitals
and opening quotes, fused dropped-period joins (including
possessive names), promo-block removal, and the no-false-
positive guards (dotted abbreviations, role prefixes, quoted
phrases that are not sentence starts).

Run with:  .venv/bin/python -m pytest tests/test_formatter.py -q
"""
import re

from src.formatter import repair_sentence_boundaries, split_sentences


class TestFeedTextRepair:
    """The economic-times style concatenation repairs."""

    def test_dropped_space_after_period(self):
        text = (
            "through pastel suits, wide-legged trousers and "
            "softer silhouettes.When Meloni became Italy's prime "
            "minister in October 2022, she wore dark suits."
        )
        cleaned = repair_sentence_boundaries(text)
        assert "silhouettes. When Meloni" in cleaned
        assert "silhouettes.When" not in cleaned

    def test_dropped_space_after_close_quote(self):
        text = (
            "She described herself as \u201ca woman, a mother, a "
            "Christian.\u201dSince becoming prime minister, Meloni "
            "has rarely appeared in all-black suits."
        )
        cleaned = repair_sentence_boundaries(text)
        assert (
            "Christian.\u201d Since becoming prime minister"
        ) in cleaned

    def test_injected_numeric_id_removed(self):
        text = (
            "how she presents herself on the world "
            "stage.133197844\u201cShe understands how important "
            "clothing is in nonverbal communication in "
            "politics,\u201d Antonio Mancinelli told AP."
        )
        cleaned = repair_sentence_boundaries(text)
        assert "133197844" not in cleaned
        assert "stage. \u201cShe understands" in cleaned

    def test_injected_id_removed_straight_capital(self):
        text = (
            "comments that have continued to follow her "
            "political career.133197829Over the years, Meloni "
            "has worked to distance herself."
        )
        cleaned = repair_sentence_boundaries(text)
        assert "133197829" not in cleaned
        assert "career. Over the years" in cleaned

    def test_dropped_period_before_open_quote(self):
        text = (
            "She also lightened her hair.\u201cLighter colors "
            "suggest a loyalty,\u201d a fashion professor told AP."
        )
        cleaned = repair_sentence_boundaries(text)
        assert "hair. \u201cLighter colors" in cleaned

    def test_fused_dropped_period_join(self):
        text = (
            "The G7 moment that captured Meloni's new "
            "imageOne of the clearest examples of her "
            "transformation came at the 2024 summit."
        )
        cleaned = repair_sentence_boundaries(text)
        assert "image. One of the clearest" in cleaned

    def test_fused_dropped_period_possessive_name(self):
        text = (
            "From black boots to pastel suitsMeloni's current "
            "style is a sharp departure from her younger "
            "political years."
        )
        cleaned = repair_sentence_boundaries(text)
        assert "suits. Meloni's current style" in cleaned

    def test_promo_block_quoted_resume(self):
        text = (
            "a remark from German Chancellor Friedrich Merz "
            "that was caught on an open microphone.Also Read: "
            "\u2018We are the most famous couple on "
            "Instagram\u2019: Modi\u2013Meloni\u2019s \u2018Melodi\u2019 "
            "moment steals G7 spotlight\u201cYou\u2019re wearing a "
            "tie today,\u201d Merz told her."
        )
        cleaned = repair_sentence_boundaries(text)
        assert "microphone.Also Read" not in cleaned
        assert "Also Read" not in cleaned
        assert "microphone. \u201cYou\u2019re wearing a tie" in cleaned

    def test_promo_block_fused_resume(self):
        text = (
            "pursue a more hardline foreign policy."
            "Also Read: \u2018If you ran in New Delhi, you\u2019d win "
            "a million votes\u2019: Italian PM Giorgia Meloni "
            "recalls 2023 India visitInstead, Meloni has adopted "
            "a pragmatic approach toward the EU."
        )
        cleaned = repair_sentence_boundaries(text)
        assert "Also Read" not in cleaned
        assert "foreign policy. Instead, Meloni" in cleaned

    def test_related_stories_promo_block_removed(self):
        text = (
            "Officials confirmed the evacuation overnight."
            "Related Stories: \u2018Three dead as quake hits "
            "north India\u2019: report\u201cRescue teams are "
            "searching the rubble,\u201d Col. A. Sharma said."
        )
        cleaned = repair_sentence_boundaries(text)
        assert "Related Stories" not in cleaned
        assert "overnight. \u201cRescue teams" in cleaned

    def test_role_prefix_not_split(self):
        # "Italian-Haitian designer Stella Jean said" is one
        # clause; the name+verb rule must not fire after a role.
        text = (
            "much more serious,\u201d Italian-Haitian designer "
            "Stella Jean said."
        )
        cleaned = repair_sentence_boundaries(text)
        assert cleaned == "much more serious,\u201d Italian-Haitian designer Stella Jean said."
        assert "designer. Stella" not in cleaned

    def test_quoted_phrase_not_sentence_start(self):
        # A single quoted phrase with no sentence-fusion
        # evidence: the quote is not a dropped boundary.
        text = (
            "I have read the book \u201cThe Great Gatsby\u201d "
            "twice this year."
        )
        assert repair_sentence_boundaries(text) == text

    def test_camel_case_brand_not_split(self):
        text = (
            "The launch covered iPhone, eBay and McDonald's "
            "restaurants across the region."
        )
        assert repair_sentence_boundaries(text) == text

    def test_word_preservation(self):
        text = (
            "over a confident image through pastel suits, "
            "wide-legged trousers and softer "
            "silhouettes.When Meloni became Italy's prime "
            "minister in October 2022."
        )
        tokens = lambda t: re.findall(
            r"[A-Za-z0-9\u2019'-]+", t.lower()
        )
        assert tokens(text) == tokens(
            repair_sentence_boundaries(text)
        )


class TestDottedRunGuards:
    """Periods inside dotted abbreviations must never become
    sentence boundaries."""

    def test_phd_not_split(self):
        text = "Ph.D. holders earn more than most graduates."
        assert split_sentences(text) == [text]

    def test_dotted_abbreviations_not_split(self):
        for text in [
            "The flight departs at 6 a.m. and arrives by noon.",
            "It happened in the U.S. and the U.K. yesterday.",
            "They brought apples, oranges, etc. The rest came.",
        ]:
            assert len(split_sentences(text)) == 1, text

    def test_initial_not_split(self):
        text = "J. Smith was arrested. The police confirmed it."
        parts = split_sentences(text)
        assert len(parts) == 2
        assert parts[0] == "J. Smith was arrested."

    def test_no_false_quote_splits(self):
        # A boundary candidate that follows a closing quote is
        # treated conservatively: the quote span is never
        # interrupted, so the text is left unchanged.
        text = (
            "She said \u201cthe last time I bought red meat\u201d "
            "More than a million people were moved to safety."
        )
        assert repair_sentence_boundaries(text) == text