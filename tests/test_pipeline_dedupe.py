"""Tests for cross-source deduplication + wire/echo detection."""
import pytest
from datetime import datetime, timedelta, timezone

from src.models.article import Article
from src.pipeline.dedupe import Deduplicator

T0 = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


def _art(source_id, title, summary="", url=None, published=None,
         tier=None, role=None, states=None):
    a = Article(
        source_id=source_id,
        source_name=source_id.replace("-", " ").title(),
        tier=tier if tier is not None else (1 if source_id == "pib" else 2),
        source_role=role or (
            "official-primary" if source_id == "pib" else "journalism"
        ),
        url=url or f"https://{source_id}.example.in/a/{len(title)}.html",
        title=title,
        summary=summary or None,
        published=published or T0,
    )
    if states:
        a.dedupe_states = states
    return a


@pytest.fixture(scope="module")
def deduper():
    return Deduplicator()


def _dedupe(deduper, articles, categories=None, states=None):
    if states is None:
        states = [getattr(a, "dedupe_states", None) for a in articles]
    return deduper.dedupe(articles, categories=categories, states=states)


# --- A. exact duplicates ----------------------------------------------------

def test_exact_url_duplicate(deduper):
    u = "https://www.thehindu.com/news/national/rbi-rate.html"
    a = _art("the-hindu", "RBI cuts repo rate by 25 bps", url=u, published=T0)
    b = _art("the-hindu", "RBI cuts repo rate by 25 bps", url=u, published=T0)
    r = _dedupe(deduper, [a, b])
    assert len(r.events) == 1
    assert r.event_of(0) == r.event_of(1)
    assert r.decisions[1].relationship == "exact_url"


def test_exact_url_duplicate_ignores_number_conflict(deduper):
    u = "https://www.ndtv.com/economy/rbi-rate.html"
    a = _art("ndtv", "RBI cuts repo rate to 6.0 percent", summary="RBI cut the repo rate to 6.0 percent.", url=u, published=T0)
    b = _art("ndtv", "RBI cuts repo rate to 6.0 percent", summary="Correction: RBI cut the repo rate to 6.0 percent.", url=u, published=T0 + timedelta(minutes=30))
    r = _dedupe(deduper, [a, b])
    assert len(r.events) == 1
    assert r.event_of(0) == r.event_of(1)


# --- B. URL variants --------------------------------------------------------

def test_url_variants_tracking_params(deduper):
    u1 = "https://www.thehindu.com/news/national/rbi-rate.html"
    u2 = "https://www.thehindu.com/news/national/rbi-rate.html?utm_source=feedburner&utm_medium=feed"
    a = _art("the-hindu", "RBI cuts repo rate by 25 bps", url=u1, published=T0)
    b = _art("the-hindu", "RBI cuts repo rate by 25 bps", url=u2, published=T0)
    r = _dedupe(deduper, [a, b])
    assert len(r.events) == 1
    assert r.event_of(0) == r.event_of(1)


def test_url_variants_trailing_slash(deduper):
    u1 = "https://www.thehindu.com/news/national/rbi-rate"
    u2 = "https://www.thehindu.com/news/national/rbi-rate/"
    a = _art("the-hindu", "RBI cuts repo rate by 25 bps", url=u1, published=T0)
    b = _art("the-hindu", "RBI cuts repo rate by 25 bps", url=u2, published=T0)
    r = _dedupe(deduper, [a, b])
    assert len(r.events) == 1


# --- C. same headline, different URLs ---------------------------------------

def test_same_headline_different_sources(deduper):
    a = _art("ndtv", "RBI cuts repo rate by 25 bps", url="https://ndtv.com/e/1.html", published=T0)
    b = _art("indian-express", "RBI cuts repo rate by 25 bps", url="https://indianexpress.com/e/2.html", published=T0 + timedelta(minutes=5))
    r = _dedupe(deduper, [a, b])
    assert len(r.events) == 1
    assert r.decisions[1].relationship == "same_headline"
    assert r.decisions[1].same_source is False


def test_same_headline_within_window_only(deduper):
    a = _art("ndtv", "Sensex hits record high", url="https://ndtv.com/e/1.html", published=T0)
    b = _art("ndtv", "Sensex hits record high", url="https://ndtv.com/e/2.html", published=T0 + timedelta(days=10))
    r = _dedupe(deduper, [a, b])
    assert len(r.events) == 2, "same headline months apart must be separate events"


# --- D. small headline wording changes --------------------------------------

def test_small_wording_changes(deduper):
    a = _art("the-hindu", "RBI cuts repo rate by 25 bps", url="https://thehindu.com/e/1.html", published=T0)
    b = _art("indian-express", "RBI cuts repo rate by 25 bps, effective immediately", url="https://indianexpress.com/e/2.html", published=T0 + timedelta(minutes=5))
    r = _dedupe(deduper, [a, b])
    assert len(r.events) == 1
    assert r.event_of(0) == r.event_of(1)


# --- E. same event, substantially different headlines -----------------------

def test_same_event_different_headlines(deduper):
    s1 = "The RBI cut the repo rate to 6.0 percent on Friday."
    s2 = "The Reserve Bank of India cut the repo rate to 6.0 percent on Friday."
    a = _art("the-hindu", "RBI cuts repo rate by 25 basis points", summary=s1, published=T0)
    b = _art("ndtv", "Central bank reduces key lending rate", summary=s2, published=T0 + timedelta(minutes=3))
    c = _art("moneycontrol", "RBI lowers repo rate", summary=s1, published=T0 + timedelta(minutes=6))
    r = _dedupe(deduper, [a, b, c], categories=["economy-finance"] * 3)
    assert len(r.events) == 1
    ev = r.events[0]
    assert ev.independent_source_groups == 3
    assert ev.confidence == "high"
    assert ev.category == "economy-finance"
    assert set(ev.states) == set()


# --- F. PTI republishing ----------------------------------------------------

def test_pti_republishing_is_one_group(deduper):
    t = "Centre launches new scheme to boost farmers' income"
    s = "Government launches new scheme to boost farmers' income, PTI reported on Thursday."
    a = _art("ndtv", t, summary=s, published=T0)
    b = _art("indian-express", t, summary=s, published=T0 + timedelta(minutes=2))
    c = _art("times-of-india", t, summary=s, published=T0 + timedelta(minutes=4))
    r = _dedupe(deduper, [a, b, c])
    assert len(r.events) == 1
    ev = r.events[0]
    assert ev.independent_source_groups == 1
    assert ev.wire_group == "pti"
    assert ev.wire_kind == "wire"
    assert ev.is_wire_echo is True
    assert ev.confidence == "low"


# --- G. ANI republishing ----------------------------------------------------

def test_ani_republishing_is_one_group(deduper):
    t = "Delhi records season's coldest morning"
    s = "Delhi recorded the season's coldest morning on Tuesday, ANI reported."
    a = _art("india-today", t, summary=s, published=T0)
    b = _art("the-print", t, summary=s, published=T0 + timedelta(minutes=3))
    r = _dedupe(deduper, [a, b])
    assert len(r.events) == 1
    ev = r.events[0]
    assert ev.wire_group == "ani"
    assert ev.wire_kind == "wire"
    assert ev.independent_source_groups == 1


# --- H. Reuters republishing ------------------------------------------------

def test_reuters_republishing_is_one_group(deduper):
    a = _art("reuters-india", "India's economy grows 7.2 percent in 2024-25",
             summary="India's economy grew 7.2 percent in the last fiscal year.", tier=4,
             role="international", published=T0)
    b = _art("ndtv", "India's economy grows 7.2% in 2024-25",
             summary="(Reuters) India's economy grew 7.2 percent in the last fiscal year.",
             published=T0 + timedelta(minutes=4))
    c = _art("economic-times", "India's economy grows 7.2 percent",
             summary="Reuters: India's economy grew 7.2 percent in the last fiscal year.",
             published=T0 + timedelta(minutes=8))
    r = _dedupe(deduper, [a, b, c])
    assert len(r.events) == 1
    ev = r.events[0]
    assert ev.wire_group == "reuters"
    assert ev.wire_kind == "wire"
    assert ev.independent_source_groups == 1


# --- I. official-release echoing --------------------------------------------

def test_official_release_echoing_is_one_group(deduper):
    t = "Cabinet approves new telecom policy"
    s = "The Union Cabinet approved the new telecom policy on Thursday."
    a = _art("pib", t, summary=s, tier=1, role="official-primary", published=T0)
    b = _art("ndtv", t, summary=s, published=T0 + timedelta(minutes=5))
    c = _art("indian-express", t, summary=s, published=T0 + timedelta(minutes=10))
    r = _dedupe(deduper, [a, b, c])
    assert len(r.events) == 1
    ev = r.events[0]
    assert ev.independent_source_groups == 1
    assert ev.wire_group == "pib"
    assert ev.wire_kind == "official"
    assert ev.is_wire_echo is True
    assert ev.primary_source == "pib"


# --- J. same person, different event ----------------------------------------

def test_same_person_different_event(deduper):
    a = _art("the-hindu", "Modi launches new healthcare scheme in Gujarat",
             summary="PM Modi announced a new healthcare scheme in Gandhinagar.", published=T0)
    b = _art("ndtv", "Modi addresses massive rally in Gujarat",
             summary="PM Modi addressed a public rally in Ahmedabad on Sunday.", published=T0)
    r = _dedupe(deduper, [a, b])
    assert len(r.events) == 2
    assert r.event_of(0) != r.event_of(1)


# --- K. same organization, different event ----------------------------------

def test_same_organization_different_event(deduper):
    a = _art("economic-times", "TCS signs deal with central bank",
             summary="TCS won a technology contract from the Reserve Bank of India.", published=T0)
    b = _art("moneycontrol", "TCS reports quarterly profit jump",
             summary="TCS posted higher profit for the last quarter.", published=T0)
    r = _dedupe(deduper, [a, b])
    assert len(r.events) == 2


# --- L. same city, different event ------------------------------------------

def test_same_city_different_event(deduper):
    a = _art("times-of-india", "Mumbai metro line 3 extension approved",
             summary="The Mumbai metro extension was approved.", published=T0,
             states=["maharashtra"])
    b = _art("ndtv", "Mumbai police bust drug racket",
             summary="Two people were arrested in the city.", published=T0,
             states=["maharashtra"])
    r = _dedupe(deduper, [a, b])
    assert len(r.events) == 2


# --- M. same topic, different event -----------------------------------------

def test_same_topic_opposite_action_is_separate(deduper):
    a = _art("the-hindu", "RBI hikes repo rate to 6.5 percent", published=T0)
    b = _art("ndtv", "RBI cuts repo rate to 6.5 percent", published=T0)
    r = _dedupe(deduper, [a, b])
    assert len(r.events) == 2
    assert r.event_of(0) != r.event_of(1)


def test_same_category_alone_does_not_merge(deduper):
    a = _art("the-hindu", "NITI Aayog releases new strategy paper",
             summary="NITI Aayog released a strategy paper on growth.", published=T0)
    b = _art("ndtv", "NITI Aayog approves fresh funding for states",
             summary="NITI Aayog approved additional funding for states.", published=T0)
    r = _dedupe(deduper, [a, b], categories=["government-policy", "government-policy"])
    assert len(r.events) == 2


# --- N. same event on different days ----------------------------------------

def test_same_headline_on_different_days_separate(deduper):
    a = _art("ndtv", "Cyclone Dana makes landfall in Odisha",
             summary="Cyclone Dana crossed the Odisha coast.", published=T0)
    b = _art("the-hindu", "Cyclone Dana makes landfall in Odisha",
             summary="Cyclone Dana crossed the Odisha coast.",
             published=T0 + timedelta(days=10))
    r = _dedupe(deduper, [a, b])
    assert len(r.events) == 2


# --- O. follow-up story vs original event -----------------------------------

def test_followup_vs_original_event(deduper):
    a = _art("ndtv", "Cyclone Dana makes landfall in Odisha",
             summary="Cyclone Dana crossed the coast near Bhubaneswar on Thursday night.", published=T0)
    b = _art("the-hindu", "Odisha seeks funds for coastal reconstruction",
             summary="The state government asked the Centre for financial aid after the disaster.",
             published=T0 + timedelta(days=2))
    r = _dedupe(deduper, [a, b])
    assert len(r.events) == 2
    assert r.event_of(0) != r.event_of(1)


# --- P. corrections / updates -----------------------------------------------

def test_updated_article_same_url_is_same_event(deduper):
    u = "https://www.ndtv.com/economy/rbi-rate.html"
    a = _art("ndtv", "RBI cuts repo rate to 6.0 percent",
             summary="RBI cut the repo rate to 6.0 percent.", url=u, published=T0)
    b = _art("ndtv", "RBI cuts repo rate to 6.0 percent",
             summary="RBI cut the repo rate to 6.0 percent (updated).", url=u,
             published=T0 + timedelta(hours=1))
    r = _dedupe(deduper, [a, b])
    assert len(r.events) == 1


# --- Q. breaking news then detailed report ----------------------------------

def test_breaking_then_detailed_report(deduper):
    a = _art("ndtv", "Breaking: RBI cuts repo rate",
             summary="RBI cut the repo rate to 6.0 percent on Friday.", published=T0)
    b = _art("the-hindu", "RBI cuts repo rate by 25 bps in surprise move",
             summary="The Reserve Bank of India cut the repo rate to 6.0 percent on Friday, in a surprise move.",
             published=T0 + timedelta(minutes=40))
    r = _dedupe(deduper, [a, b])
    assert len(r.events) == 1
    assert r.event_of(0) == r.event_of(1)


# --- R. unrelated stories, similar wording ----------------------------------

def test_similar_wording_different_story(deduper):
    a = _art("ndtv", "India announces new policy on semiconductor chips",
             summary="India announced a new policy on semiconductor chips.", published=T0)
    b = _art("the-hindu", "India protests foreign policy on semiconductor chips",
             summary="India protested a foreign policy stance on semiconductor chips.", published=T0)
    r = _dedupe(deduper, [a, b])
    assert len(r.events) == 2


# --- S. multiple states in same event ---------------------------------------

def test_multiple_states_same_event(deduper):
    s = "Cyclone Dana is expected to hit Odisha and West Bengal this week."
    a = _art("ndtv", "Cyclone Dana: Odisha and West Bengal on alert", summary=s,
             published=T0, states=["odisha", "west bengal"])
    b = _art("the-hindu", "Odisha, West Bengal brace for Cyclone Dana", summary=s,
             published=T0 + timedelta(minutes=10), states=["odisha", "west bengal"])
    r = _dedupe(deduper, [a, b])
    assert len(r.events) == 1
    ev = r.events[0]
    assert set(ev.states) == {"odisha", "west bengal"}


# --- T. international event affecting India ---------------------------------

def test_international_event_affecting_india(deduper):
    s = "India brought back 800 citizens from Sudan as part of Operation Kaveri."
    a = _art("reuters-india", "India evacuates citizens from Sudan", summary=s,
             tier=4, role="international", published=T0)
    b = _art("the-hindu", "Operation Kaveri: India brings home citizens from Sudan",
             summary=s, published=T0 + timedelta(minutes=20))
    r = _dedupe(deduper, [a, b])
    assert len(r.events) == 1
    assert r.event_of(0) == r.event_of(1)
    ev = r.events[0]
    assert ev.independent_source_groups == 2


# --- U. false-positive entity matches ---------------------------------------

def test_surat_thani_not_merged_with_indian_surat(deduper):
    a = _art("ndtv", "Surat Thani province hit by cyclone",
             summary="Cyclone hit the Surat Thani province in Thailand.", published=T0)
    b = _art("the-hindu", "Surat in Gujarat sees new industrial park",
             summary="Surat city in Gujarat inaugurated a new industrial park.", published=T0)
    r = _dedupe(deduper, [a, b])
    assert len(r.events) == 2


def test_hyderabad_pakistan_not_merged_with_telangana(deduper):
    a = _art("times-of-india", "Hyderabad in Pakistan braces for floods",
             summary="Hyderabad, Pakistan, prepared for floods.", published=T0)
    b = _art("ndtv", "Hyderabad in Telangana faces heavy rains",
             summary="Hyderabad, Telangana, saw heavy rains.", published=T0)
    r = _dedupe(deduper, [a, b])
    assert len(r.events) == 2


def test_indian_surat_city_articles_merge(deduper):
    a = _art("the-hindu", "Surat gets new diamond bourse", published=T0)
    b = _art("ndtv", "Surat diamond bourse opens", published=T0 + timedelta(minutes=5))
    r = _dedupe(deduper, [a, b])
    assert len(r.events) == 1


# --- wire detection details -------------------------------------------------

def test_attribution_in_author_line_detected(deduper):
    a = _art("ndtv", "Govt announces new farm support scheme",
             summary="The scheme was announced on Thursday.",
             published=T0)
    b = _art("indian-express", "Govt announces new farm support scheme",
             summary="The scheme was announced on Thursday.",
             published=T0 + timedelta(minutes=2))
    # give both an author attribution (byline) of PTI
    a.author = "By PTI"
    b.author = "By PTI"
    r = _dedupe(deduper, [a, b])
    assert len(r.events) == 1
    assert r.events[0].wire_group == "pti"


def test_no_wire_attribution_means_independent_groups(deduper):
    t = "Union Cabinet clears defence procurement plan"
    s = "The Union Cabinet cleared the defence procurement plan on Wednesday."
    a = _art("the-hindu", t, summary=s, published=T0)
    b = _art("ndtv", t, summary=s, published=T0 + timedelta(minutes=3))
    c = _art("indian-express", t, summary=s, published=T0 + timedelta(minutes=6))
    r = _dedupe(deduper, [a, b, c])
    assert len(r.events) == 1
    ev = r.events[0]
    assert ev.independent_source_groups == 1, "identical reporting w/o attribution is not independent"


# --- event identity & metadata ----------------------------------------------

def test_event_id_is_content_derived_and_stable(deduper):
    a = _art("the-hindu", "RBI cuts repo rate by 25 bps", url="https://a.in/1", published=T0)
    b = _art("ndtv", "RBI cuts repo rate by 25 bps", url="https://b.in/2", published=T0)
    r1 = _dedupe(deduper, [a, b])
    c = _art("the-hindu", "RBI cuts repo rate by 25 bps", url="https://a.in/1", published=T0)
    d = _art("ndtv", "RBI cuts repo rate by 25 bps", url="https://b.in/2", published=T0)
    r2 = _dedupe(deduper, [c, d])
    assert r1.events[0].event_id == r2.events[0].event_id
    assert len(r1.events[0].event_id) >= 16


def test_representative_is_highest_tier(deduper):
    t = "Government announces new port development policy"
    s = "The government announced a new port development policy on Monday."
    a = _art("mint", t, summary=s, tier=3, role="specialist", published=T0)
    b = _art("the-hindu", t, summary=s, tier=2, role="journalism", published=T0 + timedelta(minutes=2))
    r = _dedupe(deduper, [a, b])
    assert len(r.events) == 1
    ev = r.events[0]
    assert ev.primary_source == "the-hindu"


def test_same_source_near_duplicate_flag(deduper):
    a = _art("the-hindu", "RBI cuts repo rate by 25 bps", url="https://a.in/1", published=T0)
    b = _art("the-hindu", "RBI cuts repo rate by 25 bps", url="https://a.in/2", published=T0 + timedelta(minutes=1))
    r = _dedupe(deduper, [a, b])
    assert len(r.events) == 1
    assert r.decisions[1].same_source is True


def test_single_article_is_single_event_medium_confidence(deduper):
    a = _art("the-hindu", "Vande Bharat train flagged off from Varanasi", published=T0)
    r = _dedupe(deduper, [a])
    assert len(r.events) == 1
    ev = r.events[0]
    assert len(ev.member_indices) == 1
    assert ev.confidence == "medium"
    assert ev.independent_source_groups == 1


def test_explainability(deduper):
    a = _art("ndtv", "Centre launches new scheme to boost farmers' income",
             summary="Government launches new scheme to boost farmers' income, PTI reported on Thursday.",
             published=T0)
    b = _art("indian-express", "Centre launches new scheme to boost farmers' income",
             summary="Government launches new scheme to boost farmers' income, PTI reported on Thursday.",
             published=T0 + timedelta(minutes=2))
    r = _dedupe(deduper, [a, b])
    assert len(r.events) == 1
    ev = r.events[0]
    assert ev.reasons
    assert ev.entities or ev.reasons
    assert ev.event_time == T0


# --- live-update duplication repro (diagnosed 2026-08-13) -----------------
# Two different outlets covering the same sitting with their own live-update
# copy (as in the Parliament Monsoon Session diagnosis) share entities but
# have far-below-threshold text similarity: dedupe must stay conservative
# and keep them as separate events, while verbatim wire echoes must merge.


def _parliament_live_pair():
    return [
        _art(
            "indian-express",
            "Parliament Monsoon Session Live: Amit Shah vs Opposition in focus,"
            " last day of Monsoon Session today",
            summary=(
                "Amit Shah is expected to speak in the Lok Sabha on the last"
                " day of the Monsoon Session. The opposition demanded a"
                " discussion on the recent protests and the FCRA bill."
            ),
        ),
        _art(
            "ndtv",
            'Parliament Monsoon Session 2026 Highlights: Amit Shah "Ready To'
            ' Answer" Opposition On Delhi Protests',
            summary=(
                "Amit Shah said he is ready to answer the opposition on the"
                " Delhi protests. Parliament proceedings continued on the last"
                " day of the Monsoon Session."
            ),
        ),
    ]


def test_live_update_posts_below_threshold_stay_separate(deduper):
    # The two outlets' own live-update copy shares entities but its text
    # similarity is far below the merge thresholds; conservative dedupe
    # must NOT force-merge on topic alone.
    articles = _parliament_live_pair()
    r = _dedupe(deduper, articles)
    assert len(r.events) == 2
    assert len({r.event_of(i) for i in range(len(articles))}) == 2
    for ev in r.events:
        assert len(ev.member_indices) == 1


def test_live_update_wire_echo_merges(deduper):
    # A live-update bulletin syndicated verbatim by multiple outlets is a
    # single wire echo and must collapse into one event.
    text = (
        "Parliament Monsoon Session ended on its last day. "
        "Amit Shah answered opposition questions during the proceedings."
    )
    articles = [
        _art("ndtv", "Live: Parliament Monsoon Session", summary=text),
        _art("indian-express", "Parliament Monsoon Session ends, Shah answers opposition",
             summary=text),
    ]
    r = _dedupe(deduper, articles)
    assert len(r.events) == 1
    assert len(r.events[0].member_indices) == 2


# --- M. same-publisher feeds share one corroboration group -------------------

def test_same_publisher_feeds_share_one_independent_group(deduper):
    s = "The Reserve Bank of India cut the repo rate to 6.0 percent on Friday."
    a = _art("the-hindu", "RBI cuts repo rate by 25 basis points", summary=s, published=T0)
    b = _art("businessline", "RBI lowers repo rate", summary=s, published=T0 + timedelta(minutes=2))
    c = _art("sportstar", "RBI lowers repo rate", summary=s, published=T0 + timedelta(minutes=4))
    r = _dedupe(deduper, [a, b, c], categories=["economy-finance"] * 3)
    assert len(r.events) == 1
    ev = r.events[0]
    assert ev.independent_source_groups == 1
    assert ev.wire_group == "thehindu"
    assert ev.confidence != "high"


def test_different_publishers_still_count_independently(deduper):
    s = "The Reserve Bank of India cut the repo rate to 6.0 percent on Friday."
    a = _art("the-hindu", "RBI cuts repo rate by 25 basis points", summary=s, published=T0)
    b = _art("ndtv", "Central bank reduces key lending rate", summary=s, published=T0 + timedelta(minutes=3))
    r = _dedupe(deduper, [a, b], categories=["economy-finance"] * 2)
    assert len(r.events) == 1
    ev = r.events[0]
    assert ev.independent_source_groups == 2
    assert ev.confidence == "high"
