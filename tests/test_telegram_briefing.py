"""Tests for the conservative briefing builder.

Run with:  .venv/bin/python -m pytest src/test_briefing.py -q
"""
from datetime import datetime, timedelta, timezone

from src.telegram_briefing import (
    BREAKING,
    JUST_IN,
    NEWS,
    UPDATE,
    aggregate_sentences,
    build_briefing,
    clean_headline,
    clean_sentence_text,
    decode_html_entities,
    group_items,
    is_headline_paraphrase,
    public_label,
    same_event,
)
from src.formatter import clean_sentence, repair_sentence_boundaries, split_sentences
from src.telegram_formatter import build_message

NOW = datetime.now(timezone.utc)


def item(
    title,
    summary,
    source="BBC World",
    event_id="e1",
    score=76,
    confidence="medium",
    urgency=None,
    category="world",
    primary=False,
    strong=0,
    status="NEW",
    minutes_ago=30,
    tier=2,
    url="https://example.com/a",
    **extra,
):
    base = {
        "id": "id-" + str(abs(hash(title + source))),
        "story_id": "story-" + str(abs(hash(title + source))),
        "item_id": "item-" + str(abs(hash(title + source))),
        "event_id": event_id,
        "title": title,
        "summary": summary,
        "source": source,
        "category": category,
        "score": score,
        "confidence": confidence,
        "urgency_terms": urgency or [],
        "primary_source": primary,
        "strong_corroboration": strong,
        "event_status": status,
        "tier": tier,
        "url": url,
        "effective_at": (
            NOW - timedelta(minutes=minutes_ago)
        ).isoformat(),
    }
    base.update(extra)
    return base


def bbc():
    return item(
        title="State of emergency declared as fast-moving "
        "Canada wildfire doubles in size",
        summary=(
            'The Bald Range wildfire in British Columbia, still '
            'considered "out of control", has spread over more '
            "than 36 sq miles (95 sq km)."
        ),
        event_id="e-wildfire-1",
        score=76,
        confidence="medium",
        urgency=["wildfire", "state of emergency"],
    )


def aljazeera():
    return item(
        title="Wildfire forces thousands of evacuations "
        "in Western Canada",
        summary=(
            "A fast-moving wildfire has forced more than 20,000 "
            "people to evacuate parts of British Columbia's "
            "Okanagan region."
        ),
        source="Al Jazeera",
        event_id="e-wildfire-2",
        score=72,
        confidence="medium",
        urgency=["wildfire", "evacuations"],
    )


# ---------------------------------------------------------
# Conservative grouping
# ---------------------------------------------------------


def test_no_merge_on_geography_alone():
    a = item(
        title="Canada wildfire spreads",
        summary="A wildfire in Canada.",
        source="BBC World",
        event_id="ea",
        urgency=["wildfire"],
        region="Canada",
    )
    b = item(
        title="Canada economy grows",
        summary="The Canadian economy grew.",
        source="Reuters",
        event_id="eb",
        urgency=[],
        region="Canada",
    )
    assert not same_event(a, b)
    assert len(group_items([a, b])) == 2


def test_no_merge_on_urgency_alone():
    a = item(
        title="Wildfire forces evacuations in Greece",
        summary="Fires near Athens.",
        source="BBC World",
        event_id="ea",
        urgency=["wildfire"],
    )
    b = item(
        title="Wildfire threatens suburbs of Sydney",
        summary="Fires near Sydney.",
        source="Reuters",
        event_id="eb",
        urgency=["wildfire"],
    )
    assert not same_event(a, b)


def test_merge_on_multiple_signals():
    assert same_event(bbc(), aljazeera())
    assert len(group_items([bbc(), aljazeera()])) == 1


def test_merge_with_different_urgent_categories():
    a = item(
        "State of emergency declared as fast-moving "
        "Canada wildfire doubles in size",
        "The Bald Range wildfire spread over 36 sq miles.",
        event_id="e1",
        category="world",
        urgency=["wildfire", "state of emergency"],
    )
    b = item(
        "Wildfire forces thousands of evacuations "
        "in Western Canada",
        "More than 20,000 people evacuated.",
        source="Al Jazeera",
        event_id="e2",
        category="disaster",
        urgency=["wildfire", "evacuation"],
    )
    assert same_event(a, b)


def test_non_urgent_categories_never_merge():
    a = item(
        "Swimmer wins gold in final race",
        "A swimmer won a race.",
        category="sports",
        urgency=[],
        event_id="e1",
    )
    b = item(
        "Swimmer wins silver in another race",
        "A swimmer placed second.",
        source="Reuters",
        category="sports",
        urgency=[],
        event_id="e2",
    )
    assert not same_event(a, b)


def test_merge_on_same_event_id():
    a = item("Alpha event", "Alpha happened.", event_id="shared")
    b = item("Beta event", "Beta happened.", event_id="shared")
    assert same_event(a, b)


def test_grouping_avoids_chain_drift():
    a = item(
        "Iran strike on Tehran oil refinery",
        "Refinery hit.",
        event_id="ea",
        urgency=["strike"],
    )
    b = item(
        "Tehran oil refinery strike kills workers",
        "Workers killed.",
        event_id="eb",
        urgency=["strike"],
    )
    c = item(
        "Workers strike over wages in Germany",
        "Wage strike.",
        event_id="ec",
        urgency=[],
    )
    groups = group_items([a, b, c])
    assert len(groups) == 2


# ---------------------------------------------------------
# Public labels
# ---------------------------------------------------------


def test_breaking_requires_all_signals():
    x = item(
        "Major earthquake hits coastal city",
        "A strong earthquake struck the coast.",
        score=90,
        confidence="high",
        urgency=["earthquake"],
        primary=True,
        minutes_ago=5,
    )
    assert public_label(x, 15, NOW) == BREAKING


def test_high_priority_alone_is_not_breaking():
    x = item(
        "Company announces new product",
        "A company made an announcement.",
        score=95,
        confidence="high",
        urgency=[],
        primary=True,
        category="business",
        minutes_ago=5,
    )
    assert public_label(x, 15, NOW) != BREAKING


def test_breaking_requires_verification():
    x = item(
        "Major earthquake hits coastal city",
        "A strong earthquake struck the coast.",
        score=90,
        confidence="high",
        urgency=["earthquake"],
        primary=False,
        strong=0,
        minutes_ago=30,
    )
    assert public_label(x, 15, NOW) == NEWS


def test_low_confidence_never_breaking():
    x = item(
        "Major earthquake hits coastal city",
        "A strong earthquake struck the coast.",
        score=90,
        confidence="low",
        urgency=["earthquake"],
        primary=True,
        minutes_ago=5,
    )
    assert public_label(x, 15, NOW) == NEWS


def test_just_in_requires_freshness_and_importance():
    fresh_but_low = item(
        "Minor sports result",
        "A sports team won.",
        score=50,
        confidence="high",
        minutes_ago=2,
        category="sports",
    )
    important_but_old = item(
        "Government announces new policy",
        "A policy was announced.",
        score=80,
        confidence="high",
        minutes_ago=60,
    )
    just_in = item(
        "Government announces new policy",
        "A policy was announced.",
        score=80,
        confidence="high",
        minutes_ago=2,
    )
    assert public_label(fresh_but_low, 15, NOW) == NEWS
    assert public_label(important_but_old, 15, NOW) == NEWS
    assert public_label(just_in, 15, NOW) == JUST_IN


def test_update_label_for_update_status():
    x = item(
        "Wildfire update: crews gain control",
        "Firefighters have made progress.",
        status="UPDATE",
        score=70,
    )
    assert public_label(x, 15, NOW) == UPDATE


# ---------------------------------------------------------
# Aggregation provenance + conflicts
# ---------------------------------------------------------


def test_aggregate_keeps_provenance():
    rows = aggregate_sentences(
        [bbc(), aljazeera()],
        bbc(),
    )
    assert len(rows) == 2
    assert rows[0]["source"] == "BBC World"
    assert rows[1]["source"] == "Al Jazeera"
    assert rows[0]["text"].startswith("The Bald Range")


def test_aggregate_primary_first():
    rows = aggregate_sentences(
        [aljazeera(), bbc()],
        bbc(),
    )
    assert rows[0]["source"] == "BBC World"


def test_aggregate_dedupes_identical_sentences():
    a = item("X event", "Same sentence here.", source="BBC World")
    b = item(
        "X event details",
        "Same sentence here.",
        source="Reuters",
    )
    rows = aggregate_sentences([a, b], a)
    assert len(rows) == 1


def test_conflicting_numbers_are_dropped():
    primary = item(
        "Protest turnout in capital",
        "Officials estimate 20,000 people joined the protest.",
        source="BBC World",
        score=80,
    )
    rival = item(
        "Protest turnout in capital city",
        "Organizers claim 30,000 people joined the protest.",
        source="Reuters",
        score=70,
    )
    rows = aggregate_sentences([primary, rival], primary)
    texts = [r["text"] for r in rows]
    assert len(texts) == 1
    assert "20,000" in texts[0]
    assert "30,000" not in texts[0]


def test_different_units_do_not_conflict():
    primary = item(
        "Fire expands",
        "The fire has spread over 36 sq miles.",
        source="BBC World",
    )
    rival = item(
        "Fire forces evacuations",
        "More than 20,000 people have been evacuated.",
        source="Al Jazeera",
    )
    rows = aggregate_sentences([primary, rival], primary)
    assert len(rows) == 2


def test_no_invented_facts():
    briefing = build_briefing(
        bbc(),
        [bbc(), aljazeera()],
        15,
        NOW,
    )
    all_text = " ".join(
        briefing["opening"]
        + briefing["body"]
    )
    source_texts = (
        bbc()["summary"] + " " + aljazeera()["summary"]
    )
    for sentence in (
        briefing["opening"] + briefing["body"]
    ):
        assert sentence in source_texts


def test_briefing_corroborating_attribution():
    briefing = build_briefing(
        bbc(),
        [bbc(), aljazeera()],
        15,
        NOW,
    )
    assert briefing["source"] == "BBC World"
    assert "Al Jazeera" in briefing["corroborating"]


def test_briefing_bullets_are_literal():
    briefing = build_briefing(
        bbc(),
        [bbc(), aljazeera()],
        15,
        NOW,
    )
    bullets = briefing["bullets"]
    bullet_text = " ".join(
        b["text"] for b in bullets
    )
    combined = (
        bbc()["summary"] + " " + aljazeera()["summary"]
    )
    for b in bullets:
        assert b["text"].lower() in combined.lower()


def test_short_source_stays_short():
    single = item(
        "Small local story",
        summary="A single fact sentence.",
    )
    briefing = build_briefing(
        single,
        [single],
        15,
        NOW,
    )
    assert len(briefing["opening"]) == 1
    assert briefing["body"] == []
    assert briefing["corroborating"] == []


def test_location_bullet_stops_at_place_name():
    # Regression: the Location bullet must not swallow
    # following lowercase words ("Port Vila overn").
    storm = item(
        title="Tropical Storm Kei makes landfall near Port Vila",
        summary=(
            "Tropical Storm Kei made landfall near Port Vila "
            "overnight, bringing torrential rain and gusts of "
            "140 km/h."
        ),
        source="Pacific Times",
        minutes_ago=8,
    )
    briefing = build_briefing(
        storm,
        [storm],
        15,
        NOW,
    )
    location = [
        b for b in briefing["bullets"]
        if b["label"] == "Location"
    ]
    assert len(location) == 1
    assert location[0]["text"] == "Port Vila"


def test_bullet_sentence_not_repeated_in_body():
    # Regression: France 24 Typhoon Dolphin — the
    # "expected to make landfall" sentence must appear
    # exactly once (as the ➡️ Next bullet), never again
    # as a body paragraph.
    typhoon = item(
        title="China cancels flights, issues evacuation orders "
        "as Typhoon Dolphin looms",
        summary=(
            "China's National Meteorological Centre on Sunday "
            "issued a red typhoon alert, its most severe warning, "
            "as Typhoon Dolphin barrelled towards the country's "
            "densely populated eastern coast. Flights in and out "
            "of the region were cancelled and nearly 100,000 "
            "people were relocated to less risky areas, according "
            "to authorities. Typhoon Dolphin is expected to make "
            "landfall late Sunday or early Monday."
        ),
        source="France 24",
        score=73,
        urgency=["evacuation"],
    )
    briefing = build_briefing(
        typhoon,
        [typhoon],
        15,
        NOW,
    )
    next_bullets = [
        b for b in briefing["bullets"]
        if b["label"] == "Next"
    ]
    assert len(next_bullets) == 1
    landfall = next_bullets[0]["text"]
    assert "expected to make landfall" in landfall
    assert landfall not in briefing["body"]

    msg = build_message(
        dict(
            typhoon,
            public_label=briefing["label"],
            headline=briefing["headline"],
            briefing=briefing,
        ),
        {
            "target_message_chars": 1500,
            "max_message_chars": 3000,
        },
    )["text"]
    assert msg.count("expected to make landfall") == 1
    assert (
        msg.find("red typhoon alert")
        < msg.find("expected to make landfall")
    )


# ---------------------------------------------------------
# Headline cleaning
# ---------------------------------------------------------


def test_headline_strips_source_suffix():
    assert clean_headline(
        "Storm hits coast - BBC World"
    ) == "Storm hits coast"


def test_headline_strips_live_prefix():
    assert clean_headline(
        "Live: Storm hits coast"
    ) == "Storm hits coast"


def test_headline_never_paraphrases():
    assert clean_headline(
        "Canada wildfire doubles in size"
    ) == "Canada wildfire doubles in size"


# ---------------------------------------------------------
# Cross-source same-event merging (real-world scenarios)
#
# Stories from different feeds reporting the SAME real-world
# development must become ONE post; stories that merely share
# a topic or a place must stay separate.
# ---------------------------------------------------------

# --- MUST MERGE: same event, different feeds ---------------


def test_merge_hormuz_tanker_seizure():
    a = item(
        "Seven oil tankers seized near Strait of Hormuz",
        "Iranian forces boarded seven tankers in the strait.",
        source="Reuters",
        event_id="ea-hormuz",
        urgency=["seizure"],
        minutes_ago=25,
    )
    b = item(
        "Iran seizes 7 oil tankers in Hormuz Strait",
        "Commandos took control of seven vessels near Hormuz.",
        source="BBC World",
        event_id="eb-hormuz",
        urgency=["seizure"],
        minutes_ago=40,
    )
    assert same_event(a, b)
    assert len(group_items([a, b])) == 1


def test_merge_saudi_iran_pact():
    a = item(
        "Saudi Arabia and Iran sign historic pact",
        "The two rivals sealed a normalization agreement.",
        source="Al Jazeera",
        event_id="ea-pact",
    )
    b = item(
        "Riyadh and Tehran ink peace agreement",
        "A deal to restore ties was signed in Beijing.",
        source="Reuters",
        event_id="eb-pact",
    )
    assert same_event(a, b)
    assert len(group_items([a, b])) == 1


def test_merge_typhoon_landfall():
    a = item(
        "Typhoon Yagi hits the Philippines",
        "The storm slammed into the northern coast.",
        source="BBC World",
        event_id="ea-yagi",
        urgency=["typhoon"],
        minutes_ago=15,
    )
    b = item(
        "Super Typhoon Yagi makes landfall in the "
        "Philippines, 6 dead",
        "At least six people were killed as Yagi came ashore.",
        source="Al Jazeera",
        event_id="eb-yagi",
        urgency=["typhoon"],
        minutes_ago=35,
    )
    assert same_event(a, b)
    assert len(group_items([a, b])) == 1


def test_merge_ukraine_refinery_strikes():
    a = item(
        "Ukraine strikes Russian oil refinery",
        "Drones hit a refinery outside Moscow.",
        source="BBC World",
        event_id="ea-refinery",
        urgency=["strike"],
    )
    b = item(
        "Drones hit oil refinery near Moscow",
        "Ukrainian drones attacked a refinery in Russia.",
        source="Reuters",
        event_id="eb-refinery",
        urgency=["strike"],
    )
    assert same_event(a, b)
    assert len(group_items([a, b])) == 1


def test_merge_meta_fine():
    a = item(
        "EU fines Meta €1.3 billion over privacy breach",
        "The regulator handed Facebook parent a record penalty.",
        source="Reuters",
        event_id="ea-meta",
    )
    b = item(
        "Facebook parent hit with €1.3 billion EU fine",
        "Meta was ordered to pay for breaching data rules.",
        source="BBC World",
        event_id="eb-meta",
    )
    assert same_event(a, b)
    assert len(group_items([a, b])) == 1


def test_merge_hunter_biden_verdict():
    a = item(
        "Hunter Biden convicted in Delaware gun trial",
        "The president's son was found guilty on three counts.",
        source="BBC World",
        event_id="ea-biden",
    )
    b = item(
        "Biden's son found guilty in Delaware gun trial",
        "Hunter Biden faces sentencing after the verdict.",
        source="Reuters",
        event_id="eb-biden",
    )
    assert same_event(a, b)
    assert len(group_items([a, b])) == 1


def test_merge_thailand_mall_shooting():
    a = item(
        "Shooting at Bangkok mall kills 2",
        "A gunman opened fire in a shopping centre.",
        source="Reuters",
        event_id="ea-bangkok",
        urgency=["shooting"],
    )
    b = item(
        "Gunman opens fire at Thailand mall, 2 dead",
        "Police said two people were killed in the attack.",
        source="BBC World",
        event_id="eb-bangkok",
        urgency=["shooting"],
    )
    assert same_event(a, b)
    assert len(group_items([a, b])) == 1


def test_no_merge_on_corroboration_without_shared_action():
    a = item(
        "US adds 272,000 jobs in May",
        "Payrolls beat expectations for the month.",
        source="Reuters",
        event_id="ea-jobs",
        category="finance",
    )
    b = item(
        "US payrolls jump 272,000 in May",
        "The labour market stayed strong.",
        source="BBC World",
        event_id="eb-jobs",
        category="finance",
    )
    assert not same_event(a, b)
    assert len(group_items([a, b])) == 2


def test_merge_zelensky_serbia_visit():
    a = item(
        "Zelensky arrives in Serbia",
        "The Ukrainian president met his Serbian counterpart.",
        source="Al Jazeera",
        event_id="ea-zelensky",
    )
    b = item(
        "Ukrainian President Zelensky visits Belgrade",
        "A first trip to Belgrade since the war began.",
        source="BBC World",
        event_id="eb-zelensky",
    )
    assert same_event(a, b)
    assert len(group_items([a, b])) == 1


def test_merge_cuba_blackout():
    a = item(
        "Cuba blackout leaves 3 million in the dark",
        "The island's grid failed island-wide.",
        source="BBC World",
        event_id="ea-cuba",
        urgency=["blackout"],
    )
    b = item(
        "Power grid collapses in Havana, 3 million affected",
        "Cuba suffered its worst outage in decades.",
        source="Reuters",
        event_id="eb-cuba",
        urgency=["blackout"],
    )
    assert same_event(a, b)
    assert len(group_items([a, b])) == 1


# --- MUST NOT MERGE: shared topic/place, different events ---


def test_no_merge_colombia_car_bomb_and_inauguration():
    bomb = item(
        "Car bomb kills 5 in Colombia capital",
        "An explosion hit a police station.",
        source="Reuters",
        event_id="ea-colombia-bomb",
        urgency=["bomb"],
        minutes_ago=20,
    )
    inauguration = item(
        "Colombia swears in President Petro",
        "The new president took the oath in Bogota.",
        source="BBC World",
        event_id="eb-colombia-swearing",
        minutes_ago=20,
    )
    assert not same_event(bomb, inauguration)
    assert len(group_items([bomb, inauguration])) == 2


def test_no_merge_colombia_eln_and_inauguration():
    eln = item(
        "ELN attacks military base in Colombia",
        "Rebels bombed an army post overnight.",
        source="Reuters",
        event_id="ea-colombia-eln",
        minutes_ago=20,
    )
    inauguration = item(
        "Colombia swears in President Petro",
        "The new president took the oath in Bogota.",
        source="BBC World",
        event_id="eb-colombia-swearing",
        minutes_ago=20,
    )
    assert not same_event(eln, inauguration)
    assert len(group_items([eln, inauguration])) == 2


def test_no_merge_tps_bills_c16_and_c83():
    c16 = item(
        "Senate C16 extends TPS protections",
        "The bill would renew protected status.",
        source="Reuters",
        event_id="ea-tps-c16",
    )
    c83 = item(
        "House C83 ends TPS program",
        "The legislation would cancel the programme.",
        source="BBC World",
        event_id="eb-tps-c83",
    )
    assert not same_event(c16, c83)
    assert len(group_items([c16, c83])) == 2


def test_no_merge_different_ceuta_developments():
    surge = item(
        "Migrant surge in Ceuta as 800 cross into Spain",
        "Hundreds arrived at the border overnight.",
        source="Reuters",
        event_id="ea-ceuta-surge",
        minutes_ago=10,
    )
    exercise = item(
        "Spain deploys navy exercise off Ceuta",
        "Warships held drills near the enclave.",
        source="BBC World",
        event_id="eb-ceuta-exercise",
        minutes_ago=10,
    )
    assert not same_event(surge, exercise)
    assert len(group_items([surge, exercise])) == 2


def test_no_merge_canada_and_spokane_wildfires():
    canada = item(
        "Canada wildfire forces 20,000 evacuations",
        "The fire spread across the province.",
        source="BBC World",
        event_id="ea-canada-fire",
        urgency=["wildfire"],
        minutes_ago=30,
    )
    spokane = item(
        "Spokane wildfire forces 12,000 people evacuated",
        "The blaze grew near the city.",
        source="Reuters",
        event_id="eb-spokane-fire",
        urgency=["wildfire"],
        minutes_ago=30,
    )
    assert not same_event(canada, spokane)
    assert len(group_items([canada, spokane])) == 2


def test_no_merge_1999_eclipse_archive_and_current():
    archive = item(
        "Rare 1999 solar eclipse photos archived",
        "A look back at last century's eclipse.",
        source="BBC World",
        event_id="ea-eclipse-1999",
        effective_at="1999-08-11T10:00:00+00:00",
    )
    current = item(
        "Total solar eclipse visible in August",
        "Skywatchers prepare for the big day.",
        source="Reuters",
        event_id="eb-eclipse-2026",
        minutes_ago=60,
    )
    assert not same_event(archive, current)
    assert len(group_items([archive, current])) == 2


# ---------------------------------------------------------
# Regression tests: real-audit merge pairs (must still merge)
# ---------------------------------------------------------


def test_must_merge_ukraine_pow_kostiantynivka_strike():
    a = item(
        "Russian missile strike on Kostiantynivka "
        "kills 12 Ukrainian prisoners of war",
        "The strike hit a POW holding site.",
        source="BBC World",
        event_id="ea-pow",
        minutes_ago=30,
    )
    b = item(
        "Kostiantynivka strike deadliest attack of the "
        "year, 12 POWs killed, Ukraine says",
        "Twelve captured soldiers died.",
        source="Al Jazeera",
        event_id="eb-pow",
        minutes_ago=35,
    )
    assert same_event(a, b)
    assert len(group_items([a, b])) == 1


def test_must_merge_france_finance_minister():
    a = item(
        "France appoints Eric Lombard as new finance minister",
        "Lombard takes over the finance ministry.",
        source="BBC World",
        event_id="ea-lombard",
        minutes_ago=30,
    )
    b = item(
        "New French finance minister Eric Lombard "
        "meets EU partners",
        "Lombard discussed the budget with EU peers.",
        source="Reuters",
        event_id="eb-lombard",
        minutes_ago=40,
    )
    assert same_event(a, b)
    assert len(group_items([a, b])) == 1


def test_must_merge_hungary_greece_bulgaria_migration():
    a = item(
        "Hungary seals border as 30,000 migrants "
        "mass at southern fence",
        "The closure follows days of arrivals.",
        source="Hungarian press",
        event_id="ea-border",
        minutes_ago=30,
    )
    b = item(
        "Hungary to keep border sealed after 30,000 "
        "migrants arrived",
        "Officials extended the closure.",
        source="Reuters",
        event_id="eb-border",
        minutes_ago=50,
    )
    assert same_event(a, b)
    assert len(group_items([a, b])) == 1


def test_must_merge_us_mexico_police():
    a = item(
        "US police officer accused of killing three people "
        "in Mexico arrested at border",
        "The officer was held while crossing into the US.",
        source="The Guardian World",
        event_id="ea-eberle",
        minutes_ago=30,
    )
    b = item(
        "US police officer accused of killing three in Mexico "
        "arrested at border, officials say",
        "Eberle was arrested in Texas.",
        source="AP News",
        event_id="eb-eberle",
        minutes_ago=45,
    )
    assert same_event(a, b)
    assert len(group_items([a, b])) == 1


def test_must_merge_haiti_tps():
    a = item(
        "Haitians face arrest and deportation after US "
        "removes TPS protections",
        "Tens of thousands could be returned.",
        source="The Guardian World",
        event_id="ea-tps",
        minutes_ago=30,
    )
    b = item(
        "US ends TPS for Haitians as 100,000 face "
        "deportation",
        "DHS said protections are over.",
        source="BBC World",
        event_id="eb-tps",
        minutes_ago=60,
    )
    assert same_event(a, b)
    assert len(group_items([a, b])) == 1


# ---------------------------------------------------------
# Regression tests: real-audit split pairs (must stay separate)
# ---------------------------------------------------------


def test_no_merge_haiti_tps_and_mexico_police():
    haiti = item(
        "Haitians face arrest and deportation after US "
        "removes TPS protections",
        "Tens of thousands could be returned.",
        source="The Guardian World",
        event_id="ea-tps",
        minutes_ago=30,
    )
    mexico = item(
        "US police officer accused of killing three people "
        "in Mexico arrested at border",
        "The officer was held while crossing into the US.",
        source="The Guardian World",
        event_id="eb-mexico",
        minutes_ago=35,
    )
    assert not same_event(haiti, mexico)
    assert len(group_items([haiti, mexico])) == 2


def test_no_merge_ukraine_strike_and_obituary():
    strike = item(
        "Russian strikes kill four as Kyiv hits oil refinery",
        "Four people died in the attack.",
        source="France 24",
        event_id="ea-strike",
        minutes_ago=30,
    )
    obituary = item(
        "Ukraine mourns 'collector of souls' Oleksiy Yukov, "
        "killed recovering war dead",
        "Yukov recovered fallen soldiers' bodies.",
        source="NPR World",
        event_id="eb-obit",
        minutes_ago=45,
    )
    assert not same_event(strike, obituary)
    assert len(group_items([strike, obituary])) == 2


def test_no_merge_hungary_greece_bulgaria_migration():
    hungary = item(
        "Hungary seals southern border as 30,000 "
        "migrants gather",
        "The fence is being reinforced.",
        source="Hungarian press",
        event_id="ea-hungary",
        minutes_ago=30,
    )
    greece = item(
        "Greece shuts southern islands to migrant "
        "arrivals",
        "Ferries were diverted.",
        source="Reuters",
        event_id="eb-greece",
        minutes_ago=40,
    )
    bulgaria = item(
        "Bulgaria detains migrants along southern "
        "route",
        "Border patrols were increased.",
        source="AP News",
        event_id="ec-bulgaria",
        minutes_ago=50,
    )
    assert not same_event(hungary, greece)
    assert not same_event(hungary, bulgaria)
    assert not same_event(greece, bulgaria)
    assert len(group_items([hungary, greece, bulgaria])) == 3


def test_one_shared_entity_needs_two_actions():
    a = item(
        "Drone strike hits Kyiv oil refinery",
        "A refinery near the capital was hit.",
        source="BBC World",
        event_id="ea-a",
        minutes_ago=30,
    )
    b = item(
        "Russian drone attack on Kyiv refinery "
        "kills two",
        "Two workers died in the strike.",
        source="Reuters",
        event_id="eb-b",
        minutes_ago=40,
    )
    assert same_event(a, b)
    assert len(group_items([a, b])) == 1


def test_one_shared_entity_and_one_action_need_topic():
    a = item(
        "Wildfire forces thousands of evacuations "
        "in Western Canada",
        "More than 20,000 people evacuated.",
        source="Al Jazeera",
        event_id="ea-wf",
        minutes_ago=30,
    )
    b = item(
        "State of emergency declared as fast-moving "
        "Canada wildfire doubles in size",
        "The Bald Range wildfire spread over 36 sq miles.",
        source="BBC World",
        event_id="eb-wf",
        minutes_ago=45,
    )
    assert same_event(a, b)
    assert len(group_items([a, b])) == 1



# ---------------------------------------------------------
# Headline-paraphrase safety: overlap alone must never drop
# a sentence that carries a genuinely new fact
# ---------------------------------------------------------

# --- MUST KEEP: real-data cases where the removed sentence
# --- carried a genuinely new fact beyond the headline ------


def test_paraphrase_keeps_cyclist_withdrawal_reason():
    # Audit case 15: overlap > 0.5, but the sentence adds the
    # reason (feeling unwell) and the stage/timing.
    assert not is_headline_paraphrase(
        "French cyclist and defending champion Pauline "
        "Ferrand-Prévot pulled out of the Women's Tour de "
        "France ahead of Saturday's 8th and penultimate "
        "stage due to feeling unwell.",
        "Women's Tour de France: Defending champion Pauline "
        "Ferrand-Prévot pulls out of race",
    )


def test_paraphrase_keeps_swimming_venue():
    # Audit case 17: adds the venue (European Championships,
    # Paris) that the headline lacks.
    assert not is_headline_paraphrase(
        "France have finally won medals in open water "
        "swimming at the European Swimming Championships "
        "in Paris.",
        "Finally, medals for France in open water swimming",
    )


def test_paraphrase_keeps_hormuz_agreement_status():
    # Audit case 18: adds the status of a possible agreement.
    assert not is_headline_paraphrase(
        "An agreement over the Strait of Hormuz is still "
        "far away.",
        "Tanker sailors still face danger in the Strait of "
        "Hormuz",
    )


def test_paraphrase_keeps_biden_medical_detail():
    # Audit case 2: adds the medical specificity that the
    # headline lacks.
    assert not is_headline_paraphrase(
        "Joe Biden’s son, Hunter Biden, says his father’s "
        "prostate cancer has “metastasised into his bones "
        "and further”.",
        "‘Very painful’: Joe Biden’s cancer has spread, son "
        "Hunter says",
    )


def test_paraphrase_keeps_colombia_toll_booth():
    # Audit case 3: adds the toll-booth detail.
    assert not is_headline_paraphrase(
        "A car bomb destroyed a toll booth on a highway in "
        "Colombia a day after President Abelardo de la "
        "Espriella was sworn in.",
        "Car bomb hits Colombia highway day after new "
        "president sworn in",
    )


def test_paraphrase_keeps_sydney_near_miss_time():
    # Audit case 22: adds the exact time (7am) and day that
    # the headline lacks.
    assert not is_headline_paraphrase(
        "The Jetstar and Qatar Airways planes almost "
        "collided at Sydney airport at 7am on Sunday.",
        "‘Not enough air traffic controllers’: safety "
        "concerns at Sydney airport after Jetstar and Qatar "
        "planes involved in near-miss",
    )


def test_paraphrase_keeps_womens_march_timing():
    # Audit case 28: adds the day (Sunday) the headline lacks.
    assert not is_headline_paraphrase(
        "South African women will commemorate the "
        "achievements of the 1956 Women's March on Sunday.",
        "South Africa commemorates 1956 Women's March - but "
        "fight for freedom isn't over",
    )


# --- MUST STILL DROP: true headline restatements -----------


def test_paraphrase_drops_zelenskyy_restatement():
    # Audit case 25: identical fact, no new information.
    assert is_headline_paraphrase(
        "Ukrainian President Zelenskyy is making his first "
        "visit to Serbia.",
        "Ukraine's Zelenskyy makes first visit to "
        "Russia-friendly Serbia",
    )


def test_paraphrase_drops_zelenskyy_official_trip():
    # Same event restated with synonyms only.
    assert is_headline_paraphrase(
        "Zelenskyy visits Serbia in first official trip",
        "Ukraine's Zelenskyy makes first visit to "
        "Russia-friendly Serbia",
    )


def test_paraphrase_drops_police_bust_restatement():
    assert is_headline_paraphrase(
        "Police have arrested 78 people in a smuggling bust.",
        "Police arrest 78 people in smuggling bust",
    )


def test_paraphrase_drops_battering_restatement():
    assert is_headline_paraphrase(
        "A storm is battering the coast of Japan.",
        "Storm batters the coast of Japan",
    )


# --- Mechanism proofs --------------------------------------


def test_paraphrase_new_number_keeps_sentence():
    # A number absent from the headline is a new fact.
    assert not is_headline_paraphrase(
        "Farmers across the country protested against farm "
        "taxes with 400 tractors in the capital.",
        "Farmers across the country protest against farm taxes",
    )


def test_paraphrase_new_day_keeps_sentence():
    # A day/time absent from the headline is a new fact.
    assert not is_headline_paraphrase(
        "The new transport rules take effect in the capital "
        "on Monday.",
        "Transport rules take effect in the capital",
    )


def test_paraphrase_new_named_entity_keeps_sentence():
    # A named place absent from the headline is a new fact.
    assert not is_headline_paraphrase(
        "Diplomatic talks between Iran and Oman resumed in "
        "Muscat.",
        "Diplomatic talks resume between Iran and Oman",
    )


def test_paraphrase_unchanged_restatement_removed():
    # Same facts, same words: still a paraphrase even with a
    # new day when the headline already contains that day.
    assert is_headline_paraphrase(
        "The new transport rules take effect in the capital "
        "on Monday.",
        "Transport rules take effect in the capital on Monday",
    )


def test_near_duplicate_still_removed_after_paraphrase_fix():
    # Near-dup protection is untouched by the paraphrase fix.
    primary = item(
        "Wildfire spreads across the valley",
        "The fire has spread across more than 36 square miles.",
        source="BBC World",
    )
    corroborator = item(
        "Wildfire spreads across the valley",
        "The fire has spread over more than 36 square miles.",
        source="Al Jazeera",
    )
    rows = aggregate_sentences(
        [primary, corroborator],
        primary,
    )
    assert len(rows) == 1


def test_kept_paraphrase_sentence_is_verbatim():
    # The relaxed rule must never invent text: kept sentences
    # are exactly the source sentences.
    single = item(
        "Women's Tour de France: Defending champion Pauline "
        "Ferrand-Prévot pulls out of race",
        summary=(
            "French cyclist and defending champion Pauline "
            "Ferrand-Prévot pulled out of the Women's Tour de "
            "France ahead of Saturday's 8th and penultimate "
            "stage due to feeling unwell. An Olympic champion "
            "in mountain biking at the 2024 Paris Olympics, "
            "Ferrand-Prévot won the Women's Tour de France in "
            "her debut last year."
        ),
    )
    briefing = build_briefing(single, [single], 15, NOW)
    texts = briefing["opening"] + briefing["body"]
    assert len(texts) == 2
    assert "feeling unwell" in texts[0]
    for sentence in texts:
        assert sentence in single["summary"]


def test_zelenskyy_restatement_dropped_in_briefing():
    # End to end: the restatement is dropped but the second
    # sentence (real context) survives.
    single = item(
        "Ukraine's Zelenskyy makes first visit to "
        "Russia-friendly Serbia",
        summary=(
            "Ukrainian President Zelenskyy is making his first "
            "visit to Serbia. His host, President Aleksandar "
            "Vucic, hopes to score points with the EU, while "
            "Zelenskyy seeks more arms from a country that "
            "remains friendly to Russia."
        ),
        source="DW World",
    )
    briefing = build_briefing(single, [single], 15, NOW)
    texts = briefing["opening"] + briefing["body"]
    assert len(texts) == 1
    assert "Vucic" in texts[0]
    assert "first visit to Serbia" not in texts[0]


# ---------------------------------------------------------------------------
# FIX 1 - HTML entity cleanup
# ---------------------------------------------------------------------------

class TestHtmlEntityDecoding:
    def test_quot_decodes_to_quote(self):
        assert decode_html_entities(
            '&quot;We have to brace ourselves&quot;'
        ) == '"We have to brace ourselves"'
        assert clean_sentence_text(
            '&quot;Go,&quot; she said.'
        ) == '"Go," she said.'

    def test_apostrophe_entities_decode(self):
        assert decode_html_entities("can&#039;t") == "can't"
        assert decode_html_entities("can&#39;t") == "can't"
        assert decode_html_entities("can&apos;t") == "can't"

    def test_amp_decodes_to_ampersand(self):
        assert decode_html_entities("R &amp; D") == "R & D"

    def test_numeric_html_entities(self):
        assert decode_html_entities("&#34;quote&#34;") == '"quote"'
        assert decode_html_entities("&#x27;apos&#x27;") == "'apos'"
        assert decode_html_entities("&#8217;") == "\u2019"

    def test_no_double_decoding(self):
        # "&amp;lt;" means a literal "&lt;" in the source; it
        # must never become "<".
        assert decode_html_entities("use &amp;lt;b&amp;gt;") == (
            "use &lt;b&gt;"
        )
        assert decode_html_entities("&amp;amp;") == "&amp;"
        # Single-encoded entities decode exactly once.
        assert decode_html_entities("5 &amp; 10") == "5 & 10"

    def test_double_encoded_quotes_repair(self):
        # Feed text that was escaped twice ("&amp;quot;") must
        # still end up as a real quotation mark.
        assert decode_html_entities(
            "he said &amp;quot;no&amp;quot;"
        ) == 'he said "no"'

    def test_clean_sentence_keeps_quote_period(self):
        # "...let it burn." behind a closing quote must not
        # gain a second period.
        assert clean_sentence(
            'The firefighter said "let it burn."'
        ) == 'The firefighter said "let it burn."'
        assert clean_sentence(
            "He said \"go\""
        ) == 'He said "go".'


# ---------------------------------------------------------------------------
# FIX 2 - missing-punctuation sentence-boundary repair
# ---------------------------------------------------------------------------

class TestSentenceBoundaryRepair:
    GUARDIAN = (
        "Authorities warn that days of intense rain could "
        "trigger widespread flooding and landslides as storm "
        "moves north and further inland More than a million "
        "people were moved to safety across eastern China by "
        "Sunday evening."
    )

    def test_typhoon_dolphin_boundary_repaired(self):
        parts = split_sentences(self.GUARDIAN)
        assert len(parts) == 2
        assert parts[0] == (
            "Authorities warn that days of intense rain could "
            "trigger widespread flooding and landslides as "
            "storm moves north and further inland."
        )
        assert parts[1] == (
            "More than a million people were moved to safety "
            "across eastern China by Sunday evening."
        )

    def test_boundary_repair_preserves_words(self):
        import re as _re
        before = sorted(
            _re.findall(r"[a-z0-9\u2019'-]+",
                        self.GUARDIAN.lower())
        )
        after = sorted(
            _re.findall(
                r"[a-z0-9\u2019'-]+",
                " ".join(split_sentences(self.GUARDIAN)).lower(),
            )
        )
        assert before == after

    def test_boundary_repair_invents_no_text(self):
        import re as _re
        # Compare word tokens with trailing punctuation
        # stripped: only the whitespace between the two
        # sentences may change, never the words themselves.
        tokens = lambda t: _re.findall(
            r"[A-Za-z0-9\u2019'-]+", t.lower()
        )
        assert tokens(self.GUARDIAN) == tokens(
            repair_sentence_boundaries(self.GUARDIAN)
        )

    def test_no_false_splits_on_legit_transitions(self):
        for text in [
            "Officials visited New York City to inspect the damage.",
            "The Bald Range wildfire has spread over more than "
            "36 sq miles.",
            "A resident said the fire was the worst he had "
            "seen in 40 years.",
            "He said More than 20,000 people have been "
            "evacuated from the region.",
            "The tax covers houses worth more than $5m.",
            "The mayor John Smith said the plan was rejected.",
            "She told reporters the airport had reopened "
            "overnight.",
        ]:
            parts = split_sentences(text)
            assert len(parts) == 1, (text, parts)

    def test_multiple_boundaries_repaired(self):
        text = (
            "Train operators are cracking down after the move "
            "away from paper tickets increases fare dodging "
            "There is more than one way to commit fraud on a "
            "train journey."
        )
        parts = split_sentences(text)
        assert len(parts) == 2
        assert parts[1].startswith(
            "There is more than one way"
        )


class TestAbbreviationSentenceBoundaries:
    """A period that ends an abbreviation is never a sentence
    boundary, so feeds containing "Dr.", "No. 266", "a.m.",
    "U.S." etc. are not shredded into dropped rows."""

    def test_title_abbreviation_not_split(self):
        text = (
            "The court was hearing a PIL filed by Dr. Anurag "
            "Srivastava, through advocate Gaurav Kumar Bansal, "
            "seeking a direction to declare cancer a notifiable "
            "disease nationwide."
        )
        assert split_sentences(text) == [text]

    def test_number_marker_abbreviation_not_split(self):
        text = (
            '"During the verification it was found that Mr Prakash '
            'Raj had shifted from the registered address at No. '
            '266, Garden Apartment," the statement said.'
        )
        assert split_sentences(text) == [text]

    def test_dotted_run_abbreviations_not_split(self):
        for text in [
            "The flight departs at 6 a.m. and arrives before noon.",
            "It happened in the U.S. and the U.K. yesterday.",
            "Ph.D. holders earn more than most graduates.",
            "They brought apples, oranges, etc. The rest came later.",
        ]:
            assert len(split_sentences(text)) == 1, text

    def test_single_initial_not_split(self):
        text = "J. Smith was arrested. The police confirmed it."
        parts = split_sentences(text)
        assert len(parts) == 2
        assert parts[0] == "J. Smith was arrested."

    def test_lowercase_sentence_end_still_splits(self):
        # "no." as a genuine sentence end (lowercase) must split.
        parts = split_sentences(
            "He said no. There is nothing more to add."
        )
        assert len(parts) == 2
        assert parts[0] == "He said no."

    def test_real_split_after_abbreviation_sentence(self):
        # The sentence after the abbreviation sentence still splits.
        parts = split_sentences(
            "Prof. Sharma teaches at the university. "
            "He arrived late."
        )
        assert len(parts) == 2
        assert parts[0].startswith("Prof. Sharma")
        assert parts[1] == "He arrived late."



class TestDryRunEditorialCleanup:
    """Regression tests for the six editorial/copy problems
    found in the real-data dry run of 2026-08-10.

    1. live-blog boilerplate ("Follow our live blog for the
       latest developments.")
    2. truncated feed fragments ("...a Qatar Airways plane t")
    3. Guardian subheading/body fusion
    4. article/navigation series fragments ("Total solar
       eclipse (2/4) A total solar eclipse...")
    5. quote-boundary handling (never split inside a quote)
    6. missing-boundary repair ("...an hour later The annual
       Perseid meteor shower...")
    """

    # --- 3. Guardian subheading/body fusion -----------------
    def test_subheading_body_fusion_split(self):
        text = (
            "Prime minister and opposition leader meet in "
            "Canberra to advance agreement on gambling "
            "advertising Anthony Albanese has bowed to "
            "pressure and will limit gambling inducements "
            "as part of a deal between Labor and Angus "
            "Taylor, even as some Liberals warn it might "
            "not do enough to help vulnerable people."
        )
        parts = split_sentences(text)
        assert len(parts) == 2, parts
        assert parts[0].endswith("gambling advertising.")
        assert parts[1].startswith(
            "Anthony Albanese has bowed to pressure"
        )

    def test_multi_word_names_never_split(self):
        text = (
            "Mexico City Andres Manuel Lopez Obrador has "
            "faced mounting criticism over his handling of "
            "the crisis."
        )
        parts = split_sentences(text)
        assert len(parts) == 1, parts
        assert parts[0] == text

    def test_name_as_reported_speech_object_not_split(self):
        text = (
            "Police said John Smith has been arrested in "
            "connection with the burglary."
        )
        parts = split_sentences(text)
        assert len(parts) == 1, parts

    def test_name_boundary_respects_titles(self):
        text = (
            "The prime minister Anthony Albanese has "
            "announced the plan to parliament."
        )
        parts = split_sentences(text)
        assert len(parts) == 1, parts

    # --- 5. quote-boundary handling -------------------------
    def test_no_boundary_inserted_inside_quote(self):
        text = (
            "For Mina, a housewife in Iran, the war is no "
            "longer measured by air strikes or military "
            "statements. \u201cI honestly can\u2019t remember "
            "the last time I bought red meat. We replaced it "
            "with chicken, but even chicken has become a "
            "luxury,\u201d she told DW."
        )
        parts = split_sentences(text)
        joined = " ".join(parts)
        assert "the last time. I bought" not in joined
        assert (
            "\u201cI honestly can\u2019t remember the last "
            "time I bought red meat." in joined
        )
        assert parts[2].endswith(
            "luxury,\u201d she told DW."
        )

    def test_quote_boundary_outside_quote_still_repaired(self):
        text = (
            "Officials said the fire was out of control "
            "More than 500 homes were destroyed in the "
            "blaze."
        )
        parts = split_sentences(text)
        assert len(parts) == 2, parts
        assert parts[1].startswith("More than 500 homes")

    # --- 6. missing-boundary repair -------------------------
    def test_starwatch_missing_boundary_repaired(self):
        text = (
            "The eclipse will begin just after 6pm on 12 "
            "August, reaching its maximum just over an hour "
            "later The annual Perseid meteor shower reaches "
            "its peak this week on the night of 12-13 "
            "August."
        )
        parts = split_sentences(text)
        assert len(parts) == 2, parts
        assert parts[0].endswith("an hour later.")
        assert parts[1].startswith("The annual Perseid")

    # --- 2. truncated feed fragments ------------------------
    def test_truncated_single_letter_tail_dropped(self):
        text = (
            "King called in the Airservices Australia chief "
            "executive officer, Rob Sharp, for a meeting in "
            "Canberra on Monday, the day after a Jetstar "
            "plane narrowly avoided hitting a Qatar Airways "
            "plane t."
        )
        parts = split_sentences(text)
        assert parts == [
            "King called in the Airservices Australia chief "
            "executive officer, Rob Sharp, for a meeting in "
            "Canberra on Monday, the day after a Jetstar "
            "plane narrowly avoided hitting a Qatar Airways "
            "plane."
        ]

    def test_truncated_letter_without_period_dropped(self):
        assert clean_sentence(
            "A Jetstar plane narrowly avoided hitting a "
            "Qatar Airways plane t"
        ) == (
            "A Jetstar plane narrowly avoided hitting a "
            "Qatar Airways plane."
        )

    def test_legit_abbreviations_not_dropped(self):
        assert clean_sentence(
            "The flight departs at 6 a.m. and arrives "
            "before noon."
        ) == (
            "The flight departs at 6 a.m. and arrives "
            "before noon."
        )
