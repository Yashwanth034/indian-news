"""Tests for India news category classification."""
import pytest

from src.models.article import Article
from src.pipeline.classify import CategoryClassifier


@pytest.fixture(scope="module")
def classifier():
    return CategoryClassifier()


def _article(title, summary=""):
    return Article(
        source_id="the-hindu",
        source_name="The Hindu",
        tier=2,
        source_role="journalism",
        url="https://www.thehindu.com/news/a1.html",
        title=title,
        summary=summary or None,
    )


def _primary(classifier, title, summary=""):
    return classifier.classify(_article(title, summary)).primary


# --- all 18 categories ------------------------------------------------------

@pytest.mark.parametrize("title,expected", [
    ("BJP wins big in state elections", "national-politics"),
    ("NITI Aayog releases new policy document", "government-policy"),
    ("RBI keeps repo rate unchanged", "economy-finance"),
    ("Reliance Jio revenue and profit jump", "business-industry"),
    ("India signs defence procurement deal with US", "defence-security"),
    ("UPI transactions cross 10 billion", "technology-digital"),
    ("ISRO launches Chandrayaan-4 mission", "science-space"),
    ("Cyclone Dana makes landfall in Odisha", "weather-disasters"),
    ("Supreme Court upholds electoral bonds verdict", "courts-law"),
    ("Dengue cases rise in Delhi hospitals", "health"),
    ("NEET results announced, admissions begin", "education"),
    ("Rural employment scheme gets more funds", "jobs-employment"),
    ("Vande Bharat train flagged off", "transport-infrastructure"),
    ("India beat Australia in World Cup final", "sports"),
    ("G20 leaders meet in New Delhi", "international-affairs"),
    ("MSP hike for kharif crops announced", "agriculture-rural"),
    ("Solar power capacity crosses 100 GW", "energy-environment"),
    ("Delhi police bust drug smuggling ring", "crime-public-safety"),
])
def test_all_18_categories(classifier, title, expected):
    assert _primary(classifier, title) == expected


# --- ambiguous / multi-category ---------------------------------------------

def test_multi_category_story_keeps_secondary(classifier):
    r = classifier.classify(_article("Supreme Court quashes GST notice against startup"))
    assert r.primary == "courts-law"
    assert "economy-finance" in r.secondary
    assert "technology-digital" in r.secondary
    assert r.secondary[0] == "economy-finance"


def test_monsoon_shared_between_weather_and_agriculture(classifier):
    assert _primary(classifier, "Monsoon rains boost kharif sowing") == "agriculture-rural"
    assert _primary(classifier, "Monsoon to hit Kerala this week") == "weather-disasters"


def test_train_accident_transport_over_crime(classifier):
    assert _primary(classifier, "Odisha train accident kills 20") == "transport-infrastructure"


# --- generic-word false positives -------------------------------------------

@pytest.mark.parametrize("title", [
    "Chip shortage hits food industry",
    "Bank of the river eroded after rains",
    "Power of the executive expands",
    "Final decision on the bill expected",
    "New space for offices announced",
    "Who gets the credit for the deal",
    "Record attendance at the fair",
])
def test_generic_words_do_not_classify(classifier, title):
    r = classifier.classify(_article(title))
    assert r.primary is None
    assert r.classified is False


# --- official announcements by subject --------------------------------------

def test_rbi_decision_is_economy_not_government(classifier):
    assert _primary(classifier, "RBI hikes repo rate by 25 bps") == "economy-finance"


def test_isro_announcement_is_science_not_government(classifier):
    assert _primary(classifier, "ISRO announces Gaganyaan crew selection") == "science-space"


def test_cabinet_defence_approval_is_defence(classifier):
    r = classifier.classify(_article("Cabinet approves new defence procurement"))
    assert r.primary == "defence-security"
    assert r.primary != "government-policy"
    assert r.primary != "national-politics"


def test_centre_electricity_rules_is_energy(classifier):
    assert _primary(classifier, "Centre notifies new electricity rules") == "energy-environment"


# --- international stories affecting India ----------------------------------

def test_g20_climate_story_is_international(classifier):
    assert _primary(classifier, "G20 summit: India pushes climate agenda") == "international-affairs"


def test_bilateral_summit_is_international(classifier):
    assert _primary(classifier, "India-US hold annual summit") == "international-affairs"


# --- state / local stories --------------------------------------------------

def test_state_flood_story_is_weather(classifier):
    assert _primary(classifier, "Kerala flood: 20 dead, rescue under way") == "weather-disasters"


def test_local_civic_story_has_no_category(classifier):
    r = classifier.classify(_article("Mysuru gets new city park"))
    assert r.primary is None


# --- sports-heavy volume does not leak --------------------------------------

@pytest.mark.parametrize("title,expected", [
    ("India beat Australia in World Cup final", "sports"),
    ("IPL auction sees record bids", "sports"),
    ("India wins hockey World Cup", "sports"),
])
def test_sports_heavy_stories(classifier, title, expected):
    assert _primary(classifier, title) == expected


def test_record_profit_is_business_not_sports(classifier):
    assert _primary(classifier, "Record profit for India's largest telecom") == "business-industry"


# --- business vs economy ----------------------------------------------------

def test_business_vs_economy(classifier):
    assert _primary(classifier, "Reliance Jio posts profit") == "business-industry"
    assert _primary(classifier, "RBI keeps repo rate unchanged") == "economy-finance"
    assert _primary(classifier, "Bank stocks rally after RBI rate cut") == "economy-finance"


# --- technology vs science --------------------------------------------------

def test_technology_vs_science(classifier):
    assert _primary(classifier, "AI startup raises funding") == "technology-digital"
    assert _primary(classifier, "ISRO launches Chandrayaan-4") == "science-space"
    assert _primary(classifier, "ISRO partners with startup for satellite") == "science-space"


# --- government vs policy ---------------------------------------------------

def test_government_vs_policy(classifier):
    assert _primary(classifier, "RBI hikes repo rate") == "economy-finance"
    assert _primary(classifier, "Cabinet approves defence procurement") == "defence-security"


# --- defence vs international affairs ---------------------------------------

def test_defence_vs_international(classifier):
    assert _primary(classifier, "India-US sign defence cooperation pact") == "defence-security"
    assert _primary(classifier, "India-US hold annual summit") == "international-affairs"


# --- disaster vs crime / public safety --------------------------------------

def test_disaster_vs_crime(classifier):
    assert _primary(classifier, "Cyclone Tauktae: rescue operations under way") == "weather-disasters"
    assert _primary(classifier, "Delhi police arrest two in smuggling case") == "crime-public-safety"


# --- India itself does not determine a category -----------------------------

def test_india_alone_does_not_classify(classifier):
    assert _primary(classifier, "India signs new agreement with partners") is None
    assert _primary(classifier, "India's economy grows at record pace") is None


# --- explainability & summary usage -----------------------------------------

def test_summary_signals_are_used(classifier):
    r = classifier.classify(
        _article("Global markets update", "RBI kept repo rate unchanged at 6.5 percent.")
    )
    assert r.primary == "economy-finance"
    assert any(sig.location == "summary" for sig in r.signals)


def test_result_is_explainable(classifier):
    r = classifier.classify(_article("RBI keeps repo rate unchanged"))
    assert r.primary == "economy-finance"
    assert r.primary_label == "Economy & Finance"
    assert r.primary_score >= 2.0
    assert isinstance(r.primary_importance_weight, float)
    assert isinstance(r.primary_urgent, bool)
    assert r.scores["economy-finance"] >= 2.0
    assert r.reasons
    assert r.signals
    assert 0.0 <= r.primary_share <= 1.0


def test_no_category_when_no_signals(classifier):
    r = classifier.classify(_article("The quick brown fox jumps over the lazy dog"))
    assert r.primary is None
    assert r.scores == {}
    assert r.classified is False
