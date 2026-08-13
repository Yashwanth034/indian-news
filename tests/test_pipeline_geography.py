"""Tests for India geo classification (scope, state, national significance)."""
import pytest

from src.models.article import Article
from src.pipeline.geography import GeoClassifier


@pytest.fixture(scope="module")
def classifier():
    return GeoClassifier()


def _article(title, summary=""):
    return Article(
        source_id="the-hindu",
        source_name="The Hindu",
        tier=2,
        source_role="journalism",
        url="https://www.thehindu.com/news/g1.html",
        title=title,
        summary=summary or None,
    )


def _geo(classifier, title, summary=""):
    return classifier.classify(_article(title, summary))


# --- national scope ----------------------------------------------------------

@pytest.mark.parametrize("title", [
    "Union Cabinet approves new policy",
    "RBI keeps repo rate unchanged",
    "Supreme Court strikes down law",
    "India signs trade agreement with EU",
    "ISRO launches new navigation satellite",
    "G20 leaders meet in New Delhi",
    "Clash at LAC in Ladakh",
    "NHAI to build new national highway",
    "Inter-state water dispute tribunal",
    "Indian economy grows 7 percent",
    "Railways announces new trains across India",
])
def test_national_scope(classifier, title):
    r = _geo(classifier, title)
    assert r.scope == "national"
    assert r.national_significance is True
    assert r.is_national_story is True


def test_national_election(classifier):
    r = _geo(classifier, "Election Commission announces Lok Sabha election schedule")
    assert r.scope == "national"
    assert r.national_significance is True


def test_no_geo_defaults_to_national(classifier):
    r = _geo(classifier, "Markets rally as IT stocks surge")
    assert r.scope == "national"
    assert r.state is None
    assert r.state_identifiable is False
    assert r.national_significance is False


# --- state scope -------------------------------------------------------------

@pytest.mark.parametrize("title,state", [
    ("Telangana government announces new scheme", "telangana"),
    ("Andhra Pradesh cabinet reshuffle", "andhra pradesh"),
    ("Maharashtra presents state budget", "maharashtra"),
    ("Karnataka launches new policy", "karnataka"),
])
def test_state_scope(classifier, title, state):
    r = _geo(classifier, title)
    assert r.scope == "state"
    assert r.state == state
    assert r.state_identifiable is True
    assert r.national_significance is False


# --- local scope -------------------------------------------------------------

@pytest.mark.parametrize("title,state", [
    ("Hyderabad road closure from Monday", "telangana"),
    ("Mumbai police launch operation", "maharashtra"),
    ("Pune gets new city park", "maharashtra"),
])
def test_local_scope(classifier, title, state):
    r = _geo(classifier, title)
    assert r.scope == "local"
    assert r.state == state
    assert r.state_identifiable is True
    assert r.national_significance is False


def test_two_cities_same_state_is_local(classifier):
    r = _geo(classifier, "Mumbai-Pune metro connectivity plan")
    assert r.scope == "local"
    assert r.state == "maharashtra"


# --- state/local with national significance ----------------------------------

def test_local_terror_attack_is_significant(classifier):
    r = _geo(classifier, "Mumbai terror attack kills 10")
    assert r.scope == "local"
    assert r.state == "maharashtra"
    assert r.national_significance is True


def test_state_train_accident_is_significant(classifier):
    r = _geo(classifier, "Odisha train accident: 20 dead")
    assert r.scope == "state"
    assert r.state == "odisha"
    assert r.national_significance is True


def test_state_earthquake_is_significant(classifier):
    r = _geo(classifier, "Earthquake in Gujarat")
    assert r.scope == "state"
    assert r.state == "gujarat"
    assert r.national_significance is True


def test_state_election_is_significant(classifier):
    r = _geo(classifier, "Election Commission announces Maharashtra polls")
    assert r.scope == "state"
    assert r.state == "maharashtra"
    assert r.national_significance is True


def test_state_pandemic_is_significant(classifier):
    r = _geo(classifier, "Kerala reports new pandemic wave")
    assert r.scope == "state"
    assert r.state == "kerala"
    assert r.national_significance is True


# --- multi-state / India-wide ------------------------------------------------

def test_multi_state_cities_is_national(classifier):
    r = _geo(classifier, "Vande Bharat connects Mumbai and Ahmedabad")
    assert r.scope == "national"
    assert r.state is None
    assert r.state_identifiable is False
    assert set(r.states) == {"maharashtra", "gujarat"}
    assert r.national_significance is True


def test_multi_state_cyclone_is_national(classifier):
    r = _geo(classifier, "Cyclone lashes Odisha and West Bengal")
    assert r.scope == "national"
    assert r.state is None
    assert set(r.states) == {"odisha", "west bengal"}


def test_multi_state_election_is_national(classifier):
    r = _geo(classifier, "BJP sweeps Bihar, Uttar Pradesh and Madhya Pradesh")
    assert r.scope == "national"
    assert r.state is None
    assert r.national_significance is True


# --- national context overrides state/local ----------------------------------

def test_pm_visit_is_national_with_state_context(classifier):
    r = _geo(classifier, "PM Modi visits Bihar")
    assert r.scope == "national"
    assert r.state == "bihar"


def test_national_term_with_state_in_summary(classifier):
    r = _geo(classifier, "RBI keeps rates unchanged", "Telangana industry welcomes the move.")
    assert r.scope == "national"
    assert r.state == "telangana"


def test_national_agency_launch_with_state(classifier):
    r = _geo(classifier, "ISRO launches PSLV from Andhra Pradesh")
    assert r.scope == "national"
    assert r.state == "andhra pradesh"


def test_sensex_with_mumbai(classifier):
    r = _geo(classifier, "Sensex hits record, Mumbai bourses rally")
    assert r.scope == "national"
    assert r.state == "maharashtra"


def test_budget_with_state_context_is_national(classifier):
    r = _geo(classifier, "Union Budget 2024: Lessons from Gujarat")
    assert r.scope == "national"
    assert r.state == "gujarat"


# --- union territories as state nodes ----------------------------------------

def test_chandigarh_ut_is_state_scope(classifier):
    r = _geo(classifier, "Chandigarh: new administrative hub")
    assert r.scope == "state"
    assert r.state == "chandigarh"


def test_new_delhi_is_state_scope(classifier):
    r = _geo(classifier, "New Delhi gets new metro line")
    assert r.scope == "state"
    assert r.state == "delhi"


def test_delhi_ut_is_state_scope(classifier):
    r = _geo(classifier, "Delhi metro expansion approved")
    assert r.scope == "state"
    assert r.state == "delhi"


def test_port_blair_ut_city_is_state_scope(classifier):
    r = _geo(classifier, "Port Blair gets new power plant")
    assert r.scope == "state"
    assert r.state == "andaman and nicobar islands"


def test_city_derived_state_patna_is_local(classifier):
    r = _geo(classifier, "Patna gets new flyover")
    assert r.scope == "local"
    assert r.state == "bihar"


# --- false positives / foreign places that look Indian -----------------------

def test_surat_thani_geo_suppressed(classifier):
    r = _geo(classifier, "Surat Thani province gets new port")
    assert r.state is None
    assert r.states == []
    assert r.scope == "national"
    assert any(s.group == "false_positive" for s in r.signals)


def test_surat_thani_event_still_significant(classifier):
    r = _geo(classifier, "Surat Thani braced for cyclone")
    assert r.state is None
    assert r.national_significance is True


def test_hyderabad_pakistan_geo_suppressed(classifier):
    r = _geo(classifier, "Hyderabad in Pakistan opens new cricket stadium")
    assert r.state is None
    assert r.scope == "national"


def test_indian_wells_geo_suppressed(classifier):
    r = _geo(classifier, "Indian Wells tennis open begins")
    assert r.scope == "national"
    assert r.state is None
    assert r.national_significance is False


def test_indian_ocean_geo_suppressed(classifier):
    r = _geo(classifier, "Indian Ocean temperatures rise")
    assert r.state is None
    assert r.national_significance is False


# --- summary fallback --------------------------------------------------------

def test_summary_geo_fallback(classifier):
    r = _geo(classifier, "Rescue operations under way", "Cyclone Dana makes landfall in Odisha.")
    assert r.scope == "state"
    assert r.state == "odisha"
    assert r.national_significance is True
    assert any(s.location == "summary" for s in r.signals)


# --- explainability ----------------------------------------------------------

def test_result_is_explainable(classifier):
    r = _geo(classifier, "Telangana government announces new scheme")
    assert r.scope == "state"
    assert r.reasons
    assert r.signals
    assert any(s.group == "state" and s.location == "title" for s in r.signals)
    assert all(s.location in ("title", "summary") for s in r.signals)


def test_has_title_geo_flag(classifier):
    r = _geo(classifier, "Telangana government announces new scheme")
    assert r.has_title_geo is True
    r2 = _geo(classifier, "The quick brown fox jumps over the lazy dog")
    assert r2.has_title_geo is False


def test_no_geo_signals_explainable(classifier):
    r = _geo(classifier, "The quick brown fox jumps over the lazy dog")
    assert r.scope == "national"
    assert r.state is None
    assert r.reasons
    assert r.signals == []
