"""Tests for the editorial gate pipeline.

Covers the hard-exclusion / soft-penalty / filler / major-override model:
clickbait, gossip, celebrity, astrology, rumour, opinion, routine notices,
minor sports, filler, and major-story override behaviour.
"""
import pytest

from src.models.article import Article
from src.pipeline.classify import CategoryClassifier
from src.pipeline.editorial import EditorialGate, GATE_SOURCES
from src.pipeline.geography import GeoClassifier


@pytest.fixture(scope="module")
def gate():
    return EditorialGate()


@pytest.fixture(scope="module")
def classifier():
    return CategoryClassifier()


@pytest.fixture(scope="module")
def geo():
    return GeoClassifier()


def _art(title, summary="", **kw):
    return Article(
        source_id="the-hindu",
        source_name="The Hindu",
        tier=2,
        source_role="journalism",
        url="https://www.thehindu.com/news/e.html",
        title=title,
        summary=summary or None,
        **kw,
    )


def _evaluate(gate, title, summary="", category=None, geo=None):
    return gate.evaluate_text(title, summary, category=category, geo=geo)


# --- 1/2. clickbait vs legitimate headlines ---------------------------------

def test_obvious_clickbait_is_rejected(gate):
    r = _evaluate(gate, "You won't believe what this minister did next")
    assert r.decision == "reject"
    assert "clickbait" in r.gates
    assert any("clickbait" in s.gate for s in r.signals)


def test_legitimate_headline_with_clickbait_word_passes(gate, classifier):
    title = "What happens next for India's Chandrayaan mission after landing"
    cat = classifier.classify_text(title)
    r = _evaluate(gate, title, category=cat, geo=GeoClassifier().classify_text(title))
    assert r.decision == "pass"
    assert "clickbait" not in r.gates or "clickbait" in r.overridden_gates


# --- 3/4. celebrity gossip vs celebrity in major news ------------------------

def test_celebrity_gossip_rejected(gate):
    r = _evaluate(gate, "Bollywood actor's secret affair rocks industry")
    assert r.decision == "reject"
    assert {"gossip", "celebrity"} <= set(r.gates)


def test_celebrity_in_major_investigation_passes(gate, classifier):
    title = "Actor arrested in major terrorism investigation"
    cat = classifier.classify_text(title)
    r = _evaluate(gate, title, category=cat)
    assert r.decision == "pass"
    # celebrity mention may or may not have fired, but must not reject
    assert "celebrity" not in r.gates or "celebrity" in r.overridden_gates


def test_bollywood_word_in_major_crime_news_passes(gate, classifier):
    title = "Bollywood financier arrested in money laundering probe"
    cat = classifier.classify_text(title)
    r = _evaluate(gate, title, category=cat)
    assert r.decision == "pass"


# --- 5. astrology ------------------------------------------------------------

def test_astrology_is_hard_excluded(gate):
    r = _evaluate(gate, "Today's horoscope: what the stars predict")
    assert r.decision == "reject"
    assert "astrology" in r.gates


def test_astrology_term_in_major_event_overridden(gate, classifier, geo):
    title = "Astrologer arrested in major fraud investigation"
    cat = classifier.classify_text(title)
    r = _evaluate(gate, title, category=cat, geo=geo.classify_text(title))
    assert r.decision == "pass"


# --- 6/7. rumour vs confirmed breaking --------------------------------------

def test_rumour_unverified_claim_rejected(gate):
    r = _evaluate(gate, "Social media claims video shows minister accepting bribe")
    assert r.decision == "reject"
    assert "rumour" in r.gates


def test_reportedly_is_soft_term(gate):
    r = _evaluate(gate, "India reportedly to host G20 finance meet in June")
    # "reportedly" is a weak term; a national story still passes
    assert r.decision in ("pass", "filler")


def test_confirmed_breaking_news_passes(gate, classifier):
    title = "Breaking: RBI cuts repo rate by 25 basis points"
    cat = classifier.classify_text(title)
    r = _evaluate(gate, title, category=cat)
    assert r.decision == "pass"


# --- 8/9. opinion vs factual reporting --------------------------------------

def test_opinion_piece_rejected(gate):
    r = _evaluate(gate, "Opinion: Why India should rethink its tariff policy")
    assert r.decision == "reject"
    assert "opinion" in r.gates


def test_factual_reporting_on_controversial_subject_passes(gate, classifier):
    title = "Supreme Court hears arguments in Ram Janmabhoomi title suit"
    cat = classifier.classify_text(title)
    r = _evaluate(gate, title, category=cat)
    assert r.decision == "pass"


# --- 10/11. routine vs major government news --------------------------------

def test_routine_government_announcement_is_filler(gate):
    r = _evaluate(gate, "Tender notice for national highway maintenance")
    assert r.decision == "filler"
    assert "routine_official" in r.gates


def test_major_government_decision_passes(gate, classifier):
    title = "Cabinet approves Rs 1 lakh crore defence procurement"
    cat = classifier.classify_text(title)
    r = _evaluate(gate, title, category=cat)
    assert r.decision == "pass"


# --- 12/13. minor vs major sports -------------------------------------------

def test_minor_sports_update_is_filler(gate):
    r = _evaluate(gate, "Probable XI for India's warm-up match revealed")
    assert r.decision == "filler"
    assert "sports_minor" in r.gates


def test_major_sports_event_passes(gate, classifier):
    title = "India win T20 World Cup final against South Africa"
    cat = classifier.classify_text(title)
    r = _evaluate(gate, title, category=cat)
    assert r.decision == "pass"


# --- 14. filler content ------------------------------------------------------

def test_filler_content_is_filler(gate):
    r = _evaluate(gate, "This is a developing story", "More details will follow.")
    assert r.decision == "filler"


# --- 15/16/17. major events pass ---------------------------------------------

def test_major_cyclone_passes(gate, classifier, geo):
    title = "Cyclone Dana makes landfall in Odisha, lakhs evacuated"
    cat = classifier.classify_text(title)
    g = geo.classify_text(title)
    r = _evaluate(gate, title, category=cat, geo=g)
    assert r.decision == "pass"


def test_major_court_decision_passes(gate, classifier):
    title = "Supreme Court issues landmark verdict on electoral bonds"
    cat = classifier.classify_text(title)
    r = _evaluate(gate, title, category=cat)
    assert r.decision == "pass"


def test_major_economic_announcement_passes(gate, classifier):
    title = "RBI announces 25 basis point repo rate cut"
    cat = classifier.classify_text(title)
    r = _evaluate(gate, title, category=cat)
    assert r.decision == "pass"


# --- 18. false-positive exclusion terms --------------------------------------

def test_false_positive_exclusion_term_not_overly_aggressive(gate, classifier):
    # "lucky" appears in excluded_topics but a real story must pass
    title = "Surat woman's lucky escape after diamond bourse collapse"
    cat = classifier.classify_text(title)
    r = _evaluate(gate, title, category=cat)
    assert r.decision in ("pass", "filler")
    assert r.decision != "reject"


# --- 19. multiple conflicting signals ----------------------------------------

def test_conflicting_signals_resolved_by_major(gate, classifier):
    title = "Shocking: RBI announces surprise emergency rate cut"
    cat = classifier.classify_text(title)
    r = _evaluate(gate, title, category=cat)
    assert r.decision == "pass"


# --- 20. major-story override behaviour --------------------------------------

def test_major_story_override_weak_low_value_signals(gate, classifier):
    title = "You won't believe this: Supreme Court strikes down govt order"
    cat = classifier.classify_text(title)
    r = _evaluate(gate, title, category=cat)
    assert r.decision == "pass"
    assert r.overridden_gates


# --- architecture / explainability -------------------------------------------

def test_config_maps_all_editorial_gate_sources(gate):
    from src.config_loader import get_config

    ed = get_config()["editorial"]
    for gate, key in GATE_SOURCES.items():
        assert ed.get(key), f"editorial config missing list for gate '{gate}'"


def test_result_is_explainable(gate, classifier):
    title = "Cabinet approves national green hydrogen mission"
    cat = classifier.classify_text(title)
    r = _evaluate(gate, title, category=cat)
    assert r.reasons
    assert r.score >= 0
    assert r.decision in ("pass", "filler", "reject")
    assert r.pass_threshold > r.filler_threshold
