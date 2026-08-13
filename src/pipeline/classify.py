"""India news category classification.

Maps an article title + summary to a primary news category using the
taxonomy in config/categories.json. All category rules and weights live in
config (categories.json terms + config.json ``classification`` block), not in
code.

Key behaviours:
  * headlines outweigh summaries (title 1.0x, summary 0.6x)
  * multi-word terms and specific single-word terms classify; a curated
    generic-word list (config.json) stops lone ambiguous words (bank, bill,
    court, final, record, power, oil, gas, space, launch, research, ...)
    from causing false positives
  * process words (government, ministry, cabinet, parliament, policy,
    scheme, bill, mandate, regulation, notified) are deliberately weak so
    official announcements are classified by their actual subject
    (RBI decision -> Economy & Finance, ISRO mission -> Science & Space,
    Cabinet defence approval -> Defence & National Security)
  * international-affairs is context, not subject: when a concrete subject
    category also matches, international-affairs is damped so the subject
    wins (India-US defence deal -> Defence & National Security)
  * "India" itself is never a category signal
  * terms are stemmed lightly (s/es/ed/ing/d) so "launches", "arrested",
    "trains", "banks" match their base term
"""
import re
from dataclasses import dataclass, field
from typing import Optional

from src.config_loader import get_config
from src.models.article import Article

STEM_SUFFIX = "(?:s|es|ed|ing|d)?"
INTERNATIONAL_CATEGORY = "international-affairs"


@dataclass
class CategorySignal:
    category_id: str
    term: str
    location: str  # title | summary
    weight: float
    generic: bool = False


@dataclass
class ClassificationResult:
    primary: Optional[str]
    primary_label: Optional[str]
    primary_importance_weight: Optional[float]
    primary_urgent: Optional[bool]
    primary_score: float
    scores: dict
    secondary: list
    signals: list
    reasons: list
    primary_share: float

    @property
    def classified(self) -> bool:
        return self.primary is not None


class CategoryClassifier:
    """Classify articles into the configured news categories."""

    def __init__(self, config=None):
        config = config or get_config()
        cfg = config["config"].get("classification", {})
        self.title_multiplier = float(cfg.get("title_multiplier", 1.0))
        self.summary_multiplier = float(cfg.get("summary_multiplier", 0.6))
        self.multiword_weight = float(cfg.get("multiword_weight", 3.0))
        self.single_word_weight = float(cfg.get("single_word_weight", 2.0))
        self.generic_single_word_weight = float(cfg.get("generic_single_word_weight", 0.6))
        self.primary_min = float(cfg.get("primary_min_score", 2.0))
        self.secondary_min = float(cfg.get("secondary_min_score", 1.0))
        self.max_score = float(cfg.get("max_score", 100.0))
        self.intl_damp = float(cfg.get("international_damp", 0.6))
        self.generic_words = set(cfg.get("generic_single_words", []))
        self.excluded_terms = set(cfg.get("excluded_terms", []))

        self.categories = list(config["categories"]["categories"])
        self._order = [c["id"] for c in self.categories]
        self._meta = {c["id"]: c for c in self.categories}

        self._term_weights = {}
        self._term_regexes = {}
        for cat in self.categories:
            cid = cat["id"]
            weights = {}
            for raw_term in cat.get("terms", []):
                term = (raw_term or "").strip().lower()
                if not term or term in self.excluded_terms:
                    continue
                weight = self._term_weight(term)
                weights[term] = weight
                self._term_regexes[(cid, term)] = self._compile(term)
            self._term_weights[cid] = weights

    # -- public API ---------------------------------------------------------

    def classify(self, article: Article) -> ClassificationResult:
        return self.classify_text(article.title or "", article.summary or "")

    def classify_text(self, title: str, summary: str = "") -> ClassificationResult:
        raw = {}
        signals = []
        used = set()

        for cid, weights in self._term_weights.items():
            for term, weight in weights.items():
                regex = self._term_regexes[(cid, term)]
                for text, location, mult in (
                    (title or "", "title", self.title_multiplier),
                    (summary or "", "summary", self.summary_multiplier),
                ):
                    if not text:
                        continue
                    key = (cid, term, location)
                    if key in used:
                        continue
                    for m in regex.finditer(text):
                        used.add(key)
                        contributed = weight * mult
                        raw[cid] = raw.get(cid, 0.0) + contributed
                        signals.append(
                            CategorySignal(
                                category_id=cid,
                                term=term,
                                location=location,
                                weight=round(contributed, 3),
                                generic=weight == self.generic_single_word_weight,
                            )
                        )
                        break

        effective = dict(raw)
        damped_intl = False
        if (
            raw.get(INTERNATIONAL_CATEGORY, 0.0) > 0.0
            and any(cid != INTERNATIONAL_CATEGORY and score >= self.primary_min for cid, score in raw.items())
        ):
            effective[INTERNATIONAL_CATEGORY] = raw[INTERNATIONAL_CATEGORY] * self.intl_damp
            damped_intl = True

        reasons = []
        for sig in sorted(
            signals, key=lambda s: (s.location, s.category_id, s.term)
        ):
            label = self._meta[sig.category_id]["label"]
            generic = " (generic)" if sig.generic else ""
            reasons.append(
                f"'{sig.term}' -> {label} ({sig.location}, weight {sig.weight:g}){generic}"
            )
        for cid in self._order:
            if raw.get(cid, 0.0) > 0.0:
                reasons.append(
                    f"{self._meta[cid]['label']} raw score {round(raw[cid], 1)}"
                )
        if damped_intl:
            reasons.append(
                f"international-affairs damped x{self.intl_damp:g} (subject category present)"
            )

        best_id, best_score = None, -1.0
        for cid in self._order:
            score = effective.get(cid, 0.0)
            if score >= self.primary_min and score > best_score:
                best_id, best_score = cid, score

        secondary = [
            cid
            for cid in self._order
            if cid != best_id and effective.get(cid, 0.0) >= self.secondary_min
        ]
        secondary.sort(key=lambda cid: (effective.get(cid, 0.0), -self._order.index(cid)), reverse=True)

        total = sum(effective.values())
        share = round(best_score / total, 3) if total > 0 else 0.0

        if best_id:
            meta = self._meta[best_id]
            reasons.append(
                f"primary: {meta['label']} (score {round(best_score, 1)})"
            )
        else:
            top = max(effective.values(), default=0.0)
            reasons.append(
                f"no category (best {round(top, 1)} < minimum {self.primary_min})"
            )

        meta = self._meta.get(best_id, {})
        return ClassificationResult(
            primary=best_id,
            primary_label=meta.get("label"),
            primary_importance_weight=meta.get("importance_weight"),
            primary_urgent=meta.get("urgent"),
            primary_score=round(best_score, 1) if best_id else 0.0,
            scores={cid: round(sc, 1) for cid, sc in effective.items() if sc > 0.0},
            secondary=secondary,
            signals=signals,
            reasons=reasons,
            primary_share=share,
        )

    # -- internals ----------------------------------------------------------

    def _term_weight(self, term: str) -> float:
        if " " in term:
            return self.multiword_weight
        if term in self.generic_words:
            return self.generic_single_word_weight
        return self.single_word_weight

    @staticmethod
    def _compile(term: str):
        pattern = r"(?<!\w)" + re.escape(term) + STEM_SUFFIX + r"(?!\w)"
        return re.compile(pattern, re.IGNORECASE)
