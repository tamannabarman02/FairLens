"""Transparent rule-based detector: lexicon and pattern checks with per-match explanations."""
from __future__ import annotations

import re
import time
from functools import lru_cache

from . import lexicon as L
from .base import DetectionResult, Detector, Span, label_for


def _alt(words) -> str:
    return "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))


@lru_cache(maxsize=1)
def _compiled():
    rules: list[tuple[re.Pattern, str, float, str]] = []
    rules.append((re.compile(rf"\b(?:{_alt(L.GENDERED_TERMS)})\b", re.I), "gender", 0.35,
                  "Gender-marked term where a neutral alternative exists."))
    rules.append((re.compile(rf"\b(?:{_alt(L.AGE_CODED)})\b", re.I), "age", 0.3,
                  "Age-coded wording; can signal preference for or against an age group."))
    rules.append((re.compile(rf"\b(?:{_alt(L.GENDER_CODED_TRAITS)})\b", re.I), "gender", 0.15,
                  "Gender-coded trait word (common in job adverts)."))
    rules.append((re.compile(rf"\b(?:{_alt(L.LOADED_LANGUAGE)})\b", re.I), "loaded_language", 0.25,
                  "Loaded or derogatory wording that frames a group or event subjectively."))
    for pat, cat, w, expl in L.PHRASE_PATTERNS:
        rules.append((re.compile(pat, re.I), cat, w, expl))

    # Generalisations: "[quantifier] <group> <predicate>"
    group_rules = []
    for cat, groups in L.GROUPS.items():
        gs = [g for g in groups if g not in {"she", "he", "woman", "man", "a woman", "a man"}]
        g_alt = _alt(gs)
        pred = _alt(L.GENERALIZING_PREDICATES)
        trait = _alt(L.NEGATIVE_TRAITS)
        group_rules.append((
            re.compile(rf"\b(?P<q>{_alt(L.QUANTIFIERS)}\s+)?(?P<g>{g_alt})\s+(?:{pred})\b", re.I), cat,
            "Makes a broad generalisation about a social group."))
        group_rules.append((
            re.compile(rf"\b(?P<g>{g_alt})\b[^.!?]{{0,60}}?\b(?P<t>{trait})\b", re.I), cat,
            "Associates a social group with a negative or limiting trait."))
    # Individual gendered subject + stereotyped trait, e.g. "she is too emotional to lead"
    group_rules.append((
        re.compile(rf"\b(?P<g>she|he|women|men|a woman|a man)\b[^.!?]{{0,40}}?\b(?P<t>too emotional|hysterical|bossy|nagging|not tough enough|can't handle|cannot handle)\b", re.I),
        "gender", "Attributes a gender-stereotyped trait."))
    return rules, group_rules


class RuleBasedDetector(Detector):
    name = "rule_based"
    display = "Rule-based"

    def explain(self, text: str) -> list[Span]:
        rules, group_rules = _compiled()
        spans: list[Span] = []
        for rx, cat, w, expl in rules:
            for m in rx.finditer(text):
                spans.append(Span(m.start(), m.end(), m.group(0), cat, expl, w))
        for rx, cat, expl in group_rules:
            for m in rx.finditer(text):
                w = 0.55
                if m.groupdict().get("q"):
                    w = 0.7
                if m.groupdict().get("t"):
                    w = 0.6
                prefix = text[max(0, m.start() - 14) : m.start()].lower()
                if re.search(r"\bnot(?:\s+(?:all|every|most))?\s*$", prefix):  # "not all women are ..." pushes back on the generalisation
                    w *= 0.3
                spans.append(Span(m.start(), m.end(), m.group(0), cat, expl, w))
        return _dedupe(spans)

    def _score_spans(self, spans: list[Span]) -> float:
        p = 1.0
        for s in spans:
            p *= 1 - s.weight
        return 1 - p

    def _scores(self, texts):
        return [self._score_spans(self.explain(t)) for t in texts]

    def predict(self, text: str) -> DetectionResult:
        t0 = time.perf_counter()
        spans = self.explain(text)
        score = self._score_spans(spans)
        cats: dict[str, float] = {}
        for s in spans:
            cats[s.category] = max(cats.get(s.category, 0.0), s.weight)
        return DetectionResult(self.name, score, label_for(score), (time.perf_counter() - t0) * 1000, cats, spans)

    def info(self) -> dict:
        rules, group_rules = _compiled()
        return {**super().info(), "type": "lexicon+regex", "n_rules": len(rules) + len(group_rules),
                "license": "project code", "size_mb": 0}


def _dedupe(spans: list[Span]) -> list[Span]:
    """Drop spans fully contained in a stronger-or-equal span."""
    spans.sort(key=lambda s: (s.start, -(s.end - s.start)))
    kept: list[Span] = []
    for s in spans:
        if any(k.start <= s.start and s.end <= k.end and k.weight >= s.weight for k in kept):
            continue
        kept = [k for k in kept if not (s.start <= k.start and k.end <= s.end and s.weight >= k.weight)]
        kept.append(s)
    return sorted(kept, key=lambda s: s.start)
