"""Rule-based rewrite baseline: lexicon substitution plus generalisation softening."""
from __future__ import annotations

import re

from ..detection import lexicon as L
from .base import Rewriter


def _match_case(src: str, repl: str) -> str:
    if not repl:
        return repl
    if src.isupper() and len(src) > 1:
        return repl.upper()
    if src[0].isupper():
        return repl[0].upper() + repl[1:]
    return repl


def _sub_dict(text: str, mapping: dict[str, str]) -> str:
    keys = sorted(mapping, key=len, reverse=True)
    rx = re.compile(r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b", re.I)
    return rx.sub(lambda m: _match_case(m.group(0), mapping[m.group(0).lower()]), text)


def _tidy(text: str) -> str:
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s*,", ",", text)
    text = re.sub(r"\b(a|an|the)\s*,\s*", r"\1 ", text, flags=re.I)   # "a , proactive" -> "a proactive"
    text = re.sub(r"\s{2,}", " ", text).strip()
    # fix indefinite article after deletions/substitutions
    text = re.sub(r"\b([Aa])n? (?=[aeiouAEIOU])", lambda m: m.group(1) + "n ", text)
    text = re.sub(r"\b([Aa])n (?=[^aeiouAEIOU\W])", lambda m: m.group(1) + " ", text)
    return text[0].upper() + text[1:] if text else text


class RuleBasedRewriter(Rewriter):
    name = "rule_based"
    display = "Rule-based"

    def _rewrite(self, text, strict, facts):
        out = _sub_dict(text, L.GENDERED_TERMS)
        out = _sub_dict(out, L.AGE_CODED)
        out = _sub_dict(out, L.GENDER_CODED_TRAITS)
        out = _sub_dict(out, L.LOADED_LANGUAGE)
        # "All/Most women are X" -> "Some people are X"  (softens a universal claim about a group)
        groups = [g for gs in L.GROUPS.values() for g in gs if g not in {"he", "she", "man", "woman", "a man", "a woman"}]
        g_alt = "|".join(re.escape(g) for g in sorted(groups, key=len, reverse=True))
        out = re.sub(rf"\b(?:all|every|most|typical)\s+(?:{g_alt})\b", "some people", out, flags=re.I)
        out = re.sub(rf"^(?:{g_alt})\s+(are|always|never|can't|cannot|should|tend to)\b", r"Some people \1", out, flags=re.I)
        out = re.sub(r"\b(?:he or she|she or he)\b", "they", out, flags=re.I)
        out = re.sub(r"\b(?:his or her|her or his)\b", "their", out, flags=re.I)
        out = re.sub(r"\b(native|mother[- ]tongue) english speakers?\b", "fluent English speaker", out, flags=re.I)
        out = re.sub(r"\b(?:under|below|no older than) (?:the age of )?\d{2}\b ?", "", out, flags=re.I)
        return _tidy(out), None
