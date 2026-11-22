"""Pure helpers that turn pipeline outputs into Review evidence (no Streamlit, unit-testable).

Inputs are the dicts produced by `pipeline.analyze_text(...).to_dict()` and `pipeline.mitigate_text(...).to_dict()`.
Nothing here computes new scores; it only regroups what the detectors, fact checker and Gate already returned.
"""
from __future__ import annotations

from ..detection.lexicon import CATEGORY_LABELS

SENTENCE_FLAG = 0.5
FACT_TYPES = [("entities", "Names / entities"), ("numbers", "Numbers"), ("dates", "Dates"), ("roles", "Roles"),
              ("negation", "Negation"), ("quotes", "Quotations")]


def category_label(cat: str) -> str:
    if cat == "model_attribution":
        return "Model attribution"
    return CATEGORY_LABELS.get(cat, cat.replace("_", " ").title())


def _sentence_of(analysis: dict, start: int) -> dict | None:
    for s in analysis.get("sentences") or []:
        if s["start"] <= start < s["end"]:
            return s
    return None


def flag_evidence(analysis: dict) -> list[dict]:
    """One row per flagged phrase: category, trigger explanation, rule weight and which detectors flag its sentence."""
    displays = {k: v["display"] for k, v in analysis["detector_results"].items()}
    rows: dict[tuple[int, int], dict] = {}
    for sp in analysis.get("spans") or []:
        key = (sp["start"], sp["end"])
        sent = _sentence_of(analysis, sp["start"])
        scores = (sent or {}).get("scores") or {}
        if not scores:
            scores = {k: v["score"] for k, v in analysis["detector_results"].items() if v.get("score") is not None}
        detectors = [{"detector": k, "display": displays.get(k, k), "score": s, "flags": s is not None and s >= SENTENCE_FLAG}
                     for k, s in scores.items()]
        source = "model" if sp["category"] == "model_attribution" else "rule"
        row = rows.setdefault(key, {"phrase": sp["text"], "start": sp["start"], "end": sp["end"], "category": sp["category"],
                                    "category_label": category_label(sp["category"]), "explanation": sp.get("explanation", ""),
                                    "weight": sp.get("weight"), "source": source,
                                    "sentence": (sent or {}).get("text", analysis["text"][:200]), "detectors": detectors})
        row["flagged_by"] = [d["display"] for d in row["detectors"] if d["flags"]]
    out = sorted(rows.values(), key=lambda r: r["start"])
    for c in analysis.get("categories") or []:
        if c.get("zero_shot") and not c.get("phrases"):
            out.append({"phrase": "(whole text)", "start": -1, "end": -1, "category": c["category"],
                        "category_label": category_label(c["category"]), "explanation": c.get("explanation", ""),
                        "weight": None, "source": "zero_shot", "sentence": analysis["text"][:200],
                        "detectors": [{"detector": "bart_mnli", "display": displays.get("bart_mnli", "BART-MNLI"),
                                       "score": c["zero_shot"], "flags": True}],
                        "flagged_by": [displays.get("bart_mnli", "BART-MNLI")]})
    return out


def _fact_detail(key: str, x: dict) -> str:
    if key == "negation":
        return f"{x.get('original', 0)} cue(s) in original, {x.get('rewrite', 0)} in rewrite"
    if not x.get("kept") and not x.get("missing"):
        return "none in original"
    parts = []
    if x.get("kept"):
        parts.append("kept: " + ", ".join(map(str, x["kept"])))
    if x.get("missing"):
        parts.append("missing: " + ", ".join(map(str, x["missing"])))
    return " · ".join(parts)


def preservation_items(candidate: dict, min_similarity: float | None = None, max_contradiction: float | None = None
                       ) -> list[tuple[str, bool | None, str]]:
    """Critical-fact checks for one candidate from the existing fact checker output (fact_checks)."""
    fc = candidate.get("fact_checks") or {}
    items: list[tuple[str, bool | None, str]] = []
    for key, label in FACT_TYPES:
        x = fc.get(key, {})
        nothing = key != "negation" and not x.get("kept") and not x.get("missing")
        items.append((label, None if nothing else x.get("pass"), _fact_detail(key, x)))
    if min_similarity is not None and max_contradiction is not None:
        ok = candidate["semantic_similarity"] >= min_similarity and candidate["contradiction"] <= max_contradiction
        items.append(("Overall meaning", ok, f"similarity {candidate['semantic_similarity']:.2f}, "
                                             f"contradiction {candidate['contradiction']:.2f}"))
    return items


def reason_rows(mitigation: dict, reasons: dict[str, str], name_of=lambda k: k) -> list[dict]:
    rows = []
    for c in mitigation["candidates"]:
        rows.append({"Version": name_of(c["candidate"]), "Verdict": c["verdict"],
                     "Reason codes": ", ".join(c["reasons"]),
                     "What it means": " ".join(reasons.get(r, r) for r in c["reasons"])})
    return rows


def entailment_signal(candidate: dict, mode: str) -> tuple[str, float]:
    """Label and value of the entailment signal the Gate actually checks ('forward' or 'both' directions)."""
    if mode == "forward":
        return "Entailment (original ⇒ rewrite)", candidate["entail_forward"]
    return "Bidirectional entailment", candidate["entailment_both_ways"]
