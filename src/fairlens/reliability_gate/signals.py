"""Gate signals computed for each candidate rewrite."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import numpy as np

SIGNAL_GROUPS = ["bias", "semantic", "entailment", "facts", "agreement", "uncertainty", "fluency"]


@dataclass
class Signals:
    candidate: str
    text: str
    changed: bool
    bias_before: float
    bias_after: float
    bias_reduction: float
    detector_scores_before: dict[str, float]
    detector_scores_after: dict[str, float]
    detector_std: float                 # disagreement across detectors on the rewrite
    confidence_margin: float            # |ensemble - 0.5| * 2 on the rewrite
    uncertainty: float                  # 0.5 * normalised std + 0.5 * (1 - margin)
    semantic_similarity: float
    entail_forward: float
    entail_backward: float
    entailment_both_ways: float         # min(forward, backward)
    contradiction: float
    fact_retention: float
    fact_checks: dict = field(default_factory=dict)
    fluency: float = 1.0
    length_ratio: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)

    def feature_vector(self, input_biased: bool) -> list[float]:
        return [self.bias_before, self.bias_after, self.bias_reduction, self.detector_std, self.confidence_margin,
                self.semantic_similarity, self.entail_forward, self.entail_backward, self.contradiction,
                self.fact_retention, self.fluency, self.length_ratio, float(self.changed), float(input_biased)]


FEATURE_NAMES = ["bias_before", "bias_after", "bias_reduction", "detector_std", "confidence_margin",
                 "semantic_similarity", "entail_forward", "entail_backward", "contradiction",
                 "fact_retention", "fluency", "length_ratio", "changed", "input_biased"]

# Which feature columns belong to which ablation group (used by the learned gate ablations).
FEATURE_GROUPS = {
    "bias": ["bias_after", "bias_reduction"],
    "semantic": ["semantic_similarity"],
    "entailment": ["entail_forward", "entail_backward", "contradiction"],
    "facts": ["fact_retention"],
    "agreement": ["detector_std"],
    "uncertainty": ["confidence_margin"],
    "fluency": ["fluency", "length_ratio"],
}


def normalise(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().lower()).rstrip(".")


def ensemble(scores: dict[str, float]) -> float:
    return float(np.mean(list(scores.values()))) if scores else 0.0


def compute_signals(original: str, candidates: dict[str, str], detectors: dict, suite,
                    detector_cache: dict[str, dict[str, float]] | None = None) -> list[Signals]:
    """Score `original` and every candidate with all detectors (one detector model at a time) and run the
    preservation suite. `detector_cache` maps text -> {detector: score} to avoid recomputation."""
    from ..evaluation.preservation import fluency as fluency_check
    from ..evaluation import facts as fact_checker

    cache = detector_cache if detector_cache is not None else {}
    texts = [original] + [t for t in candidates.values()]
    todo = [t for t in dict.fromkeys(texts) if t not in cache or set(detectors) - set(cache[t])]
    for name, det in detectors.items():
        need = [t for t in todo if name not in cache.get(t, {})]
        if need:
            for t, r in zip(need, det.predict_batch(need)):
                cache.setdefault(t, {})[name] = r.score
    changed_names = [n for n, t in candidates.items() if normalise(t) != normalise(original)]
    pres = {}
    if changed_names:
        evals = suite.evaluate([original] * len(changed_names), [candidates[n] for n in changed_names])
        pres = dict(zip(changed_names, evals))
    before = {k: cache[original][k] for k in detectors}
    b0 = ensemble(before)
    out = []
    for name, text in candidates.items():
        after = {k: cache[text][k] for k in detectors}
        b1 = ensemble(after)
        std = float(np.std(list(after.values()))) if len(after) > 1 else 0.0
        margin = abs(b1 - 0.5) * 2
        if name in pres:
            p = pres[name]
            sim, e = p["semantic_similarity"], p["entailment"]
            fc, fl = p["facts"], p["fluency"]
            ef, eb, eboth, contra = e["forward"], e["backward"], e["both"], e["contradiction"]
        else:  # unchanged candidate: preservation is trivially perfect
            sim, ef, eb, eboth, contra = 1.0, 1.0, 1.0, 1.0, 0.0
            fc = fact_checker.check(original, text)
            fl = fluency_check(original, text)
        out.append(Signals(
            candidate=name, text=text, changed=name in pres, bias_before=b0, bias_after=b1, bias_reduction=b0 - b1,
            detector_scores_before=before, detector_scores_after=after, detector_std=std, confidence_margin=margin,
            uncertainty=0.5 * min(1.0, std / 0.5) + 0.5 * (1 - margin), semantic_similarity=float(sim),
            entail_forward=float(ef), entail_backward=float(eb), entailment_both_ways=float(eboth),
            contradiction=float(contra), fact_retention=float(fc["fact_retention"]), fact_checks=fc["checks"],
            fluency=float(fl["fluency"]), length_ratio=float(fl["length_ratio"])))
    return out
