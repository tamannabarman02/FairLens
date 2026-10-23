"""Transparent threshold rules of the FairLens Reliability Gate.

Candidate-level verdict precedence:  REJECT  >  RETRY  >  HUMAN_REVIEW  >  ACCEPT
  REJECT       a preservation check failed (meaning, facts, entailment, fluency) or a neutral input was changed
  RETRY        meaning preserved but bias not reduced enough / residual bias remains
  HUMAN_REVIEW checks pass but the detectors disagree or are not confident about the rewrite
  ACCEPT       every enabled check passed (or: neutral input left unchanged -> NO CHANGE)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ..config import load_config
from .signals import Signals

ACCEPT, REJECT, RETRY, HUMAN_REVIEW = "ACCEPT", "REJECT", "RETRY", "HUMAN_REVIEW"
PRECEDENCE = {REJECT: 3, RETRY: 2, HUMAN_REVIEW: 1, ACCEPT: 0}

REASONS = {
    "NEUTRAL_UNCHANGED": "Input is not biased and was left unchanged (NO CHANGE).",
    "UNNECESSARY_CHANGE": "Input is not biased, so any rewrite is an unnecessary change.",
    "FACT_LOSS": "A fact (entity, number, date, role, quote or negation) was lost or changed.",
    "ENTAILMENT_FAIL": "The rewrite and original do not entail each other in both directions.",
    "CONTRADICTION": "The NLI model judges the rewrite to contradict the original.",
    "LOW_SIMILARITY": "Semantic similarity to the original is below the threshold.",
    "LENGTH_OUT_OF_BOUNDS": "The rewrite is much shorter or longer than the original.",
    "FLUENCY_FAIL": "The rewrite failed fluency/sanity checks (repetition, prompt leakage, garbage).",
    "INSUFFICIENT_BIAS_REDUCTION": "The ensemble bias score did not drop enough.",
    "RESIDUAL_BIAS": "The rewrite is still classified as biased.",
    "DETECTOR_DISAGREEMENT": "Detectors disagree strongly about the rewrite.",
    "LOW_CONFIDENCE": "The detector ensemble is not confident about the rewrite.",
    "ALL_CHECKS_PASSED": "Bias reduced and meaning, facts and fluency preserved.",
}

# Which ablation group switches off which reason codes.
GROUP_OF_REASON = {
    "FACT_LOSS": "facts", "ENTAILMENT_FAIL": "entailment", "CONTRADICTION": "entailment",
    "LOW_SIMILARITY": "semantic", "LENGTH_OUT_OF_BOUNDS": "fluency", "FLUENCY_FAIL": "fluency",
    "INSUFFICIENT_BIAS_REDUCTION": "bias", "RESIDUAL_BIAS": "bias",
    "DETECTOR_DISAGREEMENT": "agreement", "LOW_CONFIDENCE": "uncertainty",
}
VERDICT_OF_REASON = {
    "FACT_LOSS": REJECT, "ENTAILMENT_FAIL": REJECT, "CONTRADICTION": REJECT, "LOW_SIMILARITY": REJECT,
    "LENGTH_OUT_OF_BOUNDS": REJECT, "FLUENCY_FAIL": REJECT, "UNNECESSARY_CHANGE": REJECT,
    "INSUFFICIENT_BIAS_REDUCTION": RETRY, "RESIDUAL_BIAS": RETRY,
    "DETECTOR_DISAGREEMENT": HUMAN_REVIEW, "LOW_CONFIDENCE": HUMAN_REVIEW,
}


@dataclass
class GateConfig:
    biased_input_threshold: float = 0.5
    min_bias_reduction: float = 0.15
    max_residual_bias: float = 0.5
    min_semantic_similarity: float = 0.70
    require_bidirectional_entailment: bool = True
    min_entailment_prob: float = 0.5
    max_contradiction: float = 0.5
    min_fact_retention: float = 1.0
    max_detector_std: float = 0.35
    min_confidence_margin: float = 0.10
    length_ratio_bounds: tuple[float, float] = (0.5, 2.0)
    min_fluency: float = 0.5
    max_retries: int = 2
    disabled: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def from_config(cls, **overrides) -> "GateConfig":
        g = dict(load_config()["gate"])
        g["length_ratio_bounds"] = tuple(g["length_ratio_bounds"])
        g.update(overrides)
        return cls(**{k: v for k, v in g.items() if k in cls.__dataclass_fields__})


@dataclass
class Verdict:
    candidate: str
    decision: str
    reasons: list[str]
    checks: dict[str, bool]

    def to_dict(self):
        return asdict(self)


def input_is_biased(sig: Signals, cfg: GateConfig) -> bool:
    return sig.bias_before >= cfg.biased_input_threshold


def judge(sig: Signals, cfg: GateConfig, input_biased: bool | None = None) -> Verdict:
    biased = input_is_biased(sig, cfg) if input_biased is None else input_biased
    if not biased:
        if not sig.changed:
            return Verdict(sig.candidate, ACCEPT, ["NEUTRAL_UNCHANGED"], {"unchanged": True})
        return Verdict(sig.candidate, REJECT, ["UNNECESSARY_CHANGE"], {"unchanged": False})

    lo, hi = cfg.length_ratio_bounds
    checks = {
        "FACT_LOSS": sig.fact_retention >= cfg.min_fact_retention,
        "ENTAILMENT_FAIL": (not cfg.require_bidirectional_entailment) or sig.entailment_both_ways >= cfg.min_entailment_prob,
        "CONTRADICTION": sig.contradiction <= cfg.max_contradiction,
        "LOW_SIMILARITY": sig.semantic_similarity >= cfg.min_semantic_similarity,
        "LENGTH_OUT_OF_BOUNDS": lo <= sig.length_ratio <= hi,
        "FLUENCY_FAIL": sig.fluency >= cfg.min_fluency,
        "INSUFFICIENT_BIAS_REDUCTION": sig.bias_reduction >= cfg.min_bias_reduction,
        "RESIDUAL_BIAS": sig.bias_after < cfg.max_residual_bias,
        "DETECTOR_DISAGREEMENT": sig.detector_std <= cfg.max_detector_std,
        "LOW_CONFIDENCE": sig.confidence_margin >= cfg.min_confidence_margin,
    }
    active = {k: v for k, v in checks.items() if GROUP_OF_REASON[k] not in cfg.disabled}
    failed = [k for k, ok in active.items() if not ok]
    if not failed:
        return Verdict(sig.candidate, ACCEPT, ["ALL_CHECKS_PASSED"], active)
    decision = max((VERDICT_OF_REASON[f] for f in failed), key=PRECEDENCE.get)
    failed.sort(key=lambda f: -PRECEDENCE[VERDICT_OF_REASON[f]])
    return Verdict(sig.candidate, decision, failed, active)


def candidate_quality(sig: Signals) -> float:
    """Tie-break among ACCEPTed candidates: bias reduction + semantic similarity + fact retention."""
    return sig.bias_reduction + sig.semantic_similarity + sig.fact_retention


@dataclass
class GateDecision:
    decision: str
    chosen: str | None
    chosen_text: str | None
    reason_code: str
    reason: str
    verdicts: list[Verdict]
    input_biased: bool
    attempts: int = 0

    def to_dict(self):
        d = asdict(self)
        return d


def decide(signals: list[Signals], cfg: GateConfig, allow_retry: bool = False, attempts: int = 0) -> GateDecision:
    """Input-level decision over all candidates (the original, unchanged text should be among them)."""
    if not signals:
        raise ValueError("no candidates")
    biased = input_is_biased(signals[0], cfg)
    verdicts = [judge(s, cfg, biased) for s in signals]
    by_name = {s.candidate: s for s in signals}
    acc = [v for v in verdicts if v.decision == ACCEPT]
    if acc:
        best = max(acc, key=lambda v: candidate_quality(by_name[v.candidate]))
        code = best.reasons[0]
        return GateDecision(ACCEPT, best.candidate, by_name[best.candidate].text, code, REASONS[code], verdicts, biased, attempts)
    review = [v for v in verdicts if v.decision == HUMAN_REVIEW]
    if review:
        best = max(review, key=lambda v: candidate_quality(by_name[v.candidate]))
        code = best.reasons[0]
        return GateDecision(HUMAN_REVIEW, best.candidate, by_name[best.candidate].text, code, REASONS[code], verdicts, biased, attempts)
    changed = [v for v in verdicts if by_name[v.candidate].changed]
    pool = changed or verdicts
    best = min(pool, key=lambda v: (PRECEDENCE[v.decision], -candidate_quality(by_name[v.candidate])))
    code = best.reasons[0]
    if allow_retry and attempts < cfg.max_retries:
        return GateDecision(RETRY, best.candidate, by_name[best.candidate].text, code, REASONS[code], verdicts, biased, attempts)
    return GateDecision(HUMAN_REVIEW, None, None, "RETRIES_EXHAUSTED" if allow_retry else code,
                        ("No candidate passed after retries; escalated to a human. Last failure: " if allow_retry else
                         "No candidate passed; needs a human. Main failure: ") + REASONS[code], verdicts, biased, attempts)
