"""Simpler selection policies the Gate is compared against.

Each policy receives the candidate signals for one input (the unchanged original included, named
"original") and returns the chosen candidate name, or None to defer to a human.
"""
from __future__ import annotations

import random
from typing import Callable

from .rules import ACCEPT, GateConfig, decide
from .signals import Signals

Policy = Callable[[list[Signals]], "str | None"]


def lowest_bias(sigs: list[Signals]) -> str | None:
    """(a) choose the candidate with the lowest ensemble bias score (ties -> first)."""
    return min(sigs, key=lambda s: s.bias_after).candidate


def highest_similarity(sigs: list[Signals]) -> str | None:
    """(b) choose the changed candidate most similar to the original (the unchanged copy is excluded,
    otherwise this policy would trivially never rewrite anything)."""
    changed = [s for s in sigs if s.changed] or sigs
    return max(changed, key=lambda s: s.semantic_similarity).candidate


def always_first(first: str) -> Policy:
    """(c) always accept the first model's rewrite."""
    def pol(sigs):
        names = [s.candidate for s in sigs]
        return first if first in names else next((n for n in names if n != "original"), names[0])
    pol.__name__ = f"always_{first}"
    return pol


def random_policy(seed: int) -> Policy:
    """(d) pick a random changed candidate."""
    rng = random.Random(seed)

    def pol(sigs):
        changed = [s.candidate for s in sigs if s.changed] or [s.candidate for s in sigs]
        return rng.choice(changed)
    pol.__name__ = "random"
    return pol


def no_change(sigs: list[Signals]) -> str | None:
    """Keep the original text (the do-nothing baseline)."""
    return "original"


def llm_judge_policy(judge_scores: dict[str, float], threshold: float, biased_input_threshold: float = 0.5) -> Policy:
    """LLM-as-judge baseline. Uses the same input-bias check as the Gate (ensemble >= threshold; otherwise keep the
    original), then replaces all of the Gate's preservation signals with one judge: among changed candidates with
    P(yes) >= threshold it picks the lowest bias score, and defers to a human when none qualifies.
    `judge_scores` maps candidate name -> P(yes) for one input."""
    def pol(sigs):
        if sigs and sigs[0].bias_before < biased_input_threshold:
            return "original"
        ok = [s for s in sigs if s.changed and judge_scores.get(s.candidate, 0.0) >= threshold]
        return min(ok, key=lambda s: s.bias_after).candidate if ok else None
    pol.__name__ = "llm_judge"
    return pol


def gate_policy(cfg: GateConfig) -> Policy:
    def pol(sigs):
        d = decide(sigs, cfg, allow_retry=False)
        return d.chosen if d.decision == ACCEPT else None
    pol.__name__ = "fairlens_gate"
    return pol
