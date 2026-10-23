"""Retry logic: on failure, re-prompt the LLM rewriters with stricter preserve-facts instructions,
up to cfg.max_retries times, then escalate to HUMAN_REVIEW."""
from __future__ import annotations

from typing import Callable

from ..evaluation import facts as fact_checker
from .rules import ACCEPT, RETRY, GateConfig, GateDecision, decide
from .signals import Signals, compute_signals


def gate_with_retry(original: str, rewriters: dict, detectors: dict, suite, cfg: GateConfig,
                    detector_cache: dict | None = None, progress: Callable[[str], None] | None = None,
                    initial: dict[str, str] | None = None) -> tuple[GateDecision, list[Signals], list[dict]]:
    """Returns (final decision, signals of all candidates over all attempts, rewrite log)."""
    cache = detector_cache if detector_cache is not None else {}
    hints = fact_checker.extract(original).as_hints()
    log: list[dict] = []
    candidates: dict[str, str] = {"original": original}
    if initial is not None:
        candidates.update(initial)
    else:
        for name, rw in rewriters.items():
            if progress:
                progress(f"Rewriting with {rw.display}")
            r = rw.rewrite(original, attempt=0)
            log.append(r.to_dict())
            if r.text:
                candidates[name] = r.text
    all_signals = compute_signals(original, candidates, detectors, suite, cache)
    decision = decide(all_signals, cfg, allow_retry=True, attempts=0)
    attempt = 0
    llm_names = [n for n, rw in rewriters.items() if n != "rule_based"]
    while decision.decision == RETRY and attempt < cfg.max_retries and llm_names:
        attempt += 1
        new: dict[str, str] = {}
        for name in llm_names:
            if progress:
                progress(f"Retry {attempt}: strict prompt for {rewriters[name].display}")
            r = rewriters[name].rewrite(original, strict=True, facts=hints, attempt=attempt)
            log.append(r.to_dict())
            if r.text:
                new[f"{name}_retry{attempt}"] = r.text
        if not new:
            break
        all_signals += compute_signals(original, new, detectors, suite, cache)
        decision = decide(all_signals, cfg, allow_retry=True, attempts=attempt)
    if decision.decision == RETRY:  # budget exhausted
        decision = decide(all_signals, cfg, allow_retry=False, attempts=attempt)
        if decision.decision != ACCEPT:
            decision.reason_code = "RETRIES_EXHAUSTED"
            decision.reason = "No candidate passed after strict retries; escalated to human review."
    return decision, all_signals, log
