from conftest import FakeRewriter

from fairlens.reliability_gate import baselines
from fairlens.reliability_gate.learned import LearnedGate
from fairlens.reliability_gate.retry import gate_with_retry
from fairlens.reliability_gate.rules import ACCEPT, HUMAN_REVIEW, REJECT, RETRY, GateConfig, decide, judge
from fairlens.reliability_gate.signals import Signals, compute_signals

SRC = "Looking for a young salesman with 5 years of experience."
GOOD = "Looking for a salesperson with 5 years of experience."
BAD_NUM = "Looking for a salesperson with 2 years of experience."
NEUTRAL = "The committee approved the budget on Tuesday."


def cfg(**kw):
    base = dict(min_semantic_similarity=0.6, min_entailment_prob=0.4, max_detector_std=0.4, min_confidence_margin=0.1)
    base.update(kw)
    return GateConfig.from_config(**base)


def sigs(fake_detectors, fake_suite, cands, original=SRC):
    return {s.candidate: s for s in compute_signals(original, {"original": original, **cands}, fake_detectors, fake_suite)}


def test_accepts_good_rewrite(fake_detectors, fake_suite):
    s = sigs(fake_detectors, fake_suite, {"good": GOOD})
    assert judge(s["good"], cfg()).decision == ACCEPT
    d = decide(list(s.values()), cfg())
    assert d.decision == ACCEPT and d.chosen == "good" and d.reason_code == "ALL_CHECKS_PASSED"


def test_rejects_fact_change(fake_detectors, fake_suite):
    s = sigs(fake_detectors, fake_suite, {"bad": BAD_NUM})
    v = judge(s["bad"], cfg())
    assert v.decision == REJECT and "FACT_LOSS" in v.reasons


def test_unchanged_biased_input_is_retry(fake_detectors, fake_suite):
    s = sigs(fake_detectors, fake_suite, {})
    v = judge(s["original"], cfg())
    assert v.decision == RETRY and "INSUFFICIENT_BIAS_REDUCTION" in v.reasons


def test_neutral_input_no_change(fake_detectors, fake_suite):
    s = sigs(fake_detectors, fake_suite, {"para": "The committee passed the budget on Tuesday."}, original=NEUTRAL)
    assert judge(s["original"], cfg()).reasons == ["NEUTRAL_UNCHANGED"]
    assert judge(s["para"], cfg()).decision == REJECT
    d = decide(list(s.values()), cfg())
    assert d.decision == ACCEPT and d.chosen == "original"


def test_disagreement_goes_to_human_review(fake_detectors, fake_suite):
    s = sigs(fake_detectors, fake_suite, {"good": GOOD})["good"]
    s.detector_std = 0.45
    assert judge(s, cfg()).decision == HUMAN_REVIEW
    assert judge(s, cfg(disabled=frozenset({"agreement"}))).decision == ACCEPT


def test_ablation_disables_fact_check(fake_detectors, fake_suite):
    s = sigs(fake_detectors, fake_suite, {"bad": BAD_NUM})["bad"]
    assert judge(s, cfg(disabled=frozenset({"facts"}))).decision == ACCEPT


def test_retry_then_accept_with_strict_prompt(fake_detectors, fake_suite):
    llm = FakeRewriter("llm", {SRC: BAD_NUM}, strict_mapping={SRC: GOOD})
    d, all_sigs, log = gate_with_retry(SRC, {"llm": llm}, fake_detectors, fake_suite, cfg(max_retries=2))
    assert d.decision == ACCEPT and d.chosen == "llm_retry1" and d.attempts == 1
    assert [c[1] for c in llm.calls] == [False, True]


def test_retry_exhausted_escalates(fake_detectors, fake_suite):
    llm = FakeRewriter("llm", {SRC: BAD_NUM}, strict_mapping={SRC: BAD_NUM})
    d, _, log = gate_with_retry(SRC, {"llm": llm}, fake_detectors, fake_suite, cfg(max_retries=2))
    assert d.decision == HUMAN_REVIEW and d.reason_code == "RETRIES_EXHAUSTED" and len(log) == 3


def test_baselines(fake_detectors, fake_suite):
    s = list(sigs(fake_detectors, fake_suite, {"good": GOOD, "bad": BAD_NUM}).values())
    assert baselines.lowest_bias(s) in {"good", "bad"}
    assert baselines.highest_similarity(s) in {"good", "bad"}
    assert baselines.always_first("bad")(s) == "bad"
    assert baselines.random_policy(0)(s) in {"good", "bad"}
    assert baselines.gate_policy(cfg())(s) == "good"


def test_learned_gate_fits():
    import numpy as np

    rng = np.random.default_rng(0)
    X = rng.random((60, 14))
    y = ["ACCEPT" if x[9] > 0.5 else "REJECT" for x in X]
    g = LearnedGate("tree").fit(X, y)
    assert set(g.predict(X)) <= {"ACCEPT", "REJECT", "HUMAN_REVIEW"}
    assert LearnedGate("logreg", drop_groups=("facts",)).fit(X, y).cols.__len__() == 13
