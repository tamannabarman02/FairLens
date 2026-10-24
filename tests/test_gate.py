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


def test_no_change_baseline(fake_detectors, fake_suite):
    s = list(sigs(fake_detectors, fake_suite, {"good": GOOD}).values())
    assert baselines.no_change(s) == "original"


def test_llm_judge_baseline(fake_detectors, fake_suite):
    s = list(sigs(fake_detectors, fake_suite, {"good": GOOD, "bad": BAD_NUM}).values())
    pol = baselines.llm_judge_policy({"good": 0.9, "bad": 0.2}, threshold=0.5)
    assert pol(s) == "good"
    assert baselines.llm_judge_policy({"good": 0.4, "bad": 0.3}, threshold=0.5)(s) is None  # defer
    neutral = list(sigs(fake_detectors, fake_suite, {"p": "The committee passed the budget on Tuesday."}, original=NEUTRAL).values())
    assert baselines.llm_judge_policy({"p": 0.99}, 0.5)(neutral) == "original"  # same input check as the Gate


def test_llm_judge_scores_yes_no(monkeypatch):
    """Judge maps next-token logits for Yes/No to P(yes); checked with a tiny stand-in model."""
    import torch

    from fairlens.reliability_gate import judge as J

    class Tok:
        def encode(self, w, add_special_tokens=False):
            return [1] if w.strip().lower() == "yes" else [2] if w.strip().lower() == "no" else [0]

        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
            return msgs[-1]["content"]

        def __call__(self, text, return_tensors="pt"):
            return {"input_ids": torch.tensor([[0, 0]])}

    class Model:
        device = "cpu"

        def __call__(self, **kw):
            logits = torch.zeros(1, 2, 3)
            logits[0, -1, 1], logits[0, -1, 2] = 2.0, 0.0
            return type("O", (), {"logits": logits})

    j = J.LLMJudge("dummy")
    monkeypatch.setattr(j, "_load", lambda: (Tok(), Model()))
    p = j.p_yes("a", "b")
    assert abs(p - 1 / (1 + 2.718281828 ** -2)) < 1e-6


def test_learned_gate_fits():
    import numpy as np

    rng = np.random.default_rng(0)
    X = rng.random((60, 14))
    y = ["ACCEPT" if x[9] > 0.5 else "REJECT" for x in X]
    g = LearnedGate("tree").fit(X, y)
    assert set(g.predict(X)) <= {"ACCEPT", "REJECT", "HUMAN_REVIEW"}
    assert LearnedGate("logreg", drop_groups=("facts",)).fit(X, y).cols.__len__() == 13
