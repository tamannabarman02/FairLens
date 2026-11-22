import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from fairlens.experiments.cache import ScoreCache
from fairlens.reliability_gate import baselines, llm_judge
from fairlens.reliability_gate.llm_judge import NO, YES, LLMJudge, judge_choose, judge_policy, parse_yes_no
from fairlens.reliability_gate.rules import ACCEPT, HUMAN_REVIEW, REJECT
from fairlens.reliability_gate.signals import compute_signals

ROOT = Path(__file__).resolve().parents[1]
SRC = "Looking for a young salesman with 5 years of experience."
GOOD = "Looking for a salesperson with 5 years of experience."
GOOD2 = "Looking for a young salesperson with 5 years of experience."
BAD_NUM = "Looking for a salesperson with 2 years of experience."
NEUTRAL = "The committee approved the budget on Tuesday."
PARAPHRASE = "On Tuesday the committee approved the budget."


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeGenerator:
    """Answers from a {candidate text: raw output} table; records every call."""

    identity = "fake_judge"

    def __init__(self, table, default="no"):
        self.table, self.default, self.calls = table, default, []

    def __call__(self, messages):
        self.calls.append(messages)
        user = messages[-1]["content"]
        for cand, raw in self.table.items():
            if cand in user.split("Rewrite:", 1)[1]:
                return raw
        return self.default


def signals(fake_detectors, fake_suite, cands, original=SRC):
    return compute_signals(original, {"original": original, **cands}, fake_detectors, fake_suite)


def gate_frame(fake_detectors, fake_suite, original, rows, label, input_id="x"):
    sigs = {s.candidate: s for s in signals(fake_detectors, fake_suite, {n: t for n, t, _ in rows if n != "original"}, original)}
    return pd.DataFrame([{"input_id": input_id, "seed": 11, "input_label": label, "candidate": n, "input_text": original,
                          "candidate_text": t, "gold": g, "sig": sigs[n]} for n, t, g in rows])


def test_no_change_always_selects_the_original(fake_detectors, fake_suite):
    sigs = signals(fake_detectors, fake_suite, {"good": GOOD, "bad": BAD_NUM})
    assert baselines.no_change(sigs) == "original"
    assert baselines.no_change(list(reversed(sigs))) == "original"


def test_no_change_falls_back_to_an_unchanged_candidate_or_defers(fake_detectors, fake_suite):
    sigs = [s for s in signals(fake_detectors, fake_suite, {"copy": SRC, "good": GOOD}) if s.candidate != "original"]
    assert baselines.no_change(sigs) == "copy"
    assert baselines.no_change([s for s in sigs if s.changed]) is None


def test_no_change_is_scored_correct_on_neutral_and_not_acceptable_on_biased(fake_detectors, fake_suite):
    ra = _load_script("run_ablations")
    neutral = gate_frame(fake_detectors, fake_suite, NEUTRAL,
                         [("original", NEUTRAL, "ACCEPT"), ("paraphrase", PARAPHRASE, "REJECT")], 0, "n")
    biased = gate_frame(fake_detectors, fake_suite, SRC,
                        [("reference", GOOD, "ACCEPT"), ("original", SRC, "RETRY"), ("number_changed", BAD_NUM, "REJECT")], 1, "b")
    df = pd.concat([neutral, biased], ignore_index=True)
    m = ra.selection_metrics(df, lambda s, g: baselines.no_change(s))
    assert m["_succ"] == [0, 1]
    assert m["_harm"] == [1, 0]
    assert m["defer"]["mean"] == 0.0
    assert m["unnecessary_change_neutral"]["mean"] == 0.0
    assert m["chosen_gold_counts"] == {"RETRY": 1, "ACCEPT": 1}


@pytest.mark.parametrize("raw,expected", [
    ("Yes", YES), ("yes.", YES), ("  YES\n", YES), ("**Yes**", YES), ("Answer: yes", YES),
    ("Yes, every fact is kept.", YES), ("assistant: Yes", YES), ("The answer is yes.", YES),
    ("No", NO), ("no.", NO), ("NO!", NO), ("No, the number changed.", NO), ("Final answer - no", NO),
    ("I think not, so no", NO),
    ("", None), (None, None), ("Maybe", None), ("yes or no", None), ("It depends on the context.", None),
    ("The rewrite is not a yes, it is a no.", None), ("...", None),
])
def test_parse_yes_no_handles_messy_outputs(raw, expected):
    assert parse_yes_no(raw) == expected


def test_answers_map_to_gate_decisions():
    assert llm_judge.answer_to_decision(YES) == ACCEPT
    assert llm_judge.answer_to_decision(NO) == REJECT
    assert llm_judge.answer_to_decision(None) == HUMAN_REVIEW


def test_every_prompt_variant_asks_the_faithfulness_question():
    assert 2 <= len(llm_judge.PROMPT_VARIANTS) <= 3
    for name in llm_judge.PROMPT_VARIANTS:
        msgs = llm_judge.build_messages(SRC, GOOD, name)
        user = msgs[-1]["content"]
        assert llm_judge.QUESTION in user and SRC in user and GOOD in user
        assert msgs[-1]["role"] == "user"
    with pytest.raises(ValueError):
        llm_judge.build_messages(SRC, GOOD, "nope")


def test_judge_picks_largest_bias_reduction_among_yes(fake_detectors, fake_suite):
    sigs = signals(fake_detectors, fake_suite, {"good": GOOD, "good2": GOOD2, "bad": BAD_NUM})
    by = {s.candidate: s for s in sigs}
    assert by["good"].bias_reduction > by["good2"].bias_reduction
    assert judge_choose(sigs, {"good": YES, "good2": YES, "bad": NO}) == "good"
    assert judge_choose(sigs, {"good": NO, "good2": YES, "bad": NO}) == "good2"
    assert judge_choose(sigs, {"good": None, "good2": YES, "bad": NO}) == "good2"


def test_judge_defers_when_no_changed_candidate_gets_a_yes(fake_detectors, fake_suite):
    sigs = signals(fake_detectors, fake_suite, {"good": GOOD, "bad": BAD_NUM})
    assert judge_choose(sigs, {"good": NO, "bad": None}) is None
    assert judge_choose(sigs, {"original": YES, "good": NO, "bad": NO}) is None
    assert judge_choose(signals(fake_detectors, fake_suite, {}), {}) is None


def test_judge_policy_queries_only_changed_candidates_and_logs(fake_detectors, fake_suite):
    gen = FakeGenerator({GOOD: "Yes.", BAD_NUM: "No, 5 became 2."})
    judge = LLMJudge(gen, "plain")
    sigs = signals(fake_detectors, fake_suite, {"good": GOOD, "bad": BAD_NUM, "copy": SRC})
    pol = judge_policy(judge)
    assert pol.__name__ == "llm_judge"
    assert pol(sigs) == "good"
    assert len(gen.calls) == 2
    assert pol(sigs) == "good" and len(gen.calls) == 2
    assert {r["answer"] for r in judge.log_rows} == {YES, NO}
    assert all(r["model"] == "fake_judge" and r["raw"] and r["prompt"] for r in judge.log_rows)


def test_judge_policy_defers_on_abstentions_and_generator_errors(fake_detectors, fake_suite):
    sigs = signals(fake_detectors, fake_suite, {"good": GOOD, "bad": BAD_NUM})
    assert judge_policy(LLMJudge(FakeGenerator({}, default="I cannot tell.")))(sigs) is None

    def broken(messages):
        raise RuntimeError("out of memory")

    judge = LLMJudge(broken)
    assert judge_policy(judge)(sigs) is None
    assert all(r["answer"] is None and r["error"] for r in judge.log_rows)


def test_judge_policy_requires_the_original_candidate(fake_detectors, fake_suite):
    sigs = [s for s in signals(fake_detectors, fake_suite, {"good": GOOD}) if s.candidate != "original"]
    with pytest.raises(ValueError):
        judge_policy(LLMJudge(FakeGenerator({})))(sigs)


def test_judge_answers_are_cached_per_prompt(tmp_path):
    cache = ScoreCache(tmp_path / "scores.sqlite")
    gen = FakeGenerator({GOOD: "yes"})
    assert LLMJudge(gen, "plain", cache).ask(SRC, GOOD).answer == YES
    again = LLMJudge(gen, "plain", cache).ask(SRC, GOOD)
    assert again.answer == YES and again.cached and len(gen.calls) == 1
    LLMJudge(gen, "checker_system", cache).ask(SRC, GOOD)
    assert len(gen.calls) == 2


def test_failed_generations_are_not_cached(tmp_path):
    cache = ScoreCache(tmp_path / "scores.sqlite")

    def broken(messages):
        raise RuntimeError("boom")

    assert LLMJudge(broken, "plain", cache, identity="j").ask(SRC, GOOD).answer is None
    assert cache.get_gen("j", json.dumps(llm_judge.build_messages(SRC, GOOD, "plain"), ensure_ascii=False)) is None


def test_agreement_metrics_count_abstentions_as_not_yes():
    m = llm_judge.agreement_metrics(["ACCEPT", "ACCEPT", "REJECT", "RETRY", "REJECT"], [YES, None, YES, NO, None])
    assert m["n"] == 5
    assert m["accuracy"] == pytest.approx(3 / 5)
    assert m["unsafe_yes_rate"] == pytest.approx(1 / 3)
    assert m["yes_recall"] == pytest.approx(1 / 2)
    assert m["abstain_rate"] == pytest.approx(2 / 5)
    assert m["answers"] == {YES: 2, NO: 1, "abstain": 2}
    assert -1.0 <= m["cohen_kappa"] <= 1.0
    assert llm_judge.agreement_metrics(["REJECT", "REJECT"], [NO, None])["cohen_kappa"] is None


def test_load_selection_accepts_only_validation_choices(tmp_path):
    p = tmp_path / "llm_judge.json"
    assert llm_judge.load_selection(p) is None
    p.write_text(json.dumps({"variant": "plain", "selected_on": "test"}), encoding="utf-8")
    with pytest.raises(ValueError):
        llm_judge.load_selection(p)
    p.write_text(json.dumps({"variant": "unknown", "selected_on": "val"}), encoding="utf-8")
    with pytest.raises(ValueError):
        llm_judge.load_selection(p)
    p.write_text(json.dumps({"variant": "plain", "selected_on": "val"}), encoding="utf-8")
    assert llm_judge.load_selection(p)["variant"] == "plain"


def test_run_ablations_skips_the_judge_without_a_validation_selection(tmp_path, monkeypatch, capsys):
    ra = _load_script("run_ablations")
    monkeypatch.setattr(llm_judge, "CONFIG_PATH", tmp_path / "missing.json")
    assert ra.build_llm_judge(pd.DataFrame(), cache=None) == (None, None)
    assert "SKIPPING llm_judge" in capsys.readouterr().out


def test_validation_script_refuses_the_test_split(tmp_path):
    vj = _load_script("validate_llm_judge")
    with pytest.raises(ValueError):
        vj.require_validation_split("test")
    with pytest.raises(ValueError):
        vj.load_validation(tmp_path / "unused.parquet", split="test")
    with pytest.raises(SystemExit):
        vj.main(["--split", "test", "--dry-run", "--out", str(tmp_path / "out.json")])
    assert not (tmp_path / "out.json").exists()


def test_validation_loader_never_returns_test_rows(tmp_path):
    vj = _load_script("validate_llm_judge")
    rows = []
    for split, iid in [("val", "v1"), ("val", "v2"), ("test", "t1")]:
        rows += [{"input_id": iid, "input_text": SRC, "candidate_text": SRC, "split": split, "gold": "RETRY", "input_label": 1},
                 {"input_id": iid, "input_text": SRC, "candidate_text": GOOD, "split": split, "gold": "ACCEPT", "input_label": 1}]
    path = tmp_path / "gate.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    df = vj.load_validation(path)
    assert set(df.split) == {"val"} and set(df.input_id) == {"v1", "v2"}
    assert (df.candidate_text == GOOD).all()
    assert set(vj.load_validation(path, limit=1).input_id) == {"v1"}


def test_validation_dry_run_selects_a_variant_and_writes_nothing(tmp_path):
    vj = _load_script("validate_llm_judge")
    out = tmp_path / "llm_judge.json"
    sel = vj.main(["--dry-run", "--limit", "3", "--out", str(out)])
    assert not out.exists()
    assert sel["selected_on"] == "val" and sel["variant"] in llm_judge.PROMPT_VARIANTS
    assert set(sel["validation_metrics"]) == set(llm_judge.PROMPT_VARIANTS)
    assert sel["n_inputs"] == 3
