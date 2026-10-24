"""V4 G.1: validate the LLM-as-judge baseline before using it.

1. Validation (controlled gold set, val split, seed 11, stratified sample): score every judge prompt variant,
   choose the prompt + P(yes) threshold that maximise agreement (balanced accuracy) of "judge yes" with gold ACCEPT.
2. Store agreement (accuracy, balanced accuracy, Cohen's kappa, confusion) as a `judge_validation` run.
3. Score all controlled-set candidates (val + test, every seed) with the frozen prompt -> judge_scores.parquet,
   consumed by scripts/run_ablations.py. Optionally score real mitigation candidates (--mitigation-run).

  python scripts/run_judge.py [--val-n 240] [--mitigation-run <run_id>]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fairlens.experiments.tracking import RunTracker  # noqa: E402
from fairlens.reliability_gate.judge import PROMPTS, LLMJudge  # noqa: E402

PROC = ROOT / "datasets" / "processed"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def agreement(gold_accept: np.ndarray, p: np.ndarray, thr: float) -> dict:
    from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, confusion_matrix

    pred = (p >= thr).astype(int)
    return {"threshold": thr, "accuracy": float((pred == gold_accept).mean()),
            "balanced_accuracy": float(balanced_accuracy_score(gold_accept, pred)),
            "kappa": float(cohen_kappa_score(gold_accept, pred)),
            "confusion": confusion_matrix(gold_accept, pred, labels=[0, 1]).tolist(), "n": int(len(pred))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-n", type=int, default=240)
    ap.add_argument("--mitigation-run")
    a = ap.parse_args()
    gate = pd.read_parquet(PROC / "gate_controlled.parquet")
    pairs = gate[gate.candidate_text != gate.input_text]  # unchanged candidates need no judge (handled by policies)
    val = pairs[(pairs.split == "val") & (pairs.seed == 11)]
    rng = np.random.default_rng(0)
    # stratified: all gold-ACCEPT rows + an equal-sized random sample of the rest
    acc, rest = val[val.gold == "ACCEPT"], val[val.gold != "ACCEPT"]
    k = min(len(rest), max(len(acc), a.val_n - len(acc)))
    sample = pd.concat([acc, rest.iloc[rng.choice(len(rest), size=k, replace=False)]])
    judge = LLMJudge()
    tracker = RunTracker("judge_validation", config={"judge_model": judge.model_id, "prompts": PROMPTS, "val_n": len(sample),
                                                     "selection": "max balanced accuracy on val; thresholds 0.05..0.95"},
                         seed=0, dataset_files=[PROC / "gate_controlled.parquet"], models={"judge": f"{judge.model_id}@{judge.info()['revision']}"})
    y = (sample.gold == "ACCEPT").astype(int).to_numpy()
    per_prompt = {}
    for name in PROMPTS:
        judge.prompt = name
        t0 = time.perf_counter()
        p = np.array([judge.p_yes(o, r) for o, r in zip(sample.input_text, sample.candidate_text)])
        best = max((agreement(y, p, t) for t in np.round(np.arange(0.05, 0.96, 0.05), 2)), key=lambda m: m["balanced_accuracy"])
        per_prompt[name] = {**best, "ms_per_pair": (time.perf_counter() - t0) * 1000 / len(p)}
        log(f"{name}: {per_prompt[name]}")
    chosen = max(per_prompt, key=lambda n: per_prompt[n]["balanced_accuracy"])
    thr = per_prompt[chosen]["threshold"]
    judge.prompt = chosen
    tracker.save_artifact("judge_validation.json", {"per_prompt": per_prompt, "chosen_prompt": chosen, "threshold": thr,
                                                    "judge": judge.info()})
    log(f"chosen {chosen} @ {thr}; scoring all controlled candidates")
    uniq = pairs.drop_duplicates(["input_text", "candidate_text"])[["input_text", "candidate_text"]]
    scores = []
    for i, (o, r) in enumerate(zip(uniq.input_text, uniq.candidate_text)):
        scores.append(judge.p_yes(o, r))
        if i % 200 == 199:
            log(f"  {i + 1}/{len(uniq)}")
    uniq = uniq.assign(p_yes=scores)
    tracker.save_artifact("judge_scores.parquet", uniq)
    if a.mitigation_run:
        mp = ROOT / "outputs" / "runs" / a.mitigation_run / "mitigation_candidates.parquet"
        cand = pd.read_parquet(mp)
        cand = cand[cand.changed].drop_duplicates(["input_text", "text"])
        cand = cand.assign(p_yes=[judge.p_yes(o, r) for o, r in zip(cand.input_text, cand.text)])
        tracker.save_artifact("judge_scores_mitigation.parquet", cand[["input_id", "candidate", "input_text", "text", "p_yes"]])
    val_all = pairs[pairs.split == "val"].merge(uniq, on=["input_text", "candidate_text"])
    tracker.finish(metrics={"chosen_prompt": chosen, "threshold": thr, "val_sample": per_prompt[chosen],
                            "val_all_seeds": agreement((val_all.gold == "ACCEPT").astype(int).to_numpy(), val_all.p_yes.to_numpy(), thr)})
    log(f"done: {tracker.run_id}")


if __name__ == "__main__":
    main()
