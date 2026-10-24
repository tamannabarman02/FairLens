"""V4 Stage 4 analysis on stored runs (no new model inference except none): A.1, A.2, A.4, A.5.

Inputs: the latest completed `benchmark_gate` run (gate_signals.parquet), the latest `judge_validation` run, and
for A.2 the latest `benchmark_mitigation` run (+ its Gate decisions).
  A.1 risk-coverage: sweep each policy's strictness on VAL; unsafe-accept vs human-review (deferral) rate; the
      operating point is chosen on val (lowest deferral with val unsafe-accept <= target) and its TEST result is
      reported separately.
  A.2 red-team: corrupt rewrites the Gate ACCEPTED on real mitigation outputs (number, name, role, negation drop/add)
      with 3 seeds; catch rate = share of corrupted rewrites the Gate no longer accepts; CIs across seeds and items.
      Requires preservation/detector scoring of the corrupted texts (models are loaded for that part only).
  A.4 threshold sensitivity: vary each tuned threshold around its value on val and test (reported separately).
  A.5 conformal risk control (Angelopoulos et al. 2022) for the unsafe-accept loss on the controlled set, val as
      calibration, test as evaluation; exchangeability holds only within this synthetic set (stated in results).
  python scripts/run_analysis.py [--skip-redteam]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fairlens.evaluation import metrics as M  # noqa: E402
from fairlens.experiments.tracking import RunTracker, latest_run, load_manifest  # noqa: E402
from fairlens.reliability_gate.rules import ACCEPT, GateConfig, judge  # noqa: E402
from fairlens.reliability_gate.signals import Signals  # noqa: E402

SIG_FIELDS = [f for f in Signals.__dataclass_fields__]
TARGET_UNSAFE = 0.05


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def to_sig(r) -> Signals:
    d = {f: getattr(r, f) for f in SIG_FIELDS if f not in {"fact_checks", "detector_scores_before", "detector_scores_after", "candidate", "text"}}
    return Signals(candidate=r.candidate, text=r.candidate_text, fact_checks=json.loads(r.fact_checks),
                   detector_scores_before={}, detector_scores_after={}, **d)


def select(df: pd.DataFrame, accept_fn) -> tuple[np.ndarray, np.ndarray]:
    """Input-level: pick the accepted candidate with best quality, or defer. Returns per-input (harmful, deferred)."""
    harm, defer = [], []
    for _, g in df.groupby(["input_id", "seed"]):
        acc = [(r, s) for r, s in zip(g.itertuples(), g.sig) if accept_fn(s, r)]
        if not acc:
            harm.append(0); defer.append(1)
            continue
        r, s = max(acc, key=lambda x: x[1].bias_reduction + x[1].semantic_similarity + x[1].fact_retention)
        harm.append(int(r.gold != ACCEPT)); defer.append(0)
    return np.array(harm), np.array(defer)


def strictness_path(cfg: GateConfig, t: float) -> GateConfig:
    """1-D path from lenient (t=0) to strict (t=1) through the preservation thresholds."""
    return replace(cfg, min_semantic_similarity=0.3 + 0.65 * t, min_entailment_prob=0.0 + 0.9 * t,
                   max_contradiction=1.0 - 0.8 * t, min_fact_retention=0.5 + 0.5 * t)


def risk_coverage(val, test, tuned, judge_scores, judge_thr):
    curves, points = {}, {}
    grids = {"fairlens_gate": np.round(np.linspace(0, 1, 21), 3), "llm_judge": np.round(np.linspace(0.02, 0.98, 25), 3)}

    def gate_fn(t):
        c = strictness_path(tuned, t)
        return lambda s, r: judge(s, c).decision == ACCEPT

    def judge_fn(t):
        return lambda s, r: (s.bias_before < tuned.biased_input_threshold and not s.changed) or (
            s.bias_before >= tuned.biased_input_threshold and s.changed and judge_scores.get((r.input_text, r.candidate_text), 0) >= t)

    fns = {"fairlens_gate": gate_fn}
    if judge_scores:
        fns["llm_judge"] = judge_fn
    for name, fn in fns.items():
        rows = []
        for t in grids[name]:
            h, d = select(val, fn(t))
            rows.append({"knob": float(t), "unsafe_accept": float(h.mean()), "review_rate": float(d.mean())})
        curves[name] = rows
        ok = [r for r in rows if r["unsafe_accept"] <= TARGET_UNSAFE]
        op = min(ok, key=lambda r: (r["review_rate"], r["unsafe_accept"])) if ok else min(rows, key=lambda r: r["unsafe_accept"])
        ht, dt = select(test, fn(op["knob"]))
        points[name] = {"chosen_on_val": op, "test": {"unsafe_accept": M.bootstrap_ci(ht), "review_rate": M.bootstrap_ci(dt)}}
    h, d = select(val, lambda s, r: s.candidate == min([x for x in val[(val.input_id == r.input_id) & (val.seed == r.seed)].sig],
                                                          key=lambda x: x.bias_after).candidate)
    curves["lowest_bias"] = [{"knob": None, "unsafe_accept": float(h.mean()), "review_rate": float(d.mean())}]
    return curves, points


def sensitivity(df_by_split, tuned):
    knobs = {"min_semantic_similarity": [-0.2, -0.1, -0.05, 0, 0.05, 0.1], "min_entailment_prob": [-0.2, -0.1, 0, 0.1, 0.2],
             "min_bias_reduction": [-0.1, -0.05, 0, 0.05, 0.1], "max_detector_std": [-0.15, -0.05, 0, 0.1, 0.3],
             "min_confidence_margin": [-0.1, -0.05, 0, 0.05, 0.1]}
    out = {}
    for k, deltas in knobs.items():
        rows = []
        for dlt in deltas:
            v = float(np.clip(getattr(tuned, k) + dlt, 0, 1))
            c = replace(tuned, **{k: v})
            row = {"value": v, "delta": dlt}
            for split, df in df_by_split.items():
                pred = [judge(s, c).decision for s in df.sig]
                g = df.gold.to_numpy()
                p = np.array(pred)
                row[f"{split}_unsafe_accept"] = float(((p == ACCEPT) & (g != ACCEPT)).mean() / max(1e-9, (g != ACCEPT).mean()))
                row[f"{split}_accept_recall"] = float((p[g == ACCEPT] == ACCEPT).mean())
            rows.append(row)
        out[k] = rows
    return out


def conformal(val, test, alpha_list=(0.05, 0.10)):
    """CRC over a 1-D family of gates (strictness t): smallest-risk-guarantee threshold on calibration (val)."""
    tuned_like = GateConfig.from_config()
    ts = np.round(np.linspace(0, 1, 41), 3)
    def losses(df, t):
        c = strictness_path(tuned_like, t)
        return np.array([int(judge(s, c).decision == ACCEPT and g != ACCEPT) for s, g in zip(df.sig, df.gold)])
    cal = {t: losses(val, t) for t in ts}
    n = len(val)
    res = {}
    for a in alpha_list:
        ok = [t for t in ts if (n / (n + 1)) * cal[t].mean() + 1 / (n + 1) <= a]
        if not ok:
            res[str(a)] = {"status": "no threshold satisfies the bound"}
            continue
        t = min(ok)  # least strict gate that still satisfies the bound (loss is non-increasing in t on this path)
        lt = losses(test, t)
        acc_rec = np.mean([judge(s, strictness_path(tuned_like, t)).decision == ACCEPT for s, g in zip(test.sig, test.gold) if g == ACCEPT])
        res[str(a)] = {"t": float(t), "cal_risk": float(cal[t].mean()), "test_risk": M.bootstrap_ci(lt),
                       "test_risk_le_alpha": bool(lt.mean() <= a), "test_accept_recall": float(acc_rec), "n_cal": n, "n_test": len(test)}
    mono = all(cal[ts[i]].mean() >= cal[ts[i + 1]].mean() - 1e-12 for i in range(len(ts) - 1))
    return {"alphas": res, "loss": "1[accepted and gold != ACCEPT] per candidate", "risk_monotone_in_t_on_cal": mono,
            "exchangeability": "val and test inputs are a random split of one synthetic perturbation set; this does not "
                               "extend to real rewrites or other distributions."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-redteam", action="store_true")
    a = ap.parse_args()
    gr = latest_run("benchmark_gate")
    if not gr:
        sys.exit("no completed benchmark_gate run")
    gdir = Path(gr["manifest_path"]).parent
    gs = json.loads((gdir / "gate_summary.json").read_text(encoding="utf-8"))
    df = pd.read_parquet(gdir / "gate_signals.parquet")
    df["sig"] = [to_sig(r) for r in df.itertuples()]
    tuned = GateConfig.from_config(**gs["tuned_rules"])
    val, test = df[df.split == "val"], df[df.split == "test"]
    jr = latest_run("judge_validation")
    jscores, jthr = {}, None
    if jr:
        jd = Path(jr["manifest_path"]).parent
        sc = pd.read_parquet(jd / "judge_scores.parquet")
        jscores = {(o, t): p for o, t, p in zip(sc.input_text, sc.candidate_text, sc.p_yes)}
        jthr = json.loads((jd / "judge_validation.json").read_text(encoding="utf-8"))["threshold"]
    tracker = RunTracker("analysis", config={"gate_run": gr["run_id"], "judge_run": jr["run_id"] if jr else None,
                                             "target_unsafe_accept": TARGET_UNSAFE, "tuned_rules": gs["tuned_rules"]},
                         seed=0, dataset_files=[ROOT / "datasets/processed/gate_controlled.parquet"])
    out = {"gate_run": gr["run_id"], "judge_run": jr["run_id"] if jr else None}
    log("A.1 risk-coverage")
    out["risk_coverage"], out["operating_points"] = risk_coverage(val, test, tuned, jscores, jthr)
    log("A.4 sensitivity")
    out["sensitivity"] = sensitivity({"val": val, "test": test}, tuned)
    log("A.5 conformal")
    out["conformal"] = conformal(val, test)
    if not a.skip_redteam:
        log("A.2 red-team")
        out["redteam"] = redteam(tuned)
    tracker.save_artifact("analysis_summary.json", out)
    tracker.finish(metrics={"operating_points": out["operating_points"], "conformal": out["conformal"]["alphas"],
                            "redteam": (out.get("redteam") or {}).get("catch_rate")})
    log(f"done: {tracker.run_id}")


def redteam(tuned: GateConfig) -> dict:
    """Corrupt real rewrites the Gate accepted and check whether it still accepts them."""
    import random

    sys.path.insert(0, str(ROOT / "scripts"))
    from build_mitigation_sets import p_entity, p_negation, p_number, p_role

    from fairlens.detection.registry import build_detectors
    from fairlens.evaluation.preservation import PreservationSuite
    from fairlens.experiments.cache import ScoreCache
    from fairlens.experiments.mitigation_runner import PrecomputedSuite, preserve_all, score_all
    from fairlens.reliability_gate.signals import compute_signals

    mr = latest_run("benchmark_mitigation")
    if not mr:
        return {"status": "no completed benchmark_mitigation run"}
    cand = pd.read_parquet(Path(mr["manifest_path"]).parent / "mitigation_candidates.parquet")
    acc = cand[(cand.verdict == ACCEPT) & cand.changed]
    def neg_drop(t, rng):
        out = p_negation(t, rng)
        return out if out and len(out) < len(t) else None
    def neg_add(t, rng):
        out = p_negation(t, rng)
        return out if out and len(out) > len(t) else None
    corruptions = {"number_changed": p_number, "name_swapped": p_entity, "role_swapped": p_role,
                   "negation_dropped": neg_drop, "negation_added": neg_add}
    rows = []
    for seed in [11, 22, 33]:
        rng = random.Random(seed)
        for r in acc.itertuples():
            for cname, fn in corruptions.items():
                c = fn(r.text, rng)
                if c and c.strip() != r.text.strip():
                    rows.append({"seed": seed, "input_id": r.input_id, "original": r.input_text, "accepted": r.text,
                                 "corruption": cname, "corrupted": c})
    if not rows:
        return {"status": "no accepted rewrites to corrupt", "mitigation_run": mr["run_id"]}
    rt = pd.DataFrame(rows)
    detectors = build_detectors()
    det_scores = score_all(detectors, list(set(rt.original) | set(rt.corrupted)), ScoreCache())
    pres = preserve_all(list(zip(rt.original, rt.corrupted)), PreservationSuite())
    caught = []
    for r in rt.itertuples():
        s = compute_signals(r.original, {"original": r.original, "corrupted": r.corrupted}, detectors, PrecomputedSuite(pres), det_scores)[1]
        caught.append(int(judge(s, tuned).decision != ACCEPT))
    rt["caught"] = caught
    by = {}
    for cname, g in rt.groupby("corruption"):
        per_seed = g.groupby("seed").caught.mean()
        by[cname] = {"n": len(g), "catch_rate": M.bootstrap_ci(g.caught.to_numpy()),
                     "per_seed": {int(k): float(v) for k, v in per_seed.items()}, "seed_sd": float(per_seed.std(ddof=0))}
    return {"mitigation_run": mr["run_id"], "n_accepted_rewrites": int(len(acc)), "catch_rate": by,
            "examples_missed": rt[rt.caught == 0].head(10)[["corruption", "accepted", "corrupted"]].to_dict("records")}


if __name__ == "__main__":
    main()
