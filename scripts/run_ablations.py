"""Reliability Gate evaluation on the controlled gold set: rules vs val-tuned rules vs learned gate vs
simpler selection baselines, plus signal ablations with bootstrap CIs over 3 perturbation seeds.

  python scripts/run_ablations.py
Outputs (outputs/runs/<run_id>/): gate_signals.parquet, gate_summary.json
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fairlens.detection.registry import build_detectors  # noqa: E402
from fairlens.evaluation import metrics as M  # noqa: E402
from fairlens.evaluation.preservation import PreservationSuite  # noqa: E402
from fairlens.experiments.cache import ScoreCache, detector_identity  # noqa: E402
from fairlens.experiments.mitigation_runner import PrecomputedSuite, preserve_all, score_all  # noqa: E402
from fairlens.experiments.tracking import RunTracker  # noqa: E402
from fairlens.reliability_gate import baselines as B  # noqa: E402
from fairlens.reliability_gate.learned import LearnedGate  # noqa: E402
from fairlens.reliability_gate.rules import ACCEPT, HUMAN_REVIEW, GateConfig, judge  # noqa: E402
from fairlens.reliability_gate.signals import SIGNAL_GROUPS, compute_signals  # noqa: E402

PROC = ROOT / "datasets" / "processed"
GOLD = ["ACCEPT", "REJECT", "RETRY"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build_signals(gate: pd.DataFrame, detectors, cache) -> pd.DataFrame:
    texts = list(dict.fromkeys(gate.input_text.tolist() + gate.candidate_text.tolist()))
    det_scores = score_all(detectors, texts, cache)
    pres = preserve_all(list(zip(gate.input_text, gate.candidate_text)), PreservationSuite())
    rows = []
    for (iid, seed), g in gate.groupby(["input_id", "seed"]):
        orig = g.input_text.iloc[0]
        cands = dict(zip(g.candidate, g.candidate_text))
        cands.setdefault("original", orig)
        sigs = {s.candidate: s for s in compute_signals(orig, cands, detectors, PrecomputedSuite(pres), det_scores)}
        for r in g.itertuples():
            rows.append({**r._asdict(), "sig": sigs[r.candidate]})
    return pd.DataFrame(rows).drop(columns=["Index"])


def candidate_metrics(gold: list[str], pred: list[str], perturb: list[str] | None = None) -> dict:
    g, p = np.array(gold), np.array(pred)
    acc_pred = p == ACCEPT
    out = {
        "n": len(g),
        "accuracy_3class": float((g == p).mean()),
        "macro_f1_3class": float(np.mean([_f1(g == c, p == c) for c in GOLD])),
        "accept_precision": float((g[acc_pred] == ACCEPT).mean()) if acc_pred.any() else None,
        "accept_recall": float(acc_pred[g == ACCEPT].mean()) if (g == ACCEPT).any() else None,
        "unsafe_accept_rate": float(acc_pred[g != ACCEPT].mean()),
        "human_review_rate": float((p == HUMAN_REVIEW).mean()),
        "confusion": {gl: dict(Counter(p[g == gl])) for gl in GOLD},
    }
    if perturb is not None:
        pt = np.array(perturb)
        out["catch_rate_by_perturbation"] = {k: float((p[pt == k] != ACCEPT).mean()) for k in sorted(set(pt)) if k not in {"none"}}
        out["accept_rate_reference"] = float((p[pt == "none"] == ACCEPT).mean()) if (pt == "none").any() else None
    return out


def _f1(t, p):
    tp = (t & p).sum()
    return 0.0 if tp == 0 else 2 * tp / (t.sum() + p.sum())


def selection_metrics(df: pd.DataFrame, choose) -> dict:
    """Input-level: each policy picks a candidate (or None = defer). Success = chosen gold ACCEPT."""
    succ, harm, defer, unnec = [], [], [], []
    for (iid, seed), g in df.groupby(["input_id", "seed"]):
        sigs = list(g.sig)
        c = choose(sigs, g)
        if c is None:
            succ.append(0); harm.append(0); defer.append(1)
        else:
            gold = g.set_index("candidate").gold[c]
            succ.append(int(gold == ACCEPT)); harm.append(int(gold != ACCEPT)); defer.append(0)
        if g.input_label.iloc[0] == 0:
            unnec.append(int(c is not None and c != "original"))
    return {"success": M.bootstrap_ci(succ), "harmful_selection": M.bootstrap_ci(harm), "defer": M.bootstrap_ci(defer),
            "unnecessary_change_neutral": M.bootstrap_ci(unnec), "_succ": succ, "_harm": harm}


def rule_choose(cfg, oracle=False):
    def f(sigs, g):
        biased = bool(g.input_label.iloc[0]) if oracle else None
        vs = [(s, judge(s, cfg, biased)) for s in sigs]
        acc = [s for s, v in vs if v.decision == ACCEPT]
        return max(acc, key=lambda s: s.bias_reduction + s.semantic_similarity + s.fact_retention).candidate if acc else None
    return f


def learned_choose(model: LearnedGate):
    def f(sigs, g):
        X = np.array([s.feature_vector(s.bias_before >= 0.5) for s in sigs])
        P = model.model.predict_proba(X[:, model.cols])
        j = list(model.model.classes_).index(ACCEPT)
        best = int(np.argmax(P[:, j]))
        return sigs[best].candidate if P[best, j] >= model.abstain_below else None
    return f


def load_judge() -> dict | None:
    """Frozen judge (prompt + threshold chosen on val) and its scores, from the latest judge_validation run."""
    from fairlens.experiments.tracking import latest_run, load_manifest

    r = latest_run("judge_validation")
    if not r:
        return None
    d = Path(r["manifest_path"]).parent
    v = json.loads((d / "judge_validation.json").read_text(encoding="utf-8"))
    sc = pd.read_parquet(d / "judge_scores.parquet")
    m = load_manifest(r)
    return {"run_id": r["run_id"], "prompt": v["chosen_prompt"], "threshold": v["threshold"],
            "val_agreement": m.get("metrics", {}).get("val_all_seeds"),
            "scores": {(o, t): p for o, t, p in zip(sc.input_text, sc.candidate_text, sc.p_yes)}}


def tune_rules(val: pd.DataFrame, base: GateConfig) -> tuple[GateConfig, list[dict]]:
    grid = {"min_semantic_similarity": [0.6, 0.7, 0.8, 0.9], "entailment_mode": ["both", "forward"],
            "min_entailment_prob": [0.3, 0.5, 0.7], "min_bias_reduction": [0.05, 0.15, 0.25],
            "max_detector_std": [0.35, 1.0]}
    trials = []
    for combo in itertools.product(*grid.values()):
        cfg = replace(base, **dict(zip(grid, combo)))
        pred = [judge(s, cfg).decision for s in val.sig]
        m = candidate_metrics(val.gold.tolist(), pred)
        trials.append({**dict(zip(grid, combo)), "macro_f1_3class": m["macro_f1_3class"], "unsafe_accept_rate": m["unsafe_accept_rate"]})
    best = max(trials, key=lambda t: (t["macro_f1_3class"], -t["unsafe_accept_rate"]))
    return replace(base, **{k: best[k] for k in grid}), trials


def main():
    gate = pd.read_parquet(PROC / "gate_controlled.parquet")
    detectors = build_detectors()
    cache = ScoreCache()
    base = GateConfig.from_config()
    tracker = RunTracker("benchmark_gate", config={"gate_default": {k: v for k, v in base.__dict__.items() if k != "disabled"},
                                                   "detectors": list(detectors), "seeds": sorted(gate.seed.unique().tolist())},
                         seed=0, dataset_files=[PROC / "gate_controlled.parquet"],
                         models={k: detector_identity(d) for k, d in detectors.items()})
    log(f"gate run {tracker.run_id}: {len(gate)} candidates, detectors {list(detectors)}")
    df = build_signals(gate, detectors, cache)
    val, test = df[df.split == "val"], df[df.split == "test"]
    summary: dict = {"run_id": tracker.run_id, "n_candidates": len(df), "n_inputs": int(gate.input_id.nunique()),
                     "seeds": sorted(gate.seed.unique().tolist()), "gold_distribution_test": dict(Counter(test.gold))}

    tuned, trials = tune_rules(val, base)
    summary["tuned_rules"] = {k: getattr(tuned, k) for k in ["min_semantic_similarity", "entailment_mode", "min_entailment_prob",
                                                             "min_bias_reduction", "max_detector_std"]}
    summary["tuning_trials_top10"] = sorted(trials, key=lambda t: -t["macro_f1_3class"])[:10]
    log(f"val-tuned rules: {summary['tuned_rules']}")

    variants = {"rules_default": base, "rules_val_tuned": tuned}
    cand = {}
    for name, cfg in variants.items():
        cand[name] = candidate_metrics(test.gold.tolist(), [judge(s, cfg).decision for s in test.sig], test.perturbation.tolist())
        cand[name + "_oracle_input"] = candidate_metrics(
            test.gold.tolist(), [judge(s, cfg, bool(l)).decision for s, l in zip(test.sig, test.input_label)])
    Xv = np.array([s.feature_vector(s.bias_before >= 0.5) for s in val.sig])
    Xt = np.array([s.feature_vector(s.bias_before >= 0.5) for s in test.sig])
    learned = {}
    for kind in ["logreg", "tree"]:
        lg = LearnedGate(kind).fit(Xv, val.gold.tolist())
        learned[kind] = lg
        cand[f"learned_{kind}"] = candidate_metrics(test.gold.tolist(), lg.predict(Xt), test.perturbation.tolist())
        cand[f"learned_{kind}"]["model"] = lg.describe()
    summary["candidate_level"] = cand

    policies = {"fairlens_gate_default": rule_choose(base), "fairlens_gate_tuned": rule_choose(tuned),
                "learned_logreg": learned_choose(learned["logreg"]), "learned_tree": learned_choose(learned["tree"]),
                "lowest_bias": lambda s, g: B.lowest_bias(s), "highest_similarity": lambda s, g: B.highest_similarity(s)}
    policies["no_change"] = lambda s, g: B.no_change(s)
    judge = load_judge()
    if judge:
        jscores, jthr = judge["scores"], judge["threshold"]
        summary["judge"] = {k: judge[k] for k in ["run_id", "prompt", "threshold", "val_agreement"]}

        def judge_pol(s, g):
            per = {c: jscores.get((g.input_text.iloc[0], t)) for c, t in zip(g.candidate, g.candidate_text)}
            return B.llm_judge_policy({c: v for c, v in per.items() if v is not None}, jthr, base.biased_input_threshold)(s)
        policies["llm_judge"] = judge_pol
    else:
        summary["judge"] = {"status": "no completed judge_validation run; LLM-judge baseline not evaluated"}
    for k in range(5):
        pol = B.random_policy(100 + k)
        policies[f"random_s{k}"] = lambda s, g, pol=pol: pol(s)
    sel = {n: selection_metrics(test, f) for n, f in policies.items()}
    rnd = [k for k in sel if k.startswith("random_s")]
    sel["random_mean"] = {m: {"mean": float(np.mean([sel[k][m]["mean"] for k in rnd]))} for m in
                          ["success", "harmful_selection", "defer", "unnecessary_change_neutral"]}
    ref = sel["fairlens_gate_tuned"]
    for n in [k for k in sel if k != "fairlens_gate_tuned" and not k.startswith("random")]:
        sel[n]["diff_success_vs_tuned_gate"] = M.paired_bootstrap_diff(sel[n]["_succ"], ref["_succ"])
        sel[n]["diff_harm_vs_tuned_gate"] = M.paired_bootstrap_diff(sel[n]["_harm"], ref["_harm"])
    summary["selection"] = {k: {m: v for m, v in d.items() if not m.startswith("_")} for k, d in sel.items() if k not in rnd}

    abl = {}
    full = candidate_metrics(test.gold.tolist(), [judge(s, tuned).decision for s in test.sig])
    full_unsafe = [int(judge(s, tuned).decision == ACCEPT and g != ACCEPT) for s, g in zip(test.sig, test.gold)]
    full_correct = [int(judge(s, tuned).decision == g) for s, g in zip(test.sig, test.gold)]
    for grp in SIGNAL_GROUPS:
        cfg = replace(tuned, disabled=frozenset({grp}))
        pred = [judge(s, cfg).decision for s in test.sig]
        m = candidate_metrics(test.gold.tolist(), pred, test.perturbation.tolist())
        unsafe = [int(p == ACCEPT and g != ACCEPT) for p, g in zip(pred, test.gold)]
        correct = [int(p == g) for p, g in zip(pred, test.gold)]
        per_seed = {}
        for sd, gs in test.assign(pred=pred).groupby("seed"):
            per_seed[int(sd)] = candidate_metrics(gs.gold.tolist(), gs.pred.tolist())["unsafe_accept_rate"]
        abl[f"no_{grp}"] = {"metrics": m, "delta_unsafe_accept": M.paired_bootstrap_diff(unsafe, full_unsafe),
                            "delta_accuracy": M.paired_bootstrap_diff(correct, full_correct),
                            "unsafe_accept_by_seed": per_seed}
        lg = LearnedGate("logreg", drop_groups=(grp,)).fit(Xv, val.gold.tolist())
        abl[f"no_{grp}"]["learned_logreg_macro_f1"] = candidate_metrics(test.gold.tolist(), lg.predict(Xt))["macro_f1_3class"]
    summary["ablations"] = {"full": full, **abl}

    sig_rows = []
    for r in df.itertuples():
        s = r.sig
        sig_rows.append({"input_id": r.input_id, "seed": r.seed, "split": r.split, "input_label": r.input_label,
                         "candidate": r.candidate, "perturbation": r.perturbation, "gold": r.gold, "input_text": r.input_text,
                         "candidate_text": r.candidate_text, **{k: v for k, v in s.to_dict().items()
                                                               if k not in {"candidate", "text", "fact_checks", "detector_scores_before", "detector_scores_after"}},
                         "fact_checks": json.dumps(s.fact_checks),
                         "pred_default": judge(s, base).decision, "pred_tuned": judge(s, tuned).decision,
                         "reasons_tuned": ",".join(judge(s, tuned).reasons)})
    tracker.save_artifact("gate_signals.parquet", pd.DataFrame(sig_rows))
    tracker.save_artifact("gate_summary.json", summary)
    tracker.finish(metrics={"tuned_macro_f1": cand["rules_val_tuned"]["macro_f1_3class"],
                            "tuned_unsafe_accept": cand["rules_val_tuned"]["unsafe_accept_rate"],
                            "gate_success": summary["selection"]["fairlens_gate_tuned"]["success"]["mean"],
                            "lowest_bias_success": summary["selection"]["lowest_bias"]["success"]["mean"]})
    for k, v in cand.items():
        log(f"{k:28s} macroF1 {v['macro_f1_3class']:.3f} unsafe-accept {v['unsafe_accept_rate']:.3f} review {v['human_review_rate']:.3f}")
    for k, v in summary["selection"].items():
        log(f"select {k:24s} success {v['success']['mean']:.3f} harmful {v['harmful_selection']['mean']:.3f} defer {v['defer']['mean']:.3f}")
    for k, v in abl.items():
        log(f"ablate {k:16s} unsafe {v['metrics']['unsafe_accept_rate']:.3f} (delta {v['delta_unsafe_accept']['diff']:+.3f} "
            f"[{v['delta_unsafe_accept']['lo']:+.3f},{v['delta_unsafe_accept']['hi']:+.3f}])")
    log(f"done: {tracker.run_id}")


if __name__ == "__main__":
    main()
