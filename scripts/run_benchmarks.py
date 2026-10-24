"""Benchmarks. Detection: every detector on the same test split, same metrics, plus probes.

  python scripts/run_benchmarks.py --task detection [--detectors bert,deberta,...] [--zs-val-n 300]
  python scripts/run_benchmarks.py --task mitigation [--limit N] [--lite]

Detection outputs (run dir under outputs/runs/<run_id>/): detection_scores.parquet, detection_summary.json
Mitigation outputs: mitigation_candidates.parquet, mitigation_summary.json, prompts_log.jsonl
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
from fairlens.config import load_config  # noqa: E402
from fairlens.detection.hf_classifier import ZeroShotNLIDetector  # noqa: E402
from fairlens.detection.registry import build_detector  # noqa: E402
from fairlens.evaluation import metrics as M  # noqa: E402
from fairlens.experiments.cache import ScoreCache, detector_identity, score_texts  # noqa: E402
from fairlens.experiments.tracking import RunTracker  # noqa: E402
from fairlens.models import manager, rss_mb  # noqa: E402

PROC = ROOT / "datasets" / "processed"
ZS_TEMPLATES = {
    "T1_single_stereotype": {"biased": "This text contains a stereotype or biased generalization about a social group."},
    "T2_single_biased": {"biased": "This text is biased."},
    "T3_pair_biased_neutral": {"biased": "This text is biased.", "neutral": "This text is neutral."},
    "T4_pair_stereotype_fact": {"biased": "This text stereotypes a group of people.", "neutral": "This text states a fact."},
    "T5_pair_original": dict(load_config()["detectors"]["bart_mnli"]["hypotheses"]),
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def choose_zero_shot_template(det: ZeroShotNLIDetector, val: pd.DataFrame, n: int, cache: ScoreCache) -> tuple[str, dict]:
    sample = pd.concat([d.sample(min(len(d), n // 2), random_state=0) for _, d in val.groupby("label")])
    results = {}
    for name, hyp in ZS_TEMPLATES.items():
        det.hyp_biased, det.hyp_neutral = hyp["biased"], hyp.get("neutral")
        sc, _ = score_texts(det, sample.text.tolist(), cache)
        s = sample.text.map(sc).to_numpy()
        m = M.classification(sample.label.to_numpy(), s)
        results[name] = {"macro_f1": m["macro_f1"], "f1": m["f1"], "roc_auc": m.get("roc_auc"), "hypotheses": hyp}
        log(f"  zero-shot template {name}: macro-F1 {m['macro_f1']:.3f} AUC {m.get('roc_auc', float('nan')):.3f}")
    best = max(results, key=lambda k: results[k]["macro_f1"])
    det.hyp_biased, det.hyp_neutral = ZS_TEMPLATES[best]["biased"], ZS_TEMPLATES[best].get("neutral")
    return best, results


def probes(score_of) -> dict:
    out = {}
    wb = pd.read_parquet(PROC / "winobias_probe.parquet")
    wb["score"] = wb.text.map(score_of)
    pro, anti = wb[wb.condition == "pro"], wb[wb.condition == "anti"]
    gap = pro.sort_values(["type", "pair_index"]).score.to_numpy() - anti.sort_values(["type", "pair_index"]).score.to_numpy()
    out["winobias"] = {"n": len(wb), "fpr_pro": float((pro.score >= 0.5).mean()), "fpr_anti": float((anti.score >= 0.5).mean()),
                       "mean_score_pro": float(pro.score.mean()), "mean_score_anti": float(anti.score.mean()),
                       "mean_gap_pro_minus_anti": M.bootstrap_ci(gap)}
    cp = pd.read_parquet(PROC / "crows_pairs.parquet")
    cp = cp[cp.split == "test"]
    pref = (cp.sent_more.map(score_of) > cp.sent_less.map(score_of)).astype(float)
    out["crows_pairs"] = {"n_pairs": len(cp), "pref_more_stereo": M.bootstrap_ci(pref[cp.stereo_antistereo == "stereo"].to_numpy()),
                          "pref_more_antistereo": M.bootstrap_ci(pref[cp.stereo_antistereo == "antistereo"].to_numpy())}
    ss = pd.read_parquet(PROC / "stereoset_pairs.parquet")
    ss = ss[ss.split == "test"]
    st, an, un = ss.stereotype.map(score_of), ss.anti_stereotype.map(score_of), ss.unrelated.map(score_of)
    out["stereoset"] = {"n_items": len(ss), "pref_stereo_over_anti": M.bootstrap_ci((st > an).astype(float).to_numpy()),
                        "pref_stereo_over_unrelated": M.bootstrap_ci((st > un).astype(float).to_numpy()),
                        "flag_rate_unrelated": float((un >= 0.5).mean())}
    return out


def probe_texts() -> list[str]:
    wb = pd.read_parquet(PROC / "winobias_probe.parquet")
    cp = pd.read_parquet(PROC / "crows_pairs.parquet")
    cp = cp[cp.split == "test"]
    ss = pd.read_parquet(PROC / "stereoset_pairs.parquet")
    ss = ss[ss.split == "test"]
    return list(dict.fromkeys(wb.text.tolist() + cp.sent_more.tolist() + cp.sent_less.tolist()
                              + ss.stereotype.tolist() + ss.anti_stereotype.tolist() + ss.unrelated.tolist()))


def run_detection(keys: list[str], zs_val_n: int, skip_probes: bool) -> None:
    det_df = pd.read_parquet(PROC / "detection.parquet")
    test, val = det_df[det_df.split == "test"].reset_index(drop=True), det_df[det_df.split == "val"]
    cache = ScoreCache()
    tracker = RunTracker("benchmark_detection", config={"detectors": keys, "zs_val_n": zs_val_n, "threshold": 0.5,
                                                        "timing_subset": 100}, seed=0,
                         dataset_files=[PROC / "detection.parquet", PROC / "winobias_probe.parquet",
                                        PROC / "crows_pairs.parquet", PROC / "stereoset_pairs.parquet"])
    summary: dict = {"run_id": tracker.run_id, "n_test": len(test), "detectors": {}}
    scores = pd.DataFrame({"text": test.text, "label": test.label, "source": test.source, "bias_type": test.bias_type})
    ptexts = [] if skip_probes else probe_texts()
    timing = test.sample(100, random_state=1).text.tolist()
    for key in keys:
        det = build_detector(key)
        if det is None:
            log(f"{key}: unavailable (not trained / not configured) - skipped")
            summary["detectors"][key] = {"status": "unavailable"}
            continue
        log(f"== {key} ({det.display})")
        manager().clear()
        r0 = rss_mb()
        t0 = time.perf_counter()
        det.predict_batch(timing[:2])
        load_s = time.perf_counter() - t0
        rec: dict = {"status": "ok", "display": det.display, "load_s": round(load_s, 2), "info": det.info()}
        if key == "bart_mnli":
            best, tpl = choose_zero_shot_template(det, val, zs_val_n, cache)
            rec["template_selection"] = {"chosen": best, "candidates": tpl, "val_n": zs_val_n}
            rec["info"] = det.info()
        t1 = time.perf_counter()
        det.predict_batch(timing, batch_size=16)
        rec["ms_per_sample"] = (time.perf_counter() - t1) * 1000 / len(timing)
        rec["rss_delta_mb"] = round(rss_mb() - r0)
        rec["model_mb"] = round(manager().resident_mb())
        log(f"   {rec['ms_per_sample']:.1f} ms/sample, model {rec['model_mb']} MB, RSS +{rec['rss_delta_mb']} MB")
        sc, _ = score_texts(det, test.text.tolist() + ptexts, cache, progress=lambda i, n: log(f"   scored {i}/{n}") if i % 512 < 128 else None)
        s = test.text.map(sc).to_numpy()
        scores[key] = s
        y = test.label.to_numpy()
        rec["metrics"] = M.classification(y, s)
        rec["ci"] = {"f1": M.bootstrap_metric(y, s, M.f1_from_scores()),
                     "macro_f1": M.bootstrap_metric(y, s, lambda a, b: M.classification(a, b)["macro_f1"], n=300)}
        rec["curves"] = M.curves(y, s)
        rec["by_source"] = {src: M.classification(g.label.to_numpy(), g[key].to_numpy()) for src, g in scores.groupby("source")}
        for v in rec["by_source"].values():
            v.pop("reliability", None)
        if not skip_probes:
            rec["probes"] = probes(lambda t: sc[t])
        rec["cache_identity"] = detector_identity(det)
        summary["detectors"][key] = rec
        m = rec["metrics"]
        log(f"   F1 {m['f1']:.3f} macro-F1 {m['macro_f1']:.3f} AUC {m.get('roc_auc', 0):.3f} ECE {m['ece']:.3f}")
        tracker.save_artifact("detection_summary.json", summary)
    ok = [k for k in keys if k in scores]
    if len(ok) > 1:
        ens = scores[ok].mean(axis=1).to_numpy()
        summary["ensemble"] = {"members": ok, "metrics": M.classification(test.label.to_numpy(), ens)}
    tracker.add_models({k: summary["detectors"][k].get("cache_identity") for k in ok})
    tracker.save_artifact("detection_scores.parquet", scores)
    tracker.save_artifact("detection_summary.json", summary)
    tracker.finish(metrics={k: {"f1": summary["detectors"][k]["metrics"]["f1"],
                                "macro_f1": summary["detectors"][k]["metrics"]["macro_f1"]} for k in ok})
    log(f"done: {tracker.run_id}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="detection", choices=["detection", "mitigation"])
    ap.add_argument("--detectors", default="rule_based,bert,deberta,bart_mnli,hub_pretrained")
    ap.add_argument("--zs-val-n", type=int, default=300)
    ap.add_argument("--skip-probes", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--lite", action="store_true")
    a = ap.parse_args()
    if a.task == "detection":
        run_detection(a.detectors.split(","), a.zs_val_n, a.skip_probes)
    else:
        from fairlens.experiments.mitigation_runner import run_mitigation

        run_mitigation(limit=a.limit, lite=a.lite)


if __name__ == "__main__":
    main()
