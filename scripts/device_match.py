"""V4 1.2 acceptance: one detector and one rewriter give matching outputs on every available device.

Runs the Hub RoBERTa detector on 50 validation sentences and FLAN-T5-small (greedy) on 10, once per device
(CPU and, where present, CUDA/MPS), and compares against CPU.
Tolerance (stated before the run): detector |score difference| <= 1e-3; rewriter outputs identical text.

  python scripts/device_match.py [--out outputs/audit/device_match_<platform>.json]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DETECTOR = "himel7/bias-detector"
REWRITER = "google/flan-t5-small"
SCORE_TOL = 1e-3


def run_on(dev: str, texts: list[str]) -> dict:
    os.environ["FAIRLENS_DEVICE"] = dev
    from fairlens.detection.hf_classifier import HFClassifierDetector
    from fairlens.mitigation.llm_adapters import Seq2SeqRewriter
    from fairlens.models import ModelManager, set_manager

    set_manager(ModelManager(max_resident_mb=10_000))
    d = HFClassifierDetector("hub", "Hub", DETECTOR, biased_label_ids=[1])
    scores = [r.score for r in d.predict_batch(texts, batch_size=16)]
    rw = Seq2SeqRewriter("flan", "FLAN-T5-small", REWRITER)
    outs = [rw.rewrite(t).text for t in texts[:10]]
    return {"scores": scores, "rewrites": outs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    a = ap.parse_args()
    import pandas as pd

    from fairlens.hardware import available_devices

    df = pd.read_parquet(ROOT / "datasets/processed/detection.parquet")
    texts = df[df.split == "val"].sample(50, random_state=0).text.tolist()
    devs = available_devices()
    res = {d: run_on(d, texts) for d in devs}
    cpu = res["cpu"]
    cmp = {}
    for d in devs:
        if d == "cpu":
            continue
        diffs = [abs(x - y) for x, y in zip(res[d]["scores"], cpu["scores"])]
        same = [x == y for x, y in zip(res[d]["rewrites"], cpu["rewrites"])]
        cmp[d] = {"max_abs_score_diff": max(diffs), "score_within_tol": max(diffs) <= SCORE_TOL,
                  "rewrite_identical_rate": sum(same) / len(same),
                  "rewrite_mismatches": [{"cpu": c, d: o} for c, o, s in zip(cpu["rewrites"], res[d]["rewrites"], same) if not s]}
    out = {"platform": platform.platform(), "devices": devs, "detector": DETECTOR, "rewriter": REWRITER,
           "tolerance": {"score_abs": SCORE_TOL, "rewrite": "identical text"}, "n_detector": len(texts), "n_rewriter": 10,
           "comparison_vs_cpu": cmp, "passed": all(c["score_within_tol"] and c["rewrite_identical_rate"] == 1.0 for c in cmp.values()) if cmp else None,
           "note": "Only CPU available on this machine; nothing to compare." if not cmp else ""}
    p = Path(a.out) if a.out else ROOT / "outputs" / "audit" / f"device_match_{sys.platform}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "comparison_vs_cpu"}, indent=1))
    print(json.dumps(cmp, indent=1)[:2000])


if __name__ == "__main__":
    main()
