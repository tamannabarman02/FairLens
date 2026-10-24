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
    diag = {"batch1": [r.score for r in d.predict_batch(texts, batch_size=1)]}
    # diagnostic: same model with eager attention (rules out fused-SDPA padding issues on MPS)
    import torch
    from transformers import AutoModelForSequenceClassification

    from fairlens.config import pinned_revision
    from fairlens.hardware import place, to_model

    tok, _ = d._load()
    eager = place(AutoModelForSequenceClassification.from_pretrained(DETECTOR, revision=pinned_revision(DETECTOR),
                                                                     attn_implementation="eager").eval())
    ev = []
    for i in range(0, len(texts), 16):
        enc = to_model(tok(texts[i:i + 16], padding=True, truncation=True, max_length=128, return_tensors="pt"), eager)
        with torch.inference_mode():
            ev += torch.softmax(eager(**enc).logits.float(), -1)[:, 1].cpu().tolist()
    diag["eager_batch16"] = ev
    # layer-by-layer hidden states for the first sentence (compared across devices by the caller)
    enc = to_model(tok(texts[:1], return_tensors="pt"), eager)
    with torch.inference_mode():
        hs = eager(**enc, output_hidden_states=True).hidden_states
    diag["_hidden_first"] = [h[0].float().cpu().numpy().tolist() for h in hs]
    rw = Seq2SeqRewriter("flan", "FLAN-T5-small", REWRITER)
    outs = [rw.rewrite(t).text for t in texts[:10]]
    return {"scores": scores, "rewrites": outs, "diag": diag}


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
        import numpy as np

        diag = {k: max(abs(x - y) for x, y in zip(res[d]["diag"][k], cpu["scores"])) for k in res[d]["diag"] if not k.startswith("_")}
        diag["n_scores_over_tol"] = sum(x > SCORE_TOL for x in diffs)
        diag["score_pairs_worst5"] = sorted(([round(x, 4), round(y, 4), len(t.split())] for x, y, t in
                                             zip(res[d]["scores"], cpu["scores"], texts)), key=lambda r: -abs(r[0] - r[1]))[:5]
        diag["hidden_max_abs_diff_by_layer"] = [float(np.max(np.abs(np.array(a) - np.array(b)))) for a, b in
                                                zip(res[d]["diag"]["_hidden_first"], cpu["diag"]["_hidden_first"])]
        diag["cpu_batch1_vs_cpu_batch16"] = max(abs(x - y) for x, y in zip(cpu["diag"]["batch1"], cpu["scores"]))
        cmp[d] = {"max_abs_score_diff": max(diffs), "score_within_tol": max(diffs) <= SCORE_TOL, "diagnostics_vs_cpu": diag,
                  "rewrite_identical_rate": sum(same) / len(same),
                  "rewrite_mismatches": [{"cpu": c, d: o} for c, o, s in zip(cpu["rewrites"], res[d]["rewrites"], same) if not s]}
    out = {"platform": platform.platform(), "devices": devs, "detector": DETECTOR, "rewriter": REWRITER,
           "tolerance": {"score_abs": SCORE_TOL, "rewrite": "identical text"}, "n_detector": len(texts), "n_rewriter": 10,
           "comparison_vs_cpu": cmp, "passed": all(c["score_within_tol"] and c["rewrite_identical_rate"] == 1.0 for c in cmp.values()) if cmp else None,
           "note": "Only CPU available on this machine; nothing to compare." if not cmp else ""}
    p = Path(a.out) if a.out else ROOT / "outputs" / "audit" / f"device_match_{sys.platform}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    for r in res.values():
        r["diag"].pop("_hidden_first", None)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "comparison_vs_cpu"}, indent=1))
    print(json.dumps(cmp, indent=1)[:2000])


if __name__ == "__main__":
    main()
