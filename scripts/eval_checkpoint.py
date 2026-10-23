"""Re-evaluate a fine-tuned detector checkpoint on a split and compare with its training record.

  python scripts/eval_checkpoint.py --model bert [--split val]
Writes outputs/audit/eval_<model>_<split>.json. Validation only by default (test stays sealed).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fairlens.config import load_config  # noqa: E402
from fairlens.detection.registry import build_detector  # noqa: E402
from fairlens.evaluation import metrics as M  # noqa: E402
from fairlens.hardware import device  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert")
    ap.add_argument("--split", default="val", choices=["val", "train"])
    a = ap.parse_args()
    det = build_detector(a.model)
    if det is None:
        print(f"{a.model}: no checkpoint")
        return
    df = pd.read_parquet(ROOT / "datasets/processed/detection.parquet")
    df = df[df.split == a.split]
    t0 = time.perf_counter()
    scores = [r.score for r in det.predict_batch(df.text.tolist(), batch_size=32)]
    m = M.classification(df.label.to_numpy(), scores)
    rec_path = ROOT / load_config()["detectors"][a.model]["checkpoint"] / "fairlens_training.json"
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    out = {"model": a.model, "split": a.split, "n": len(df), "device": device(), "seconds": round(time.perf_counter() - t0, 1),
           "macro_f1": m["macro_f1"], "accuracy": m["accuracy"], "recorded_best_val_macro_f1": rec["best_val_macro_f1"],
           "abs_diff": abs(m["macro_f1"] - rec["best_val_macro_f1"]), "trained_at": rec["trained_at"]}
    p = ROOT / "outputs" / "audit" / f"eval_{a.model}_{a.split}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
