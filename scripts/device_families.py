"""Which model families compute the same on MPS/CUDA as on CPU? (follow-up to scripts/device_match.py)

For each model used by FairLens: forward 8 validation sentences on CPU and on every other available device and
report the max absolute difference of the final hidden state / logits. Tolerance stated before the run: 1e-3.
  python scripts/device_families.py --out device_families.json
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
TOL = 1e-3
MODELS = {  # family: (model id, loader)
    "distilbert": ("distilbert-base-uncased", "encoder"),
    "roberta": ("himel7/bias-detector", "classifier"),
    "deberta-v2": ("microsoft/deberta-v3-small", "encoder"),
    "bart": ("facebook/bart-large-mnli", "classifier"),
    "bert(minilm)": ("sentence-transformers/all-MiniLM-L6-v2", "encoder"),
    "deberta-v2(nli)": ("cross-encoder/nli-deberta-v3-small", "classifier"),
    "t5": ("google/flan-t5-base", "seq2seq"),
    "qwen2": ("Qwen/Qwen2.5-0.5B-Instruct", "causal"),
}


def outputs(model_id: str, kind: str, dev: str, texts: list[str]):
    import torch
    from transformers import (AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
                              AutoModelForSequenceClassification, AutoTokenizer)

    from fairlens.config import pinned_revision

    rev = pinned_revision(model_id)
    tok = AutoTokenizer.from_pretrained(model_id, revision=rev)
    cls = {"encoder": AutoModel, "classifier": AutoModelForSequenceClassification,
           "seq2seq": AutoModelForSeq2SeqLM, "causal": AutoModelForCausalLM}[kind]
    m = cls.from_pretrained(model_id, revision=rev, dtype=torch.float32).eval().to(dev)
    enc = tok(texts, padding=True, truncation=True, max_length=64, return_tensors="pt").to(dev)
    with torch.inference_mode():
        if kind == "seq2seq":
            out = m(**enc, decoder_input_ids=torch.zeros((len(texts), 1), dtype=torch.long, device=dev)).logits
        elif kind in {"classifier", "causal"}:
            out = m(**enc).logits
        else:
            out = m(**enc).last_hidden_state
    return out.float().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="device_families.json")
    a = ap.parse_args()
    import pandas as pd

    from fairlens.hardware import available_devices

    df = pd.read_parquet(ROOT / "datasets/processed/detection.parquet")
    texts = df[df.split == "val"].sample(8, random_state=1).text.tolist()
    devs = [d for d in available_devices() if d != "cpu"]
    res = {}
    for fam, (mid, kind) in MODELS.items():
        try:
            ref = outputs(mid, kind, "cpu", texts)
            res[fam] = {"model": mid}
            for d in devs:
                diff = float((outputs(mid, kind, d, texts) - ref).abs().max())
                res[fam][d] = {"max_abs_diff": diff, "match": diff <= TOL}
        except Exception as exc:  # noqa: BLE001
            res[fam] = {"model": mid, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
        print(fam, res[fam], flush=True)
    import torch

    out = {"platform": platform.platform(), "torch": torch.__version__, "devices": devs, "tolerance": TOL, "families": res}
    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
