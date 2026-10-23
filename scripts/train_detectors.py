"""Fine-tune the BERT-family and DeBERTa-family detectors on datasets/processed/detection.parquet.

Plain PyTorch loop (CPU-friendly): dynamic padding, length-bucketed batches, linear warmup/decay,
best checkpoint by validation macro-F1. Saves to checkpoints/<name>/ with fairlens_training.json.

Usage: python scripts/train_detectors.py --model bert|deberta|all [--epochs 2] [--max-train N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fairlens.config import PROJECT_ROOT, load_config  # noqa: E402
from fairlens.experiments.tracking import RunTracker  # noqa: E402

DATA = ROOT / "datasets" / "processed" / "detection.parquet"
HPARAMS = {
    "bert": {"lr": 5e-5, "epochs": 3, "batch_size": 16, "max_length": 128},
    "deberta": {"lr": 5e-5, "epochs": 3, "batch_size": 16, "max_length": 128},
}


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def batches(enc_lens: list[int], bs: int, rng: random.Random, shuffle: bool):
    idx = list(range(len(enc_lens)))
    if shuffle:
        rng.shuffle(idx)
        # bucket by length within chunks of 50 batches to cut padding while staying random
        chunk = bs * 50
        idx = [i for c in range(0, len(idx), chunk) for i in sorted(idx[c:c + chunk], key=lambda j: enc_lens[j])]
        bl = [idx[i:i + bs] for i in range(0, len(idx), bs)]
        rng.shuffle(bl)
        return bl
    idx.sort(key=lambda j: enc_lens[j])
    return [idx[i:i + bs] for i in range(0, len(idx), bs)]


def evaluate(model, tok, texts, labels, max_length, bs=32):
    import torch
    from sklearn.metrics import f1_score

    model.eval()
    probs = np.zeros(len(texts))
    lens = [len(t) for t in texts]
    for b in batches(lens, bs, random.Random(0), shuffle=False):
        enc = tok([texts[i] for i in b], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        with torch.inference_mode():
            p = torch.softmax(model(**enc).logits.float(), -1)[:, 1].numpy()
        probs[b] = p
    preds = (probs >= 0.5).astype(int)
    return {"macro_f1": float(f1_score(labels, preds, average="macro")), "acc": float((preds == labels).mean())}, probs


def train(name: str, epochs: int | None, max_train: int | None, seed: int = 42) -> dict:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    from fairlens.config import pinned_revision

    spec = load_config()["detectors"][name]
    hp = dict(HPARAMS[name])
    if epochs:
        hp["epochs"] = epochs
    base = spec["base_model"]
    out_dir = PROJECT_ROOT / spec["checkpoint"]
    set_seed(seed)
    df = pd.read_parquet(DATA)
    tr = df[df.split == "train"]
    if max_train and len(tr) > max_train:
        tr = tr.sample(max_train, random_state=seed)
    va = df[df.split == "val"]
    rev = pinned_revision(base)
    tok = AutoTokenizer.from_pretrained(base, revision=rev)
    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=2, id2label={0: "neutral", 1: "biased"}, label2id={"neutral": 0, "biased": 1}, revision=rev)

    texts, labels = tr.text.tolist(), tr.label.to_numpy()
    lens = [len(tok.tokenize(t)) for t in texts]
    rng = random.Random(seed)
    steps_per_epoch = math.ceil(len(texts) / hp["batch_size"])
    total = steps_per_epoch * hp["epochs"]
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)

    tracker = RunTracker("train_detector", config={"detector": name, "base_model": base, "revision": rev, **hp,
                                                   "n_train": len(texts), "n_val": len(va), "seed": seed},
                         seed=seed, dataset_files=[DATA])
    best, history, t_start = -1.0, [], time.perf_counter()
    step = 0
    for ep in range(hp["epochs"]):
        model.train()
        ep_loss, t_ep = 0.0, time.perf_counter()
        for bi, b in enumerate(batches(lens, hp["batch_size"], rng, shuffle=True)):
            enc = tok([texts[i] for i in b], padding=True, truncation=True, max_length=hp["max_length"], return_tensors="pt")
            out = model(**enc, labels=torch.tensor(labels[b]))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            ep_loss += out.loss.item(); step += 1
            if bi % 50 == 0:
                el = time.perf_counter() - t_ep
                print(f"[{name}] epoch {ep+1}/{hp['epochs']} step {bi}/{steps_per_epoch} loss {ep_loss/(bi+1):.4f} "
                      f"({el/(bi+1):.2f}s/step, eta epoch {el/(bi+1)*(steps_per_epoch-bi-1)/60:.1f} min)", flush=True)
        vm, _ = evaluate(model, tok, va.text.tolist(), va.label.to_numpy(), hp["max_length"])
        history.append({"epoch": ep + 1, "train_loss": ep_loss / steps_per_epoch, **{f"val_{k}": v for k, v in vm.items()},
                        "epoch_s": round(time.perf_counter() - t_ep, 1)})
        print(f"[{name}] epoch {ep+1} val {vm}", flush=True)
        if vm["macro_f1"] > best:
            best = vm["macro_f1"]
            out_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)
    meta = {"detector": name, "base_model": base, "base_revision": rev, "hparams": hp, "seed": seed,
            "n_train": len(texts), "n_val": len(va), "best_val_macro_f1": best, "history": history,
            "train_seconds": round(time.perf_counter() - t_start, 1),
            "data_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest()[:16],
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "run_id": tracker.run_id}
    (out_dir / "fairlens_training.json").write_text(json.dumps(meta, indent=2))
    tracker.finish(metrics={"best_val_macro_f1": best}, artifacts={"checkpoint": str(out_dir), "history": history})
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--max-train", type=int)
    a = ap.parse_args()
    for name in (["bert", "deberta"] if a.model == "all" else [a.model]):
        meta = train(name, a.epochs, a.max_train)
        print(json.dumps({k: meta[k] for k in ["detector", "best_val_macro_f1", "train_seconds"]}))


if __name__ == "__main__":
    main()
