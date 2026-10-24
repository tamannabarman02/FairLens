"""V4 1.4: runtime and memory profiling before full runs (also re-runs one real inference per model, audit item 3).

Each item runs in its own subprocess so peak memory is clean and a failure or timeout cannot affect the others.
  training: 20 optimisation steps per fine-tuned model (+ DeBERTa fix-ladder variants)
  inference: 100 samples per detector, rewriter, preservation model and captioner
Inputs: validation-split sentences for detectors/rewriters (test stays sealed); the verification COCO sample image,
repeated, for captioners (timing only). Results: a `profile` run in outputs/runs/<run_id>/profile.json.

  python scripts/profile_models.py [--only key1,key2] [--n 100] [--steps 20] [--timeout 900]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TRAIN_ITEMS = {
    "train:bert": {"model": "bert"},
    "train:deberta": {"model": "deberta"},
    "train:deberta+freeze": {"model": "deberta", "freeze_embeddings": True},
    "train:deberta+freeze+len64": {"model": "deberta", "freeze_embeddings": True, "max_length": 64},
}
INFER_ITEMS = ["det:rule_based", "det:bert", "det:deberta", "det:bart_mnli", "det:bart_mnli_lite", "det:hub_pretrained",
               "rw:rule_based", "rw:flan_t5", "rw:flan_t5_lite", "rw:qwen", "pres:minilm", "pres:nli",
               "cap:blip", "cap:florence", "cap:vit_gpt2", "clip"]


def peak_mb() -> float:
    import psutil

    mi = psutil.Process().memory_info()
    if hasattr(mi, "peak_wset"):
        return mi.peak_wset / 2**20
    import resource

    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 1024


def val_texts(n: int) -> list[str]:
    import pandas as pd

    df = pd.read_parquet(ROOT / "datasets/processed/detection.parquet")
    return df[df.split == "val"].sample(n, random_state=0).text.tolist()


def child_train(spec: dict, steps: int) -> dict:
    import math
    import random

    import numpy as np
    import pandas as pd
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from fairlens.config import load_config, pinned_revision
    from fairlens.hardware import device, place

    sys.path.insert(0, str(ROOT / "scripts"))
    from train_detectors import HPARAMS, batches

    name = spec["model"]
    hp = {**HPARAMS[name], **{k: v for k, v in spec.items() if k != "model"}}
    base = load_config()["detectors"][name]["base_model"]
    rev = pinned_revision(base)
    torch.manual_seed(42)
    df = pd.read_parquet(ROOT / "datasets/processed/detection.parquet")
    tr = df[df.split == "train"].sample(steps * hp["batch_size"], random_state=42)
    tok = AutoTokenizer.from_pretrained(base, revision=rev)
    model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=2, revision=rev)
    if hp.get("freeze_embeddings"):
        for p in model.base_model.embeddings.parameters():
            p.requires_grad = False
    model = place(model).train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=hp["lr"], weight_decay=0.01)
    texts, labels = tr.text.tolist(), tr.label.to_numpy()
    lens = [len(tok.tokenize(t)) for t in texts]
    times = []
    for b in batches(lens, hp["batch_size"], random.Random(0), shuffle=True)[:steps]:
        t0 = time.perf_counter()
        enc = tok([texts[i] for i in b], padding=True, truncation=True, max_length=hp["max_length"], return_tensors="pt").to(model.device)
        out = model(**enc, labels=torch.tensor(labels[b], device=model.device))
        out.loss.backward()
        opt.step(); opt.zero_grad()
        times.append(time.perf_counter() - t0)
    n_train = int((df.split == "train").sum())
    s_step = float(np.median(times[2:] or times))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"device": device(), "steps": len(times), "median_s_per_step": s_step, "mean_s_per_step": float(np.mean(times)),
            "first_step_s": times[0], "hparams": hp, "trainable_params_m": round(trainable / 1e6, 1),
            "est_epoch_min": s_step * math.ceil(n_train / hp["batch_size"]) / 60, "peak_mb": peak_mb()}


def child_infer(key: str, n: int) -> dict:
    import numpy as np

    from fairlens.config import load_config
    from fairlens.hardware import device

    cfg = load_config()
    texts = val_texts(n)
    sample_out = None
    t_load = time.perf_counter()
    kind, name = key.split(":", 1) if ":" in key else (key, key)
    if kind == "det":
        from fairlens.detection.hf_classifier import ZeroShotNLIDetector
        from fairlens.detection.registry import build_detector

        if name == "bart_mnli_lite":
            d = build_detector("bart_mnli", lite=True)
        else:
            d = build_detector(name)
        if d is None:
            return {"status": "unavailable", "reason": "no trained checkpoint"}
        d.predict_batch(texts[:2])
        load_s = time.perf_counter() - t_load
        t0 = time.perf_counter()
        res = d.predict_batch(texts, batch_size=16)
        total = time.perf_counter() - t0
        sample_out = {"text": texts[0][:80], "score": round(res[0].score, 3)}
        model = getattr(d, "model_id", None) or getattr(d, "model_path", None) or "rules"
    elif kind == "rw":
        from fairlens.mitigation.llm_adapters import build_rewriters

        lite = name.endswith("_lite")
        rw = build_rewriters([name.replace("_lite", "")], lite=lite)[name.replace("_lite", "")]
        texts = [t for t in texts if 5 <= len(t.split()) <= 40][:n]
        r0 = rw.rewrite(texts[0])
        load_s = time.perf_counter() - t_load
        t0 = time.perf_counter()
        outs = [rw.rewrite(t) for t in texts]
        total = time.perf_counter() - t0
        errs = [o.error for o in outs if o.error]
        changed = float(np.mean([bool(o.text) and o.text.strip().rstrip(".").lower() != t.strip().rstrip(".").lower()
                                 for o, t in zip(outs, texts)]))
        sample_out = {"input": texts[0][:80], "output": (r0.text or r0.error or "")[:120], "changed_rate": changed,
                      "errors": len(errs)}
        model = getattr(rw, "model_id", "rules")
    elif kind == "pres":
        from fairlens.evaluation.preservation import Entailment, SemanticSimilarity

        m = SemanticSimilarity() if name == "minilm" else Entailment()
        fn = (lambda a, b: m.similarity(a, b)) if name == "minilm" else (lambda a, b: m.both_ways(a, b))
        fn(texts[:2], texts[:2])
        load_s = time.perf_counter() - t_load
        t0 = time.perf_counter()
        out = fn(texts, list(reversed(texts)))
        total = time.perf_counter() - t0
        sample_out = {"pair_0": out[0] if isinstance(out[0], float) else {k: round(v, 3) for k, v in out[0].items()}}
        model = m.model_id
    else:
        from PIL import Image

        img = Image.open(ROOT / "outputs/model_verification/sample_coco_000000039769.jpg").convert("RGB")
        if kind == "clip":
            from fairlens.multimodal.caption_evaluation import ClipAlignment

            c = ClipAlignment()
            c.score(img, ["two cats"])
            load_s = time.perf_counter() - t_load
            t0 = time.perf_counter()
            for _ in range(n):
                v = c.score(img, ["two cats lying on a pink couch", "a man riding a motorcycle"])
            total = time.perf_counter() - t0
            sample_out = {"matching": round(v[0], 3), "wrong": round(v[1], 3)}
            model = c.model_id
        else:
            from fairlens.multimodal.captioners import build_captioners

            caps = build_captioners(use_fallback_for_florence=(name == "vit_gpt2"))
            c = caps["blip" if name == "blip" else "florence"]
            first = c.caption(img)
            if first.error:
                return {"status": "failed", "error": first.error}
            load_s = time.perf_counter() - t_load
            t0 = time.perf_counter()
            for _ in range(n):
                cap = c.caption(img)
            total = time.perf_counter() - t0
            sample_out = {"caption": cap.text, "first_call_ms": round(first.runtime_ms)}
            model = c.model_id
    return {"status": "ok", "device": device(), "model": model, "n": n, "load_s": round(load_s, 2),
            "ms_per_sample": total * 1000 / n, "peak_mb": peak_mb(), "sample": sample_out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=900, help="seconds per item before it is recorded as a timeout")
    ap.add_argument("--child")
    ap.add_argument("--run-dir")
    a = ap.parse_args()
    if a.child:
        try:
            rec = child_train(TRAIN_ITEMS[a.child], a.steps) if a.child.startswith("train:") else child_infer(a.child, a.n)
        except Exception as exc:  # noqa: BLE001
            rec = {"status": "failed", "error": f"{type(exc).__name__}: {str(exc)[:400]}"}
        print("@@RESULT@@" + json.dumps(rec, default=str), flush=True)
        return

    from fairlens.experiments.tracking import RunTracker

    keys = a.only.split(",") if a.only else list(TRAIN_ITEMS) + INFER_ITEMS
    tracker = RunTracker("profile", config={"items": keys, "n": a.n, "steps": a.steps, "timeout_s": a.timeout,
                                            "inputs": "validation sentences; COCO sample image repeated for captioners"},
                         seed=0, dataset_files=[ROOT / "datasets/processed/detection.parquet"])
    results = {}
    for k in keys:
        print(f"== {k}", flush=True)
        t0 = time.perf_counter()
        try:
            # child output goes to a log file (no pipes, no inherited stdin): avoids stalls seen with captured pipes
            child_log = tracker.dir / f"child_{k.replace(':', '_').replace('+', '_')}.log"
            with open(child_log, "w", encoding="utf-8") as fh:
                proc = subprocess.Popen([sys.executable, __file__, "--child", k, "--n", str(a.n), "--steps", str(a.steps)],
                                        stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=ROOT,
                                        env={**os.environ, "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TQDM_DISABLE": "1"})
                try:
                    proc.wait(timeout=a.timeout)
                except subprocess.TimeoutExpired:
                    # On Windows a venv python.exe is a launcher with the real interpreter as its child: kill the tree.
                    import psutil

                    for c in psutil.Process(proc.pid).children(recursive=True):
                        c.kill()
                    proc.kill()
                    raise
            text = child_log.read_text(encoding="utf-8", errors="replace")
            line = next((l for l in text.splitlines() if l.startswith("@@RESULT@@")), None)
            rec = json.loads(line[10:]) if line else {"status": "failed", "error": text[-600:]}
        except subprocess.TimeoutExpired:
            rec = {"status": "timeout", "error": f"exceeded {a.timeout} s"}
        rec.setdefault("status", "ok")
        rec["wall_s"] = round(time.perf_counter() - t0, 1)
        results[k] = rec
        print("   " + json.dumps({x: rec.get(x) for x in ["status", "median_s_per_step", "est_epoch_min", "ms_per_sample",
                                                         "peak_mb", "sample", "error"]}, default=str)[:400], flush=True)
        tracker.save_artifact("profile.json", results)
        tracker.add_models({k: rec.get("model") or (rec.get("hparams") or {}).get("model")})
    tracker.finish(metrics={k: {"status": v["status"], "s_per_step": v.get("median_s_per_step"), "ms_per_sample": v.get("ms_per_sample"),
                                "peak_mb": v.get("peak_mb")} for k, v in results.items()})
    print(f"done: {tracker.run_id}")


if __name__ == "__main__":
    main()
