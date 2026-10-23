"""Phase 1: verify every candidate model on this machine.

For each model: Hub metadata (license, commit, weight size), download, load, one real inference on a
sample, load time, latency and peak process memory. Each model runs in its own subprocess so memory
numbers are clean and a failure cannot take the others down. Results:
  outputs/model_verification/verification.json   (all records)
  configs/model_lock.json                        (pinned commit per verified model)

Usage:  python scripts/verify_models.py [--only key1,key2] [--skip-download-over-mb 4000]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SAMPLE_TEXT = "Looking for a young, aggressive salesman with 5 years of experience."
SAMPLE_NEUTRAL = "The committee approved the budget on Tuesday."
SAMPLE_IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"

# key: (role, model_id, kind)
CANDIDATES = {
    "distilbert-base-uncased": ("detector base (BERT family)", "distilbert-base-uncased", "encoder"),
    "bert-base-uncased": ("detector base (BERT family, larger)", "bert-base-uncased", "encoder"),
    "deberta-v3-small": ("detector base (DeBERTa family)", "microsoft/deberta-v3-small", "encoder"),
    "hub-bias-roberta": ("pretrained Hub bias classifier", "himel7/bias-detector", "classifier"),
    "bart-large-mnli": ("zero-shot detector", "facebook/bart-large-mnli", "zeroshot"),
    "distilbart-mnli": ("zero-shot detector (lite)", "valhalla/distilbart-mnli-12-1", "zeroshot"),
    "flan-t5-base": ("mitigation LLM A", "google/flan-t5-base", "seq2seq"),
    "flan-t5-large": ("mitigation LLM A (larger)", "google/flan-t5-large", "seq2seq"),
    "qwen2.5-0.5b": ("mitigation LLM B", "Qwen/Qwen2.5-0.5B-Instruct", "chat"),
    "qwen2.5-1.5b": ("mitigation LLM B (larger)", "Qwen/Qwen2.5-1.5B-Instruct", "chat"),
    "minilm": ("semantic similarity", "sentence-transformers/all-MiniLM-L6-v2", "embedding"),
    "nli-deberta-small": ("bidirectional entailment", "cross-encoder/nli-deberta-v3-small", "nli"),
    "blip": ("captioner A", "Salesforce/blip-image-captioning-base", "blip"),
    "florence2": ("captioner B (native port)", "florence-community/Florence-2-base", "florence"),
    "vit-gpt2": ("captioner B fallback", "nlpconnect/vit-gpt2-image-captioning", "vitgpt2"),
    "clip": ("image-text alignment", "openai/clip-vit-base-patch32", "clip"),
}


def hub_meta(model_id: str) -> dict:
    from huggingface_hub import HfApi

    info = HfApi().model_info(model_id, files_metadata=True)
    lic = info.card_data.get("license") if info.card_data else None
    weights = {s.rfilename: s.size or 0 for s in info.siblings
               if s.rfilename.endswith((".safetensors", ".bin")) and "/" not in s.rfilename}
    st = [k for k in weights if k.endswith(".safetensors")]
    size = sum(weights[k] for k in (st or weights))
    return {"license": lic, "sha": info.sha, "weights_mb": round(size / 2**20), "gated": bool(info.gated),
            "custom_code": "custom_code" in (info.tags or []), "pipeline_tag": info.pipeline_tag}


def sample_image():
    import io
    import urllib.request

    from PIL import Image

    cache = ROOT / "outputs" / "model_verification" / "sample_coco_000000039769.jpg"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(urllib.request.urlopen(SAMPLE_IMAGE_URL, timeout=30).read())
    return Image.open(io.BytesIO(cache.read_bytes()))


def run_one(key: str) -> dict:
    """Executed in a child process."""
    import psutil

    from fairlens.models import ModelManager, set_manager

    set_manager(ModelManager(max_resident_mb=10_000))
    role, mid, kind = CANDIDATES[key]
    proc = psutil.Process()
    rss0 = proc.memory_info().rss / 2**20
    rec: dict = {"key": key, "role": role, "model_id": mid, "kind": kind}
    t0 = time.perf_counter()

    if kind in {"encoder"}:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(mid, revision=rec.get("sha"))
        model = AutoModelForSequenceClassification.from_pretrained(mid, num_labels=2).eval()
        rec["load_s"] = time.perf_counter() - t0
        import torch

        t1 = time.perf_counter()
        with torch.inference_mode():
            logits = model(**tok([SAMPLE_TEXT], return_tensors="pt")).logits
        rec["latency_ms"] = (time.perf_counter() - t1) * 1000
        rec["sample_output"] = f"untrained 2-class head forward pass OK, logits shape {tuple(logits.shape)}; tokenizer {type(tok).__name__}, tokens={tok.tokenize(SAMPLE_TEXT)[:8]}"
    elif kind == "classifier":
        from fairlens.detection.hf_classifier import HFClassifierDetector

        d = HFClassifierDetector("hub", "Hub", mid)
        tok, model = d._load()
        rec["load_s"] = time.perf_counter() - t0
        rec["id2label"] = model.config.id2label
        t1 = time.perf_counter()
        s = d._scores([SAMPLE_TEXT, SAMPLE_NEUTRAL])
        rec["latency_ms"] = (time.perf_counter() - t1) * 1000 / 2
        rec["sample_output"] = {"biased_sample": round(s[0], 3), "neutral_sample": round(s[1], 3), "biased_ids": d.biased_ids(model)}
    elif kind == "zeroshot":
        from fairlens.config import load_config
        from fairlens.detection.hf_classifier import ZeroShotNLIDetector

        d = ZeroShotNLIDetector("z", "Z", mid, load_config()["detectors"]["bart_mnli"]["hypotheses"])
        d._load()
        rec["load_s"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        s = d._scores([SAMPLE_TEXT, SAMPLE_NEUTRAL])
        rec["latency_ms"] = (time.perf_counter() - t1) * 1000 / 2
        rec["sample_output"] = {"biased_sample": round(s[0], 3), "neutral_sample": round(s[1], 3)}
    elif kind in {"seq2seq", "chat"}:
        from fairlens.mitigation.llm_adapters import ChatRewriter, Seq2SeqRewriter

        r = (Seq2SeqRewriter if kind == "seq2seq" else ChatRewriter)("m", "M", mid)
        r._load()
        rec["load_s"] = time.perf_counter() - t0
        out = r.rewrite(SAMPLE_TEXT)
        if out.error:
            raise RuntimeError(out.error)
        rec["latency_ms"] = out.runtime_ms
        rec["sample_output"] = out.text
    elif kind == "embedding":
        from fairlens.evaluation.preservation import SemanticSimilarity

        s = SemanticSimilarity(mid)
        s._load()
        rec["load_s"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        v = s.similarity([SAMPLE_TEXT], ["We are hiring a proactive salesperson with 5 years of experience."])
        rec["latency_ms"] = (time.perf_counter() - t1) * 1000
        rec["sample_output"] = {"cosine": round(v[0], 3)}
    elif kind == "nli":
        from fairlens.evaluation.preservation import Entailment

        e = Entailment(mid)
        e._load()
        rec["load_s"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        v = e.both_ways([SAMPLE_TEXT, SAMPLE_TEXT],
                        ["We are hiring a proactive salesperson with 5 years of experience.",
                         "We are hiring a salesperson with no experience."])
        rec["latency_ms"] = (time.perf_counter() - t1) * 1000 / 4
        rec["sample_output"] = {"paraphrase_both": round(v[0]["both"], 3), "changed_fact_both": round(v[1]["both"], 3),
                                "changed_fact_contradiction": round(v[1]["contradiction"], 3)}
    elif kind in {"blip", "florence", "vitgpt2"}:
        from fairlens.multimodal.captioners import BlipCaptioner, FlorenceCaptioner, VitGpt2Captioner

        cls = {"blip": BlipCaptioner, "florence": FlorenceCaptioner, "vitgpt2": VitGpt2Captioner}[kind]
        c = cls(mid)
        img = sample_image()
        c._load()
        rec["load_s"] = time.perf_counter() - t0
        cap = c.caption(img)
        if cap.error:
            raise RuntimeError(cap.error)
        rec["latency_ms"] = cap.runtime_ms
        rec["sample_output"] = cap.text
    elif kind == "clip":
        from fairlens.multimodal.caption_evaluation import ClipAlignment

        c = ClipAlignment(mid)
        img = sample_image()
        c._load()
        rec["load_s"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        v = c.score(img, ["two cats lying on a pink couch", "a man riding a motorcycle"])
        rec["latency_ms"] = (time.perf_counter() - t1) * 1000
        rec["sample_output"] = {"matching_caption": round(v[0], 3), "wrong_caption": round(v[1], 3)}
    mi = proc.memory_info()
    rec["rss_delta_mb"] = round(mi.rss / 2**20 - rss0)
    rec["peak_mb"] = round(getattr(mi, "peak_wset", mi.rss) / 2**20)
    rec["load_s"] = round(rec["load_s"], 2)
    rec["latency_ms"] = round(rec["latency_ms"], 1)
    rec["status"] = "verified"
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--child")
    args = ap.parse_args()

    if args.child:
        try:
            rec = run_one(args.child)
        except Exception as exc:  # noqa: BLE001
            rec = {"key": args.child, "status": "failed", "error": f"{type(exc).__name__}: {str(exc)[:500]}"}
        print("@@RESULT@@" + json.dumps(rec, default=str))
        return

    out_dir = ROOT / "outputs" / "model_verification"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "verification.json"
    results = json.loads(out_file.read_text()) if out_file.exists() else {}
    keys = args.only.split(",") if args.only else list(CANDIDATES)
    import fairlens.config  # noqa: F401  (cache env)

    for key in keys:
        role, mid, kind = CANDIDATES[key]
        print(f"== {key} ({mid})", flush=True)
        meta = {}
        try:
            meta = hub_meta(mid)
        except Exception as exc:  # noqa: BLE001
            meta = {"meta_error": str(exc)[:200]}
        t0 = time.perf_counter()
        p = subprocess.run([sys.executable, __file__, "--child", key], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", cwd=ROOT)
        line = next((l for l in p.stdout.splitlines() if l.startswith("@@RESULT@@")), None)
        rec = json.loads(line[10:]) if line else {"key": key, "status": "failed",
                                                   "error": (p.stderr or p.stdout)[-800:]}
        rec.update({"role": role, "model_id": mid, **meta, "wall_s": round(time.perf_counter() - t0, 1),
                    "verified_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        results[key] = rec
        print(json.dumps({k: rec.get(k) for k in ["status", "license", "weights_mb", "load_s", "latency_ms",
                                                  "peak_mb", "sample_output", "error"]}, default=str), flush=True)
        out_file.write_text(json.dumps(results, indent=2, default=str))

    lock = {r["model_id"]: {"sha": r.get("sha"), "license": r.get("license"), "verified_at": r["verified_at"]}
            for r in results.values() if r.get("status") == "verified" and r.get("sha")}
    (ROOT / "configs" / "model_lock.json").write_text(json.dumps(lock, indent=2))
    print(f"wrote {out_file} and configs/model_lock.json ({len(lock)} pinned)")


if __name__ == "__main__":
    main()
