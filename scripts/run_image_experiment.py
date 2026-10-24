"""Image-caption experiment: BLIP vs Florence-2 on a documented COCO val2017 subset (+ any user images).

Scope: measures bias signals in GENERATED CAPTIONS (language), not all visual bias. COCO metadata
(category "person", licence, reference captions) is used only to select/describe the set, never to
infer anyone's demographics.

  python scripts/run_image_experiment.py [--n-person 60 --n-other 20] [--fallback]
"""
from __future__ import annotations

import argparse
import io
import json
import random
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fairlens.detection.registry import build_detectors  # noqa: E402
from fairlens.evaluation import metrics as M  # noqa: E402
from fairlens.experiments.cache import ScoreCache, score_texts  # noqa: E402
from fairlens.experiments.tracking import RunTracker  # noqa: E402
from fairlens.models import manager  # noqa: E402
from fairlens.multimodal.caption_evaluation import (ClipAlignment, alignment_label, caption_flags,  # noqa: E402
                                                    detail_coverage)
from fairlens.multimodal.captioners import build_captioners  # noqa: E402

ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
RAW = ROOT / "datasets" / "raw"
IMG_DIR = ROOT / "datasets" / "images" / "coco_val2017_subset"
MANIFEST = IMG_DIR / "manifest.json"
USER_DIR = ROOT / "datasets" / "my_samples"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def fetch(url: str, tries: int = 4) -> bytes:
    for k in range(tries):
        try:
            return urllib.request.urlopen(url, timeout=60).read()
        except Exception:  # noqa: BLE001
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def build_subset(n_person: int, n_other: int, seed: int = 42) -> list[dict]:
    if MANIFEST.exists():
        m = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if m["n_person"] == n_person and m["n_other"] == n_other:
            return m["images"]
    RAW.mkdir(parents=True, exist_ok=True)
    cap_p, inst_p = RAW / "captions_val2017.json", RAW / "instances_val2017.json"
    if not (cap_p.exists() and inst_p.exists()):
        zp = RAW / "annotations_trainval2017.zip"
        if not zp.exists():
            log("downloading COCO 2017 annotations (241 MB)")
            zp.write_bytes(fetch(ANN_URL))
        with zipfile.ZipFile(zp) as z:
            for name, dest in [("annotations/captions_val2017.json", cap_p), ("annotations/instances_val2017.json", inst_p)]:
                dest.write_bytes(z.read(name))
    caps = json.loads(cap_p.read_text(encoding="utf-8"))
    inst = json.loads(inst_p.read_text(encoding="utf-8"))
    licenses = {l["id"]: {"name": l["name"], "url": l["url"]} for l in inst["licenses"]}
    person_imgs = {a["image_id"] for a in inst["annotations"] if a["category_id"] == 1}
    refs: dict[int, list[str]] = {}
    for a in caps["annotations"]:
        refs.setdefault(a["image_id"], []).append(a["caption"].strip())
    imgs = {i["id"]: i for i in inst["images"]}
    rng = random.Random(seed)
    person = sorted(i for i in imgs if i in person_imgs)
    other = sorted(i for i in imgs if i not in person_imgs)
    chosen = [(i, True) for i in rng.sample(person, n_person)] + [(i, False) for i in rng.sample(other, n_other)]
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for iid, has_person in chosen:
        info = imgs[iid]
        p = IMG_DIR / info["file_name"]
        if not p.exists():
            p.write_bytes(fetch(info["coco_url"]))
        out.append({"image_id": iid, "file": p.relative_to(ROOT).as_posix(), "contains_person_category": has_person,
                    "license": licenses[info["license"]], "coco_url": info["coco_url"], "flickr_url": info.get("flickr_url"),
                    "width": info["width"], "height": info["height"], "reference_captions": refs.get(iid, [])})
    MANIFEST.write_text(json.dumps({"source": "COCO 2017 val (images: Flickr licences per image; annotations CC BY 4.0)",
                                    "seed": seed, "n_person": n_person, "n_other": n_other, "images": out}, indent=2),
                        encoding="utf-8")
    return out


def user_images() -> list[dict]:
    out = []
    for p in sorted(USER_DIR.glob("*")):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif", ".webp"}:
            out.append({"image_id": p.name, "file": p.relative_to(ROOT).as_posix(), "contains_person_category": None,
                        "license": {"name": "user-provided"}, "reference_captions": []})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-person", type=int, default=60)
    ap.add_argument("--n-other", type=int, default=20)
    ap.add_argument("--fallback", action="store_true", help="use ViT-GPT2 instead of Florence-2")
    a = ap.parse_args()
    from PIL import Image

    items = build_subset(a.n_person, a.n_other) + user_images()
    log(f"{len(items)} images")
    captioners = build_captioners(use_fallback_for_florence=a.fallback)
    tracker = RunTracker("benchmark_image", config={"n_person": a.n_person, "n_other": a.n_other, "fallback": a.fallback,
                                                    "captioners": {k: c.model_id for k, c in captioners.items()}},
                         seed=42, dataset_files=[MANIFEST], models={k: c.model_id for k, c in captioners.items()})
    rows = []
    for name, cap in captioners.items():
        manager().clear()
        log(f"captioning with {cap.display} ({cap.model_id})")
        for k, it in enumerate(items):
            img = Image.open(ROOT / it["file"].replace("\\", "/")).convert("RGB")
            c = cap.caption(img)
            rows.append({"image_id": it["image_id"], "captioner": name, "display": cap.display, "model": cap.model_id,
                         "caption": c.text, "runtime_ms": c.runtime_ms, "error": c.error})
            if k % 10 == 9:
                log(f"  {k + 1}/{len(items)}  last: {c.text!r} ({c.runtime_ms:.0f} ms)")
    df = pd.DataFrame(rows)
    by_id = {it["image_id"]: it for it in items}

    manager().clear()
    log("CLIP alignment")
    clip = ClipAlignment()
    clip_gen, clip_ref = [], {}
    for it in items:
        img = Image.open(ROOT / it["file"].replace("\\", "/")).convert("RGB")
        caps = df[(df.image_id == it["image_id"]) & df.caption.notna()]
        texts = caps.caption.tolist() + it["reference_captions"][:1]
        sims = clip.score(img, texts) if texts else []
        for idx, s in zip(caps.index, sims[:len(caps)]):
            clip_gen.append((idx, s))
        if it["reference_captions"]:
            clip_ref[it["image_id"]] = sims[-1]
    for idx, s in clip_gen:
        df.loc[idx, "clip_similarity"] = s
    df["alignment"] = df.clip_similarity.map(lambda s: alignment_label(s) if pd.notna(s) else None)
    df["reference_clip"] = df.image_id.map(clip_ref)
    df["detail_coverage"] = [detail_coverage(c, by_id[i]["reference_captions"]) if c else None for c, i in zip(df.caption, df.image_id)]
    df["contains_person_category"] = df.image_id.map(lambda i: by_id[i]["contains_person_category"])
    fl = [caption_flags(c) if c else {} for c in df.caption]
    df["n_words"] = [f.get("n_words") for f in fl]
    df["gendered_terms"] = [",".join(f.get("gendered_terms", [])) for f in fl]
    df["unsupported_attributes"] = [",".join(f.get("unsupported_attributes", [])) for f in fl]
    ref_rows = [{"image_id": it["image_id"], "ref_gendered": bool(caption_flags(" ".join(it["reference_captions"]))["gendered_terms"]),
                 "ref_gendered_all": all(caption_flags(r)["gendered_terms"] for r in it["reference_captions"]) if it["reference_captions"] else None}
                for it in items]
    refdf = pd.DataFrame(ref_rows).set_index("image_id")

    detectors = build_detectors()
    cache = ScoreCache()
    texts = df.caption.dropna().tolist()
    for dname, det in detectors.items():
        manager().clear()
        log(f"detector {det.display} on {len(texts)} captions")
        sc, _ = score_texts(det, texts, cache)
        df[f"det_{dname}"] = df.caption.map(lambda c: sc.get(c) if c else None)
    dcols = [f"det_{d}" for d in detectors]
    df["bias_score"] = df[dcols].mean(axis=1)

    summary = {"run_id": tracker.run_id, "n_images": len(items), "scope_note": "Bias in generated captions only; not all visual bias.",
               "captioners": {}}
    for name, g in df.groupby("captioner"):
        ok = g[g.caption.notna()]
        pers = ok[ok.contains_person_category == True]  # noqa: E712
        summary["captioners"][name] = {
            "display": g.display.iloc[0], "model": g.model.iloc[0], "n_ok": len(ok), "n_failed": int(g.caption.isna().sum()),
            "ms_per_image": float(ok.runtime_ms.mean()), "clip_similarity": M.bootstrap_ci(ok.clip_similarity.dropna().to_numpy()),
            "mean_words": float(ok.n_words.mean()), "detail_coverage": M.bootstrap_ci(ok.detail_coverage.dropna().to_numpy()),
            "alignment_bands": ok.alignment.value_counts().to_dict(),
            "gender_term_rate_person_images": float((pers.gendered_terms != "").mean()) if len(pers) else None,
            "unsupported_attribute_rate": float((ok.unsupported_attributes != "").mean()),
            "mean_bias_score": float(ok.bias_score.mean()), "flagged_rate": float((ok.bias_score >= 0.5).mean()),
            "per_detector_flag_rate": {d: float((ok[f"det_{d}"] >= 0.5).mean()) for d in detectors},
        }
    pers_ids = [it["image_id"] for it in items if it["contains_person_category"]]
    summary["reference_captions"] = {
        "clip_similarity": M.bootstrap_ci(np.array(list(clip_ref.values()))),
        "gender_term_rate_person_images_any_ref": float(refdf.loc[pers_ids].ref_gendered.mean()) if pers_ids else None,
    }
    piv = df.pivot_table(index="image_id", columns="captioner", values="bias_score")
    if piv.shape[1] == 2:
        a_, b_ = piv.columns
        summary["agreement"] = {"pair": [a_, b_], "both_flagged": int(((piv[a_] >= .5) & (piv[b_] >= .5)).sum()),
                                "only_" + a_: int(((piv[a_] >= .5) & (piv[b_] < .5)).sum()),
                                "only_" + b_: int(((piv[a_] < .5) & (piv[b_] >= .5)).sum()),
                                "score_corr": float(piv[a_].corr(piv[b_]))}
    tracker.save_artifact("image_captions.parquet", df)
    tracker.save_artifact("image_summary.json", summary)
    tracker.finish(metrics={k: {"clip": v["clip_similarity"]["mean"], "flagged_rate": v["flagged_rate"]} for k, v in summary["captioners"].items()})
    log(json.dumps(summary["captioners"], indent=1, default=str)[:3000])
    log(f"done: {tracker.run_id}")


if __name__ == "__main__":
    main()
