"""Phase 2: download benchmark datasets and build the unified detection schema + probes.

Outputs (datasets/processed/):
  detection.parquet       text, label, bias_type, source, split, group_id   (train/val/test, seeded, grouped)
  crows_pairs.parquet     paired probe (sent_more vs sent_less), all 1508 pairs, with split
  stereoset_pairs.parquet paired probe (stereotype vs anti-stereotype vs unrelated) with split
  winobias_probe.parquet  WinoBias pro/anti sentences (neutral event descriptions; FPR + gap probe)
  dataset_manifest.json   source URLs, revisions, counts, seed, sha256 of each processed file
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import fairlens.config  # noqa: E402,F401  (cache env)

SEED = 42
RAW = ROOT / "datasets" / "raw"
OUT = ROOT / "datasets" / "processed"
CROWS_URL = "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv"

SS_LABELS = {0: "anti-stereotype", 1: "stereotype", 2: "unrelated"}
CROWS_TYPE = {"race-color": "race_ethnicity", "sexual-orientation": "sexual_orientation",
              "physical-appearance": "physical_appearance"}
SS_TYPE = {"race": "race_ethnicity", "profession": "profession", "gender": "gender", "religion": "religion"}


def hf_parquet(repo: str, path: str) -> tuple[pd.DataFrame, str]:
    from huggingface_hub import HfApi, hf_hub_download

    sha = HfApi().dataset_info(repo).sha
    return pd.read_parquet(hf_hub_download(repo, path, repo_type="dataset", revision=sha)), sha


def group_split(groups: pd.Series, seed: int, fracs=(0.7, 0.15, 0.15)) -> pd.Series:
    uniq = np.array(sorted(groups.unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    a, b = int(fracs[0] * n), int((fracs[0] + fracs[1]) * n)
    m = {g: "train" for g in uniq[:a]} | {g: "val" for g in uniq[a:b]} | {g: "test" for g in uniq[b:]}
    return groups.map(m)


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", str(t)).strip()


def detok(tokens) -> str:
    s = " ".join(tokens)
    s = re.sub(r" ([.,!?;:'])", r"\1", s)
    s = s.replace(" n't", "n't")
    return s


def build() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    sources: dict[str, dict] = {}
    rows: list[dict] = []

    # --- BABE (media bias, sentence-level expert labels) -------------------------------------
    babe_tr, sha = hf_parquet("mediabiasgroup/BABE", "data/train-00000-of-00001.parquet")
    babe_te, _ = hf_parquet("mediabiasgroup/BABE", "data/test-00000-of-00001.parquet")
    sources["BABE"] = {"repo": "mediabiasgroup/BABE", "revision": sha, "license": "CC BY-NC-SA 4.0 (per paper; card has no license field)"}
    rng = np.random.default_rng(SEED)
    val_mask = rng.random(len(babe_tr)) < 0.15
    for split, df in [("train", babe_tr[~val_mask]), ("val", babe_tr[val_mask]), ("test", babe_te)]:
        for r in df.itertuples():
            rows.append({"text": norm(r.text), "label": int(r.label), "bias_type": "media_framing",
                         "source": "babe", "split": split, "group_id": f"babe:{r.uuid}",
                         "biased_words": r.biased_words, "subset": str(r.topic)})

    # --- StereoSet (intra + inter, validation split = the public labelled split) ---------------
    ss_pairs = []
    for sub in ["intrasentence", "intersentence"]:
        df, sha = hf_parquet("McGill-NLP/stereoset", f"{sub}/validation-00000-of-00001.parquet")
        sources["StereoSet"] = {"repo": "McGill-NLP/stereoset", "revision": sha, "license": "CC BY-SA 4.0"}
        for r in df.itertuples():
            sents = dict(zip(r.sentences["gold_label"], r.sentences["sentence"]))
            def txt(s):
                return norm(s) if sub == "intrasentence" else norm(f"{r.context} {s}")
            ss_pairs.append({"id": r.id, "subset": sub, "bias_type": SS_TYPE.get(r.bias_type, r.bias_type),
                             "target": r.target, "stereotype": txt(sents[1]), "anti_stereotype": txt(sents[0]),
                             "unrelated": txt(sents[2])})
    ss = pd.DataFrame(ss_pairs)
    ss["split"] = group_split(ss["id"], SEED)
    for r in ss.itertuples():
        for col, lab in [("stereotype", 1), ("anti_stereotype", 0)]:
            rows.append({"text": getattr(r, col), "label": lab, "bias_type": r.bias_type, "source": f"stereoset_{r.subset[:5]}",
                         "split": r.split, "group_id": f"ss:{r.id}", "biased_words": None, "subset": col})

    # --- CrowS-Pairs ---------------------------------------------------------------------------
    cp_path = RAW / "crows_pairs_anonymized.csv"
    if not cp_path.exists():
        urllib.request.urlretrieve(CROWS_URL, cp_path)
    cp = pd.read_csv(cp_path, encoding="utf-8")
    sources["CrowS-Pairs"] = {"url": CROWS_URL, "sha256": hashlib.sha256(cp_path.read_bytes()).hexdigest(), "license": "CC BY-SA 4.0"}
    cp["pair_id"] = "cp:" + cp["Unnamed: 0"].astype(str)
    cp["bias_type_n"] = cp["bias_type"].map(lambda b: CROWS_TYPE.get(b, b))
    cp["split"] = group_split(cp["pair_id"], SEED)
    for r in cp[cp.stereo_antistereo == "stereo"].itertuples():
        rows.append({"text": norm(r.sent_more), "label": 1, "bias_type": r.bias_type_n, "source": "crows_pairs",
                     "split": r.split, "group_id": r.pair_id, "biased_words": None, "subset": "sent_more"})
        rows.append({"text": norm(r.sent_less), "label": 0, "bias_type": r.bias_type_n, "source": "crows_pairs",
                     "split": r.split, "group_id": r.pair_id, "biased_words": None, "subset": "sent_less"})

    det = pd.DataFrame(rows)
    # Leakage guard: identical text may not appear in two splits (keep the earliest split by priority test>val>train).
    prio = {"test": 0, "val": 1, "train": 2}
    det["_p"] = det["split"].map(prio)
    det = det.sort_values("_p")
    before = len(det)
    det = det.drop_duplicates(subset=["text"], keep="first").drop(columns="_p")
    dupes = before - len(det)
    det = det.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # --- WinoBias probe ------------------------------------------------------------------------
    wb_rows = []
    for t in ["type1", "type2"]:
        for cond in ["pro", "anti"]:
            df, sha = hf_parquet("uclanlp/wino_bias", f"{t}_{cond}/test-00000-of-00001.parquet")
            sources["WinoBias"] = {"repo": "uclanlp/wino_bias", "revision": sha, "license": "MIT"}
            for i, r in enumerate(df.itertuples()):
                wb_rows.append({"text": detok(r.tokens), "label": 0, "condition": cond, "type": t, "pair_index": i,
                                "source": "winobias"})
    wb = pd.DataFrame(wb_rows)

    cp_probe = cp[["pair_id", "sent_more", "sent_less", "stereo_antistereo", "bias_type_n", "split"]].rename(
        columns={"bias_type_n": "bias_type"})

    files = {
        "detection.parquet": det, "crows_pairs.parquet": cp_probe,
        "stereoset_pairs.parquet": ss, "winobias_probe.parquet": wb,
    }
    manifest = {"built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seed": SEED,
                "sources": sources, "files": {}, "duplicates_removed": int(dupes)}
    for name, df in files.items():
        p = OUT / name
        df.to_parquet(p, index=False)
        manifest["files"][name] = {"rows": len(df), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()[:16]}
    manifest["detection_counts"] = (det.groupby(["source", "split", "label"]).size()
                                    .rename("n").reset_index().to_dict(orient="records"))
    # leakage check across splits by group
    g = det.groupby("group_id")["split"].nunique()
    manifest["groups_spanning_splits"] = int((g > 1).sum())
    (OUT / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "detection_counts"}, indent=2))
    print(det.groupby(["split", "label"]).size().unstack())
    print(det.groupby(["source", "split"]).size().unstack())


if __name__ == "__main__":
    build()
