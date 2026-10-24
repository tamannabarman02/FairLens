"""Choose the shared mitigation instruction on a DEVELOPMENT sample (train split only, never the eval set).

Each instruction variant is run with every LLM rewriter on the same 30 biased train-split sentences.
Score per (variant, model): changed_rate x share of changed outputs that keep facts and similarity >= 0.7,
plus mean rule-detector bias drop. The variant with the best mean score across models is written to
configs/prompt_selection.json and used by fairlens.mitigation.prompts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fairlens.detection.rule_based import RuleBasedDetector  # noqa: E402
from fairlens.evaluation import facts  # noqa: E402
from fairlens.evaluation.preservation import SemanticSimilarity  # noqa: E402
from fairlens.mitigation import prompts  # noqa: E402
from fairlens.mitigation.llm_adapters import build_rewriters  # noqa: E402
from fairlens.models import manager  # noqa: E402

VARIANTS = {
    "V1_base": prompts.DEFAULT_INSTRUCTION,
    "V2_inclusive": "Rewrite the following text in neutral, inclusive language that avoids stereotypes about any group. "
                    "Keep all facts, names, numbers and negations exactly. Return only the rewritten text.",
    "V3_remove_bias": "Remove the stereotype or biased wording from this text and write a fair version with the same facts. "
                      "Return only the rewritten text.",
}


def main(n: int = 30):
    det = pd.read_parquet(ROOT / "datasets" / "processed" / "detection.parquet")
    dev = det[(det.split == "train") & (det.label == 1) & det.text.str.split().str.len().between(6, 35)]
    dev = pd.concat([d.sample(n // 4 + 1, random_state=7) for _, d in dev.groupby("source")]).head(n)
    texts = dev.text.tolist()
    rule = RuleBasedDetector()
    sim = SemanticSimilarity()
    results, outputs = {}, {}
    rewriters = {k: v for k, v in build_rewriters().items() if k != "rule_based"}
    variants = {**{k: (v, False) for k, v in VARIANTS.items()}, **{k + "+demos": (v, True) for k, v in VARIANTS.items()}}
    for vname, (instr, demos) in variants.items():
        prompts.BASE_INSTRUCTION, prompts.USE_DEMOS = instr, demos
        for rname, rw in rewriters.items():
            manager().clear()
            outs = [rw.rewrite(t).text or "" for t in texts]
            outputs[f"{vname}/{rname}"] = outs
            changed = [o.strip().rstrip(".").lower() != t.strip().rstrip(".").lower() for o, t in zip(outs, texts)]
            sims = sim.similarity(texts, [o or "." for o in outs])
            keep = [facts.check(t, o)["fact_retention"] == 1.0 for t, o in zip(texts, outs)]
            ok = [c and s >= 0.7 and k for c, s, k in zip(changed, sims, keep)]
            drop = [rule.predict(t).score - rule.predict(o).score for t, o in zip(texts, outs)]
            score = float(np.mean(changed)) * (float(np.sum(ok)) / max(1, np.sum(changed)))
            results[f"{vname}/{rname}"] = {"changed_rate": float(np.mean(changed)), "ok_rate": float(np.mean(ok)),
                                           "mean_similarity": float(np.mean(sims)), "fact_keep_rate": float(np.mean(keep)),
                                           "rule_bias_drop": float(np.mean(drop)), "score": score}
            print(vname, rname, results[f"{vname}/{rname}"], flush=True)
    per_variant = {v: float(np.mean([results[f"{v}/{r}"]["score"] for r in rewriters])) for v in variants}
    best = max(per_variant, key=per_variant.get)
    out = {"chosen": best, "instruction": variants[best][0], "use_demos": variants[best][1], "per_variant_score": per_variant, "results": results,
           "dev_sample": texts, "outputs": outputs, "n": len(texts),
           "note": "Development sample from the TRAIN split; the mitigation evaluation set was not used."}
    (ROOT / "configs" / "prompt_selection.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("chosen:", best, per_variant)


if __name__ == "__main__":
    main()
