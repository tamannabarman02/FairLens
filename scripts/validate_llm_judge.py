"""Choose the LLM-judge prompt variant on the VALIDATION split of the controlled gate set.

Each variant answers the yes/no faithfulness question for every changed (input, candidate) pair of the
val split; agreement with gold (yes <=> gold ACCEPT) is measured as accuracy, Cohen's kappa and unsafe-yes
rate (P(yes | gold != ACCEPT)). The variant with the highest kappa (ties: lower unsafe-yes, then lower
abstain rate) is written to configs/llm_judge.json. The test split is never loaded.

  python scripts/validate_llm_judge.py                 # loads Qwen2.5-0.5B-Instruct
  python scripts/validate_llm_judge.py --dry-run --limit 3   # stub generator, writes nothing
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fairlens.experiments.cache import ScoreCache  # noqa: E402
from fairlens.experiments.tracking import RunTracker  # noqa: E402
from fairlens.reliability_gate.llm_judge import (CONFIG_PATH, PROMPT_VARIANTS, LLMJudge,  # noqa: E402
                                                 QwenJudgeGenerator, agreement_metrics)
from fairlens.reliability_gate.signals import normalise  # noqa: E402

GATE_PATH = ROOT / "datasets" / "processed" / "gate_controlled.parquet"
VALIDATION = "val"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def require_validation_split(split: str) -> str:
    if split != VALIDATION:
        raise ValueError(f"the judge prompt is selected on the {VALIDATION!r} split only; refusing split {split!r}")
    return split


def load_validation(path: Path = GATE_PATH, split: str = VALIDATION, limit: int | None = None) -> pd.DataFrame:
    """Changed candidates of the val split, read with a parquet row filter so test rows are never loaded."""
    require_validation_split(split)
    df = pd.read_parquet(path, filters=[("split", "==", VALIDATION)])
    if set(df.split) - {VALIDATION}:
        raise RuntimeError("non-validation rows were loaded")
    if limit:
        keep = sorted(df.input_id.unique())[:limit]
        df = df[df.input_id.isin(keep)]
    changed = [normalise(a) != normalise(b) for a, b in zip(df.input_text, df.candidate_text)]
    return df[changed].reset_index(drop=True)


def stub_generator(messages: list[dict[str, str]]) -> str:
    """Deterministic placeholder for --dry-run: cycles through a clear yes, a clear no and an abstention."""
    h = int(hashlib.sha1(json.dumps(messages).encode("utf-8")).hexdigest(), 16)
    return ["Yes.", "no", "It depends."][h % 3]


stub_generator.identity = "stub"


def select_variant(results: dict[str, dict]) -> str:
    def key(name):
        m = results[name]["overall"]
        kappa = m["cohen_kappa"] if m["cohen_kappa"] is not None else float("-inf")
        return (kappa, -m["unsafe_yes_rate"], -m["abstain_rate"])
    return max(results, key=key)


def evaluate(df: pd.DataFrame, judge: LLMJudge) -> dict:
    pairs = list(zip(df.input_text, df.candidate_text))
    answers = judge.ask_many(pairs, progress=lambda k, n: log(f"    {k}/{n}") if k % 50 == 0 or k == n else None)
    pred = [answers[p].answer for p in pairs]
    out = {"overall": agreement_metrics(df.gold.tolist(), pred)}
    for label, g in df.assign(pred=pred).groupby("input_label"):
        out[f"input_label_{int(label)}"] = agreement_metrics(g.gold.tolist(), g.pred.tolist())
    return out


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", default=VALIDATION, help="must be 'val'; any other split is refused")
    ap.add_argument("--variants", nargs="+", default=list(PROMPT_VARIANTS), choices=list(PROMPT_VARIANTS))
    ap.add_argument("--limit", type=int, default=None, help="use only the first N val inputs")
    ap.add_argument("--large", action="store_true", help="use the large Qwen model id from configs/models.yaml")
    ap.add_argument("--dry-run", action="store_true", help="stub generator, no cache, writes no config and no run")
    ap.add_argument("--out", type=Path, default=CONFIG_PATH)
    args = ap.parse_args(argv)
    if args.split != VALIDATION:
        ap.error(f"the judge prompt is selected on the {VALIDATION!r} split only; refusing split {args.split!r}")

    df = load_validation(GATE_PATH, args.split, args.limit)
    log(f"validation pairs: {len(df)} rows, {df.input_id.nunique()} inputs, gold {df.gold.value_counts().to_dict()}")
    if args.dry_run:
        generator, cache, tracker = stub_generator, None, None
    else:
        generator, cache = QwenJudgeGenerator(large=args.large), ScoreCache()
        tracker = RunTracker("llm_judge_validation", seed=generator.seed, dataset_files=[GATE_PATH],
                             models={"judge": generator.identity},
                             config={"split": VALIDATION, "variants": args.variants, "limit": args.limit,
                                     "selection": "max cohen_kappa, then min unsafe_yes_rate, then min abstain_rate"})
    identity = getattr(generator, "identity", "judge")

    results, logs = {}, []
    for variant in args.variants:
        log(f"variant {variant}: {PROMPT_VARIANTS[variant]['description']}")
        judge = LLMJudge(generator, variant, cache, identity)
        results[variant] = evaluate(df, judge)
        logs += judge.log_rows
        m = results[variant]["overall"]
        kappa = "n/a" if m["cohen_kappa"] is None else f"{m['cohen_kappa']:.3f}"
        log(f"  accuracy {m['accuracy']:.3f} kappa {kappa} unsafe-yes {m['unsafe_yes_rate']:.3f} "
            f"abstain {m['abstain_rate']:.3f}")
    best = select_variant(results)
    log(f"selected variant on {VALIDATION}: {best}")
    selection = {"variant": best, "selected_on": VALIDATION, "model": identity, "large": args.large, "limit": args.limit,
                 "n_rows": len(df), "n_inputs": int(df.input_id.nunique()),
                 "selection_rule": "max cohen_kappa, then min unsafe_yes_rate, then min abstain_rate",
                 "validation_metrics": results}
    if tracker is None:
        log("dry run: nothing written")
        return selection
    selection["run_id"] = tracker.run_id
    tracker.save_artifact("llm_judge_validation.json", selection)
    log_path = tracker.dir / "prompts_log.jsonl"
    log_path.write_text("\n".join(json.dumps(r, default=str, ensure_ascii=False) for r in logs), encoding="utf-8")
    args.out.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    tracker.finish(metrics={"selected_variant": best, **{f"{v}_kappa": r["overall"]["cohen_kappa"] for v, r in results.items()}},
                   artifacts={"prompts_log.jsonl": log_path, "config": args.out.resolve()})
    log(f"wrote {args.out} (run {tracker.run_id})")
    return selection


if __name__ == "__main__":
    main()
