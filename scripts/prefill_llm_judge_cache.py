"""Pre-compute LLM-judge answers into the shared ScoreCache (outputs/cache/scores.sqlite).

For environments where the model can run but pandas cannot (e.g. Windows Application Control blocking pandas
in the project venv): this script reads gate_controlled.parquet with pyarrow only and asks Qwen the judge
question for every changed candidate of ONE split, so validate_llm_judge.py / run_ablations.py later find every
answer in the cache. The cache key is the judge identity plus the exact prompt, so cached answers are identical
to what those scripts would have generated themselves.

  python scripts/prefill_llm_judge_cache.py --split val  --variants plain checker_system question_first
  python scripts/prefill_llm_judge_cache.py --split test --variants <variant chosen on val>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Hide scikit-learn when Application Control blocks it: transformers then treats it as absent (find_spec -> None)
# instead of failing to import it. It is only used for assisted decoding, not for greedy judge generation.
sys.modules.setdefault("sklearn", None)

import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fairlens.experiments.cache import ScoreCache  # noqa: E402
from fairlens.experiments.tracking import RunTracker  # noqa: E402
from fairlens.reliability_gate.llm_judge import PROMPT_VARIANTS, LLMJudge, QwenJudgeGenerator  # noqa: E402
from fairlens.reliability_gate.signals import normalise  # noqa: E402

GATE_PATH = ROOT / "datasets" / "processed" / "gate_controlled.parquet"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def changed_pairs(split: str) -> list[tuple[str, str]]:
    """Changed (input, candidate) pairs of one split, read with a row filter so other splits are never loaded."""
    t = pq.read_table(GATE_PATH, columns=["split", "input_text", "candidate_text"], filters=[("split", "==", split)])
    if set(pc.unique(t["split"]).to_pylist()) - {split}:
        raise RuntimeError("rows of another split were loaded")
    pairs = zip(t["input_text"].to_pylist(), t["candidate_text"].to_pylist())
    return list(dict.fromkeys((a, b) for a, b in pairs if normalise(a) != normalise(b)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--variants", nargs="+", required=True, choices=list(PROMPT_VARIANTS))
    args = ap.parse_args()
    pairs = changed_pairs(args.split)
    gen, cache = QwenJudgeGenerator(), ScoreCache()
    tracker = RunTracker("llm_judge_prefill", seed=gen.seed, dataset_files=[GATE_PATH], models={"judge": gen.identity},
                         config={"split": args.split, "variants": args.variants, "n_pairs": len(pairs)})
    log(f"{args.split}: {len(pairs)} changed pairs, variants {args.variants}, judge {gen.identity} (run {tracker.run_id})")
    counts, logs = {}, []
    for variant in args.variants:
        judge = LLMJudge(gen, variant, cache)
        judge.ask_many(pairs, progress=lambda k, n: log(f"  {variant} {k}/{n}") if k % 100 == 0 or k == n else None)
        logs += judge.log_rows
        new = sum(not r["cached"] for r in judge.log_rows)
        errors = sum(r["error"] is not None for r in judge.log_rows)
        counts[variant] = {"generated": new, "cached": len(pairs) - new, "errors": errors}
        log(f"  {variant}: {new} generated, {len(pairs) - new} already cached, {errors} errors")
        if errors:
            tracker.finish(status="failed", metrics=counts)
            raise SystemExit(f"{errors} judge calls failed; errors are not cached and would become abstentions")
    log_path = tracker.dir / "prompts_log.jsonl"
    log_path.write_text("\n".join(json.dumps(r, default=str, ensure_ascii=False) for r in logs), encoding="utf-8")
    tracker.finish(metrics=counts, artifacts={"prompts_log.jsonl": log_path})


if __name__ == "__main__":
    main()
