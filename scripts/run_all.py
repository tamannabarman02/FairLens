"""Run the FairLens experiment pipeline in order, one heavy job at a time, on any OS.

    python scripts/run_all.py                    # every step
    python scripts/run_all.py --from gate        # resume from a step
    python scripts/run_all.py --only detection mitigation
    python scripts/run_all.py --list

Each step's output goes to outputs/logs/<step>.log. A failed step is reported and the remaining steps still run
unless --stop-on-error is given; nothing is retried or skipped silently.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

STEPS: list[tuple[str, list[str], str]] = [
    ("train_bert", ["scripts/train_detectors.py", "--model", "bert"], "Fine-tune DistilBERT"),
    ("train_deberta", ["scripts/train_detectors.py", "--model", "deberta"], "Fine-tune DeBERTa-v3-small"),
    ("prompts", ["scripts/select_prompts.py"], "Choose the rewrite prompt on the train-split sample"),
    ("detection", ["scripts/run_benchmarks.py", "--task", "detection"], "Detection benchmark"),
    ("judge", ["scripts/validate_llm_judge.py"], "Choose the LLM-judge prompt on the validation split"),
    ("gate", ["scripts/run_ablations.py"], "Gate vs baselines, threshold tuning and ablations"),
    ("mitigation", ["scripts/run_benchmarks.py", "--task", "mitigation"], "Mitigation benchmark"),
    ("image", ["scripts/run_image_experiment.py"], "Image-caption experiment"),
    ("use_cases", ["scripts/run_use_cases.py"], "Six demo use cases"),
    ("results", ["scripts/generate_results_md.py"], "Regenerate docs/RESULTS.md from stored runs"),
]
NAMES = [s[0] for s in STEPS]


def run_step(name: str, args: list[str], log_dir: Path) -> tuple[int, float]:
    """Run one step as a child process, streaming its output to a log file."""
    log = log_dir / f"{name}.log"
    t0 = time.perf_counter()
    with open(log, "w", encoding="utf-8") as fh:
        proc = subprocess.run([PY, "-u", *args], cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                              env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    return proc.returncode, time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start", choices=NAMES, help="start at this step")
    ap.add_argument("--only", nargs="+", choices=NAMES, help="run only these steps (in pipeline order)")
    ap.add_argument("--skip", nargs="+", choices=NAMES, default=[], help="steps to leave out")
    ap.add_argument("--stop-on-error", action="store_true")
    ap.add_argument("--list", action="store_true", help="list the steps and exit")
    a = ap.parse_args()

    if a.list:
        for name, args, desc in STEPS:
            print(f"{name:<14} {desc}  ({' '.join(args)})")
        return 0
    selected = [s for s in STEPS if (not a.only or s[0] in a.only) and s[0] not in a.skip]
    if a.start:
        selected = selected[[s[0] for s in selected].index(a.start):] if a.start in [s[0] for s in selected] else []

    log_dir = ROOT / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    summary: list[tuple[str, int, float]] = []
    for name, args, desc in selected:
        print(f"[run_all] {name}: {desc} ...", flush=True)
        code, secs = run_step(name, args, log_dir)
        summary.append((name, code, secs))
        status = "ok" if code == 0 else f"FAILED (exit {code}, see outputs/logs/{name}.log)"
        print(f"[run_all] {name}: {status} in {secs / 60:.1f} min", flush=True)
        if code and a.stop_on_error:
            break

    print("\n[run_all] summary")
    for name, code, secs in summary:
        print(f"  {name:<14} {'ok' if code == 0 else 'FAILED'}  {secs / 60:6.1f} min")
    return 1 if any(code for _, code, _ in summary) else 0


if __name__ == "__main__":
    sys.exit(main())
