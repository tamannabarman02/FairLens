"""Re-run the paper's live demonstration (Sect. 6.3) through the same pipeline calls as the app, and store it as a run.

The app does not store uploaded text, so the original demo (2026-11-17, screenshots in paper/figures/demo_*.png)
left no run record. This script repeats it with the app's settings (standard profile, pinned gate_app thresholds)
and saves the analysis, mitigation result and fact checks under outputs/runs/<run_id>/.

Run (Git Bash, project root):
  HF_HOME="D:\\FairLens_cache\\hf" HF_HUB_OFFLINE=1 PYTHONUTF8=1 .venv/Scripts/python.exe -u scripts/run_live_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fairlens.detection.registry import build_detectors  # noqa: E402
from fairlens.evaluation import facts  # noqa: E402
from fairlens.evaluation.preservation import PreservationSuite  # noqa: E402
from fairlens.experiments.tracking import RunTracker  # noqa: E402
from fairlens.mitigation.llm_adapters import build_rewriters  # noqa: E402
from fairlens.pipeline import analyze_text, mitigate_text  # noqa: E402
from fairlens.reliability_gate.rules import GateConfig  # noqa: E402
from fairlens.ui.state import tuned_gate_overrides  # noqa: E402

TEXT = ("We are looking for a young, energetic waitress to join our Bengaluru cafe. "
        "She must be available 6 days a week and have 2 years of experience.")


def main():
    overrides = tuned_gate_overrides()
    nlp = facts._nlp()
    tracker = RunTracker("live_demo", config={"text": TEXT, "profile": "standard", "gate_overrides": overrides,
                                              "spacy_model": None if nlp is None else
                                              f"{nlp.meta['lang']}_{nlp.meta['name']} {nlp.meta['version']}"})
    print("run", tracker.run_id, flush=True)
    dets = build_detectors(lite=False)
    a = analyze_text(TEXT, dets, progress=lambda s, m: print(s, m, flush=True)).to_dict()
    tracker.save_artifact("analysis.json", a)
    m = mitigate_text(TEXT, dets, build_rewriters(lite=False), PreservationSuite(), GateConfig.from_config(**overrides),
                      progress=lambda s, msg: print(s, msg, flush=True)).to_dict()
    tracker.save_artifact("mitigation.json", m)
    tracker.save_artifact("facts_extracted.json", facts.extract(TEXT).to_dict())
    metrics = {"overall_score": a["ensemble_score"],
               "detector_scores": {k: v["score"] for k, v in a["detector_results"].items()},
               "decision": m["decision"],
               "candidates": {c["candidate"]: {"verdict": c["verdict"], "reasons": c["reasons"], "text": c["text"],
                                               "bias_before": c["bias_before"], "bias_after": c["bias_after"],
                                               "semantic_similarity": c["semantic_similarity"],
                                               "entail_forward": c["entail_forward"],
                                               "entail_backward": c["entail_backward"],
                                               "fact_retention": c["fact_retention"]} for c in m["candidates"]}}
    tracker.finish(metrics=metrics)
    print("done", tracker.run_id, tracker.dir, flush=True)


if __name__ == "__main__":
    main()
