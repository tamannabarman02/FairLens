"""Run all demo use cases through the real pipeline (results cached in outputs/use_cases/ for the demo)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fairlens.detection.registry import build_detectors  # noqa: E402
from fairlens.evaluation.preservation import PreservationSuite  # noqa: E402
from fairlens.experiments.use_cases import run_use_case  # noqa: E402
from fairlens.mitigation.llm_adapters import build_rewriters  # noqa: E402
from fairlens.ui.state import tuned_gate_overrides  # noqa: E402
from fairlens.reliability_gate.rules import GateConfig  # noqa: E402


def main():
    cases = json.loads((ROOT / "datasets" / "synthetic_demo" / "use_cases.json").read_text(encoding="utf-8"))["cases"]
    det, rw, suite = build_detectors(), build_rewriters(), PreservationSuite()
    cfg = GateConfig.from_config(**tuned_gate_overrides())
    for c in cases:
        print(f"== {c['id']}", flush=True)
        r = run_use_case(c, det, rw, suite, cfg)
        if r["kind"] == "text":
            d = r["decision"]
            print(f"   bias {r['bias_score']:.2f} -> {d['decision']} ({d['reason_code']}): {d.get('chosen_text')}", flush=True)
        elif r["kind"] == "image":
            for cap in r["captions"]:
                print(f"   {cap['display']}: {cap['text']}", flush=True)
        else:
            print("   " + r["summary"], flush=True)
    print("done: use cases")


if __name__ == "__main__":
    main()
