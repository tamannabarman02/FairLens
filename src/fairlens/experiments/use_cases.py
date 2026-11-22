"""Run the demo use cases through the real pipeline and store results in outputs/use_cases/<id>.json."""
from __future__ import annotations

import json
import time

from ..config import PROJECT_ROOT, outputs_dir


def run_use_case(case: dict, detectors=None, rewriters=None, suite=None, cfg=None) -> dict:
    from ..detection.registry import build_detectors
    from ..pipeline import analyze_image, analyze_text, mitigate_text

    out: dict = {"id": case["id"], "kind": case["kind"], "title": case["title"], "input": case["input"],
                 "computed_at": time.strftime("%Y-%m-%d %H:%M")}
    if case["kind"] == "text":
        detectors = detectors or build_detectors()
        a = analyze_text(case["input"], detectors).to_dict()
        m = mitigate_text(case["input"], detectors, rewriters, suite, cfg).to_dict()
        out.update({"analysis": a, "mitigation": m, "bias_score": a["ensemble_score"], "decision": m["decision"]})
        must = case.get("must_preserve", [])
        chosen = m["decision"].get("chosen_text") or ""
        out["must_preserve_check"] = {k: (k.lower() in chosen.lower()) for k in must} if chosen else {}
    elif case["kind"] == "image":
        from PIL import Image

        man = json.loads((PROJECT_ROOT / "datasets" / "images" / "coco_val2017_subset" / "manifest.json").read_text(encoding="utf-8"))
        item = next(i for i in man["images"] if i["contains_person_category"])
        img = Image.open(PROJECT_ROOT / item["file"].replace("\\", "/")).convert("RGB")
        r = analyze_image(img, detectors=detectors, references=item["reference_captions"]).to_dict()
        out.update({"image": item["file"], "license": item["license"], "captions": r["captions"]})
    elif case["kind"] == "benchmark":
        from ..ui.results import latest_summary

        s = latest_summary("benchmark_detection", "detection_summary.json")
        if s:
            ok = {k: v for k, v in s["detectors"].items() if v.get("status") == "ok"}
            out["summary"] = "; ".join(f"{v['display']}: macro-F1 {v['metrics']['macro_f1']:.3f}, FPR {v['metrics']['fpr']:.2f}, "
                                       f"{v['ms_per_sample']:.0f} ms/sample" for v in ok.values()) + f" (run {s['run_id']})"
        else:
            out["summary"] = "No detection benchmark run yet."
    p = outputs_dir("use_cases") / f"{case['id']}.json"
    p.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out
