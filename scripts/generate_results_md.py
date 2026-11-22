"""Generate docs/RESULTS.md (and the measured table in docs/MODEL_CARD_REGISTRY.md) from stored runs only.

Every number written here is read from outputs/runs/<run_id>/*.json; nothing is typed by hand.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fairlens.ui.results import latest_summary, model_verification  # noqa: E402

POL = {"fairlens_gate_tuned": "FairLens Gate (val-tuned rules)", "fairlens_gate_default": "FairLens Gate (default rules)",
       "learned_logreg": "Learned gate (logistic regression)", "learned_tree": "Learned gate (decision tree)",
       "lowest_bias": "Lowest bias score", "highest_similarity": "Highest similarity", "random_mean": "Random (5 seeds)",
       "fairlens_gate": "FairLens Gate (no retry)", "fairlens_gate_retry": "FairLens Gate + strict retry",
       "no_change": "No change (keep original)", "llm_judge": "LLM-as-judge (Qwen2.5, val-selected prompt)"}


def f(v, d=3):
    return "n/a" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))


def ci(x):
    if not x or x.get("mean") is None:
        return "n/a"
    if x.get("lo") is None or x.get("hi") is None:
        return f"{x['mean']:.3f} (no CI)"
    return f"{x['mean']:.3f} [{x['lo']:.3f}, {x['hi']:.3f}]"


def table(rows: list[dict]) -> str:
    if not rows:
        return "_no rows_\n"
    cols = list(rows[0])
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(f(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(out) + "\n"


def detection(s) -> str:
    ok = {k: v for k, v in s["detectors"].items() if v.get("status") == "ok"}
    out = [f"## 1. Text-bias detection\n\nRun `{s['run_id']}`; shared test split n = {s['n_test']}; threshold 0.5 for all detectors.\n"]
    out.append(table([{"Detector": v["display"], "Type": {"rule_based": "heuristic", "bert": "fine-tuned", "deberta": "fine-tuned",
                                                          "bart_mnli": "pretrained zero-shot", "hub_pretrained": "pretrained (BABE)"}.get(k, k),
                       "Precision": v["metrics"]["precision"], "Recall": v["metrics"]["recall"], "F1": v["metrics"]["f1"],
                       "F1 95% CI": f"[{v['ci']['f1']['lo']:.3f}, {v['ci']['f1']['hi']:.3f}]", "Macro-F1": v["metrics"]["macro_f1"],
                       "ROC-AUC": v["metrics"].get("roc_auc"), "ECE": v["metrics"]["ece"], "ms/sample": round(v["ms_per_sample"], 1),
                       "Model MB": v.get("model_mb")} for k, v in ok.items()]))
    out.append("\nConfusion matrices ([[TN, FP], [FN, TP]]): " + "; ".join(f"{v['display']} {v['metrics']['confusion']}" for v in ok.values()) + "\n")
    out.append("\n### Macro-F1 by source\n\n")
    srcs = sorted({x for v in ok.values() for x in v["by_source"]})
    out.append(table([{"Detector": v["display"], **{x: v["by_source"][x]["macro_f1"] for x in srcs}} for v in ok.values()]))
    if "bart_mnli" in ok and ok["bart_mnli"].get("template_selection"):
        ts = ok["bart_mnli"]["template_selection"]
        out.append(f"\nZero-shot template chosen on {ts['val_n']} validation sentences: **{ts['chosen']}**.\n\n")
        out.append(table([{"Template": k, "Val macro-F1": v["macro_f1"], "Val AUC": v["roc_auc"]} for k, v in ts["candidates"].items()]))
    rows = []
    for v in ok.values():
        p = v.get("probes")
        if p:
            rows.append({"Detector": v["display"], "WinoBias FPR pro": p["winobias"]["fpr_pro"], "WinoBias FPR anti": p["winobias"]["fpr_anti"],
                         "WinoBias gap (pro-anti)": ci(p["winobias"]["mean_gap_pro_minus_anti"]),
                         "CrowS stereo pref": ci(p["crows_pairs"]["pref_more_stereo"]),
                         "StereoSet stereo>anti": ci(p["stereoset"]["pref_stereo_over_anti"]),
                         "StereoSet unrelated flagged": p["stereoset"]["flag_rate_unrelated"]})
    if rows:
        out.append("\n### Probes\n\nWinoBias sentences are neutral, so any flag is a false positive.\n\n" + table(rows))
    if "ensemble" in s:
        out.append(f"\nMean-score ensemble ({', '.join(s['ensemble']['members'])}): macro-F1 {s['ensemble']['metrics']['macro_f1']:.3f}.\n")
    return "".join(out)


def mitigation(s) -> str:
    out = [f"## 2. Mitigation (real rewrites)\n\nRun `{s['run_id']}`; {s['n_inputs']} inputs ({s['n_biased_gold']} biased, "
           f"{s['n_neutral_gold']} neutral by gold label). No gold labels exist for model outputs; metrics are automatic.\n\n"]
    out.append(table([{"Rewriter": k, "Mean bias after (biased)": v["mean_bias_after_biased"], "Bias reduction": v["mean_bias_reduction_biased"],
                       "Bias red. 95% CI": ci(v["bias_reduction_ci"]) if v.get("bias_reduction_ci") else "n/a",
                       "Changed rate": v["changed_rate"], "Similarity": v["semantic_similarity"], "Entail both": v["entailment_both"],
                       "Fact retention": v["fact_retention"], "Fact failures": v["fact_failure_rate"], "Fluency": v["fluency"],
                       "Changed neutral": v["unnecessary_change_rate_neutral"], "Sim. to reference": v["ref_similarity"]}
                      for k, v in s["rewriters"].items()]))
    out.append("\n### Selection policies\n\n")
    out.append(table([{"Policy": POL.get(k, k), "Deferred": v.get("defer_rate"), "Accepted w/ fact loss": v.get("accepted_fact_failure_rate"),
                       "Accepted w/ contradiction": v.get("accepted_contradiction_rate"), "Unnecessary change": v.get("unnecessary_change_rate"),
                       "Bias reduction": v.get("mean_bias_reduction_biased"), "Biased left unchanged": v.get("biased_left_unchanged_rate"),
                       "Sim. to reference": v.get("mean_ref_similarity")} for k, v in s["policies"].items()]))
    out.append(f"\nGate final decisions: {s['gate_final_distribution']}; reasons: {s['gate_final_reasons']}; retry: {s['retry']}.\n"
               f"Candidate verdicts (first attempt, excluding original): {s['candidate_verdict_distribution']}.\n"
               f"Input bias detection on this set: {s['input_detection']}.\n")
    return "".join(out)


def gate(s) -> str:
    out = [f"## 3. Reliability Gate (controlled gold set)\n\nRun `{s['run_id']}`; {s['n_inputs']} authored inputs x {len(s['seeds'])} "
           f"perturbation seeds; {s['n_candidates']} candidates; test-split gold distribution {s['gold_distribution_test']}.\n"
           f"Thresholds tuned on the **val** split only: `{s['tuned_rules']}`.\n\n### Candidate-level\n\n"]
    out.append(table([{"Variant": k, "3-class macro-F1": v["macro_f1_3class"], "Accept precision": v["accept_precision"],
                       "Accept recall": v["accept_recall"], "Unsafe accept": v["unsafe_accept_rate"], "Human review": v["human_review_rate"]}
                      for k, v in s["candidate_level"].items()]))
    cr = s["candidate_level"]["rules_val_tuned"].get("catch_rate_by_perturbation")
    if cr:
        out.append("\nCatch rate by perturbation (val-tuned rules): " + ", ".join(f"{k} {v:.2f}" for k, v in cr.items()) + "\n")
    out.append("\n### Input-level selection vs baselines\n\n")
    out.append(table([{"Policy": POL.get(k, k), "Acceptable choice": ci(v["success"]), "Harmful choice": ci(v["harmful_selection"]),
                       "Deferred": v["defer"]["mean"], "Unnecessary change (neutral)": v["unnecessary_change_neutral"]["mean"],
                       "Δ success vs tuned gate": (f"{v['diff_success_vs_tuned_gate']['diff']:+.3f} [{v['diff_success_vs_tuned_gate']['lo']:+.3f}, "
                                                   f"{v['diff_success_vs_tuned_gate']['hi']:+.3f}]") if "diff_success_vs_tuned_gate" in v else "-"}
                      for k, v in s["selection"].items()]))
    out.append("\n### Ablations (val-tuned rule gate)\n\n")
    ab = s["ablations"]
    rows = [{"Gate": "full", "Unsafe accept": ab["full"]["unsafe_accept_rate"], "Δ unsafe [95% CI]": "-",
             "Macro-F1": ab["full"]["macro_f1_3class"], "Δ accuracy [95% CI]": "-", "Learned logreg macro-F1": None}]
    for k, v in ab.items():
        if k != "full":
            du, da = v["delta_unsafe_accept"], v["delta_accuracy"]
            rows.append({"Gate": k, "Unsafe accept": v["metrics"]["unsafe_accept_rate"],
                         "Δ unsafe [95% CI]": f"{du['diff']:+.3f} [{du['lo']:+.3f}, {du['hi']:+.3f}]",
                         "Macro-F1": v["metrics"]["macro_f1_3class"], "Δ accuracy [95% CI]": f"{da['diff']:+.3f} [{da['lo']:+.3f}, {da['hi']:+.3f}]",
                         "Learned logreg macro-F1": v["learned_logreg_macro_f1"]})
    out.append(table(rows))
    return "".join(out)


def image(s) -> str:
    out = [f"## 4. Image captioning\n\nRun `{s['run_id']}`; {s['n_images']} images. {s['scope_note']}\n\n"]
    out.append(table([{"Captioner": v["display"], "Model": v["model"], "OK": v["n_ok"], "Failed": v["n_failed"], "CLIP sim": ci(v["clip_similarity"]),
                       "Words": v["mean_words"], "Detail coverage": ci(v["detail_coverage"]), "Flagged biased": v["flagged_rate"],
                       "Gendered terms (person imgs)": v["gender_term_rate_person_images"],
                       "Unverifiable attrs": v["unsupported_attribute_rate"], "s/image": v["ms_per_image"] / 1000}
                      for v in s["captioners"].values()]))
    rc = s["reference_captions"]
    out.append(f"\nCOCO human reference captions: CLIP {ci(rc['clip_similarity'])}; gendered term in >=1 reference on "
               f"{f(rc['gender_term_rate_person_images_any_ref'])} of person images. Agreement: {s.get('agreement')}.\n")
    return "".join(out)


def registry_table() -> str:
    v = model_verification() or {}
    rows = [{"Role": r.get("role"), "Model": r.get("model_id"), "License (card)": r.get("license") or "none on card",
             "Revision": (r.get("sha") or "")[:10], "Weights MB": r.get("weights_mb"), "Load s": r.get("load_s"),
             "Latency ms": r.get("latency_ms"), "Peak RAM MB": r.get("peak_mb"), "Status": r.get("status"),
             "Sample output": json.dumps(r.get("sample_output"))[:90] if r.get("sample_output") is not None else r.get("error", "")[:90]}
            for r in v.values()]
    return table(rows)


def main():
    parts = [f"# Results\n\nAuto-generated by `scripts/generate_results_md.py` on {datetime.now():%Y-%m-%d %H:%M} from stored runs in "
             "`outputs/runs/`. Do not edit by hand. Interpretation lives in `docs/RESEARCH_REPORT.md`.\n\n"]
    for kind, fn, name, render in [("benchmark_detection", "detection_summary.json", "detection", detection),
                                   ("benchmark_mitigation", "mitigation_summary.json", "mitigation", mitigation),
                                   ("benchmark_gate", "gate_summary.json", "gate", gate),
                                   ("benchmark_image", "image_summary.json", "image", image)]:
        s = latest_summary(kind, fn)
        parts.append(render(s) + "\n" if s else f"## {name}\n\n_No completed run._\n\n")
    (ROOT / "docs" / "RESULTS.md").write_text("".join(parts), encoding="utf-8")
    reg = ROOT / "docs" / "MODEL_CARD_REGISTRY.md"
    if reg.exists():
        txt = reg.read_text(encoding="utf-8")
        a, b = "<!-- MEASURED:START -->", "<!-- MEASURED:END -->"
        if a in txt and b in txt:
            txt = txt.split(a)[0] + a + "\n" + registry_table() + b + txt.split(b)[1]
            reg.write_text(txt, encoding="utf-8")
    print("wrote docs/RESULTS.md")


if __name__ == "__main__":
    main()
