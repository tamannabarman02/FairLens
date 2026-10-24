import streamlit as st

from ...reporting import charts
from .. import components as C
from ..common import fmt, fmt_pct, theme
from ..results import latest_with_manifest
from .model_comparison import failed_note, run_button

POLICY_LABEL = {"fairlens_gate_tuned": "FairLens Gate (val-tuned)", "fairlens_gate_default": "FairLens Gate (default rules)",
                "learned_logreg": "Learned gate (logistic)", "learned_tree": "Learned gate (tree)", "lowest_bias": "Lowest bias score",
                "highest_similarity": "Highest similarity", "random_mean": "Random (mean of 5 seeds)",
                "fairlens_gate": "FairLens Gate (no retry)", "fairlens_gate_retry": "FairLens Gate + retry"}


def ci_txt(d):
    return f"{d['mean']:.2f} [{d['lo']:.2f}, {d['hi']:.2f}]" if d and d.get("lo") is not None else fmt(d.get("mean") if d else None)


def render():
    C.page_header("Benchmarks & Results", "Headline results from completed runs. Every number is read from run files in "
                                          "<span class='fl-mono'>outputs/runs/</span>.", "Research results")
    det, det_m = latest_with_manifest("benchmark_detection", "detection_summary.json")
    mit, mit_m = latest_with_manifest("benchmark_mitigation", "mitigation_summary.json")
    gate, gate_m = latest_with_manifest("benchmark_gate", "gate_summary.json")
    img, img_m = latest_with_manifest("benchmark_image", "image_summary.json")

    tiles = []
    if det:
        ok = {k: v for k, v in det["detectors"].items() if v.get("status") == "ok"}
        k, v = max(ok.items(), key=lambda kv: kv[1]["metrics"]["macro_f1"])
        tiles.append(C.metric_tile("Best detector", f"{v['metrics']['macro_f1']:.3f}", f"macro-F1 · {v['display']}", "accent"))
    if gate:
        g = gate["candidate_level"]["rules_val_tuned"]
        tiles.append(C.metric_tile("Gate unsafe-accept rate", fmt_pct(g["unsafe_accept_rate"]), "unacceptable rewrites accepted (controlled set)",
                                   "green" if g["unsafe_accept_rate"] < 0.1 else "amber"))
        sg, sl = gate["selection"]["fairlens_gate_tuned"], gate["selection"]["lowest_bias"]
        tiles.append(C.metric_tile("Harmful picks: Gate vs lowest-bias", f"{sg['harmful_selection']['mean']:.0%} / {sl['harmful_selection']['mean']:.0%}",
                                   "share of inputs", "green"))
    if mit:
        dist = mit["gate_final_distribution"]
        n = sum(dist.values())
        tiles.append(C.metric_tile("Real LLM rewrites accepted", fmt_pct(dist.get("ACCEPT", 0) / n if n else None),
                                   f"{n} inputs, after strict retries", "blue"))
    if img:
        cs = list(img["captioners"].values())
        tiles.append(C.metric_tile("Captions flagged as biased", " / ".join(fmt_pct(c["flagged_rate"]) for c in cs),
                                   " / ".join(c["display"] for c in cs), "amber"))
    if not tiles:
        C.render(C.empty_state("No completed runs yet", "Run the experiments below; results appear here automatically."))
    else:
        cols = st.columns(len(tiles))
        for c, t in zip(cols, tiles):
            c.markdown(t, unsafe_allow_html=True)
    st.write("")

    if mit:
        a, b = st.columns([1.4, 1])
        rw = mit["rewriters"]
        order = [k for k in ["original", "rule_based", "flan_t5", "qwen"] if k in rw]
        lab = {"original": "Original", "rule_based": "Rule-based", "flan_t5": "FLAN-T5", "qwen": "Qwen"}
        a.markdown("##### Bias score on biased inputs: before vs after rewriting")
        before = [rw["original"]["mean_bias_after_biased"]] * len(order)
        after = [rw[k]["mean_bias_after_biased"] for k in order]
        a.plotly_chart(charts.grouped_bar([lab[k] for k in order], {"Original text": before, "After rewrite": after}, theme(),
                                          yrange=[0, 1], colors={"Original text": "#B9B7CC", "After rewrite": "#6D4AFF"}),
                       use_container_width=True, config={"displayModeBar": False})
        b.markdown("##### Gate decisions (real rewrites, per input)")
        dist = mit["gate_final_distribution"]
        ks = [k for k in ["ACCEPT", "RETRY", "REJECT", "HUMAN_REVIEW"] if k in dist]
        b.plotly_chart(charts.donut([C.DECISION_LABEL[k] for k in ks], [dist[k] for k in ks], theme(),
                                    [charts.DECISION_COLORS[k] for k in ks]), use_container_width=True, config={"displayModeBar": False})

    t1, t2, t3, t4 = st.tabs(["Text Detection", "Mitigation Results", "Reliability Gate", "Image Analysis"])
    with t1:
        failed_note("benchmark_detection")
        if not det:
            run_button("detection", "Run detection benchmark", "b_det")
        else:
            C.render(C.provenance(det, det_m))
            ok = {k: v for k, v in det["detectors"].items() if v.get("status") == "ok"}
            C.render(C.html_table([{"Detector": v["display"], "Macro-F1": v["metrics"]["macro_f1"],
                                    "F1 [95% CI]": f"{v['ci']['f1']['value']:.3f} [{v['ci']['f1']['lo']:.3f}, {v['ci']['f1']['hi']:.3f}]",
                                    "ROC-AUC": v["metrics"].get("roc_auc"), "ECE": v["metrics"]["ece"]} for v in ok.values()],
                                  best={"Macro-F1": "max", "ROC-AUC": "max", "ECE": "min"}))
            if "ensemble" in det:
                st.caption(f"Mean-score ensemble of {', '.join(det['ensemble']['members'])}: macro-F1 {det['ensemble']['metrics']['macro_f1']:.3f}")
            st.markdown("##### Probes (not training labels)")
            rows = []
            for v in ok.values():
                p = v.get("probes")
                if not p:
                    continue
                rows.append({"Detector": v["display"], "WinoBias FPR pro": p["winobias"]["fpr_pro"], "WinoBias FPR anti": p["winobias"]["fpr_anti"],
                             "WinoBias gap": p["winobias"]["mean_gap_pro_minus_anti"]["mean"],
                             "CrowS stereo pref.": p["crows_pairs"]["pref_more_stereo"]["mean"],
                             "StereoSet stereo>anti": p["stereoset"]["pref_stereo_over_anti"]["mean"],
                             "StereoSet unrelated flagged": p["stereoset"]["flag_rate_unrelated"]})
            if rows:
                C.render(C.html_table(rows, best={"WinoBias FPR pro": "min", "WinoBias FPR anti": "min", "StereoSet unrelated flagged": "min"}))
                st.caption("WinoBias sentences are neutral event descriptions: a good detector flags neither condition (low FPR) and shows "
                           "no pro/anti gap. Preference = share of pairs where the stereotypical sentence scores higher.")
    with t2:
        failed_note("benchmark_mitigation")
        if not mit:
            run_button("mitigation", "Run mitigation benchmark", "b_mit")
        else:
            C.render(C.provenance(mit, mit_m))
            st.caption(f"{mit['n_inputs']} inputs ({mit['n_biased_gold']} biased, {mit['n_neutral_gold']} neutral by gold label)")
            rows = []
            for k, v in mit["rewriters"].items():
                if k == "original":
                    continue
                rows.append({"Rewriter": {"rule_based": "Rule-based", "flan_t5": "FLAN-T5", "qwen": "Qwen"}.get(k, k),
                             "Bias reduction": v["mean_bias_reduction_biased"], "Similarity": v["semantic_similarity"],
                             "Entailment (both)": v["entailment_both"], "Fact retention": v["fact_retention"],
                             "Fact failures": v["fact_failure_rate"], "Fluency": v["fluency"],
                             "Changed neutral input": v["unnecessary_change_rate_neutral"], "Similarity to reference": v["ref_similarity"]})
            C.render(C.html_table(rows, best={"Bias reduction": "max", "Similarity": "max", "Fact retention": "max", "Fact failures": "min",
                                              "Fluency": "max", "Changed neutral input": "min", "Similarity to reference": "max"}))
            st.markdown("##### Selection policies on real rewrites")
            prow = [{"Policy": POLICY_LABEL.get(k, k), "Defer to human": v.get("defer_rate"), "Accepted with fact loss": v.get("accepted_fact_failure_rate"),
                     "Unnecessary change (neutral)": v.get("unnecessary_change_rate"), "Bias reduction": v.get("mean_bias_reduction_biased"),
                     "Biased left unchanged": v.get("biased_left_unchanged_rate"), "Similarity": v.get("mean_semantic_similarity")}
                    for k, v in mit["policies"].items()]
            C.render(C.html_table(prow, best={"Accepted with fact loss": "min", "Unnecessary change (neutral)": "min", "Bias reduction": "max"}))
            st.caption("No gold labels exist for LLM outputs, so these use automatic checks only; the controlled set (next tab) has gold labels.")
            r = mit["retry"]
            st.caption(f"Strict retries: {r['inputs_retried']} inputs retried, {r['accepted_after_retry']} accepted after retry.")
    with t3:
        failed_note("benchmark_gate")
        if not gate:
            run_button("gate", "Run gate benchmark + ablations", "b_gate")
        else:
            C.render(C.provenance(gate, gate_m))
            st.caption(f"{gate['n_inputs']} inputs × {len(gate['seeds'])} perturbation seeds · test-split gold: {gate['gold_distribution_test']}")
            st.markdown("##### Gate vs. simpler selection methods (input level, test split)")
            sel = gate["selection"]
            labels = [POLICY_LABEL.get(k, k) for k in sel]
            st.plotly_chart(charts.grouped_bar(labels, {"Acceptable choice": [v["success"]["mean"] for v in sel.values()],
                                                        "Harmful choice": [v["harmful_selection"]["mean"] for v in sel.values()],
                                                        "Deferred to human": [v["defer"]["mean"] for v in sel.values()]}, theme(),
                                               yrange=[0, 1], colors={"Acceptable choice": "#1F9D5C", "Harmful choice": "#E5484D",
                                                                      "Deferred to human": "#2F6FEB"}),
                            use_container_width=True, config={"displayModeBar": False})
            C.render(C.html_table([{"Policy": POLICY_LABEL.get(k, k), "Acceptable choice [95% CI]": ci_txt(v["success"]),
                                    "Harmful choice [95% CI]": ci_txt(v["harmful_selection"]), "Deferred": v["defer"]["mean"],
                                    "Unnecessary change (neutral)": v["unnecessary_change_neutral"]["mean"]} for k, v in sel.items()]))
            st.markdown("##### Candidate-level decision quality")
            cl = gate["candidate_level"]
            C.render(C.html_table([{"Variant": k.replace("_", " "), "3-class macro-F1": v["macro_f1_3class"], "Accept precision": v["accept_precision"],
                                    "Accept recall": v["accept_recall"], "Unsafe accept": v["unsafe_accept_rate"],
                                    "Human review": v["human_review_rate"]} for k, v in cl.items()],
                                  best={"3-class macro-F1": "max", "Accept precision": "max", "Unsafe accept": "min"}))
            st.caption(f"Val-tuned thresholds: {gate['tuned_rules']}. 'oracle input' variants use the gold bias label of the input instead of the detectors.")
            cr = cl["rules_val_tuned"].get("catch_rate_by_perturbation", {})
            if cr:
                st.markdown("##### What the Gate catches (share of each perturbation not accepted)")
                st.plotly_chart(charts.hbar([k.replace("_", " ") for k in cr], list(cr.values()), theme(), "#6D4AFF", 320),
                                use_container_width=True, config={"displayModeBar": False})
            st.markdown("##### Ablations (val-tuned rule gate without one signal group)")
            ab = gate["ablations"]
            rows = [{"Gate": "Full gate", "Unsafe accept": ab["full"]["unsafe_accept_rate"], "Δ unsafe [95% CI]": "-",
                     "3-class macro-F1": ab["full"]["macro_f1_3class"], "Δ accuracy [95% CI]": "-", "Learned (logreg) macro-F1": None}]
            for k, v in ab.items():
                if k == "full":
                    continue
                du, da = v["delta_unsafe_accept"], v["delta_accuracy"]
                rows.append({"Gate": k.replace("no_", "without "), "Unsafe accept": v["metrics"]["unsafe_accept_rate"],
                             "Δ unsafe [95% CI]": f"{du['diff']:+.3f} [{du['lo']:+.3f}, {du['hi']:+.3f}]",
                             "3-class macro-F1": v["metrics"]["macro_f1_3class"],
                             "Δ accuracy [95% CI]": f"{da['diff']:+.3f} [{da['lo']:+.3f}, {da['hi']:+.3f}]",
                             "Learned (logreg) macro-F1": v["learned_logreg_macro_f1"]})
            C.render(C.html_table(rows))
            st.caption("A signal adds measurable value if removing it raises unsafe accepts with a CI excluding 0. Signals whose CI "
                       "includes 0 showed no measurable value on this set; that is reported as a negative result.")
    with t4:
        failed_note("benchmark_image")
        if not img:
            run_button("image", "Run image experiment", "b_img")
        else:
            C.render(C.provenance(img, img_m))
            st.caption(f"{img['n_images']} images. {img['scope_note']}")
            rows = [{"Captioner": v["display"], "CLIP similarity [95% CI]": ci_txt(v["clip_similarity"]), "Words": v["mean_words"],
                     "Detail coverage": v["detail_coverage"]["mean"], "Flagged biased": v["flagged_rate"],
                     "Gendered terms (person images)": v["gender_term_rate_person_images"], "Unverifiable attributes": v["unsupported_attribute_rate"],
                     "s / image": v["ms_per_image"] / 1000} for v in img["captioners"].values()]
            C.render(C.html_table(rows, best={"Detail coverage": "max", "Flagged biased": "min"}, fmt={"s / image": ".1f", "Words": ".1f"}))
            rc = img["reference_captions"]
            st.caption(f"Human COCO references: CLIP {ci_txt(rc['clip_similarity'])}; at least one reference uses a gendered term on "
                       f"{fmt_pct(rc['gender_term_rate_person_images_any_ref'])} of person images.")
            if img.get("agreement"):
                ag = img["agreement"]
                st.caption(f"Captioner agreement on flags: {ag}")
