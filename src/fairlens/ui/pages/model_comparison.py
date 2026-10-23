import streamlit as st

from ...experiments import jobs
from ...reporting import charts
from .. import components as C
from ..common import theme
from ..results import latest_summary

SRC_LABEL = {"babe": "BABE (media bias)", "stereoset_inter": "StereoSet inter", "stereoset_intra": "StereoSet intra",
             "crows_pairs": "CrowS-Pairs"}


def run_button(job: str, label: str, key: str):
    r = jobs.running(job)
    if r:
        st.info(f"{label} is running (job {r['id']}). Progress is on Experiments & Exports.")
    elif st.button(label, type="primary", key=key, icon=":material/play_arrow:"):
        jobs.start(job)
        st.success("Started in the background. Refresh this page when it finishes.")


def render():
    C.page_header("Model Comparison", "Every detector is evaluated on the same held-out test split with the same metrics and "
                                      "threshold (0.5).", "Detection benchmark")
    s = latest_summary("benchmark_detection", "detection_summary.json")
    if not s:
        C.render(C.empty_state("No detection benchmark yet", "Run the benchmark to fill this page with real results."))
        run_button("detection", "Run detection benchmark", "mc_run")
        return
    ok = {k: v for k, v in s["detectors"].items() if v.get("status") == "ok"}
    st.caption(f"Run {s['run_id']} · test n = {s['n_test']} · finished {s.get('_finished_at') or '-'}")
    names = [v["display"] for v in ok.values()]
    t1, t2, t3, t4 = st.tabs(["Performance Metrics", "Runtime & Resources", "Calibration", "Confusion Matrices"])
    with t1:
        a, b = st.columns([1.2, 1])
        with a:
            f1 = [v["metrics"]["macro_f1"] for v in ok.values()]
            err = [(v["metrics"]["macro_f1"] - v["ci"]["macro_f1"]["lo"], v["ci"]["macro_f1"]["hi"] - v["metrics"]["macro_f1"]) for v in ok.values()]
            st.markdown("##### Macro-F1 per detector (95% bootstrap CI)")
            st.plotly_chart(charts.bar(names, f1, theme(), [charts.DETECTOR_COLORS.get(n, "#6D4AFF") for n in names], err, [0, 1]),
                            use_container_width=True, config={"displayModeBar": False})
        with b:
            st.markdown("##### ROC curves")
            roc = {v["display"]: (v["curves"]["roc"]["fpr"], v["curves"]["roc"]["tpr"]) for v in ok.values()}
            st.plotly_chart(charts.lines(roc, theme(), 300, "False positive rate", "True positive rate", diagonal=False),
                            use_container_width=True, config={"displayModeBar": False})
        rows = [{"Detector": v["display"], "Precision": v["metrics"]["precision"], "Recall": v["metrics"]["recall"],
                 "F1": v["metrics"]["f1"], "Macro-F1": v["metrics"]["macro_f1"], "ROC-AUC": v["metrics"].get("roc_auc"),
                 "PR-AUC": v["metrics"].get("pr_auc"), "FPR": v["metrics"]["fpr"], "ms / sample": v["ms_per_sample"],
                 "Model MB": v.get("model_mb")} for v in ok.values()]
        C.render(C.html_table(rows, best={"Precision": "max", "Recall": "max", "F1": "max", "Macro-F1": "max", "ROC-AUC": "max",
                                          "PR-AUC": "max", "FPR": "min", "ms / sample": "min"}, fmt={"ms / sample": ".1f"}))
        st.markdown("##### Macro-F1 by source dataset")
        st.caption("Sources measure different things (media framing vs. stereotype minimal pairs) and are not interchangeable.")
        srcs = sorted({src for v in ok.values() for src in v["by_source"]})
        st.plotly_chart(charts.grouped_bar([SRC_LABEL.get(x, x) for x in srcs],
                                           {v["display"]: [v["by_source"][x]["macro_f1"] for x in srcs] for v in ok.values()},
                                           theme(), yrange=[0, 1], colors=charts.DETECTOR_COLORS),
                        use_container_width=True, config={"displayModeBar": False})
        if "bart_mnli" in ok and ok["bart_mnli"].get("template_selection"):
            ts = ok["bart_mnli"]["template_selection"]
            with st.expander(f"BART-MNLI hypothesis templates (chosen on validation: {ts['chosen']})"):
                C.render(C.html_table([{"Template": k, "Hypotheses": " | ".join(v["hypotheses"].values()), "Val macro-F1": v["macro_f1"],
                                        "Val AUC": v["roc_auc"]} for k, v in ts["candidates"].items()], best={"Val macro-F1": "max"}))
    with t2:
        a, b = st.columns(2)
        ms = [v["ms_per_sample"] for v in ok.values()]
        a.markdown("##### Inference time per sample (ms, CPU, batch 16)")
        a.plotly_chart(charts.bar(names, ms, theme(), [charts.DETECTOR_COLORS.get(n) for n in names], fmt=".1f"),
                       use_container_width=True, config={"displayModeBar": False})
        mb = [v.get("model_mb") or 0 for v in ok.values()]
        b.markdown("##### Model weights in memory (MB)")
        b.plotly_chart(charts.bar(names, mb, theme(), [charts.DETECTOR_COLORS.get(n) for n in names], fmt=".0f"),
                       use_container_width=True, config={"displayModeBar": False})
        C.render(C.html_table([{"Detector": v["display"], "ms / sample": v["ms_per_sample"], "Load time (s)": v["load_s"],
                                "Model MB": v.get("model_mb"), "RSS increase (MB)": v.get("rss_delta_mb")} for v in ok.values()],
                              best={"ms / sample": "min", "Model MB": "min"}, fmt={"ms / sample": ".1f", "Load time (s)": ".1f"}))
        st.caption("Timing on a fixed 100-sentence subset of the test split; machine details in docs/ENVIRONMENT.md.")
    with t3:
        a, b = st.columns([1.3, 1])
        rel = {}
        for v in ok.values():
            pts = [(r["conf"], r["acc"]) for r in v["metrics"]["reliability"] if r["n"]]
            rel[v["display"]] = ([p[0] for p in pts], [p[1] for p in pts])
        a.markdown("##### Reliability diagram")
        a.plotly_chart(charts.lines(rel, theme(), 340, "Predicted bias probability", "Observed share biased", diagonal=True),
                       use_container_width=True, config={"displayModeBar": False})
        b.markdown("##### Calibration error")
        b.write("")
        C.render(C.html_table([{"Detector": v["display"], "ECE": v["metrics"]["ece"], "Brier": v["metrics"]["brier"]} for v in ok.values()],
                              best={"ECE": "min", "Brier": "min"}))
        b.caption("ECE: 10 equal-width bins. Zero-shot and rule scores are not trained probabilities, so poor calibration is expected.")
    with t4:
        cols = st.columns(min(3, len(ok)))
        for i, v in enumerate(ok.values()):
            cols[i % len(cols)].plotly_chart(charts.confusion(v["metrics"]["confusion"], theme=theme(), title=v["display"]),
                                             use_container_width=True, config={"displayModeBar": False})
