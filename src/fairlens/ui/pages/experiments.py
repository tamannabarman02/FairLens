import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from ...config import PROJECT_ROOT
from ...experiments import jobs
from ...reporting import export
from .. import components as C
from ..results import COMPLETED, local_time, manifest, provenance_pairs, run_dir, run_status, runs, status_note

JOB_LABELS = {"train": "Train BERT + DeBERTa detectors", "detection": "Detection benchmark", "mitigation": "Mitigation benchmark",
              "gate": "Gate benchmark + ablations", "image": "Image-caption experiment", "use_cases": "Use-case demo runs",
              "results_md": "Regenerate docs/RESULTS.md"}
SUMMARY_FILE = {"benchmark_detection": "detection_summary.json", "benchmark_mitigation": "mitigation_summary.json",
                "benchmark_gate": "gate_summary.json", "benchmark_image": "image_summary.json"}


def render():
    C.page_header("Experiments & Exports", "Run history with full provenance, one-click reproduction and report exports.", "More")
    t1, t2, t3 = st.tabs(["Run history & provenance", "Launch experiments", "Exports"])
    all_runs = runs()
    with t1:
        if not all_runs:
            C.render(C.empty_state("No runs recorded", "Launch an experiment in the next tab."))
        else:
            df = pd.DataFrame([{"Run ID": r["run_id"], "Kind": r["kind"], "Status": C.status_badge(run_status(r)),
                                "Manifest status": r["status"], "Started": local_time(r["started_at"]),
                                "Runtime (s)": "-" if r["runtime_s"] is None or r["runtime_s"] != r["runtime_s"] else f"{r['runtime_s']:.1f}",
                                "Seed": r["seed"] if r["seed"] is not None else "-", "Git": r["git_commit"] or "-"} for r in all_runs])
            kinds = st.multiselect("Filter kinds", sorted(df.Kind.unique()), default=[], placeholder="All kinds")
            view = df[df.Kind.isin(kinds)] if kinds else df
            C.render(C.html_table(view.astype(object).where(view.notna(), "-").to_dict("records"), raw={"Status"}))
            C.render(C.caption("Only runs labelled 'Completed run' feed the result pages. Runs still marked 'running' with no "
                               "active process were stopped and are labelled 'Failed run'."))
            view = view.rename(columns={"Run ID": "run_id"})
            rid = st.selectbox("Inspect run", view.run_id.tolist())
            run = next(r for r in all_runs if r["run_id"] == rid)
            m = manifest(run)
            a, b = st.columns([1, 1])
            with a:
                hw = m.get("hardware") or {}
                pairs = provenance_pairs(run) + [
                    ("Command", m.get("command") or "-"), ("Started", local_time(m.get("started_at"))),
                    ("Runtime", f"{m['runtime_s']:.1f} s" if m.get("runtime_s") else "-"),
                    ("Device", str(m.get("device") or ("cuda" if hw.get("cuda") else "cpu"))),
                    ("Hardware", f"{hw.get('cpu', '?')}, {hw.get('ram_total_gb', '?')} GB RAM" if hw else "-")]
                C.render(C.provenance("Provenance", run_status(run), pairs, C.esc(status_note(run))))
            with b:
                C.section("Config & metrics")
                st.json({"config": m.get("config"), "metrics": m.get("metrics")}, expanded=1)
                cmd = m.get("command")
                if cmd and st.button("Reproduce this run", icon=":material/replay:", type="primary"):
                    parts = cmd.split()
                    script = next((p for p in Path(PROJECT_ROOT / "scripts").glob(parts[0])), None)
                    if script:
                        rec = jobs.start(m["kind"], [str(script.relative_to(PROJECT_ROOT)), *parts[1:]])
                        st.success(f"Started job {rec['id']} with the recorded command and the same seed/config.")
                    else:
                        st.error(f"Script {parts[0]} not found.")
    with t2:
        st.caption("Jobs run as background processes; you can keep using the dashboard. One large model is loaded at a time.")
        for name, label in JOB_LABELS.items():
            a, b, c = st.columns([3, 1, 1])
            r = jobs.running(name)
            a.markdown(f"**{label}**  \n<span class='fl-mono'>{' '.join(jobs.SCRIPTS[name])}</span>", unsafe_allow_html=True)
            if r:
                b.markdown(C.badge("running", "blue"), unsafe_allow_html=True)
                if c.button("Cancel", key=f"cancel_{name}"):
                    jobs.cancel(r)
                    st.rerun()
            elif c.button("Run", key=f"run_{name}", type="primary"):
                jobs.start(name)
                st.rerun()
        st.divider()
        recent = jobs.all_jobs()[:8]
        if recent:
            C.section("Recent jobs")
            for j in recent:
                with st.expander(f"{j['id']} · {j['status']}"):
                    st.code(jobs.log_tail(j, 25) or "(no output yet)", language=None)
            if any(j["status"] == "running" for j in recent) and st.button("Refresh progress", icon=":material/refresh:", on_click="ignore"):
                st.rerun()
    with t3:
        done = [r for r in all_runs if run_status(r) == COMPLETED and r["kind"] in SUMMARY_FILE]
        if not done:
            C.render(C.empty_state("Nothing to export yet", "Reports are generated from completed benchmark runs. Start one in "
                                   "'Launch experiments'. To export a single analysis, use FairLens Review or Text Analysis."))
            return
        rid = st.selectbox("Run to export", [r["run_id"] for r in done], key="exp_run")
        run = next(r for r in done if r["run_id"] == rid)
        p = run_dir(run) / SUMMARY_FILE[run["kind"]]
        summary = json.loads(p.read_text(encoding="utf-8"))
        secs = export.sections_for_run(run["kind"], summary)
        meta = {"run": rid, "kind": run["kind"], "git": run["git_commit"], "exported": time.strftime("%Y-%m-%d %H:%M")}
        title = f"FairLens report - {run['kind'].replace('benchmark_', '')}"
        a, b, c, d = st.columns(4)
        a.download_button("PDF", export.to_pdf(title, secs, meta), f"{rid}.pdf", "application/pdf", icon=":material/picture_as_pdf:", on_click="ignore")
        b.download_button("HTML", export.to_html(title, secs, meta), f"{rid}.html", "text/html", icon=":material/html:", on_click="ignore")
        c.download_button("JSON", export.to_json({"manifest": manifest(run), "summary": summary}), f"{rid}.json", "application/json",
                          icon=":material/data_object:", on_click="ignore")
        tables = [b_ for _, blocks in secs for b_ in blocks if isinstance(b_, list)]
        d.download_button("CSV", export.to_csv(tables[0] if tables else []), f"{rid}.csv", "text/csv", icon=":material/table:", on_click="ignore")
        with st.expander("Preview"):
            st.components.v1.html(export.to_html(title, secs, meta).decode(), height=500, scrolling=True)
