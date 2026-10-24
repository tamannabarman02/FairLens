import json

import streamlit as st

from ...config import PROJECT_ROOT
from .. import components as C
from ..common import go, show_error
from ..results import latest_summary, use_case_results

UC = PROJECT_ROOT / "datasets" / "synthetic_demo" / "use_cases.json"


def run_case(case: dict) -> dict:
    from ...experiments.use_cases import run_use_case

    return run_use_case(case)


def render():
    C.page_header("Use Cases", "One-click demo scenarios. Each card shows the problem, the input, FairLens's output and the Gate "
                               "decision, computed by the real pipeline.", "Demo scenarios")
    C.render(C.note("<b>Review assistant, not a legal or compliance certification tool.</b> These are potential applications with "
                    "synthetic demo inputs, not claims of production validation. FairLens cannot guarantee unbiased output."))
    C.gap()
    cases = json.loads(UC.read_text(encoding="utf-8"))["cases"]
    done = use_case_results()
    for i in range(0, len(cases), 2):
        cols = st.columns(2, gap="medium")
        for col, case in zip(cols, cases[i:i + 2]):
            with col:
                res = done.get(case["id"])
                body = (f'<div class="fl-caption">For: {C.esc(case["users"])}</div>'
                        f'<p class="fl-p"><b>Problem.</b> {C.esc(case["problem"])}</p>'
                        f'<div class="fl-quote">{C.esc(case["input"])}</div>')
                if res and res.get("kind") == "text":
                    d = res["decision"]
                    body += (f'<div class="fl-row mt">{C.badge("bias " + format(res["bias_score"], ".2f"), C.tone_for_score(res["bias_score"]))} '
                             f'{C.decision_badge(d["decision"])} {C.badge(d["reason_code"], "grey")}</div>'
                             f'<p class="fl-p"><b>FairLens output:</b> {C.esc(d.get("chosen_text") or "no automatic rewrite - sent to human review")}</p>')
                elif res and res.get("kind") == "image":
                    body += "".join(f'<p class="fl-p"><b>{C.esc(c["display"])}:</b> “{C.esc(c["text"] or c["error"])}” '
                                    f'{C.badge("bias " + format(c.get("bias_score", 0), ".2f"), C.tone_for_score(c.get("bias_score")))}</p>'
                                    for c in res["captions"])
                elif res and res.get("kind") == "benchmark":
                    body += f'<p class="fl-p">{C.esc(res["summary"])}</p>'
                if res:
                    body += f'<div class="fl-caption">Computed {C.esc(res["computed_at"])}</div>'
                label = "Illustrative example" if res else "Pending"
                C.render(f'<div class="fl-card uc tone-accent"><div class="fl-row"><h4>{C.esc(case["title"])}</h4>{C.status_badge(label)}</div>{body}</div>')
                b1, b2 = st.columns([1, 1.9])
                if case["kind"] == "benchmark":
                    if st.button("Open Model Comparison", key=f"uc_{case['id']}"):
                        go("Model Comparison")
                elif b1.button("Run" if not res else "Re-run", key=f"uc_{case['id']}", type="primary", icon=":material/play_arrow:"):
                    with st.spinner("Running the full pipeline..."):
                        try:
                            run_case(case)
                            st.rerun()
                        except Exception as exc:  # noqa: BLE001
                            show_error(exc)
                if res and case["kind"] == "text" and b2.button("Open in Reliability Gate", key=f"uc_open_{case['id']}"):
                    st.session_state.text_input = case["input"]
                    st.session_state.analysis = res["analysis"]
                    st.session_state.mitigation = res["mitigation"]
                    st.session_state.selected_candidate = None
                    go("Reliability Gate")
                C.gap()
