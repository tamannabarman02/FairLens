import json

import streamlit as st

from ...config import PROJECT_ROOT
from .. import components as C
from ..common import go
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
    st.write("")
    cases = json.loads(UC.read_text(encoding="utf-8"))["cases"]
    done = use_case_results()
    for i in range(0, len(cases), 2):
        cols = st.columns(2, gap="medium")
        for col, case in zip(cols, cases[i:i + 2]):
            with col:
                res = done.get(case["id"])
                body = (f'<div style="margin-bottom:6px">{C.result_label("illustrative")} '
                        f'{C.result_label("completed") if res else C.result_label("pending")}</div>'
                        f'<div class="fl-sub" style="font-size:12.5px">For: {C.esc(case["users"])}</div>'
                        f'<p style="margin-top:6px"><b>Problem.</b> {C.esc(case["problem"])}</p>'
                        f'<div class="fl-quote" style="margin-top:8px">{C.esc(case["input"])}</div>')
                if res and res.get("kind") == "text":
                    d = res["decision"]
                    body += (f'<div style="margin-top:10px">{C.badge("bias " + format(res["bias_score"], ".2f"), C.tone_for_score(res["bias_score"]))} '
                             f'{C.decision_badge(d["decision"])} {C.badge(d["reason_code"], "grey")}</div>'
                             f'<p style="margin-top:8px"><b>FairLens output:</b> {C.esc(d.get("chosen_text") or "no automatic rewrite - sent to human review")}</p>')
                elif res and res.get("kind") == "image":
                    body += "".join(f'<p style="margin-top:6px"><b>{C.esc(c["display"])}:</b> “{C.esc(c["text"] or c["error"])}” '
                                    f'{C.badge("bias " + format(c.get("bias_score", 0), ".2f"), C.tone_for_score(c.get("bias_score")))}</p>'
                                    for c in res["captions"])
                elif res and res.get("kind") == "benchmark":
                    body += f'<p style="margin-top:8px">{C.esc(res["summary"])}</p>'
                if res:
                    body += f'<div class="fl-sub" style="font-size:11.5px;margin-top:6px">Computed {C.esc(res["computed_at"])}</div>'
                C.render(f'<div class="fl-card tone-accent"><h4>{C.esc(case["title"])}</h4>{body}</div>')
                b1, b2 = st.columns(2)
                if case["kind"] == "benchmark":
                    if b1.button("Open Model Comparison", key=f"uc_{case['id']}"):
                        go("Model Comparison")
                elif b1.button("Run" if not res else "Re-run", key=f"uc_{case['id']}", type="primary", icon=":material/play_arrow:"):
                    with st.spinner("Running the full pipeline..."):
                        try:
                            run_case(case)
                            st.rerun()
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"{type(exc).__name__}: {exc}")
                if res and case["kind"] == "text" and b2.button("Open in Text Analysis", key=f"uc_open_{case['id']}"):
                    st.session_state.text_input = case["input"]
                    st.session_state.analysis = res["analysis"]
                    st.session_state.mitigation = res["mitigation"]
                    st.session_state.selected_candidate = None
                    go("Reliability Gate")
                st.write("")
