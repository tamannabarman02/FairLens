"""FairLens dashboard (Streamlit entry point).   Run:  streamlit run app.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="FairLens", page_icon=str(ROOT / "assets" / "logo.svg"), layout="wide",
                   initial_sidebar_state="expanded")

from fairlens.ui import components as C  # noqa: E402
from fairlens.ui import state  # noqa: E402
from fairlens.ui.pages import (about, benchmarks, error_analysis, experiments, gate, home, image_analysis,  # noqa: E402
                               mitigation, model_comparison, review, text_analysis, use_cases)

state.init()
C.inject_css(st.session_state.theme)

WORKFLOW = [
    st.Page(home.render, title="Home", icon=":material/home:", url_path="home", default=True),
    st.Page(review.render, title="FairLens Review", icon=":material/fact_check:", url_path="review"),
    st.Page(text_analysis.render, title="Text Analysis", icon=":material/notes:", url_path="text"),
    st.Page(mitigation.render, title="Mitigation", icon=":material/auto_fix_high:", url_path="mitigation"),
    st.Page(gate.render, title="Reliability Gate", icon=":material/verified_user:", url_path="gate"),
    st.Page(image_analysis.render, title="Image Analysis", icon=":material/image:", url_path="image"),
]
RESEARCH = [
    st.Page(benchmarks.render, title="Benchmarks", icon=":material/bar_chart:", url_path="benchmarks"),
    st.Page(model_comparison.render, title="Model Comparison", icon=":material/compare_arrows:", url_path="models"),
    st.Page(error_analysis.render, title="Error Analysis", icon=":material/bug_report:", url_path="errors"),
    st.Page(experiments.render, title="Experiments & Exports", icon=":material/science:", url_path="experiments"),
]
MORE = [
    st.Page(use_cases.render, title="Use Cases", icon=":material/cases:", url_path="use-cases"),
    st.Page(about.render, title="About", icon=":material/info:", url_path="about"),
]
ALL = WORKFLOW + RESEARCH + MORE
nav = st.navigation(ALL, position="hidden")
st.session_state["_pages"] = {p.title: p for p in ALL}


@st.cache_resource(show_spinner=False)
def compute_device() -> str:
    from fairlens.hardware import device_label

    return device_label()


@st.cache_resource(show_spinner=False)
def hardware_profile() -> str:
    from fairlens.hardware import select_profile

    return select_profile()


with st.sidebar:
    C.render('<div class="fl-logo"><div class="fl-logo-mark">' + C.icon("lens", 20, "#fff", 2.4) +
             '</div><div><div class="fl-logo-name">FairLens</div><div class="fl-logo-sub">Detect · Reduce · Verify</div></div></div>')
    for group, pages in (("Workflow", WORKFLOW), ("Research", RESEARCH), ("More", MORE)):
        C.render(f'<div class="fl-nav-group">{group}</div>')
        for p in pages:
            st.page_link(p)
    st.divider()
    st.toggle("Lite mode", key="lite", help="Smaller, faster models for live analysis: DistilBART-MNLI instead of BART-large-MNLI "
                                            "and FLAN-T5-small instead of FLAN-T5-base. Uses less memory; scores can differ. "
                                            "Benchmarks always report the standard models.")
    C.render('<div class="fl-sidebar-note">' + ("On: smaller models, less memory, faster; results may differ slightly."
                                                if st.session_state.lite else
                                                "Off: standard models. Turn on if analysis is slow or runs out of memory.") + "</div>")
    from fairlens.models import manager  # noqa: E402

    res = manager().resident()
    st.caption("Models in memory: " + (", ".join(k.split(":", 1)[1].split("/")[-1] for k in res) if res else "none yet (loaded on first use)"))

top_l, top_r = st.columns([3, 2], vertical_alignment="center")
with top_l:
    C.render(f'<div class="fl-crumb">FairLens / {C.esc(nav.title)}</div>')
with top_r:
    b1, b2 = st.columns([1, 5], vertical_alignment="center")
    with b1:
        dark = st.session_state.theme == "dark"
        if st.button("", icon=":material/light_mode:" if dark else ":material/dark_mode:", key="theme_toggle",
                     help="Switch to light mode" if dark else "Switch to dark mode"):
            st.session_state.theme = "light" if dark else "dark"
            st.rerun()
    with b2:
        prof = "lite" if st.session_state.lite else hardware_profile()
        C.render(f'<span class="fl-chip" title="Compute device and hardware profile used for live analysis">'
                 f'<span class="fl-avatar">{C.icon("cpu", 14, "#fff", 2.2)}</span>{C.esc(compute_device())} · {C.esc(prof)}</span>')

nav.run()
