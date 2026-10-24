"""FairLens dashboard (Streamlit entry point).   Run:  streamlit run app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="FairLens", page_icon="assets/logo.svg", layout="wide", initial_sidebar_state="expanded")

from fairlens.ui import components as C  # noqa: E402
from fairlens.ui import state  # noqa: E402
from fairlens.ui.pages import (about, benchmarks, error_analysis, experiments, gate, home, image_analysis,  # noqa: E402
                               mitigation, model_comparison, review, text_analysis, use_cases)

state.init()
C.inject_css(st.session_state.theme)

MAIN = [
    st.Page(home.render, title="Home", icon=":material/home:", url_path="home", default=True),
    st.Page(review.render, title="FairLens Review", icon=":material/fact_check:", url_path="review"),
    st.Page(text_analysis.render, title="Text Analysis", icon=":material/notes:", url_path="text"),
    st.Page(mitigation.render, title="Mitigation", icon=":material/auto_fix_high:", url_path="mitigation"),
    st.Page(gate.render, title="Reliability Gate", icon=":material/verified_user:", url_path="gate"),
    st.Page(image_analysis.render, title="Image Analysis", icon=":material/image:", url_path="image"),
    st.Page(model_comparison.render, title="Model Comparison", icon=":material/compare_arrows:", url_path="models"),
    st.Page(benchmarks.render, title="Benchmarks", icon=":material/bar_chart:", url_path="benchmarks"),
    st.Page(about.render, title="About", icon=":material/info:", url_path="about"),
]
MORE = [
    st.Page(error_analysis.render, title="Error Analysis", icon=":material/bug_report:", url_path="errors"),
    st.Page(experiments.render, title="Experiments & Exports", icon=":material/science:", url_path="experiments"),
    st.Page(use_cases.render, title="Use Cases", icon=":material/cases:", url_path="use-cases"),
]
nav = st.navigation(MAIN + MORE, position="hidden")
st.session_state["_pages"] = {p.title: p for p in MAIN + MORE}

with st.sidebar:
    C.render('<div class="fl-logo"><div class="fl-logo-mark">' + C.icon("lens", 20, "#fff", 2.4) +
             '</div><div><div class="fl-logo-name">FairLens</div><div class="fl-logo-sub">Detect · Reduce · Verify</div></div></div>')
    for p in MAIN:
        st.page_link(p)
    C.render('<div class="fl-nav-group">More</div>')
    for p in MORE:
        st.page_link(p)
    st.divider()
    st.toggle("Lite mode", key="lite", help="Use smaller checkpoints (DistilBART-MNLI, FLAN-T5-small) for faster, "
                                            "lower-memory analysis. Benchmarks always report the standard models.")
    from fairlens.models import manager  # noqa: E402

    res = manager().resident()
    st.caption("Models in memory: " + (", ".join(k.split(":", 1)[1].split("/")[-1] for k in res) if res else "none yet (loaded on first use)"))

from fairlens.hardware import device, select_profile  # noqa: E402

top_l, top_r = st.columns([5, 2], vertical_alignment="center")
with top_l:
    C.render(f'<div class="fl-crumb">FairLens / {C.esc(nav.title)}</div>')
with top_r:
    b1, b2 = st.columns([1, 3], vertical_alignment="center")
    with b1:
        dark = st.session_state.theme == "dark"
        if st.button("", icon=":material/light_mode:" if dark else ":material/dark_mode:", key="theme_toggle",
                     help="Switch to light mode" if dark else "Switch to dark mode"):
            st.session_state.theme = "light" if dark else "dark"
            st.rerun()
    with b2:
        prof = "lite" if st.session_state.lite else select_profile()
        dev = device().upper()
        C.render(f'<span class="fl-chip" title="Compute device chosen at startup (CUDA, then MPS, then CPU; FAIRLENS_DEVICE overrides) '
                 f'and hardware profile"><span class="fl-avatar">FL</span>{dev} · {prof}</span>')

nav.run()
