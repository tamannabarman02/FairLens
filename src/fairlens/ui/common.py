"""Shared page helpers: navigation, running the pipeline with progress, example inputs."""
from __future__ import annotations

import json

import streamlit as st

from ..config import PROJECT_ROOT
from . import components as C
from . import state

STEPS = ["Bias Detection", "Mitigation", "Reliability Check", "Final Result"]


def go(title: str) -> None:
    st.switch_page(st.session_state["_pages"][title])


def theme() -> str:
    return st.session_state.theme


def examples() -> dict[str, str]:
    p = PROJECT_ROOT / "datasets" / "synthetic_demo" / "examples.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def current_step() -> int:
    if st.session_state.mitigation:
        return 4 if st.session_state.selected_candidate else 3
    if st.session_state.analysis:
        return 1
    return 0


def run_analysis(text: str, with_mitigation: bool) -> bool:
    """Run detection (and optionally mitigation + gate) with a live status panel. Returns success."""
    st.session_state.selected_candidate = None
    with st.status("Running FairLens pipeline...", expanded=True) as status:
        try:
            status.write("Loading detectors (first run downloads/loads models; later runs are cached)")
            st.session_state.analysis = state.run_text_analysis(text, progress=lambda s, m: status.write(m))
            st.session_state.mitigation = None
            if with_mitigation:
                status.update(label="Generating rewrites and checking reliability...")
                st.session_state.mitigation = state.run_mitigation(text, progress=lambda s, m: status.write(m))
                st.session_state.selected_candidate = None
            status.update(label="Pipeline complete", state="complete", expanded=False)
            return True
        except Exception as exc:  # noqa: BLE001
            status.update(label="Pipeline failed", state="error", expanded=True)
            st.error(f"{type(exc).__name__}: {exc}. If a model failed to load, try Lite mode or see About → Limitations.")
            return False


def require_analysis(action_label: str = "Go to Text Analysis") -> bool:
    if st.session_state.analysis:
        return True
    C.render(C.empty_state("No text analysed yet", "Analyse a text first. Every value on this page comes from a live run."))
    st.write("")
    if st.button(action_label, type="primary"):
        go("Text Analysis")
    return False


def fmt_pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v * 100:.0f}%"


def fmt(v: float | None, d: int = 2) -> str:
    return "n/a" if v is None else f"{v:.{d}f}"
