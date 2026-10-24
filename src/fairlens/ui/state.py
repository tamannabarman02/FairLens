"""Session state and cached model resources for the dashboard."""
from __future__ import annotations

import streamlit as st

from . import smoke_stubs

DEFAULTS = {
    "theme": "light", "lite": False, "text_input": "", "analysis": None, "mitigation": None,
    "selected_candidate": None, "image_result": None, "image_bytes": None, "image_name": None,
    "pending_example": None, "review_step": 0, "review_source": None, "review_warnings": [], "review_units": None, "review_unit": 0,
}


def init() -> None:
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v)


@st.cache_resource(show_spinner=False)
def detectors(lite: bool):
    from ..detection.registry import build_detectors

    if smoke_stubs.enabled():
        return smoke_stubs.detectors()
    return build_detectors(lite=lite)


@st.cache_resource(show_spinner=False)
def rewriters(lite: bool):
    from ..mitigation.llm_adapters import build_rewriters

    if smoke_stubs.enabled():
        return smoke_stubs.rewriters()
    return build_rewriters(lite=lite)


@st.cache_resource(show_spinner=False)
def suite():
    from ..evaluation.preservation import PreservationSuite

    if smoke_stubs.enabled():
        return smoke_stubs.OverlapSuite()
    return PreservationSuite()


@st.cache_resource(show_spinner=False)
def captioners(fallback: bool = False):
    from ..multimodal.captioners import build_captioners

    if smoke_stubs.enabled():
        return smoke_stubs.captioners()
    return build_captioners(use_fallback_for_florence=fallback)


@st.cache_resource(show_spinner=False)
def aligner():
    from ..multimodal.caption_evaluation import ClipAlignment

    if smoke_stubs.enabled():
        return smoke_stubs.ConstantAligner()
    return ClipAlignment()


def gate_config():
    from ..reliability_gate.rules import GateConfig

    return GateConfig.from_config(**tuned_gate_overrides())


def tuned_gate_overrides() -> dict:
    """Use the thresholds tuned on the validation split by the latest gate benchmark, if one exists."""
    from .results import latest_summary

    s = latest_summary("benchmark_gate", "gate_summary.json")
    return dict(s["tuned_rules"]) if s and s.get("tuned_rules") else {}


def run_text_analysis(text: str, progress=None):
    from ..pipeline import analyze_text

    return analyze_text(text, detectors(st.session_state.lite), progress=progress).to_dict()


def run_mitigation(text: str, progress=None):
    from ..pipeline import mitigate_text

    lite = st.session_state.lite
    return mitigate_text(text, detectors(lite), rewriters(lite), suite(), gate_config(), progress=progress).to_dict()


def run_image(image, references=None, progress=None):
    from ..pipeline import analyze_image

    return analyze_image(image, captioners(), detectors(st.session_state.lite), aligner(), references, progress=progress).to_dict()
