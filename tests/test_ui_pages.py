"""Every page renders headlessly; the Review workflow walks all 6 steps; upload controls state their limits.

Uses FAIRLENS_UI_STUBS=1 so no neural model is downloaded or loaded (rule-based detector/rewriter + lexical stand-ins).
"""
import pytest
from streamlit.testing.v1 import AppTest

from fairlens.ingestion.loaders import MAX_IMAGE_BYTES, MAX_TEXT_BYTES

ALL_PAGES = ["home", "review", "text_analysis", "mitigation", "gate", "image_analysis", "benchmarks", "model_comparison",
             "error_analysis", "experiments", "use_cases", "about"]
JD = ("We are looking for a young, aggressive salesman to join our London team. The ideal candidate is a digital native "
      "with at least 5 years of experience.")


def _app(module: str) -> AppTest:
    script = f"""
import streamlit as st
from fairlens.ui import state
from fairlens.ui.pages import {module} as page
state.init()
st.session_state["_pages"] = {{}}
page.render()
"""
    return AppTest.from_string(script, default_timeout=120)


def _text(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown) + "\n".join(c.value for c in at.caption)


@pytest.fixture(autouse=True)
def stubs(monkeypatch):
    monkeypatch.setenv("FAIRLENS_UI_STUBS", "1")


@pytest.mark.parametrize("module", ALL_PAGES)
def test_every_page_renders_without_exception(module):
    at = _app(module)
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_review_walks_all_six_steps_and_offers_exports():
    at = _app("review")
    at.run()
    assert "Review assistant, not a legal or compliance certification tool." in _text(at)
    at.text_area(key="rv_text").set_value(JD).run()
    at.button(key="rv_run").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert "2 · Flagged phrases" in _text(at) and "young" in _text(at)
    for nxt, expect in [("rv_next_1", "3 · Original vs proposed rewrites"), ("rv_next_2", "4 · Critical-fact preservation"),
                        ("rv_next_3", "5 · Reliability Gate verdict"), ("rv_next_4", "6 · Export the review")]:
        at.button(key=nxt).click().run()
        assert not at.exception, [e.value for e in at.exception]
        assert expect in _text(at)
    labels = {b.label for b in at.get("download_button")}
    assert {"PDF report", "JSON", "HTML report"} <= labels


def test_review_back_returns_to_input_with_text_kept():
    at = _app("review")
    at.run()
    at.text_area(key="rv_text").set_value(JD).run()
    at.button(key="rv_run").click().run()
    at.button(key="rv_back_1").click().run()
    assert at.text_area(key="rv_text").value == JD


@pytest.mark.parametrize("module,limit", [("text_analysis", MAX_TEXT_BYTES), ("review", MAX_TEXT_BYTES), ("image_analysis", MAX_IMAGE_BYTES)])
def test_upload_controls_state_formats_size_and_limitations(module, limit):
    at = _app(module)
    at.run()
    text = _text(at)
    assert "Accepted:" in text and f"{limit // 2**20} MB" in text and "Known limitations:" in text
    if module == "image_analysis":
        assert "JPG" in text and "PNG" in text and "GIF" in text
    else:
        assert "PDF" in text and "DOCX" in text and "scanned PDFs without a text layer are not supported" in text


def test_stub_mode_is_labelled_on_analysis_pages():
    at = _app("text_analysis")
    at.run()
    assert "Smoke-test mode." in _text(at)
