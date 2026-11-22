"""Shared page helpers: navigation, running the pipeline with progress, example inputs."""
from __future__ import annotations

import json

import streamlit as st

from ..config import PROJECT_ROOT
from ..ingestion.loaders import (IMAGE_EXTS, MAX_CHARS, MAX_IMAGE_BYTES, MAX_TEXT_BYTES, TEXT_EXTS, Document, IngestionError,
                                 csv_columns, extract_text)
from ..ingestion.segment import split_sentences
from . import components as C
from . import smoke_stubs, state

MAX_SCORED_SENTENCES = 200
TEXT_LIMITATIONS = ("scanned PDFs without a text layer are not supported (no OCR); password-protected PDFs and legacy .doc "
                    f"files cannot be read; only the first {MAX_CHARS:,} characters are analysed")
IMAGE_LIMITATIONS = "animated GIFs use the first frame only; very small images (under 16 px) are rejected; WEBP/HEIC are not supported"

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
            show_error(exc)
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


def ext_list(exts) -> list[str]:
    return list(exts)


def formats_label(exts) -> str:
    return ", ".join(e.upper() for e in exts if e != "jpeg")


def mb(n: int) -> int:
    return n // 2**20


def text_upload_hint() -> str:
    return C.upload_hint(formats_label(TEXT_EXTS), mb(MAX_TEXT_BYTES), TEXT_LIMITATIONS)


def image_upload_hint() -> str:
    return C.upload_hint(formats_label(IMAGE_EXTS), mb(MAX_IMAGE_BYTES), IMAGE_LIMITATIONS)


def text_uploader(key: str, label: str = "Upload a document") -> Document | None:
    """File uploader for TXT/PDF/DOCX/CSV built from the ingestion constants. Returns the extracted Document or None."""
    f = st.file_uploader(label, type=ext_list(TEXT_EXTS), key=key,
                         help=f"{formats_label(TEXT_EXTS)}, up to {mb(MAX_TEXT_BYTES)} MB.")
    C.render(text_upload_hint())
    if f is None:
        return None
    data = f.getvalue()
    column = None
    try:
        if f.name.lower().endswith(".csv"):
            cols = csv_columns(data)
            if cols:
                column = st.selectbox("Text column", ["(detect automatically)", *cols], key=f"{key}_col",
                                      help="Which CSV column holds the text to analyse.")
                column = None if column == "(detect automatically)" else column
        doc = extract_text(f.name, data, text_column=column)
    except IngestionError as e:
        st.error(str(e))
        return None
    pages = f" ({doc.pages} pages)" if doc.pages else ""
    col = f", column '{doc.column}'" if doc.column else ""
    st.success(f"Extracted {len(doc.text):,} characters from {doc.name}{pages}{col}.")
    for w in doc.warnings:
        st.warning(w)
    return doc


def sentence_cap_note(text: str) -> None:
    n = len(split_sentences(text, max_sentences=10**6))
    if n > MAX_SCORED_SENTENCES:
        st.warning(f"This text has {n:,} sentences. The detectors score only the first {MAX_SCORED_SENTENCES}; "
                   "split the document into parts to review the rest.")


def show_error(exc: Exception) -> None:
    """Plain-language error with a next step; technical detail is kept in an expander."""
    msg = str(exc)
    low = msg.lower()
    if isinstance(exc, MemoryError) or "out of memory" in low or "alloc" in low:
        hint = "The computer ran out of memory while loading a model. Turn on Lite mode in the sidebar and try again."
    elif any(w in low for w in ("connection", "resolve", "offline", "huggingface", "hub", "couldn't find", "not found")):
        hint = ("A model could not be loaded (it may not be downloaded yet). Run scripts/verify_models.py once with internet "
                "access, or turn on Lite mode, then try again.")
    else:
        hint = "Something went wrong while running the pipeline. Try again; if it keeps failing, turn on Lite mode or shorten the text."
    st.error(hint)
    with st.expander("Technical details"):
        st.code(f"{type(exc).__name__}: {msg}", language=None)


def stub_banner() -> None:
    if smoke_stubs.enabled():
        C.render('<div class="fl-stub"><b>Smoke-test mode.</b> Neural models are replaced by light stand-ins '
                 "(rule-based detector and rewriter, word-overlap similarity, fixed captions). Values on this screen are "
                 "for checking the interface only and are not research results.</div>")


def result_header(title: str, kind: str, has_summary: bool = True) -> bool:
    """Render the one status label + provenance for a benchmark result. Returns True when results may be shown."""
    from .results import COMPLETED, PENDING, benchmark_state, provenance_pairs, status_note

    status, run = benchmark_state(kind)
    if status == COMPLETED and not has_summary:
        status = PENDING
        note = "The run finished but its summary file is missing, so nothing is shown. Re-run the benchmark."
    elif status == COMPLETED:
        note = "Every number below is read from this run's files in outputs/runs/."
    elif run is None:
        note = "No run of this benchmark exists yet. Start it below; results appear here when it completes."
    else:
        note = status_note(run) or "This attempt did not finish, so its outputs are not shown as results."
    C.render(C.provenance(title, status, provenance_pairs(run) if run else [], C.esc(note)))
    return status == COMPLETED
