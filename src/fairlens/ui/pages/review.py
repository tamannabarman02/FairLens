"""FairLens Review (V4 U.3): a guided six-step check of a job description, built from existing components."""
import streamlit as st

from ...detection.lexicon import CATEGORY_LABELS
from ...ingestion.loaders import MAX_CHARS, IngestionError, extract_text
from ...reliability_gate.rules import REASONS
from ...reporting import export
from .. import components as C
from ..state import run_mitigation, run_text_analysis
from .mitigation import name_of

STEPS = ["Upload", "Flag", "Rewrite", "Check facts", "Decide", "Export"]


def _state():
    return st.session_state.setdefault("review", {"text": "", "source": None, "analysis": None, "mitigation": None})


def render():
    rv = _state()
    C.page_header("FairLens Review", "A second check for automatic rewrites of job adverts: what was flagged, which rewrite "
                                     "FairLens recommends, what it checked, and when a person should decide.", "Job-description workflow")
    step = 0 if not rv["text"] else 1 if not rv["analysis"] else 2 if not rv["mitigation"] else 6
    C.render(C.stepper(STEPS, step))

    st.markdown("##### 1 · Upload or paste")
    st.caption("Accepted: PDF (with a text layer; scanned PDFs are not supported), DOCX (paragraphs and tables), TXT, "
               "or pasted text. Max 10 MB, first 20,000 characters.")
    a, b = st.columns([1, 1.2], gap="large")
    with a:
        f = st.file_uploader("Job advert file", type=["pdf", "docx", "txt"], key="review_upload")
        if f is not None and rv["source"] != f.name:
            try:
                doc = extract_text(f.name, f.getvalue())
                rv.update(text=doc.text, source=f.name, analysis=None, mitigation=None)
                st.success(f"Read {len(doc.text):,} characters from {f.name}.")
            except IngestionError as e:
                st.error(f"{e} Try saving the file as DOCX or TXT, or paste the text instead.")
    with b:
        pasted = st.text_area("…or paste the advert", value=rv["text"] if rv["source"] == "pasted" else "", height=120,
                              max_chars=MAX_CHARS, key="review_paste")
        if st.button("Use pasted text", disabled=not pasted.strip()):
            rv.update(text=pasted, source="pasted", analysis=None, mitigation=None)
    if not rv["text"]:
        C.render(C.empty_state("Start with a job advert", "Upload a PDF/DOCX/TXT file or paste the text above."))
        return
    with st.expander(f"Text being reviewed ({rv['source']})", expanded=False):
        st.write(rv["text"])

    if not rv["analysis"]:
        if st.button("Run the review", type="primary", icon=":material/play_arrow:"):
            with st.status("Reviewing…", expanded=True) as stt:
                try:
                    rv["analysis"] = run_text_analysis(rv["text"], progress=lambda s, m: stt.write(m))
                    stt.write("Generating and checking rewrites")
                    rv["mitigation"] = run_mitigation(rv["text"], progress=lambda s, m: stt.write(m))
                    stt.update(label="Review complete", state="complete", expanded=False)
                except Exception as exc:  # noqa: BLE001
                    stt.update(label="Review failed", state="error")
                    st.error(f"{type(exc).__name__}: {exc}. Try Lite mode (sidebar) if a model could not be loaded.")
            st.rerun()
        return
    an, mit = rv["analysis"], rv["mitigation"]

    st.markdown("##### 2 · Flagged phrases")
    agree = [r["display"] for r in an["detector_results"].values() if r.get("label") == "biased"]
    C.render(f'<div style="margin-bottom:8px">{C.badge(C.level_for_score(an["ensemble_score"]) + " bias · " + format(an["ensemble_score"], ".2f"), C.tone_for_score(an["ensemble_score"]))} '
             + (f'{C.badge("Detectors that flag it: " + ", ".join(agree), "blue")}' if agree else C.badge("No detector flags it", "green")) + "</div>")
    C.render(C.highlighted_text(an["text"], an["spans"]))
    for c in an["categories"][:6]:
        st.caption(f"**{CATEGORY_LABELS.get(c['category'], c['category'])}** · "
                   f"{', '.join(dict.fromkeys(c['phrases'])) or 'zero-shot judgement'} — {c['explanation']}")

    if not mit:
        return
    cands = {c["candidate"]: c for c in mit["candidates"]}
    st.markdown("##### 3 · Original vs rewrites")
    rows = [{"Version": name_of(k), "Text": c["text"], "Verdict": C.decision_badge(c["verdict"]), "Bias": c["bias_after"],
             "Facts kept": c["fact_retention"], "Similarity": c["semantic_similarity"]} for k, c in cands.items()]
    C.render(C.html_table(rows, raw={"Verdict"}, best={"Bias": "min", "Facts kept": "max"}))

    d = mit["decision"]
    chosen = d.get("chosen")
    st.markdown("##### 4 · Fact-preservation checks")
    show = chosen or next((k for k, c in cands.items() if c["changed"]), "original")
    fc = cands[show]["fact_checks"]
    items = []
    for key, label in [("numbers", "Numbers and requirements"), ("entities", "Names and organisations"), ("dates", "Dates"),
                       ("roles", "Roles"), ("negation", "Negations"), ("quotes", "Quotations")]:
        x = fc.get(key, {})
        detail = (f"missing: {', '.join(x['missing'])}" if x.get("missing") else
                  f"kept: {', '.join(x.get('kept', []))}" if x.get("kept") else
                  f"{x.get('original', 0)} → {x.get('rewrite', 0)} negation cue(s)" if key == "negation" else "none in the original")
        items.append((label, x.get("pass"), detail))
    st.caption(f"Checked for: {name_of(show)}")
    C.render(C.check_grid(items))

    st.markdown("##### 5 · Decision")
    C.render(C.decision_banner(d["decision"], d["reason"], f"Reason code {d['reason_code']}" +
                               (f" · recommended: {name_of(chosen)}" if chosen else "")))
    if chosen:
        C.render(f'<div class="fl-quote">{C.esc(d["chosen_text"])}</div>')
    for k, c in cands.items():
        if c["verdict"] != "ACCEPT" and c["changed"]:
            st.caption(f"{name_of(k)}: {C.DECISION_LABEL[c['verdict']]} — " + "; ".join(REASONS.get(r, r) for r in c["reasons"][:2]))

    st.markdown("##### 6 · Export")
    secs = export.sections_for_analysis(an, mit)
    meta = {"source": rv["source"], "decision": d["decision"], "reason_code": d["reason_code"]}
    e1, e2, e3 = st.columns([1, 1, 3])
    e1.download_button("PDF report", export.to_pdf("FairLens Review", secs, meta), "fairlens_review.pdf", "application/pdf",
                       icon=":material/picture_as_pdf:")
    e2.download_button("JSON", export.to_json({"meta": meta, "analysis": an, "mitigation": mit}), "fairlens_review.json",
                       "application/json", icon=":material/data_object:")
    C.render(C.note("FairLens is a review assistant: it cannot guarantee that a text is unbiased and is not a legal or "
                    "compliance certification."))
    if st.button("Review another advert", icon=":material/restart_alt:"):
        st.session_state.pop("review", None)
        st.rerun()
