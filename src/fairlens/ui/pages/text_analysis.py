import streamlit as st

from ...detection.lexicon import CATEGORY_LABELS
from ...ingestion.loaders import MAX_CHARS, IngestionError, extract_text
from ...reporting import export
from .. import components as C
from ..common import STEPS, current_step, examples, go, run_analysis

DETECTOR_HELP = {
    "bert": "DistilBERT fine-tuned on the FairLens training split (BABE + StereoSet + CrowS-Pairs).",
    "deberta": "DeBERTa-v3-small fine-tuned on the same training split.",
    "bart_mnli": "BART-large-MNLI zero-shot: entailment of a 'biased' hypothesis (template chosen on validation data).",
    "rule_based": "Transparent lexicon and pattern rules; every match has an explanation.",
}


def render():
    C.page_header("Text Analysis", "Paste or upload text. Each sentence is scored by every detector; biased spans are highlighted "
                                   "with the reason they were flagged.", "Step 1 · Bias detection")
    C.render(C.stepper(STEPS, current_step()))

    t_enter, t_upload = st.tabs(["Enter Text", "Upload File"])
    with t_enter:
        ex = examples()
        pick = st.pills("Examples", list(ex), key="ex_pick", label_visibility="collapsed")
        if pick and st.session_state.get("_last_pick") != pick:
            st.session_state._last_pick = pick
            st.session_state.text_input = ex[pick]
        text = st.text_area("Text to analyse", key="text_input", height=150, max_chars=MAX_CHARS,
                            placeholder="Paste a job advert, article paragraph, chatbot answer...")
        st.caption(f"{len(text):,} / {MAX_CHARS:,} characters")
    with t_upload:
        f = st.file_uploader("Upload TXT, PDF, DOCX or CSV (max 10 MB)", type=["txt", "pdf", "docx", "csv"])
        if f is not None:
            try:
                doc = extract_text(f.name, f.getvalue())
                st.success(f"Extracted {len(doc.text):,} characters from {doc.name}" + (f" ({doc.pages} pages)" if doc.pages else "")
                           + (" - truncated to the limit" if doc.truncated else ""))
                if st.button("Use extracted text", key="use_upload"):
                    st.session_state.text_input = doc.text
                    st.rerun()
            except IngestionError as e:
                st.error(str(e))

    c1, c2, _ = st.columns([1, 1.4, 3])
    run_det = c1.button("Analyze", type="primary", icon=":material/search:", disabled=not text.strip())
    run_all = c2.button("Analyze & mitigate", icon=":material/auto_fix_high:", disabled=not text.strip())
    if run_det or run_all:
        if run_analysis(text, with_mitigation=run_all) and run_all:
            go("Mitigation")

    a = st.session_state.analysis
    if not a:
        st.write("")
        C.render(C.empty_state("Nothing analysed yet", "Choose an example chip or paste text, then press Analyze."))
        return
    st.write("")
    left, right = st.columns([1.5, 1], gap="large")
    with left:
        st.markdown("##### Original text")
        C.render(C.highlighted_text(a["text"], a["spans"]))
        C.render('<div class="fl-sub" style="font-size:12.5px;margin-top:6px">Highlight colour = rule strength '
                 '(red strong, amber medium, purple weak/model attribution). Hover a highlight for the explanation.</div>')
        st.markdown("##### Detected bias categories")
        if not a["categories"]:
            C.render(C.note("No category-specific findings. Model scores on the right may still flag the text."))
        for c in a["categories"]:
            phr = ", ".join(f"“{p}”" for p in dict.fromkeys(c["phrases"])) or "(zero-shot judgement, no single phrase)"
            zs = f" · zero-shot {c['zero_shot']:.2f}" if c.get("zero_shot") else ""
            C.render(f'<div class="fl-card" style="margin-bottom:10px;padding:12px 16px">{C.badge(CATEGORY_LABELS.get(c["category"], c["category"].replace("_", " ").title()), C.tone_for_score(c["weight"] + 0.2))}'
                     f'<div style="margin-top:6px;font-weight:600">{C.esc(phr)}</div><p>{C.esc(c["explanation"])}{zs}</p></div>')
    with right:
        s = a["ensemble_score"]
        C.render(f'<div class="fl-card fl-gauge"><div class="fl-sub" style="font-weight:600">Overall bias score '
                 f'{C.help_dot("Mean of the available detectors; each detector scores the most biased sentence.")}</div>'
                 f'<div class="num" style="color:var(--{C.tone_for_score(s)})">{s:.2f}</div>{C.badge(C.level_for_score(s) + " bias", C.tone_for_score(s))}'
                 f'{C.score_bar("", s)}</div>')
        st.write("")
        st.markdown("##### Per-model predictions")
        for k, r in a["detector_results"].items():
            if r["score"] is None:
                C.render(C.note(f"<b>{C.esc(r['display'])}</b> failed: {C.esc(r['error'])}", "warn"))
                continue
            lab = C.badge("Biased" if r["label"] == "biased" else "Neutral", "red" if r["label"] == "biased" else "green")
            C.render(C.score_bar(r["display"], r["score"], right=f'{lab}<span>{r["runtime_ms"]:.0f} ms</span>',
                                 sub=DETECTOR_HELP.get(k, "")))
        if a.get("missing_detectors"):
            C.render(C.note("Not available yet: " + ", ".join(a["missing_detectors"]) +
                            " (train with <span class='fl-mono'>scripts/train_detectors.py</span>).", "warn"))
    if len(a["sentences"]) > 1:
        with st.expander(f"Sentence-level scores ({len(a['sentences'])} sentences)"):
            rows = [{"sentence": x["text"][:140], **{k: v for k, v in x["scores"].items()}} for x in a["sentences"]]
            C.render(C.html_table(rows))
    st.write("")
    e1, e2, e3, e4 = st.columns([1, 1, 1, 2.4])
    secs = export.sections_for_analysis(a, st.session_state.mitigation)
    e1.download_button("JSON", export.to_json({"analysis": a, "mitigation": st.session_state.mitigation}), "fairlens_analysis.json",
                       "application/json", icon=":material/download:")
    e2.download_button("HTML report", export.to_html("FairLens analysis", secs), "fairlens_analysis.html", "text/html", icon=":material/download:")
    e3.download_button("PDF report", export.to_pdf("FairLens analysis", secs), "fairlens_analysis.pdf", "application/pdf", icon=":material/download:")
    if e4.button("Continue to Mitigation →", type="primary"):
        go("Mitigation")
