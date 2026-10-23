import streamlit as st

from .. import components as C
from ..common import examples, go, run_analysis
from ..results import latest_summary


def render():
    C.render(f"""<div class="fl-hero">
      <div class="fl-eyebrow">Bias detection · mitigation · reliability</div>
      <h1>Detect. Reduce. <span>Verify.</span></h1>
      <p>FairLens finds biased language with four detectors, proposes neutral rewrites with two language models and a
      rule baseline, and then checks every rewrite with the Reliability Gate before anything is accepted.</p></div>""")
    st.write("")
    b1, b2, _ = st.columns([1.2, 1.3, 3.5])
    if b1.button("Analyze Text", type="primary", icon=":material/notes:", use_container_width=True):
        go("Text Analysis")
    if b2.button("Analyze Image", icon=":material/image:", use_container_width=True):
        go("Image Analysis")
    st.write("")
    cols = st.columns(4)
    feats = [
        ("layers", "Multi-Model Analysis", "Fine-tuned BERT and DeBERTa, zero-shot BART-MNLI and a transparent rule baseline score every sentence."),
        ("wand", "Intelligent Mitigation", "FLAN-T5, Qwen2.5-Instruct and a rule rewriter propose neutral wording with a shared prompt family."),
        ("shield", "Reliability Verification", "The Gate checks meaning, entailment, facts, detector agreement and fluency: accept, retry, reject or human review."),
        ("image", "Multimodal Support", "BLIP and Florence-2 captions are scored for image alignment and piped into the text detectors."),
    ]
    for col, (ic, t, d) in zip(cols, feats):
        col.markdown(C.card(t, d, icon_name=ic), unsafe_allow_html=True)
    st.write("")
    C.render(C.pipeline_cards([("text", "Input", "Text, PDF, DOCX, CSV or image"), ("lens", "Bias Detection", "4 detectors + explanations"),
                               ("wand", "Mitigation", "Rule, FLAN-T5, Qwen"), ("shield", "Reliability Gate", "Signals → decision + reason"),
                               ("check", "Output", "Accepted rewrite or human review")]))
    st.write("")

    left, right = st.columns([1.35, 1])
    with left:
        C.render('<div class="fl-card"><h4>Quick Start</h4><p>Pick an example and run the full pipeline. Results are computed live.</p></div>')
        t_text, t_img = st.tabs(["Text example", "Image example"])
        with t_text:
            ex = {k: v for k, v in examples().items() if v}
            choice = st.radio("Example", list(ex), horizontal=True, label_visibility="collapsed", key="qs_choice")
            txt = st.text_area("Text", ex.get(choice, ""), height=110, key=f"qs_text_{choice}", label_visibility="collapsed")
            if st.button("Run Analysis", type="primary", key="qs_run", icon=":material/play_arrow:"):
                st.session_state.text_input = txt
                if run_analysis(txt, with_mitigation=True):
                    go("Reliability Gate")
        with t_img:
            st.caption("Use a COCO val2017 sample image or upload your own on the Image Analysis page.")
            if st.button("Open Image Analysis", key="qs_img"):
                go("Image Analysis")
    with right:
        det = latest_summary("benchmark_detection", "detection_summary.json")
        gate = latest_summary("benchmark_gate", "gate_summary.json")
        C.render('<div class="fl-card"><h4>Latest research results</h4><p>From completed benchmark runs in <span class="fl-mono">outputs/</span>.</p></div>')
        st.write("")
        if det or gate:
            tiles = []
            if det:
                ok = {k: v for k, v in det["detectors"].items() if v.get("status") == "ok"}
                if ok:
                    k, v = max(ok.items(), key=lambda kv: kv[1]["metrics"]["macro_f1"])
                    tiles.append(C.metric_tile("Best detector macro-F1", f"{v['metrics']['macro_f1']:.3f}", v["display"], "accent",
                                               "Macro-averaged F1 on the shared test split."))
            if gate:
                g, lb = gate["selection"]["fairlens_gate_tuned"], gate["selection"]["lowest_bias"]
                tiles.append(C.metric_tile("Harmful selections: Gate vs lowest-bias",
                                           f"{g['harmful_selection']['mean']:.0%} vs {lb['harmful_selection']['mean']:.0%}",
                                           "controlled gold set, test split", "green", "Share of inputs where the chosen rewrite was not acceptable."))
            c = st.columns(len(tiles))
            for col, t in zip(c, tiles):
                col.markdown(t, unsafe_allow_html=True)
            if st.button("See all benchmarks", key="home_bench"):
                go("Benchmarks")
        else:
            C.render(C.empty_state("No benchmark runs yet", "Run the experiments from Experiments & Exports."))
    st.write("")
    C.render(C.note("<b>Review assistant, not a legal or compliance certification tool.</b> FairLens cannot guarantee that "
                    "any text is unbiased; detectors make errors and bias labels are culturally dependent."))
