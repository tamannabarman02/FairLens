import streamlit as st

from .. import components as C
from ..common import go
from ..results import benchmark_state

ENTRIES = [
    ("file", "FairLens Review", "Guided 6-step check of a job description: flagged phrases with evidence, rewrites, fact checks, "
                                "Gate verdict and a PDF/JSON report.", "Start a review"),
    ("text", "Text Analysis", "Score any text with every detector and see highlighted phrases with the reason they were flagged.",
     "Analyse text"),
    ("image", "Image Analysis", "Caption an image with two models, check caption-image alignment and scan the captions for bias.",
     "Analyse an image"),
    ("wand", "Mitigation", "Compare the rule-based, FLAN-T5 and Qwen rewrites side by side with a diff view.", "Open mitigation"),
    ("shield", "Reliability Gate", "See why a rewrite is accepted, retried, rejected or sent to a person, with every signal.",
     "Open the Gate"),
    ("chart", "Benchmarks", "Research results from completed runs only, each with its run ID, models, dataset version and config.",
     "View benchmarks"),
    ("user", "Use Cases", "One-click demo scenarios with synthetic inputs for recruitment, chatbots, moderation and more.",
     "Browse use cases"),
]
BENCH_KINDS = [("benchmark_detection", "Detection"), ("benchmark_mitigation", "Mitigation"), ("benchmark_gate", "Reliability Gate"),
               ("benchmark_image", "Image captions")]


def render():
    C.render("""<div class="fl-hero">
      <div class="fl-eyebrow">Bias detection · mitigation · reliability</div>
      <h1>Detect. Reduce. <span>Verify.</span></h1>
      <p>FairLens finds biased wording in text, proposes neutral rewrites, and then checks every rewrite with the
      <b>Reliability Gate</b> so that meaning, names, numbers, dates and negations are not lost along the way.</p>
      <ol class="fl-steps-list"><li><b>Add text</b>: paste it, upload a TXT, PDF, DOCX or CSV file, or pick an example.</li>
      <li><b>Review the evidence</b>: flagged phrases, detector scores and proposed rewrites.</li>
      <li><b>Trust the verdict, or don't</b>: the Gate accepts, retries, rejects or asks a person, and says why.</li></ol></div>""")
    C.gap()
    b1, b2, _ = st.columns([1.6, 1.4, 3])
    if b1.button("Start FairLens Review", type="primary", icon=":material/fact_check:", use_container_width=True, key="home_cta"):
        go("FairLens Review")
    if b2.button("Try Text Analysis", icon=":material/notes:", use_container_width=True, key="home_cta2"):
        go("Text Analysis")

    C.section("Where do you want to start?")
    for row in (ENTRIES[:4], ENTRIES[4:]):
        cols = st.columns(4, gap="medium")
        for col, (ic, title, desc, cta) in zip(cols, row):
            with col:
                C.render(C.entry_card(ic, title, desc, primary=title == "FairLens Review"))
                if st.button(cta, key=f"home_go_{title.lower().replace(' ', '_')}", use_container_width=True, icon=":material/arrow_forward:"):
                    go(title)
        C.gap()

    C.section("Research status", "One label per benchmark, read from outputs/runs/. No benchmark numbers are shown until a run completes.")
    cols = st.columns(len(BENCH_KINDS))
    for col, (kind, label) in zip(cols, BENCH_KINDS):
        col.markdown(f'<div class="fl-card compact"><h4>{C.esc(label)}</h4>{C.status_badge(benchmark_state(kind)[0])}</div>',
                     unsafe_allow_html=True)
    C.gap()
    C.render(C.note("<b>Review assistant, not a legal or compliance certification tool.</b> FairLens cannot guarantee that "
                    "any text is unbiased; detectors make errors and bias labels are culturally dependent."))
