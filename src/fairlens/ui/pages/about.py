import streamlit as st

from ...config import PROJECT_ROOT, load_config
from .. import components as C
from ..results import dataset_manifest, model_verification


def md(name: str) -> str:
    p = PROJECT_ROOT / "docs" / name
    return p.read_text(encoding="utf-8") if p.exists() else f"_{name} not written yet._"


def render():
    C.page_header("About FairLens", "A research framework for evaluating bias mitigation and the reliability of the resulting text. "
                                    "Its key contribution is the Reliability Gate.", "Documentation")
    C.render(C.pipeline_cards([("text", "Input", "Text, documents or images"), ("lens", "Bias Detection", "BERT · DeBERTa · BART-MNLI · rules"),
                               ("wand", "Mitigation", "Rule · FLAN-T5 · Qwen"), ("shield", "Reliability Gate", "accept / retry / reject / review"),
                               ("check", "Output", "Verified rewrite + evidence")]))
    C.gap()
    tabs = st.tabs(["Overview", "Methodology", "Models", "Datasets", "Limitations", "Citation"])
    with tabs[0]:
        st.markdown(md("APPLICATIONS_AND_VIVA.md").split("## ", 2)[1] if "## " in md("APPLICATIONS_AND_VIVA.md") else md("APPLICATIONS_AND_VIVA.md"))
        st.markdown("Full documentation lives in `docs/` (README, METHODOLOGY, MODEL_CARD_REGISTRY, DATASETS, RESULTS, LIMITATIONS).")
    with tabs[1]:
        st.markdown(md("METHODOLOGY.md"))
    with tabs[2]:
        v = model_verification()
        if v:
            rows = [{"Role": r.get("role"), "Model": r.get("model_id"), "License": r.get("license") or "see registry",
                     "Weights MB": r.get("weights_mb"), "Peak RAM MB": r.get("peak_mb"), "Latency ms": r.get("latency_ms"),
                     "Status": r.get("status"), "Revision": (r.get("sha") or "")[:10]} for r in v.values()]
            C.render(C.html_table(rows, fmt={"Latency ms": ".0f"}))
            st.caption("Measured on this machine by scripts/verify_models.py (one model per process).")
        st.markdown(md("MODEL_CARD_REGISTRY.md"))
    with tabs[3]:
        m = dataset_manifest()
        if m:
            C.render(C.html_table([{"Source": k, "License": s.get("license"), "Revision / hash": (s.get("revision") or s.get("sha256") or "")[:12]}
                                   for k, s in m["sources"].items()]))
        st.markdown(md("DATASETS.md"))
    with tabs[4]:
        st.markdown(md("LIMITATIONS.md"))
    with tabs[5]:
        st.code("""@software{fairlens2026,
  title  = {FairLens: Bias Detection, Mitigation and a Reliability Gate for Rewrites},
  author = {<Your Name>},
  year   = {2026},
  note   = {Final-year project; research prototype}
}""", language="bibtex")
        st.markdown(md("PROJECT_REPORT_OUTLINE.md").split("## Related work", 1)[-1] if "## Related work" in md("PROJECT_REPORT_OUTLINE.md") else "")
