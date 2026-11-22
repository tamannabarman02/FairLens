import json

import streamlit as st

from ...config import PROJECT_ROOT
from ...experiments.tracking import abs_path
from ...ingestion.loaders import IMAGE_EXTS, MAX_IMAGE_BYTES, IngestionError, load_image
from .. import components as C
from ..common import formats_label, image_upload_hint, mb, show_error, stub_banner
from ..state import run_image

MANIFEST = PROJECT_ROOT / "datasets" / "images" / "coco_val2017_subset" / "manifest.json"


def samples() -> list[dict]:
    if not MANIFEST.exists():
        return []
    return json.loads(MANIFEST.read_text(encoding="utf-8"))["images"]


def render():
    C.page_header("Image Analysis", "Two captioners describe the image; CLIP checks how well each caption matches it; the caption "
                                    "text then goes through the same bias detectors.", "Multimodal")
    stub_banner()
    C.render(C.note("<b>Scope:</b> this measures bias in the <i>generated captions</i> (the language), not all visual bias in the "
                    "image. Captions are model outputs, not ground truth."))
    C.gap()
    left, right = st.columns([1, 1.5], gap="large")
    refs = None
    with left:
        src = st.radio("Image source", ["Upload", "COCO sample"], horizontal=True)
        img, name = None, None
        if src == "Upload":
            f = st.file_uploader("Upload an image", type=list(IMAGE_EXTS), key="img_upload",
                                 help=f"{formats_label(IMAGE_EXTS)}, up to {mb(MAX_IMAGE_BYTES)} MB.")
            C.render(image_upload_hint())
            if f is not None:
                try:
                    img, name = load_image(f.name, f.getvalue()), f.name
                except IngestionError as e:
                    st.error(str(e))
        else:
            sm = samples()
            if not sm:
                C.render(C.empty_state("COCO subset not downloaded", "Run the image experiment once (Experiments & Exports → Launch "
                                       "experiments) to fetch it, or switch to Upload."))
            else:
                labels = [f"{s['image_id']} · {'person' if s['contains_person_category'] else 'no person'} · {s['license']['name']}" for s in sm]
                i = st.selectbox("Sample", range(len(sm)), format_func=lambda k: labels[k])
                s = sm[i]
                from PIL import Image

                img, name, refs = Image.open(abs_path(s["file"])).convert("RGB"), s["file"], s["reference_captions"]
                st.caption(f"Licence: {s['license']['name']} · [source]({s.get('flickr_url') or s['coco_url']})")
        if img is not None:
            st.image(img, use_container_width=True)
            if st.button("Generate captions & analyse", type="primary", icon=":material/play_arrow:"):
                with st.status("Loading captioners and analysing (first run loads BLIP, Florence-2 and CLIP)...", expanded=True) as stt:
                    try:
                        st.session_state.image_result = run_image(img, refs, progress=lambda s_, m: stt.write(m))
                        st.session_state.image_name = name
                        stt.update(label="Captions ready", state="complete", expanded=False)
                    except Exception as exc:  # noqa: BLE001
                        stt.update(label="Captioning failed", state="error")
                        show_error(exc)
    with right:
        r = st.session_state.image_result
        if not r:
            C.render(C.empty_state("No captions yet", "Upload an image or choose a COCO sample, then generate captions."))
            return
        st.caption(f"Image: {st.session_state.image_name} · {r['runtime_ms'] / 1000:.1f} s total")
        cols = st.columns(len(r["captions"]))
        for col, c in zip(cols, r["captions"]):
            with col:
                if not c["text"]:
                    C.render(C.card(c["display"], f"Failed: {C.esc(c['error'])}", "red"))
                    continue
                al = c.get("alignment", "n/a")
                tone = {"High": "green", "Medium": "amber", "Low": "red"}.get(al, "grey")
                C.render(C.card(c["display"], f'<span class="fl-card-text">“{C.esc(c["text"])}”</span>',
                                "accent" if c["captioner"] == "blip" else "blue",
                                extra=f'<div class="fl-row mt">{C.badge("alignment " + al, tone)}'
                                      f'{C.badge("CLIP " + format(c["clip_similarity"], ".3f"), "grey")}'
                                      f'{C.badge(c["bias_level"] + " bias", C.tone_for_score(c["bias_score"]))}'
                                      f'{C.badge(format(c["runtime_ms"] / 1000, ".1f") + " s", "grey")}</div>'))
        C.section("Caption analysis")
        rows = []
        for c in r["captions"]:
            if not c["text"]:
                continue
            issues = [f"rule: {f['text']} ({f['category']})" for f in c.get("rule_findings", [])]
            if c.get("unsupported_attributes"):
                issues.append("unverifiable attribute: " + ", ".join(c["unsupported_attributes"]))
            if c.get("gendered_terms"):
                issues.append("gendered terms: " + ", ".join(c["gendered_terms"]))
            al = c.get("alignment", "n/a")
            rows.append({"Model": c["display"], "Caption": c["text"], "Bias score": c["bias_score"],
                         "Detected issues": "; ".join(issues) or "none",
                         "Image alignment": C.badge(al, {"High": "green", "Medium": "amber", "Low": "red"}.get(al, "grey")),
                         "Detail coverage": c["detail_coverage"] if c.get("detail_coverage") is not None else "n/a (no reference)"})
        C.render(C.html_table(rows, raw={"Image alignment"}, best={"Bias score": "min"}))
        with st.expander("Per-detector scores on each caption"):
            C.render(C.html_table([{"Model": c["display"], **c.get("detector_scores", {})} for c in r["captions"] if c["text"]]))
        if refs:
            with st.expander("COCO human reference captions (for detail coverage only)"):
                for x in refs:
                    st.write("• " + x)
        st.caption("Gendered terms are reported, not judged: a caption saying 'a woman' may be accurate. 'Unverifiable attribute' marks "
                   "words such as nationality or wealth that pixels alone usually cannot support.")
