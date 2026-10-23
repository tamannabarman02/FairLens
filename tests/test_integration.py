"""Integration tests: full text pipeline and full image pipeline.

Fast versions use small stand-in models (fakes) so they run anywhere; the `slow` versions load the real
models from the local cache and are skipped when the cache is missing:  pytest -m slow
"""
import io

import pytest
from conftest import FakeDetector, FakeRewriter, FakeSuite

from fairlens.pipeline import analyze_image, analyze_text, mitigate_text
from fairlens.reliability_gate.rules import GateConfig

SRC = "Looking for a young, aggressive salesman with 5 years of experience. The office is in Leeds."
GOOD = "Looking for a proactive salesperson with 5 years of experience. The office is in Leeds."
BAD = "Looking for a proactive salesperson with 2 years of experience. The office is in Leeds."


def cfg():
    return GateConfig.from_config(min_semantic_similarity=0.6, min_entailment_prob=0.4, max_detector_std=0.5)


def dets():
    t = {"salesman": 0.9, "young": 0.8, "aggressive": 0.7}
    return {"a": FakeDetector("a", t), "b": FakeDetector("b", t)}


def test_text_pipeline_end_to_end():
    a = analyze_text(SRC, dets(), explain=False)
    assert a.ensemble_score > 0.5 and a.level in {"High", "Medium"}
    assert len(a.sentences) == 2
    assert {s["text"] for s in a.spans} >= {"young", "salesman"}
    rw = {"bad_llm": FakeRewriter("bad_llm", {SRC: BAD}), "good_llm": FakeRewriter("good_llm", {SRC: GOOD})}
    m = mitigate_text(SRC, dets(), rw, FakeSuite(), cfg())
    d = m.decision
    assert d["decision"] == "ACCEPT" and d["chosen"] == "good_llm"
    verdicts = {c["candidate"]: c["verdict"] for c in m.candidates}
    assert verdicts["bad_llm"] == "REJECT" and verdicts["original"] == "RETRY"
    assert m.lowest_bias_choice in {"bad_llm", "good_llm"}


def test_text_pipeline_neutral_input_no_change():
    text = "The committee approved the budget on Tuesday."
    rw = {"llm": FakeRewriter("llm", {text: "The committee passed the budget on Tuesday."})}
    m = mitigate_text(text, dets(), rw, FakeSuite(), cfg())
    assert m.decision["decision"] == "ACCEPT" and m.decision["chosen"] == "original"
    assert m.decision["reason_code"] == "NEUTRAL_UNCHANGED"


def test_text_pipeline_from_uploaded_docx():
    from docx import Document

    from fairlens.ingestion.loaders import extract_text

    buf = io.BytesIO()
    d = Document()
    d.add_paragraph(SRC)
    d.save(buf)
    doc = extract_text("job.docx", buf.getvalue())
    assert analyze_text(doc.text, dets(), explain=False).ensemble_score > 0.5


class _FakeCaptioner:
    def __init__(self, name, text):
        self.name, self.display, self.text, self.model_id = name, name.upper(), text, name

    def caption(self, image):
        from fairlens.multimodal.captioners import Caption

        return Caption(self.name, self.name, self.text, 1.0)


class _FakeAligner:
    def score(self, image, captions):
        return [0.31 if "dog" in c else 0.2 for c in captions]


def test_image_pipeline_end_to_end():
    from PIL import Image

    img = Image.new("RGB", (64, 64), "white")
    caps = {"blip": _FakeCaptioner("blip", "a dog on a beach"), "florence": _FakeCaptioner("florence", "a young woman with a dog")}
    r = analyze_image(img, caps, dets(), _FakeAligner(), references=["A dog runs on the beach."])
    by = {c["captioner"]: c for c in r.captions}
    assert by["blip"]["alignment"] == "High" and by["blip"]["detail_coverage"] > 0
    assert "woman" in by["florence"]["gendered_terms"]
    assert by["florence"]["detector_scores"]["a"] == 0.8  # "young" flagged by the fake detector


# ---------------------------------------------------------------------------------- real models
def _have(model_id: str) -> bool:
    import os

    import fairlens.config  # noqa: F401

    name = "models--" + model_id.replace("/", "--")
    return os.path.isdir(os.path.join(os.environ["HF_HOME"], "hub", name))


@pytest.mark.slow
@pytest.mark.skipif(not _have("Qwen/Qwen2.5-0.5B-Instruct"), reason="model cache missing")
def test_real_text_pipeline():
    from fairlens.detection.registry import build_detectors
    from fairlens.evaluation.preservation import PreservationSuite
    from fairlens.mitigation.llm_adapters import build_rewriters

    det = build_detectors(["rule_based", "bert", "deberta"])
    a = analyze_text(SRC, det)
    assert set(a.detector_results) == set(det) and all(r["score"] is not None for r in a.detector_results.values())
    m = mitigate_text(SRC, det, build_rewriters(["rule_based", "qwen"]), PreservationSuite(), GateConfig.from_config())
    assert m.decision["decision"] in {"ACCEPT", "REJECT", "RETRY", "HUMAN_REVIEW"}
    assert any(c["candidate"] == "qwen" and c["text"] for c in m.candidates)


@pytest.mark.slow
@pytest.mark.skipif(not _have("Salesforce/blip-image-captioning-base"), reason="model cache missing")
def test_real_image_pipeline():
    from pathlib import Path

    from PIL import Image

    from fairlens.detection.registry import build_detectors
    from fairlens.multimodal.captioners import build_captioners

    p = Path(__file__).resolve().parents[1] / "outputs" / "model_verification" / "sample_coco_000000039769.jpg"
    if not p.exists():
        pytest.skip("sample image missing")
    r = analyze_image(Image.open(p), build_captioners(), build_detectors(["rule_based"]))
    texts = [c["text"] for c in r.captions]
    assert all(texts) and any("cat" in t.lower() for t in texts)
    assert all(c["clip_similarity"] > 0.2 for c in r.captions)
