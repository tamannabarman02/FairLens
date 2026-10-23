"""End-to-end pipelines used by the dashboard, the use-case runner and integration tests.

Text:  input -> sentence split -> detectors -> mitigation (rule, FLAN-T5, Qwen) -> Reliability Gate (+retry)
Image: image -> captioners (BLIP, Florence-2) -> CLIP alignment -> caption text -> detectors
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

from .detection.base import Span
from .detection.registry import build_detectors
from .detection.rule_based import RuleBasedDetector
from .ingestion.segment import split_sentences
from .reliability_gate.retry import gate_with_retry
from .reliability_gate.rules import GateConfig, judge
from .reliability_gate.signals import ensemble

Progress = Callable[[str, str], None]  # (stage, message)


def bias_level(score: float) -> str:
    return "High" if score >= 0.66 else "Medium" if score >= 0.4 else "Low"


@dataclass
class TextAnalysis:
    text: str
    detector_results: dict[str, dict]
    ensemble_score: float
    level: str
    spans: list[dict]
    categories: list[dict]
    sentences: list[dict]
    runtime_ms: float
    missing_detectors: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def analyze_text(text: str, detectors: dict | None = None, lite: bool = False, progress: Progress | None = None,
                 explain: bool = True) -> TextAnalysis:
    from .detection.registry import missing_detectors

    t0 = time.perf_counter()
    detectors = detectors if detectors is not None else build_detectors(lite=lite)
    sents = split_sentences(text) or []
    units = [s.text for s in sents] or [text]
    results: dict[str, dict] = {}
    per_sentence: dict[str, list[float]] = {}
    for name, det in detectors.items():
        if progress:
            progress("detect", f"Running {det.display}")
        t1 = time.perf_counter()
        try:
            whole = det.predict(text).to_dict() if name == "rule_based" else None
            sent_scores = [r.score for r in det.predict_batch(units)]
            # document score = max over sentences (a document is biased if any sentence is)
            score = float(max(sent_scores))
            per_sentence[name] = sent_scores
            res = {"detector": name, "display": det.display, "score": score,
                   "label": "biased" if score >= det.threshold else "neutral",
                   "runtime_ms": (time.perf_counter() - t1) * 1000, "error": None}
            if whole:
                res["categories"] = whole["categories"]
        except Exception as exc:  # noqa: BLE001
            res = {"detector": name, "display": det.display, "score": None, "label": "error",
                   "runtime_ms": (time.perf_counter() - t1) * 1000, "error": f"{type(exc).__name__}: {exc}"}
        results[name] = res
    ok = {k: v["score"] for k, v in results.items() if v["score"] is not None}
    ens = ensemble(ok)

    spans: list[Span] = RuleBasedDetector().explain(text)
    if explain:
        # model attribution for the strongest fine-tuned detector on the most biased sentence
        for key in ("deberta", "bert"):
            if key in detectors and key in per_sentence and per_sentence[key]:
                i = int(np.argmax(per_sentence[key]))
                if per_sentence[key][i] >= 0.5:
                    off = sents[i].start if sents else 0
                    for s in detectors[key].explain(units[i]):
                        if not any(sp.start <= s.start + off < sp.end for sp in spans):
                            spans.append(Span(s.start + off, s.end + off, s.text, s.category, s.explanation, s.weight))
                break
    cats: dict[str, dict] = {}
    for s in spans:
        if s.category == "model_attribution":
            continue
        c = cats.setdefault(s.category, {"category": s.category, "phrases": [], "explanation": s.explanation, "weight": 0.0})
        c["phrases"].append(s.text)
        c["weight"] = max(c["weight"], s.weight)
    if "bart_mnli" in detectors and ens >= 0.4 and hasattr(detectors["bart_mnli"], "category_scores"):
        try:
            zs = detectors["bart_mnli"].category_scores(text)
            for k, v in zs.items():
                if v >= 0.7:
                    cats.setdefault(k, {"category": k, "phrases": [], "explanation": "Zero-shot NLI judges the text to stereotype this group.", "weight": 0.0})
                    cats[k]["zero_shot"] = v
        except Exception:  # noqa: BLE001
            pass
    sentences = [{"text": s.text, "start": s.start, "end": s.end,
                  "scores": {k: per_sentence[k][i] for k in per_sentence}} for i, s in enumerate(sents)]
    return TextAnalysis(text, results, ens, bias_level(ens), [asdict(s) for s in sorted(spans, key=lambda s: s.start)],
                        sorted(cats.values(), key=lambda c: -c["weight"]), sentences, (time.perf_counter() - t0) * 1000,
                        missing_detectors())


@dataclass
class MitigationResult:
    original: str
    decision: dict
    candidates: list[dict]          # signals + verdict per candidate
    rewrite_log: list[dict]
    lowest_bias_choice: str | None
    runtime_ms: float

    def to_dict(self):
        return asdict(self)


def mitigate_text(text: str, detectors: dict | None = None, rewriters: dict | None = None, suite=None,
                  cfg: GateConfig | None = None, lite: bool = False, progress: Progress | None = None,
                  initial: dict[str, str] | None = None) -> MitigationResult:
    from .evaluation.preservation import PreservationSuite
    from .mitigation.llm_adapters import build_rewriters
    from .reliability_gate.baselines import lowest_bias

    t0 = time.perf_counter()
    detectors = detectors if detectors is not None else build_detectors(lite=lite)
    rewriters = rewriters if rewriters is not None else build_rewriters(lite=lite)
    suite = suite or PreservationSuite()
    cfg = cfg or GateConfig.from_config()
    decision, sigs, log = gate_with_retry(text, rewriters, detectors, suite, cfg,
                                          progress=(lambda m: progress("mitigate", m)) if progress else None,
                                          initial=initial)
    biased = decision.input_biased
    cands = []
    for s in sigs:
        v = judge(s, cfg, biased)
        cands.append({**s.to_dict(), "verdict": v.decision, "reasons": v.reasons, "checks": v.checks})
    return MitigationResult(text, decision.to_dict(), cands, log, lowest_bias(sigs), (time.perf_counter() - t0) * 1000)


@dataclass
class ImageAnalysis:
    captions: list[dict]
    runtime_ms: float

    def to_dict(self):
        return asdict(self)


def analyze_image(image, captioners: dict | None = None, detectors: dict | None = None, aligner=None,
                  references: list[str] | None = None, lite: bool = False, progress: Progress | None = None) -> ImageAnalysis:
    from .multimodal.caption_evaluation import ClipAlignment, alignment_label, caption_flags, detail_coverage
    from .multimodal.captioners import build_captioners

    t0 = time.perf_counter()
    captioners = captioners if captioners is not None else build_captioners()
    detectors = detectors if detectors is not None else build_detectors(lite=lite)
    aligner = aligner or ClipAlignment()
    caps = []
    for name, c in captioners.items():
        if progress:
            progress("caption", f"Captioning with {c.display}")
        caps.append({**c.caption(image).to_dict(), "display": c.display})
    good = [c for c in caps if c["text"]]
    if good:
        if progress:
            progress("align", "Scoring image-text alignment (CLIP)")
        sims = aligner.score(image, [c["text"] for c in good])
        for c, s in zip(good, sims):
            c["clip_similarity"] = s
            c["alignment"] = alignment_label(s)
            c["detail_coverage"] = detail_coverage(c["text"], references or [])
            c.update(caption_flags(c["text"]))
        for name, det in detectors.items():
            if progress:
                progress("detect", f"Running {det.display} on captions")
            for c, r in zip(good, det.predict_batch([c["text"] for c in good])):
                c.setdefault("detector_scores", {})[name] = r.score
        for c in good:
            c["bias_score"] = ensemble(c["detector_scores"])
            c["bias_level"] = bias_level(c["bias_score"])
            c["rule_findings"] = [asdict(s) for s in RuleBasedDetector().explain(c["text"])]
    return ImageAnalysis(caps, (time.perf_counter() - t0) * 1000)
