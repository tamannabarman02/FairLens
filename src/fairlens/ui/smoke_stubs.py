"""Light stand-ins used ONLY when the dashboard is started with FAIRLENS_UI_STUBS=1 (scripts/ui_smoke.py --flow).

They let the browser flow exercise every page, upload, Gate display and export without downloading or loading
neural models. The real rule-based detector, rule-based rewriter, fact checker, fluency check and Reliability Gate
still run; only the neural pieces are replaced:
  - semantic similarity / entailment -> word-overlap proxy
  - BLIP / Florence-2 captioners and CLIP -> fixed caption and a constant alignment score
The UI shows a banner while this mode is on, and nothing produced here is a research result.
"""
from __future__ import annotations

import os
import time

ENV = "FAIRLENS_UI_STUBS"


def enabled() -> bool:
    return os.environ.get(ENV, "").strip() in {"1", "true", "yes"}


class OverlapSuite:
    """PreservationSuite interface with lexical stand-ins for the embedding and NLI models."""

    def evaluate(self, originals: list[str], rewrites: list[str]) -> list[dict]:
        from ..evaluation import facts
        from ..evaluation.preservation import fluency

        out = []
        for o, r in zip(originals, rewrites):
            a, b = set(o.lower().split()), set(r.lower().split())
            j = len(a & b) / max(1, len(a | b))
            out.append({"semantic_similarity": 0.5 + j / 2,
                        "entailment": {"forward": j, "backward": j, "both": j, "contradiction": 0.0 if j > 0.4 else 0.8},
                        "facts": facts.check(o, r), "fluency": fluency(o, r)})
        return out


class FixedCaptioner:
    def __init__(self, name: str, display: str, text: str):
        self.name, self.display, self.text, self.model_id = name, display, text, f"stub/{name}"

    def caption(self, image):
        from ..multimodal.captioners import Caption

        t0 = time.perf_counter()
        return Caption(self.name, self.model_id, self.text, (time.perf_counter() - t0) * 1000)


class ConstantAligner:
    def score(self, image, captions: list[str]) -> list[float]:
        return [0.3 for _ in captions]


def detectors():
    from ..detection.rule_based import RuleBasedDetector

    return {"rule_based": RuleBasedDetector()}


def rewriters():
    from ..mitigation.rule_based import RuleBasedRewriter

    return {"rule_based": RuleBasedRewriter()}


def captioners():
    return {"blip": FixedCaptioner("blip", "BLIP (stand-in)", "a man in a suit standing next to a desk"),
            "florence": FixedCaptioner("florence", "Florence-2 (stand-in)", "a person standing in an office")}
