import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

INGESTION_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ingestion"


class FakeDetector:
    """Deterministic detector for tests: score from a keyword table."""

    def __init__(self, name, table, default=0.1, threshold=0.5):
        self.name, self.display, self.table, self.default, self.threshold = name, name.upper(), table, default, threshold

    def predict_batch(self, texts, batch_size=16):
        from fairlens.detection.base import DetectionResult

        out = []
        for t in texts:
            s = max([v for k, v in self.table.items() if k in t.lower()] or [self.default])
            out.append(DetectionResult(self.name, s, "biased" if s >= 0.5 else "neutral", 1.0))
        return out

    def predict(self, text):
        return self.predict_batch([text])[0]

    def explain(self, text):
        return []


class FakeSuite:
    """Preservation suite stand-in: lexical overlap for similarity, word-set containment for entailment."""

    def evaluate(self, originals, rewrites):
        from fairlens.evaluation import facts
        from fairlens.evaluation.preservation import fluency

        out = []
        for o, r in zip(originals, rewrites):
            a, b = set(o.lower().split()), set(r.lower().split())
            j = len(a & b) / max(1, len(a | b))
            out.append({"semantic_similarity": 0.5 + j / 2,
                        "entailment": {"forward": j, "backward": j, "both": j, "contradiction": 0.0 if j > 0.4 else 0.8},
                        "facts": facts.check(o, r), "fluency": fluency(o, r)})
        return out


class FakeRewriter:
    def __init__(self, name, mapping, strict_mapping=None):
        self.name, self.display, self.mapping, self.strict_mapping = name, name, mapping, strict_mapping or {}
        self.calls = []

    def rewrite(self, text, strict=False, facts=None, attempt=0):
        from fairlens.mitigation.base import Rewrite

        self.calls.append((text, strict))
        out = (self.strict_mapping if strict else self.mapping).get(text, text)
        return Rewrite(self.name, out, 1.0, "prompt", attempt, strict)


@pytest.fixture
def fake_detectors():
    table = {"salesman": 0.9, "young": 0.8, "women are": 0.95, "aggressive": 0.7}
    return {"a": FakeDetector("a", table), "b": FakeDetector("b", table), "c": FakeDetector("c", table, default=0.05)}


@pytest.fixture
def fake_suite():
    return FakeSuite()


@pytest.fixture
def upload():
    """Bytes of a file in tests/fixtures/ingestion/, as the UI receives an upload."""
    return lambda name: (INGESTION_FIXTURES / name).read_bytes()
