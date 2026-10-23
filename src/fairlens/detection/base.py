"""Common detector interface: predict, predict_batch, explain, info."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field


@dataclass
class Span:
    start: int
    end: int
    text: str
    category: str
    explanation: str
    weight: float = 1.0


@dataclass
class DetectionResult:
    detector: str
    score: float                      # probability-like score that the text is biased, in [0, 1]
    label: str                        # "biased" | "neutral"
    runtime_ms: float
    categories: dict[str, float] = field(default_factory=dict)
    spans: list[Span] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def margin(self) -> float:
        """Confidence margin in [0, 1]: distance from the 0.5 decision boundary, rescaled."""
        return abs(self.score - 0.5) * 2


def label_for(score: float, threshold: float = 0.5) -> str:
    return "biased" if score >= threshold else "neutral"


class Detector(ABC):
    name: str = "detector"
    display: str = "Detector"
    threshold: float = 0.5

    @abstractmethod
    def _scores(self, texts: list[str]) -> list[float]:
        """Return a bias probability per text."""

    def predict(self, text: str) -> DetectionResult:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str], batch_size: int = 16) -> list[DetectionResult]:
        out: list[DetectionResult] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            t0 = time.perf_counter()
            scores = self._scores(chunk)
            per = (time.perf_counter() - t0) * 1000 / max(1, len(chunk))
            out.extend(
                DetectionResult(self.name, float(s), label_for(float(s), self.threshold), per) for s in scores
            )
        return out

    def explain(self, text: str) -> list[Span]:
        return []

    def info(self) -> dict:
        return {"name": self.name, "display": self.display, "threshold": self.threshold}
