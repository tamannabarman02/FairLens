from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field


@dataclass
class Rewrite:
    rewriter: str
    text: str | None
    runtime_ms: float
    prompt: str | None = None
    attempt: int = 0
    strict: bool = False
    error: str | None = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class Rewriter(ABC):
    name = "rewriter"
    display = "Rewriter"

    @abstractmethod
    def _rewrite(self, text: str, strict: bool, facts: list[str]) -> tuple[str, str | None]:
        """Return (rewritten_text, prompt_used)."""

    def rewrite(self, text: str, strict: bool = False, facts: list[str] | None = None, attempt: int = 0) -> Rewrite:
        t0 = time.perf_counter()
        try:
            out, prompt = self._rewrite(text, strict, facts or [])
            return Rewrite(self.name, out, (time.perf_counter() - t0) * 1000, prompt, attempt, strict)
        except Exception as exc:  # noqa: BLE001  - surfaced to the UI, never replaced by fake output
            return Rewrite(self.name, None, (time.perf_counter() - t0) * 1000, None, attempt, strict,
                           error=f"{type(exc).__name__}: {exc}")

    def info(self) -> dict:
        return {"name": self.name, "display": self.display}
