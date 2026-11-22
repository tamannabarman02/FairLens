"""LLM-as-judge baseline: Qwen2.5-Instruct answers a yes/no faithfulness question per (original, rewrite).

Policy: among the changed candidates the judge answers "yes" to, choose the largest bias reduction;
if there is none, defer (None -> HUMAN_REVIEW). Anything other than a clear yes/no is an abstention.
The prompt variant is chosen on the validation split only by scripts/validate_llm_judge.py
(configs/llm_judge.json).
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from ..config import PROJECT_ROOT, load_config
from .baselines import ORIGINAL, Policy
from .rules import ACCEPT, HUMAN_REVIEW, REJECT
from .signals import Signals

YES, NO = "yes", "no"
CONFIG_PATH = PROJECT_ROOT / "configs" / "llm_judge.json"
QUESTION = ("Does the rewrite keep every fact of the original — names, numbers, dates, roles, negation — "
            "and add nothing? Answer yes or no.")

PROMPT_VARIANTS: dict[str, dict] = {
    "plain": {
        "description": "The question after the two texts, no system message.",
        "system": None,
        "user": "Original: {original}\nRewrite: {candidate}\n\n" + QUESTION,
    },
    "checker_system": {
        "description": "Fact-checker system message asking for a one-word reply, then the plain user turn.",
        "system": "You are a careful fact checker. Reply with exactly one word: yes or no.",
        "user": "Original: {original}\nRewrite: {candidate}\n\n" + QUESTION,
    },
    "question_first": {
        "description": "The question first, the texts in quotes, and a one-word answer slot at the end.",
        "system": None,
        "user": QUESTION + "\n\nOriginal: \"{original}\"\nRewrite: \"{candidate}\"\n\nAnswer (yes or no):",
    },
}
DEFAULT_VARIANT = "plain"

Generator = Callable[[list[dict[str, str]]], str]

_YES_WORDS = {"yes", "y", "yeah", "yep", "true", "correct"}
_NO_WORDS = {"no", "n", "nope", "false", "incorrect"}
_PREFIX = re.compile(r"^\s*(?:assistant|answer|final answer|response|output|judgement|judgment)\s*[:\-]\s*", re.I)


def build_messages(original: str, candidate: str, variant: str = DEFAULT_VARIANT) -> list[dict[str, str]]:
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"unknown judge prompt variant {variant!r}; choose from {sorted(PROMPT_VARIANTS)}")
    v = PROMPT_VARIANTS[variant]
    msgs = [{"role": "system", "content": v["system"]}] if v["system"] else []
    msgs.append({"role": "user", "content": v["user"].format(original=original, candidate=candidate)})
    return msgs


def parse_yes_no(raw: str | None) -> str | None:
    """'yes' / 'no' for a clear answer, None (abstain) for anything else."""
    if not raw:
        return None
    text = _PREFIX.sub("", raw.strip())
    words = re.findall(r"[a-z]+", text.lower())
    if not words:
        return None
    if words[:3] in (["yes", "or", "no"], ["no", "or", "yes"]):
        return None
    if words[0] in _YES_WORDS:
        return YES
    if words[0] in _NO_WORDS:
        return NO
    has_yes, has_no = YES in words, NO in words
    if has_yes != has_no:
        return YES if has_yes else NO
    return None


def answer_to_decision(answer: str | None) -> str:
    return {YES: ACCEPT, NO: REJECT}.get(answer, HUMAN_REVIEW)


@dataclass
class JudgeVerdict:
    variant: str
    original: str
    candidate: str
    prompt: str
    raw: str | None
    answer: str | None
    runtime_ms: float
    cached: bool = False
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class QwenJudgeGenerator:
    """Greedy, seeded generation with the Qwen chat adapter from mitigation/llm_adapters.py."""

    def __init__(self, large: bool = False, max_new_tokens: int = 4, rewriter_key: str = "qwen"):
        from ..mitigation.llm_adapters import build_rewriters

        self.rewriter = build_rewriters([rewriter_key], large=large)[rewriter_key]
        self.max_new_tokens = max_new_tokens
        self.seed = int(load_config()["mitigation"]["generation"]["seed"])

    @property
    def identity(self) -> str:
        from ..config import pinned_revision

        rev = self.rewriter.revision or pinned_revision(self.rewriter.model_id) or ""
        return f"llm_judge|{self.rewriter.model_id}|{rev}|greedy|{self.max_new_tokens}|seed{self.seed}"

    def render(self, messages: list[dict[str, str]]) -> str:
        tok, _ = self.rewriter._load()
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def __call__(self, messages: list[dict[str, str]]) -> str:
        import torch

        from ..mitigation.llm_adapters import _seed
        from ..models import run_with_cpu_fallback, to_model_device

        tok, model = self.rewriter._load()
        _seed(self.seed)
        enc = tok(self.render(messages), return_tensors="pt")

        def run():
            with torch.inference_mode():
                return model.generate(**to_model_device(enc, model), max_new_tokens=self.max_new_tokens, do_sample=False,
                                      num_beams=1, pad_token_id=tok.eos_token_id).cpu()

        ids = run_with_cpu_fallback(model, run)
        return tok.decode(ids[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


class LLMJudge:
    """Asks the generator the faithfulness question; caches answers in ScoreCache and logs every call."""

    def __init__(self, generator: Generator, variant: str = DEFAULT_VARIANT, cache=None, identity: str | None = None):
        if variant not in PROMPT_VARIANTS:
            raise ValueError(f"unknown judge prompt variant {variant!r}; choose from {sorted(PROMPT_VARIANTS)}")
        self.generator, self.variant, self.cache = generator, variant, cache
        self.identity = identity or getattr(generator, "identity", None) or getattr(generator, "__name__", "judge")
        self.log_rows: list[dict] = []
        self._memo: dict[tuple[str, str], JudgeVerdict] = {}

    def _prompt_text(self, messages: list[dict[str, str]]) -> str:
        return json.dumps(messages, ensure_ascii=False)

    def ask(self, original: str, candidate: str) -> JudgeVerdict:
        if (original, candidate) in self._memo:
            return self._memo[(original, candidate)]
        messages = build_messages(original, candidate, self.variant)
        prompt = self._prompt_text(messages)
        payload = self.cache.get_gen(self.identity, prompt) if self.cache is not None else None
        if payload is not None:
            v = JudgeVerdict(self.variant, original, candidate, prompt, payload.get("raw"), parse_yes_no(payload.get("raw")),
                             float(payload.get("runtime_ms", 0.0)), cached=True)
        else:
            t0 = time.perf_counter()
            try:
                raw, err = self.generator(messages), None
            except Exception as exc:  # noqa: BLE001  - an error is an abstention, never a fabricated answer
                raw, err = None, f"{type(exc).__name__}: {exc}"
            ms = (time.perf_counter() - t0) * 1000
            v = JudgeVerdict(self.variant, original, candidate, prompt, raw, parse_yes_no(raw), ms, error=err)
            if self.cache is not None and err is None:
                self.cache.put_gen(self.identity, prompt, {"raw": raw, "runtime_ms": ms})
        self.log_rows.append({**v.to_dict(), "model": self.identity})
        self._memo[(original, candidate)] = v
        return v

    def ask_many(self, pairs: list[tuple[str, str]], progress: Callable[[int, int], None] | None = None
                 ) -> dict[tuple[str, str], JudgeVerdict]:
        out: dict[tuple[str, str], JudgeVerdict] = {}
        uniq = list(dict.fromkeys(pairs))
        for k, (o, c) in enumerate(uniq):
            out[(o, c)] = self.ask(o, c)
            if progress:
                progress(k + 1, len(uniq))
        return out

    def write_log(self, path: Path) -> Path:
        path.write_text("\n".join(json.dumps(r, default=str, ensure_ascii=False) for r in self.log_rows), encoding="utf-8")
        return path


def judge_choose(sigs: list[Signals], answers: dict[str, str | None]) -> str | None:
    """Changed candidate with the largest bias reduction among those answered 'yes'; None defers."""
    yes = [s for s in sigs if s.changed and answers.get(s.candidate) == YES]
    return max(yes, key=lambda s: s.bias_reduction).candidate if yes else None


def original_text(sigs: list[Signals]) -> str:
    for s in sigs:
        if s.candidate == ORIGINAL:
            return s.text
    raise ValueError("the judge policy needs the unchanged 'original' candidate among the signals")


def judge_policy(judge: LLMJudge) -> Policy:
    def pol(sigs):
        orig = original_text(sigs)
        answers = {s.candidate: judge.ask(orig, s.text).answer for s in sigs if s.changed}
        return judge_choose(sigs, answers)
    pol.__name__ = "llm_judge"
    return pol


def load_selection(path: Path | None = None) -> dict | None:
    """The validation-selected judge settings, or None if scripts/validate_llm_judge.py has not been run."""
    path = path or CONFIG_PATH
    if not path.exists():
        return None
    sel = json.loads(path.read_text(encoding="utf-8"))
    if sel.get("selected_on") != "val" or sel.get("variant") not in PROMPT_VARIANTS:
        raise ValueError(f"{path} is not a validation-split selection of a known prompt variant")
    return sel


def agreement_metrics(gold: list[str], answers: list[str | None]) -> dict:
    """Judge vs gold at candidate level: 'yes' is a predicted ACCEPT; 'no' and abstentions are not."""
    from sklearn.metrics import cohen_kappa_score

    g = [x == ACCEPT for x in gold]
    p = [a == YES for a in answers]
    n = len(g)
    unacceptable = [pi for gi, pi in zip(g, p) if not gi]
    acceptable = [pi for gi, pi in zip(g, p) if gi]
    kappa = float(cohen_kappa_score(g, p)) if n and len(set(g) | set(p)) > 1 else None
    return {
        "n": n,
        "accuracy": sum(gi == pi for gi, pi in zip(g, p)) / n if n else None,
        "cohen_kappa": None if kappa is None or kappa != kappa else kappa,
        "unsafe_yes_rate": sum(unacceptable) / len(unacceptable) if unacceptable else 0.0,
        "yes_recall": sum(acceptable) / len(acceptable) if acceptable else None,
        "abstain_rate": sum(a is None for a in answers) / n if n else 0.0,
        "answers": {k: sum(a == k for a in answers) for k in (YES, NO)} | {"abstain": sum(a is None for a in answers)},
    }
