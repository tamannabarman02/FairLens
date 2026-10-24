"""LLM rewriters: a seq2seq adapter (FLAN-T5) and a chat adapter (Qwen2.5-Instruct).

Generation settings are identical for every model and seeded (configs/models.yaml: mitigation.generation).
"""
from __future__ import annotations

import re

from ..config import load_config
from ..config import pinned_revision
from ..models import manager, place, run_with_cpu_fallback, to_model_device
from . import prompts
from .base import Rewriter


def _gen_cfg() -> dict:
    return dict(load_config()["mitigation"]["generation"])


def _seed(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_output(text: str, source: str) -> str:
    """Strip common prompt echoes / wrappers ("Rewritten text:", quotes) from a generation."""
    t = text.strip()
    t = re.sub(r"^(?:rewritten text|rewrite|output|text)\s*:\s*", "", t, flags=re.I)
    t = t.split("\n\n")[0].strip()
    if len(t) > 2 and t[0] in "\"'“" and t[-1] in "\"'”" and not (source.startswith(t[0])):
        t = t[1:-1].strip()
    return t


class Seq2SeqRewriter(Rewriter):
    def __init__(self, name: str, display: str, model_id: str, revision: str | None = None):
        self.name, self.display, self.model_id, self.revision = name, display, model_id, revision

    def _load(self):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        def loader():
            rev = self.revision or pinned_revision(getattr(self, "model_id", None) or getattr(self, "model_path", None))
            kw = {"revision": rev} if rev else {}
            return (AutoTokenizer.from_pretrained(self.model_id, **kw),
                    place(AutoModelForSeq2SeqLM.from_pretrained(self.model_id, **kw).eval()))

        return manager().get(f"gen:{self.model_id}", loader)

    def _rewrite(self, text, strict, facts):
        import torch

        tok, model = self._load()
        cfg = _gen_cfg()
        _seed(cfg.pop("seed"))
        prompt = prompts.seq2seq_prompt(text, strict, facts)
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=512)

        def run():
            with torch.inference_mode():
                return model.generate(**to_model_device(enc, model), **cfg).cpu()

        ids = run_with_cpu_fallback(model, run)
        return clean_output(tok.decode(ids[0], skip_special_tokens=True), text), prompt

    def info(self):
        return {**super().info(), "model": self.model_id, "revision": self.revision}


class ChatRewriter(Rewriter):
    def __init__(self, name: str, display: str, model_id: str, revision: str | None = None):
        self.name, self.display, self.model_id, self.revision = name, display, model_id, revision

    def _load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        def loader():
            rev = self.revision or pinned_revision(getattr(self, "model_id", None) or getattr(self, "model_path", None))
            kw = {"revision": rev} if rev else {}
            return (AutoTokenizer.from_pretrained(self.model_id, **kw),
                    place(AutoModelForCausalLM.from_pretrained(self.model_id, dtype="auto", **kw).eval()))

        return manager().get(f"gen:{self.model_id}", loader)

    def _rewrite(self, text, strict, facts):
        import torch

        tok, model = self._load()
        cfg = _gen_cfg()
        _seed(cfg.pop("seed"))
        msgs = prompts.chat_messages(text, strict, facts)
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt")

        def run():
            with torch.inference_mode():
                return model.generate(**to_model_device(enc, model), **cfg, pad_token_id=tok.eos_token_id).cpu()

        ids = run_with_cpu_fallback(model, run)
        out = tok.decode(ids[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return clean_output(out, text), prompt

    def info(self):
        return {**super().info(), "model": self.model_id, "revision": self.revision}


def build_rewriters(keys: list[str] | None = None, lite: bool = False, large: bool = False) -> dict[str, Rewriter]:
    from .rule_based import RuleBasedRewriter

    specs = load_config()["mitigation"]["rewriters"]
    out: dict[str, Rewriter] = {}
    for key in keys or list(specs):
        s = specs[key]
        if s["adapter"] == "rule_rewrite":
            out[key] = RuleBasedRewriter()
            continue
        mid = s["model_id"]
        if lite and s.get("lite_model_id"):
            mid = s["lite_model_id"]
        elif large and s.get("large_model_id"):
            mid = s["large_model_id"]
        cls = Seq2SeqRewriter if s["adapter"] == "seq2seq" else ChatRewriter
        out[key] = cls(key, s["display"], mid, s.get("revision"))
    return out
