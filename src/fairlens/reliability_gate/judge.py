"""LLM-as-judge baseline (V4 G.1): ask an instruction model whether a rewrite keeps every fact of the original.

The judge returns P(yes) from the next-token distribution over "Yes"/"No" after the prompt (no sampling), so it
is deterministic and gives a score that can be thresholded. Prompt variant and threshold are chosen on the
validation split only (scripts/run_judge.py).
"""
from __future__ import annotations

import math

from ..config import load_config, pinned_revision
from ..hardware import place, to_model
from ..models import manager

PROMPTS = {
    "J1_facts": ("Original: {o}\nRewrite: {r}\n\nDoes the rewrite keep every fact of the original (names, numbers, dates, "
                 "roles, negations and quotations) without adding new facts? Answer Yes or No."),
    "J2_meaning": ("Original: {o}\nRewrite: {r}\n\nIs the rewrite an acceptable replacement for the original: it removes biased "
                   "or stereotyped wording if there is any, and otherwise says exactly the same thing? Answer Yes or No."),
    "J3_strict": ("You are checking an automatic rewrite.\nOriginal: {o}\nRewrite: {r}\n\nAnswer No if the rewrite changes, drops "
                  "or adds any name, number, date, role, negation or quoted words, or changes the meaning. Otherwise answer Yes. "
                  "Answer Yes or No."),
}
SYSTEM = "You are a careful fact-checking assistant. Answer with a single word: Yes or No."


class LLMJudge:
    def __init__(self, model_id: str | None = None, prompt: str = "J1_facts"):
        self.model_id = model_id or load_config()["mitigation"]["rewriters"]["qwen"]["large_model_id"]
        self.prompt = prompt

    def _load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        def loader():
            rev = pinned_revision(self.model_id)
            return (AutoTokenizer.from_pretrained(self.model_id, revision=rev),
                    place(AutoModelForCausalLM.from_pretrained(self.model_id, revision=rev, dtype="auto").eval()))

        return manager().get(f"judge:{self.model_id}", loader)

    def _token_ids(self, tok) -> tuple[list[int], list[int]]:
        def ids(words):
            out = set()
            for w in words:
                t = tok.encode(w, add_special_tokens=False)
                if t:
                    out.add(t[0])
            return sorted(out)

        return ids(["Yes", " Yes", "yes", " yes"]), ids(["No", " No", "no", " no"])

    def p_yes(self, original: str, rewrite: str) -> float:
        import torch

        tok, model = self._load()
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": PROMPTS[self.prompt].format(o=original, r=rewrite)}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = to_model(tok(text, return_tensors="pt"), model)
        with torch.inference_mode():
            logits = model(**enc).logits[0, -1].float()
        yes, no = self._token_ids(tok)
        ly = torch.logsumexp(logits[yes], 0).item()
        ln = torch.logsumexp(logits[no], 0).item()
        return 1.0 / (1.0 + math.exp(ln - ly))

    def info(self) -> dict:
        return {"model": self.model_id, "revision": pinned_revision(self.model_id), "prompt": self.prompt,
                "prompt_text": PROMPTS[self.prompt]}
