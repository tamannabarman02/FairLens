"""Meaning-preservation evaluators: semantic similarity, bidirectional NLI entailment, fluency."""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

from ..config import load_config, pinned_revision
from ..models import manager
from . import facts as fact_checker


class SemanticSimilarity:
    def __init__(self, model_id: str | None = None):
        self.model_id = model_id or load_config()["preservation"]["embedding_model"]

    def _load(self):
        from sentence_transformers import SentenceTransformer

        return manager().get(f"emb:{self.model_id}", lambda: SentenceTransformer(self.model_id, device="cpu", revision=pinned_revision(self.model_id)))

    def similarity(self, a: list[str], b: list[str]) -> list[float]:
        model = self._load()
        ea = model.encode(a, normalize_embeddings=True, batch_size=32)
        eb = model.encode(b, normalize_embeddings=True, batch_size=32)
        return np.clip((ea * eb).sum(axis=1), -1, 1).astype(float).tolist()


class Entailment:
    """Cross-encoder NLI. entailment_both_ways = P(orig => rewrite) and P(rewrite => orig)."""

    def __init__(self, model_id: str | None = None):
        self.model_id = model_id or load_config()["preservation"]["nli_model"]

    def _load(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        def loader():
            return (AutoTokenizer.from_pretrained(self.model_id, revision=pinned_revision(self.model_id)),
                    AutoModelForSequenceClassification.from_pretrained(self.model_id, revision=pinned_revision(self.model_id)).eval())

        return manager().get(f"nli:{self.model_id}", loader)

    def probs(self, premises: list[str], hypotheses: list[str]) -> list[dict[str, float]]:
        import torch

        tok, model = self._load()
        id2 = {int(k): v.lower() for k, v in model.config.id2label.items()}
        out = []
        for i in range(0, len(premises), 16):
            enc = tok(premises[i:i + 16], hypotheses[i:i + 16], padding=True, truncation=True,
                      max_length=256, return_tensors="pt")
            with torch.inference_mode():
                p = torch.softmax(model(**enc).logits.float(), dim=-1).numpy()
            out.extend({id2[j]: float(row[j]) for j in range(len(row))} for row in p)
        return out

    def both_ways(self, originals: list[str], rewrites: list[str]) -> list[dict]:
        fwd = self.probs(originals, rewrites)
        bwd = self.probs(rewrites, originals)
        return [{"forward": f["entailment"], "backward": b["entailment"],
                 "contradiction": max(f["contradiction"], b["contradiction"]),
                 "both": min(f["entailment"], b["entailment"])} for f, b in zip(fwd, bwd)]


def fluency(original: str, rewrite: str) -> dict:
    """Heuristic fluency / sanity checks (no language model): length ratio, repetition,
    prompt leakage and character sanity. Score in [0, 1]; documented as a heuristic."""
    ow, rw = original.split(), rewrite.split()
    length_ratio = len(rw) / max(1, len(ow))
    trigrams = [tuple(rw[i:i + 3]) for i in range(max(0, len(rw) - 2))]
    rep = 0.0 if not trigrams else 1 - len(set(trigrams)) / len(trigrams)
    leak = bool(re.search(r"(rewritten text|rewrite the text|return only|^text:|stereotypes, biased)", rewrite, re.I))
    alpha = sum(c.isalpha() for c in rewrite) / max(1, len(rewrite))
    word_counts = Counter(w.lower() for w in rw)
    max_word_share = max(word_counts.values()) / len(rw) if rw else 1.0
    penalties = [
        min(1.0, rep * 2),
        1.0 if leak else 0.0,
        0.5 if alpha < 0.6 else 0.0,
        0.5 if (len(rw) > 6 and max_word_share > 0.3) else 0.0,
        1.0 if not rw else 0.0,
    ]
    score = max(0.0, 1.0 - sum(penalties))
    return {"fluency": score, "length_ratio": length_ratio, "repetition": rep, "prompt_leak": leak}


class PreservationSuite:
    def __init__(self):
        self.sim = SemanticSimilarity()
        self.nli = Entailment()

    def evaluate(self, originals: list[str], rewrites: list[str]) -> list[dict]:
        sims = self.sim.similarity(originals, rewrites)
        ents = self.nli.both_ways(originals, rewrites)
        out = []
        for o, r, s, e in zip(originals, rewrites, sims, ents):
            fc = fact_checker.check(o, r)
            fl = fluency(o, r)
            out.append({"semantic_similarity": s, "entailment": e, "facts": fc, "fluency": fl})
        return out
