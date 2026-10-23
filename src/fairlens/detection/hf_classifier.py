"""Sequence-classification detectors: our fine-tuned BERT/DeBERTa checkpoints and Hub classifiers."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from ..config import pinned_revision
from ..hardware import place, to_model
from ..models import manager
from .base import Detector, Span


class HFClassifierDetector(Detector):
    """Wraps an AutoModelForSequenceClassification.

    `biased_label_ids` lists the class indices that mean "biased"; their probabilities are summed.
    """

    def __init__(self, name: str, display: str, model_path: str, revision: str | None = None,
                 biased_label_ids: list[int] | None = None, max_length: int = 128):
        self.name, self.display = name, display
        self.model_path, self.revision = model_path, revision
        self.max_length = max_length
        self._biased_ids = biased_label_ids

    # -- loading -----------------------------------------------------------------
    def _load(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        def loader():
            rev = self.revision or pinned_revision(getattr(self, "model_id", None) or getattr(self, "model_path", None))
            kw = {"revision": rev} if rev else {}
            tok = AutoTokenizer.from_pretrained(self.model_path, **kw)
            model = place(AutoModelForSequenceClassification.from_pretrained(self.model_path, **kw).eval())
            return tok, model

        return manager().get(f"det:{self.model_path}", loader)

    def biased_ids(self, model) -> list[int]:
        if self._biased_ids is not None:
            return self._biased_ids
        id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
        ids = [i for i, lab in id2label.items() if re.search(r"bias|stereo|label_1|^1$|toxic|subjective", lab)
               and not re.search(r"non|not|un", lab)]
        return ids or [max(id2label)]

    # -- inference ---------------------------------------------------------------
    def _probs(self, texts: list[str]) -> np.ndarray:
        import torch

        tok, model = self._load()
        enc = to_model(tok(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"), model)
        with torch.inference_mode():
            logits = model(**enc).logits
        return torch.softmax(logits.float(), dim=-1).cpu().numpy()

    def _scores(self, texts):
        tok, model = self._load()
        ids = self.biased_ids(model)
        return self._probs(texts)[:, ids].sum(axis=1).tolist()

    def explain(self, text: str, top_k: int = 3) -> list[Span]:
        """Leave-one-word-out occlusion: words whose removal lowers the bias score most."""
        words = [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"[\w'-]+", text)]
        if not words or len(words) > 80:
            return []
        base = self._scores([text])[0]
        variants = [text[:s] + text[e:] for s, e, _ in words]
        drops = []
        for i in range(0, len(variants), 16):
            drops.extend(base - np.array(self._scores(variants[i : i + 16])))
        order = np.argsort(drops)[::-1][:top_k]
        return [Span(words[i][0], words[i][1], words[i][2], "model_attribution",
                     f"Removing this word lowers {self.display}'s bias score by {drops[i]:.2f}.", float(drops[i]))
                for i in order if drops[i] > 0.05]

    def info(self) -> dict:
        meta = {}
        p = Path(self.model_path) / "fairlens_training.json"
        if p.exists():
            meta = json.loads(p.read_text(encoding="utf-8"))
        return {**super().info(), "model": self.model_path, "revision": self.revision, "training": meta}


class ZeroShotNLIDetector(Detector):
    """Zero-shot detection with an MNLI model: compares entailment of a 'biased' hypothesis
    against a 'neutral' hypothesis (softmax over the two entailment logits)."""

    CATEGORY_HYPOTHESES = {
        "gender": "This text stereotypes people based on their gender.",
        "age": "This text stereotypes people based on their age.",
        "race_ethnicity": "This text stereotypes people based on their race or ethnicity.",
        "religion": "This text stereotypes people based on their religion.",
        "disability": "This text stereotypes people with disabilities.",
        "sexual_orientation": "This text stereotypes people based on their sexual orientation.",
        "socioeconomic": "This text stereotypes people based on wealth or social class.",
        "nationality": "This text stereotypes people based on their nationality.",
    }

    def __init__(self, name: str, display: str, model_id: str, hypotheses: dict[str, str],
                 revision: str | None = None, max_length: int = 256):
        """hypotheses: {"biased": ..., "neutral": ...} -> 'pair' mode (softmax of the two entailment logits);
        {"biased": ...} only -> 'single' mode (entailment vs contradiction for the biased hypothesis)."""
        self.name, self.display, self.model_id, self.revision = name, display, model_id, revision
        self.hyp_biased, self.hyp_neutral = hypotheses["biased"], hypotheses.get("neutral")
        self.max_length = max_length

    def _load(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        def loader():
            rev = self.revision or pinned_revision(getattr(self, "model_id", None) or getattr(self, "model_path", None))
            kw = {"revision": rev} if rev else {}
            tok = AutoTokenizer.from_pretrained(self.model_id, **kw)
            model = place(AutoModelForSequenceClassification.from_pretrained(self.model_id, **kw).eval())
            return tok, model

        return manager().get(f"nli:{self.model_id}", loader)

    def _entail_logits(self, premises: list[str], hypothesis: str) -> np.ndarray:
        import torch

        tok, model = self._load()
        ent = {v.lower(): int(k) for k, v in model.config.id2label.items()}["entailment"]
        enc = tok(premises, [hypothesis] * len(premises), padding=True, truncation="only_first",
                  max_length=self.max_length, return_tensors="pt")
        with torch.inference_mode():
            return model(**to_model(enc, model)).logits[:, ent].float().cpu().numpy()

    def _ec_logits(self, premises: list[str], hypothesis: str) -> np.ndarray:
        """Return entailment-minus-contradiction logits (single-hypothesis zero-shot)."""
        import torch

        tok, model = self._load()
        lab = {v.lower(): int(k) for k, v in model.config.id2label.items()}
        enc = tok(premises, [hypothesis] * len(premises), padding=True, truncation="only_first",
                  max_length=self.max_length, return_tensors="pt")
        with torch.inference_mode():
            lg = model(**to_model(enc, model)).logits.float().cpu().numpy()
        return lg[:, lab["entailment"]] - lg[:, lab["contradiction"]]

    def _scores(self, texts):
        if self.hyp_neutral:
            b = self._entail_logits(texts, self.hyp_biased)
            n = self._entail_logits(texts, self.hyp_neutral)
            d = b - n
        else:
            d = self._ec_logits(texts, self.hyp_biased)
        return (1 / (1 + np.exp(-d))).tolist()

    def category_scores(self, text: str) -> dict[str, float]:
        """Per-category entailment probability (entailment vs contradiction) for explanation."""
        import torch

        tok, model = self._load()
        lab = {v.lower(): int(k) for k, v in model.config.id2label.items()}
        hyps = list(self.CATEGORY_HYPOTHESES.values())
        enc = tok([text] * len(hyps), hyps, padding=True, truncation="only_first",
                  max_length=self.max_length, return_tensors="pt")
        with torch.inference_mode():
            lg = model(**to_model(enc, model)).logits[:, [lab["contradiction"], lab["entailment"]]].float().cpu()
        p = torch.softmax(lg, dim=-1)[:, 1].numpy()
        return dict(zip(self.CATEGORY_HYPOTHESES, map(float, p)))

    def info(self) -> dict:
        return {**super().info(), "model": self.model_id, "revision": self.revision,
                "hypotheses": {"biased": self.hyp_biased, "neutral": self.hyp_neutral}}
