"""Caption evaluation: CLIP image-text alignment, length, detail coverage and downstream bias findings.

Scope: this measures bias in *generated captions* (language), not all visual bias in the image.
"""
from __future__ import annotations

import re

from ..config import load_config, pinned_revision
from ..hardware import place, to_model
from ..models import manager

STOP = set("a an the of on in at with and or to is are was were be its it this that his her their there "
           "some two three one near next by for from up down front back top while has have".split())

# Attributes a caption cannot normally verify from pixels alone; flagged as possible unsupported assumptions.
UNSUPPORTED_ATTRS = re.compile(
    r"\b(?:rich|poor|wealthy|homeless|immigrant|foreign|illegal|criminal|terrorist|gay|lesbian|straight|"
    r"christian|muslim|jewish|hindu|religious|married|single|mother|father|wife|husband|nationality|"
    r"american|african|asian|mexican|indian|arab|lazy|smart|stupid|angry|aggressive|dangerous|sexy|"
    r"attractive|ugly|fat|beautiful|pretty|handsome)\b", re.I)
GENDER_WORDS = re.compile(r"\b(?:man|men|woman|women|boy|boys|girl|girls|lady|ladies|gentleman|male|female|he|she|his|her)\b", re.I)


class ClipAlignment:
    def __init__(self, model_id: str | None = None):
        self.model_id = model_id or load_config()["captioning"]["alignment_model"]

    def _load(self):
        from transformers import CLIPModel, CLIPProcessor

        def loader():
            return CLIPProcessor.from_pretrained(self.model_id, revision=pinned_revision(self.model_id)), place(CLIPModel.from_pretrained(self.model_id, revision=pinned_revision(self.model_id)).eval())

        return manager().get(f"clip:{self.model_id}", loader)

    def score(self, image, captions: list[str]) -> list[float]:
        """Cosine similarity between image and each caption embedding (CLIPScore without the 2.5 scale)."""
        import torch

        proc, model = self._load()
        inputs = proc(text=captions, images=image.convert("RGB"), return_tensors="pt", padding=True, truncation=True)
        with torch.inference_mode():
            out = model(**to_model(inputs, model))
        img = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        txt = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        return (txt @ img.T).squeeze(-1).float().cpu().tolist()


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOP and len(w) > 2}


def detail_coverage(caption: str, references: list[str]) -> float | None:
    """Share of content words in the reference captions (COCO) that the generated caption mentions."""
    if not references:
        return None
    ref = set().union(*(content_words(r) for r in references))
    return len(content_words(caption) & ref) / max(1, len(ref))


def alignment_label(clip_sim: float) -> str:
    # Bands chosen from the observed distribution of matched COCO captions (docs/METHODOLOGY.md).
    if clip_sim >= 0.30:
        return "High"
    if clip_sim >= 0.25:
        return "Medium"
    return "Low"


def caption_flags(caption: str) -> dict:
    return {
        "unsupported_attributes": sorted({m.group(0).lower() for m in UNSUPPORTED_ATTRS.finditer(caption)}),
        "gendered_terms": sorted({m.group(0).lower() for m in GENDER_WORDS.finditer(caption)}),
        "n_words": len(caption.split()),
    }
