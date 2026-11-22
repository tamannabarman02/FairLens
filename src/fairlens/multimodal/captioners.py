"""Image captioners: BLIP, Florence-2 (native transformers port, no remote code) and a ViT-GPT2 fallback."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from ..config import load_config
from ..config import pinned_revision
from ..models import manager, place, run_with_cpu_fallback, to_model_device


@dataclass
class Caption:
    captioner: str
    model: str
    text: str | None
    runtime_ms: float
    error: str | None = None

    def to_dict(self):
        return asdict(self)


class Captioner:
    name = "captioner"
    display = "Captioner"

    def __init__(self, model_id: str, revision: str | None = None):
        self.model_id, self.revision = model_id, revision

    def _caption(self, image) -> str:
        raise NotImplementedError

    def caption(self, image) -> Caption:
        t0 = time.perf_counter()
        try:
            text = self._caption(image.convert("RGB"))
            return Caption(self.name, self.model_id, text.strip(), (time.perf_counter() - t0) * 1000)
        except Exception as exc:  # noqa: BLE001
            return Caption(self.name, self.model_id, None, (time.perf_counter() - t0) * 1000,
                           f"{type(exc).__name__}: {exc}")

    def info(self):
        return {"name": self.name, "display": self.display, "model": self.model_id, "revision": self.revision}


class BlipCaptioner(Captioner):
    name, display = "blip", "BLIP"

    def _load(self):
        from transformers import BlipForConditionalGeneration, BlipProcessor

        def loader():
            rev = self.revision or pinned_revision(getattr(self, "model_id", None) or getattr(self, "model_path", None))
            kw = {"revision": rev} if rev else {}
            return (BlipProcessor.from_pretrained(self.model_id, **kw),
                    place(BlipForConditionalGeneration.from_pretrained(self.model_id, **kw).eval()))

        return manager().get(f"cap:{self.model_id}", loader)

    def _caption(self, image):
        import torch

        proc, model = self._load()
        inputs = proc(images=image, return_tensors="pt")

        def run():
            with torch.inference_mode():
                return model.generate(**to_model_device(inputs, model), max_new_tokens=40, num_beams=3).cpu()

        ids = run_with_cpu_fallback(model, run)
        return proc.decode(ids[0], skip_special_tokens=True)


class FlorenceCaptioner(Captioner):
    """Florence-2 via the transformers-native `Florence2ForConditionalGeneration` port
    (florence-community/*), so `trust_remote_code` is not needed."""

    name, display = "florence", "Florence-2"
    task = "<CAPTION>"

    def _load(self):
        from transformers import AutoProcessor, Florence2ForConditionalGeneration

        def loader():
            rev = self.revision or pinned_revision(getattr(self, "model_id", None) or getattr(self, "model_path", None))
            kw = {"revision": rev} if rev else {}
            return (AutoProcessor.from_pretrained(self.model_id, **kw),
                    place(Florence2ForConditionalGeneration.from_pretrained(self.model_id, **kw).eval(), allow_mps=False))

        return manager().get(f"cap:{self.model_id}", loader)

    def _caption(self, image):
        import torch

        proc, model = self._load()
        inputs = proc(text=self.task, images=image, return_tensors="pt")

        def run():
            with torch.inference_mode():
                return model.generate(**to_model_device(inputs, model), max_new_tokens=40, num_beams=3).cpu()

        ids = run_with_cpu_fallback(model, run)
        raw = proc.batch_decode(ids, skip_special_tokens=False)[0]
        parsed = proc.post_process_generation(raw, task=self.task, image_size=image.size)
        return parsed.get(self.task, raw) if isinstance(parsed, dict) else str(parsed)


class VitGpt2Captioner(Captioner):
    name, display = "vit_gpt2", "ViT-GPT2"

    def _load(self):
        from transformers import AutoTokenizer, ViTImageProcessor, VisionEncoderDecoderModel

        def loader():
            return (ViTImageProcessor.from_pretrained(self.model_id, revision=pinned_revision(self.model_id)), AutoTokenizer.from_pretrained(self.model_id, revision=pinned_revision(self.model_id)),
                    place(VisionEncoderDecoderModel.from_pretrained(self.model_id, revision=pinned_revision(self.model_id)).eval()))

        return manager().get(f"cap:{self.model_id}", loader)

    def _caption(self, image):
        import torch

        proc, tok, model = self._load()
        px = proc(images=image, return_tensors="pt").pixel_values

        def run():
            with torch.inference_mode():
                return model.generate(to_model_device(px, model), max_new_tokens=40, num_beams=3).cpu()

        ids = run_with_cpu_fallback(model, run)
        return tok.decode(ids[0], skip_special_tokens=True)


def build_captioners(use_fallback_for_florence: bool = False) -> dict[str, Captioner]:
    spec = load_config()["captioning"]["captioners"]
    out: dict[str, Captioner] = {"blip": BlipCaptioner(spec["blip"]["model_id"], spec["blip"].get("revision"))}
    f = spec["florence"]
    if use_fallback_for_florence:
        out["florence"] = VitGpt2Captioner(f["fallback_model_id"])
    else:
        out["florence"] = FlorenceCaptioner(f["model_id"], f.get("revision"))
    return out
