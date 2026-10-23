"""Build detectors from configs/models.yaml."""
from __future__ import annotations

from pathlib import Path

from ..config import PROJECT_ROOT, load_config
from .base import Detector
from .hf_classifier import HFClassifierDetector, ZeroShotNLIDetector
from .rule_based import RuleBasedDetector

DEFAULT_ORDER = ["bert", "deberta", "bart_mnli", "rule_based"]


def build_detector(key: str, lite: bool = False) -> Detector | None:
    """Return the detector for a config key, or None if it is unavailable (e.g. not trained yet)."""
    spec = load_config()["detectors"][key]
    adapter = spec["adapter"]
    if adapter == "rule_based":
        return RuleBasedDetector()
    if adapter == "hf_finetuned":
        ckpt = PROJECT_ROOT / spec["checkpoint"]
        if not (ckpt / "config.json").exists():
            return None
        return HFClassifierDetector(key, spec["display"], str(ckpt), biased_label_ids=[1])
    if adapter == "zero_shot_nli":
        model_id = spec["lite_model_id"] if lite else spec["model_id"]
        return ZeroShotNLIDetector(key, spec["display"], model_id, spec["hypotheses"], spec.get("revision"))
    if adapter == "hf_pretrained":
        if not spec.get("model_id"):
            return None
        return HFClassifierDetector(key, spec["display"], spec["model_id"], spec.get("revision"),
                                    spec.get("biased_label_ids"))
    raise ValueError(f"unknown detector adapter {adapter}")


def build_detectors(keys: list[str] | None = None, lite: bool = False) -> dict[str, Detector]:
    out = {}
    for k in keys or DEFAULT_ORDER:
        d = build_detector(k, lite=lite)
        if d is not None:
            out[k] = d
    return out


def missing_detectors(keys: list[str] | None = None) -> list[str]:
    return [k for k in keys or DEFAULT_ORDER if build_detector(k) is None]
