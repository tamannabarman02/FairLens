"""Lazy, memory-budgeted model loading.

Only a bounded amount of model weight is kept resident. Loading a model that would exceed
the budget evicts the least-recently-used ones first, so on a CPU laptop with little free RAM
only one large model is in memory at a time.
"""
from __future__ import annotations

import gc
import threading
import time
import warnings
from collections import OrderedDict
from typing import Any, Callable, TypeVar

from . import config  # noqa: F401  (sets HF cache before transformers import)
from .hardware import device as selected_device
from .hardware import select_profile

T = TypeVar("T")


def place(model: Any, device: str | None = None, allow_mps: bool = True) -> Any:
    """Move a torch module to `device` (default: hardware.device()); fall back to CPU if that fails.

    allow_mps=False keeps models with ops unsupported on Apple MPS on the CPU.
    """
    dev = device or selected_device()
    if dev == "mps" and not allow_mps:
        dev = "cpu"
    if dev == "cpu":
        return model.to("cpu")
    try:
        return model.to(dev)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"Could not move {type(model).__name__} to {dev} ({exc}); using CPU.")
        return model.to("cpu")


def model_device(model: Any) -> Any:
    """torch.device the module's parameters live on (CPU if it has none)."""
    import torch

    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError):
        return torch.device("cpu")


def to_model_device(inputs: Any, model: Any) -> Any:
    """Move a tensor, BatchEncoding or dict of tensors to the model's device."""
    dev = model_device(model)
    if hasattr(inputs, "to"):
        return inputs.to(dev)
    if isinstance(inputs, dict):
        return {k: v.to(dev) if hasattr(v, "to") else v for k, v in inputs.items()}
    return inputs


def run_with_cpu_fallback(model: Any, fn: Callable[[], T]) -> T:
    """Run fn(); if it fails on cuda/mps (unsupported op, out of memory), move the model to CPU and retry once.

    fn must place its inputs with to_model_device so the retry uses CPU tensors.
    """
    try:
        return fn()
    except (RuntimeError, NotImplementedError) as exc:
        if model_device(model).type == "cpu":
            raise
        warnings.warn(f"{type(model).__name__} failed on {model_device(model).type} ({exc}); retrying on CPU.")
        model.to("cpu")
        return fn()


def param_mb(model: Any) -> float:
    """Approximate in-memory size of a torch model in MB."""
    try:
        return sum(p.numel() * p.element_size() for p in model.parameters()) / 2**20
    except Exception:  # noqa: BLE001
        return 0.0


class ModelManager:
    def __init__(self, max_resident_mb: float | None = None):
        if max_resident_mb is None:
            prof = select_profile()
            max_resident_mb = config.load_config()["hardware_profiles"][prof]["max_resident_mb"]
        self.max_resident_mb = max_resident_mb
        self._models: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = threading.RLock()
        self.load_log: list[dict] = []

    def get(self, key: str, loader: Callable[[], Any], size_mb: Callable[[Any], float] | None = None) -> Any:
        with self._lock:
            if key in self._models:
                self._models.move_to_end(key)
                return self._models[key][0]
            t0 = time.perf_counter()
            # Free room up-front (half the budget): the incoming model is usually large and
            # loading it next to others is what exhausts RAM on small machines.
            self._evict_until(self.max_resident_mb * 0.5)
            obj = loader()
            mb = (size_mb or _obj_mb)(obj)
            self._models[key] = (obj, mb)
            self._evict_until(self.max_resident_mb, keep=key)
            self.load_log.append({"key": key, "mb": round(mb, 1), "load_s": round(time.perf_counter() - t0, 2)})
            return obj

    def _evict_until(self, limit_mb: float, keep: str | None = None) -> None:
        """Release least-recently-used models until resident size <= limit_mb."""
        for key in list(self._models):
            if self.resident_mb() <= limit_mb:
                break
            if key != keep:
                self.release(key)

    def release(self, key: str) -> None:
        with self._lock:
            if self._models.pop(key, None) is not None:
                gc.collect()

    def clear(self) -> None:
        with self._lock:
            self._models.clear()
            gc.collect()

    def resident_mb(self) -> float:
        return sum(mb for _, mb in self._models.values())

    def resident(self) -> dict[str, float]:
        return {k: round(mb, 1) for k, (_, mb) in self._models.items()}


def _obj_mb(obj: Any) -> float:
    if isinstance(obj, (tuple, list)):
        return sum(_obj_mb(o) for o in obj)
    if isinstance(obj, dict):
        return sum(_obj_mb(o) for o in obj.values())
    if hasattr(obj, "parameters"):
        return param_mb(obj)
    if hasattr(obj, "model") and hasattr(obj.model, "parameters"):  # pipelines / wrappers
        return param_mb(obj.model)
    return 0.0


_MANAGER: ModelManager | None = None


def manager() -> ModelManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = ModelManager()
    return _MANAGER


def set_manager(m: ModelManager) -> None:
    global _MANAGER
    _MANAGER = m


def rss_mb() -> float:
    import psutil

    return psutil.Process().memory_info().rss / 2**20
