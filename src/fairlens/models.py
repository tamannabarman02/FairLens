"""Lazy, memory-budgeted model loading.

Only a bounded amount of model weight is kept resident. Loading a model that would exceed
the budget evicts the least-recently-used ones first, so on a CPU laptop with little free RAM
only one large model is in memory at a time.
"""
from __future__ import annotations

import gc
import threading
import time
from collections import OrderedDict
from typing import Any, Callable

from . import config  # noqa: F401  (sets HF cache before transformers import)
from .hardware import select_profile


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
