"""Configuration loading and project paths.

Importing this module sets the Hugging Face cache location and PyTorch device environment *before*
transformers or torch is imported anywhere, so behaviour is identical on Windows, macOS and Linux.

Cache location, first match wins:
  1. FAIRLENS_CACHE environment variable
  2. paths.cache_dir in configs/local.yaml (per-machine, gitignored)
  3. paths.cache_dir in configs/models.yaml if set (null by default)
  4. a per-user folder: ~/.cache/fairlens/hf
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "models.yaml"
LOCAL_CONFIG_PATH = PROJECT_ROOT / "configs" / "local.yaml"
LOCK_PATH = PROJECT_ROOT / "configs" / "model_lock.json"


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict[str, Any]:
    with open(path or CONFIG_PATH, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if path is None and LOCAL_CONFIG_PATH.exists():
        with open(LOCAL_CONFIG_PATH, encoding="utf-8") as fh:
            cfg = _merge(cfg, yaml.safe_load(fh) or {})
    return cfg


@lru_cache(maxsize=1)
def _lock() -> dict[str, dict]:
    return json.loads(LOCK_PATH.read_text(encoding="utf-8")) if LOCK_PATH.exists() else {}


def pinned_revision(model_id: str | None) -> str | None:
    """Commit hash recorded by scripts/verify_models.py for a Hub model (None for local paths)."""
    return _lock().get(model_id or "", {}).get("sha")


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def rel_posix(p: Path | str) -> str:
    """Path relative to the project root with forward slashes (portable across OSes); absolute if outside."""
    p = Path(p)
    try:
        return p.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def outputs_dir(*parts: str) -> Path:
    p = PROJECT_ROOT / load_config()["paths"]["outputs"]
    p = p.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def default_cache_dir() -> Path:
    return Path.home() / ".cache" / "fairlens" / "hf"


def _configure_env() -> None:
    cache = os.environ.get("FAIRLENS_CACHE") or load_config()["paths"].get("cache_dir") or default_cache_dir()
    cache_path = Path(cache).expanduser()
    try:
        cache_path.mkdir(parents=True, exist_ok=True)
    except OSError:  # configured location unavailable on this machine -> per-user default
        cache_path = default_cache_dir()
        cache_path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_path))
    os.environ.setdefault("TORCH_HOME", str(cache_path.parent / "torch"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Apple MPS: run unsupported ops on CPU instead of failing (set in code so every OS/shell behaves the same).
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


_configure_env()
