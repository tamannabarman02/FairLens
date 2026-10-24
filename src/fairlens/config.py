"""Configuration loading and project paths.

Importing this module sets the Hugging Face cache location *before* transformers is
imported anywhere, so model downloads land in one per-user cache folder on every OS.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "models.yaml"


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict[str, Any]:
    with open(path or CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


LOCK_PATH = PROJECT_ROOT / "configs" / "model_lock.json"


@lru_cache(maxsize=1)
def _lock() -> dict[str, dict]:
    import json

    return json.loads(LOCK_PATH.read_text(encoding="utf-8")) if LOCK_PATH.exists() else {}


def pinned_revision(model_id: str | None) -> str | None:
    """Commit hash recorded by scripts/verify_models.py for a Hub model (None for local paths)."""
    return _lock().get(model_id or "", {}).get("sha")


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def outputs_dir(*parts: str) -> Path:
    p = PROJECT_ROOT / load_config()["paths"]["outputs"]
    p = p.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def default_cache_dir() -> Path:
    """Per-user model cache: %LOCALAPPDATA% on Windows, ~/Library/Caches on macOS, ~/.cache elsewhere."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        return base / "FairLens" / "hf"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "FairLens" / "hf"
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "fairlens" / "hf"


def _configure_cache() -> None:
    cache = os.environ.get("FAIRLENS_CACHE") or load_config()["paths"].get("cache_dir")
    cache_path = Path(cache).expanduser() if cache else default_cache_dir()
    try:
        cache_path.mkdir(parents=True, exist_ok=True)
    except OSError:  # configured drive missing on another machine -> project-local cache
        cache_path = PROJECT_ROOT / ".cache" / "hf"
        cache_path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_path))
    os.environ.setdefault("TORCH_HOME", str(cache_path.parent / "torch"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure") and (stream.encoding or "").lower() not in {"utf-8", "utf8"}:
            stream.reconfigure(encoding="utf-8", errors="replace")


_configure_cache()
