"""Experiment tracking: SQLite run index + one JSON manifest per run (provenance).

Each manifest records: run id, task, config, seeds, dataset file hashes and versions, model IDs with pinned
revisions, compute device, hardware, git commit, start/end time, runtime, status, metrics and output paths.
Paths inside manifests are relative to the project root with forward slashes, so run folders can be copied
between Windows, macOS and Linux. The SQLite index is derived data: `rebuild_index()` recreates it from the
run folders' manifests.
"""
from __future__ import annotations

import atexit
import hashlib
import json
from contextlib import contextmanager
import sqlite3
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .. import __version__
from ..config import PROJECT_ROOT, outputs_dir, pinned_revision

DB_NAME = "runs.sqlite"
DATASET_MANIFEST = PROJECT_ROOT / "datasets" / "processed" / "dataset_manifest.json"
REQUIRED_FIELDS = (
    "run_id", "task", "kind", "status", "started_at", "finished_at", "runtime_s", "command", "config",
    "seed", "seeds", "git_commit", "datasets", "dataset_versions", "models", "model_revisions", "device",
    "hardware", "outputs", "fairlens_version",
)


def _db_path() -> Path:
    return outputs_dir() / DB_NAME


@contextmanager
def _connect(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Open the index, create the table if needed, commit on success and always close (Windows file locks)."""
    con = sqlite3.connect(path or _db_path())
    try:
        with con:
            con.execute("""CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY, kind TEXT, status TEXT, started_at TEXT, finished_at TEXT,
                runtime_s REAL, seed INTEGER, git_commit TEXT, command TEXT, manifest_path TEXT, metrics TEXT)""")
            yield con
    finally:
        con.close()


def rel_path(path: str | Path) -> str:
    """Project-relative POSIX path when inside the project, otherwise the absolute POSIX path."""
    p = Path(path)
    if not p.is_absolute():
        return p.as_posix()
    p = p.resolve()
    return p.relative_to(PROJECT_ROOT).as_posix() if p.is_relative_to(PROJECT_ROOT) else p.as_posix()


def abs_path(path: str | Path) -> Path:
    """Inverse of rel_path: resolve a stored path against the project root."""
    p = Path(str(path).replace("\\", "/"))
    return p if p.is_absolute() else PROJECT_ROOT / p


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=PROJECT_ROOT,
                               capture_output=True, text=True, timeout=5).stdout.strip()
        if out.returncode != 0 or not out.stdout.strip():
            return None
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return None


def dataset_versions(dataset_files: list[Path]) -> dict[str, Any]:
    """Source revisions of the processed datasets (from dataset_manifest.json) used by this run."""
    proc = DATASET_MANIFEST.parent
    used = [Path(p).name for p in dataset_files if Path(p).resolve().parent == proc.resolve()]
    if not used or not DATASET_MANIFEST.exists():
        return {}
    man = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8"))
    return {"manifest": rel_path(DATASET_MANIFEST), "manifest_sha": file_sha(DATASET_MANIFEST),
            "built_at": man.get("built_at"), "seed": man.get("seed"), "sources": man.get("sources", {}),
            "files": {n: man.get("files", {}).get(n) for n in used}}


def model_revision(ref: str | None) -> dict[str, str | None]:
    """Model ID and pinned Hub revision for a model reference (plain ID or a 'name|model|revision|...' cache identity)."""
    parts = (ref or "").split("|")
    model_id = parts[1] if len(parts) > 1 else parts[0]
    explicit = parts[2] if len(parts) > 2 and parts[2] else None
    if model_id and Path(model_id).is_absolute():
        model_id = rel_path(model_id)
    return {"model_id": model_id or None, "revision": explicit or pinned_revision(model_id)}


def _model_revisions(models: dict[str, str | None], config: dict[str, Any]) -> dict[str, dict]:
    out = {k: model_revision(v) for k, v in models.items()}
    base = config.get("base_model")
    if base and "base_model" not in out:
        out["base_model"] = {"model_id": base, "revision": config.get("revision") or pinned_revision(base)}
    return out


def _seeds(seed: int | None, config: dict[str, Any]) -> list:
    seeds = [seed] if seed is not None else []
    for s in config.get("seeds") or []:
        if s not in seeds:
            seeds.append(s)
    return seeds


def _portable(value: Any) -> Any:
    if isinstance(value, Path) or (isinstance(value, str) and Path(value).is_absolute()):
        return rel_path(value)
    return value


def _row(manifest: dict, manifest_path: Path) -> tuple:
    return (manifest["run_id"], manifest.get("kind") or manifest.get("task"), manifest.get("status"),
            manifest.get("started_at"), manifest.get("finished_at"), manifest.get("runtime_s"), manifest.get("seed"),
            manifest.get("git_commit"), manifest.get("command"), rel_path(manifest_path),
            json.dumps(manifest.get("metrics", {}), default=str))


class RunTracker:
    def __init__(self, kind: str, config: dict[str, Any] | None = None, seed: int | None = None,
                 dataset_files: list[Path] | None = None, models: dict[str, str | None] | None = None,
                 db_path: Path | None = None):
        from ..hardware import detect, device

        self.run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{kind}-{uuid.uuid4().hex[:6]}"
        self.kind, self.seed = kind, seed
        self.db_path = db_path or _db_path()
        self.dir = self.db_path.parent / "runs" / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._t0 = time.perf_counter()
        config = config or {}
        files = [Path(p) for p in (dataset_files or []) if Path(p).exists()]
        self.manifest: dict[str, Any] = {
            "run_id": self.run_id, "task": kind, "kind": kind, "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "finished_at": None, "runtime_s": None,
            "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]) if sys.argv else None,
            "config": config, "seed": seed, "seeds": _seeds(seed, config), "git_commit": git_commit(),
            "datasets": {rel_path(p): file_sha(p) for p in files},
            "dataset_versions": dataset_versions(files),
            "models": dict(models or {}), "model_revisions": _model_revisions(models or {}, config),
            "device": device(), "hardware": detect(check_internet=False).to_dict(),
            "outputs": {"run_dir": rel_path(self.dir), "manifest": rel_path(self.manifest_path)},
            "fairlens_version": __version__,
        }
        self._write()
        atexit.register(self._close_unfinished)

    def _close_unfinished(self) -> None:
        """Mark the run failed if the process exits (error or interrupt) before finish() was called."""
        if self.manifest.get("status") == "running":
            self.manifest["error"] = "process exited before the run finished (see the step log)"
            self.finish(metrics=self.manifest.get("metrics"), status="failed")

    @property
    def manifest_path(self) -> Path:
        return self.dir / "manifest.json"

    def _write(self) -> None:
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2, default=str), encoding="utf-8")
        with _connect(self.db_path) as con:
            con.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?)", _row(self.manifest, self.manifest_path))

    def add_models(self, models: dict[str, str | None]) -> None:
        self.manifest["models"].update(models)
        self.manifest["model_revisions"].update({k: model_revision(v) for k, v in models.items()})
        self._write()

    def save_artifact(self, name: str, obj: Any) -> Path:
        p = self.dir / name
        if hasattr(obj, "to_parquet") and name.endswith(".parquet"):
            obj.to_parquet(p, index=False)
        elif hasattr(obj, "to_csv") and name.endswith(".csv"):
            obj.to_csv(p, index=False, encoding="utf-8")
        else:
            p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
        self.manifest.setdefault("artifacts", {})[name] = rel_path(p)
        self._write()
        return p

    def finish(self, metrics: dict | None = None, artifacts: dict | None = None, status: str = "completed") -> None:
        self.manifest.update({"status": status, "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                              "runtime_s": round(time.perf_counter() - self._t0, 2), "metrics": metrics or {}})
        if artifacts:
            self.manifest.setdefault("artifacts", {}).update({k: _portable(v) for k, v in artifacts.items()})
        self._write()


def missing_fields(manifest: dict) -> list[str]:
    return [f for f in REQUIRED_FIELDS if f not in manifest]


def _resolve_manifest(stored: str, run_id: str, db: Path) -> str:
    p = abs_path(stored)
    if not p.exists():
        local = db.parent / "runs" / run_id / "manifest.json"
        if local.exists():
            p = local
    return str(p)


def list_runs(kind: str | None = None, db_path: Path | None = None) -> list[dict]:
    p = db_path or _db_path()
    if not p.exists():
        return []
    with _connect(p) as con:
        con.row_factory = sqlite3.Row
        q = "SELECT * FROM runs" + (" WHERE kind=?" if kind else "") + " ORDER BY started_at DESC"
        rows = [dict(r) for r in con.execute(q, (kind,) if kind else ())]
    for r in rows:
        r["metrics"] = json.loads(r["metrics"] or "{}")
        r["manifest_path"] = _resolve_manifest(r["manifest_path"], r["run_id"], p)
    return rows


def latest_run(kind: str, status: str = "completed", db_path: Path | None = None) -> dict | None:
    for r in list_runs(kind, db_path):
        if r["status"] == status:
            return r
    return None


def load_manifest(run: dict | str) -> dict:
    path = run["manifest_path"] if isinstance(run, dict) else run
    return json.loads(abs_path(path).read_text(encoding="utf-8"))


def rebuild_index(runs_dir: Path | None = None, db_path: Path | None = None) -> int:
    """Recreate the SQLite run index from every runs/<run_id>/manifest.json. Returns the number of runs indexed."""
    db = db_path or _db_path()
    runs_dir = runs_dir or db.parent / "runs"
    rows = []
    for mp in sorted(runs_dir.glob("*/manifest.json")):
        man = json.loads(mp.read_text(encoding="utf-8"))
        if man.get("run_id"):
            rows.append(_row(man, mp))
    db.parent.mkdir(parents=True, exist_ok=True)
    with _connect(db) as con:
        con.execute("DELETE FROM runs")
        con.executemany("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
    return len(rows)
