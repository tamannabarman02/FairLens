"""Experiment tracking: SQLite run index + one JSON manifest per run (provenance).

Each manifest records: run id, kind, config, seed, dataset file hashes, model revisions, hardware,
git commit, start/end time, runtime, status, metrics and artifact paths.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import PROJECT_ROOT, outputs_dir

DB_NAME = "runs.sqlite"


def _db_path() -> Path:
    return outputs_dir() / DB_NAME


def _connect(path: Path | None = None) -> sqlite3.Connection:
    con = sqlite3.connect(path or _db_path())
    con.execute("""CREATE TABLE IF NOT EXISTS runs (
        run_id TEXT PRIMARY KEY, kind TEXT, status TEXT, started_at TEXT, finished_at TEXT,
        runtime_s REAL, seed INTEGER, git_commit TEXT, command TEXT, manifest_path TEXT, metrics TEXT)""")
    return con


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
        return (out.stdout.strip() + ("-dirty" if dirty else "")) or None
    except (OSError, subprocess.SubprocessError):
        return None


class RunTracker:
    def __init__(self, kind: str, config: dict[str, Any] | None = None, seed: int | None = None,
                 dataset_files: list[Path] | None = None, models: dict[str, str | None] | None = None,
                 db_path: Path | None = None):
        from ..hardware import detect

        self.run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{kind}-{uuid.uuid4().hex[:6]}"
        self.kind, self.seed = kind, seed
        self.db_path = db_path or _db_path()
        self.dir = self.db_path.parent / "runs" / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._t0 = time.perf_counter()
        self.manifest: dict[str, Any] = {
            "run_id": self.run_id, "kind": kind, "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]) if sys.argv else None,
            "config": config or {}, "seed": seed, "git_commit": git_commit(),
            "datasets": {str(Path(p).relative_to(PROJECT_ROOT)) if Path(p).is_relative_to(PROJECT_ROOT) else str(p): file_sha(Path(p))
                         for p in (dataset_files or []) if Path(p).exists()},
            "models": models or {}, "hardware": detect(check_internet=False).to_dict(),
        }
        self._write()

    @property
    def manifest_path(self) -> Path:
        return self.dir / "manifest.json"

    def _write(self) -> None:
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2, default=str))
        with _connect(self.db_path) as con:
            con.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
                self.run_id, self.kind, self.manifest["status"], self.manifest["started_at"],
                self.manifest.get("finished_at"), self.manifest.get("runtime_s"), self.seed,
                self.manifest["git_commit"], self.manifest["command"], str(self.manifest_path),
                json.dumps(self.manifest.get("metrics", {}), default=str)))

    def add_models(self, models: dict[str, str | None]) -> None:
        self.manifest["models"].update(models)
        self._write()

    def save_artifact(self, name: str, obj: Any) -> Path:
        p = self.dir / name
        if hasattr(obj, "to_parquet") and name.endswith(".parquet"):
            obj.to_parquet(p, index=False)
        elif hasattr(obj, "to_csv") and name.endswith(".csv"):
            obj.to_csv(p, index=False)
        else:
            p.write_text(json.dumps(obj, indent=2, default=str))
        self.manifest.setdefault("artifacts", {})[name] = str(p.relative_to(PROJECT_ROOT)) if p.is_relative_to(PROJECT_ROOT) else str(p)
        self._write()
        return p

    def finish(self, metrics: dict | None = None, artifacts: dict | None = None, status: str = "completed") -> None:
        self.manifest.update({"status": status, "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                              "runtime_s": round(time.perf_counter() - self._t0, 2), "metrics": metrics or {}})
        if artifacts:
            self.manifest.setdefault("artifacts", {}).update(artifacts)
        self._write()


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
    return rows


def latest_run(kind: str, status: str = "completed", db_path: Path | None = None) -> dict | None:
    for r in list_runs(kind, db_path):
        if r["status"] == status:
            return r
    return None


def load_manifest(run: dict | str) -> dict:
    path = run["manifest_path"] if isinstance(run, dict) else run
    return json.loads(Path(path).read_text())
