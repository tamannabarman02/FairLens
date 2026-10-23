"""Experiment tracking: SQLite run index + one JSON manifest per run (provenance).

Each manifest records: run id, kind, config, seed, dataset file hashes, model revisions, hardware,
git commit, start/end time, runtime, status, metrics and artifact paths.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sqlite3
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import PROJECT_ROOT, outputs_dir, rel_posix

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
        hw = detect(check_internet=False).to_dict()
        self.manifest: dict[str, Any] = {
            "run_id": self.run_id, "kind": kind, "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]) if sys.argv else None,
            "config": config or {}, "seed": seed, "git_commit": git_commit(),
            "datasets": {rel_posix(p): file_sha(Path(p)) for p in (dataset_files or []) if Path(p).exists()},
            "models": models or {}, "hardware": hw, "device": hw["device"],
            "platform": platform.platform(), "python": platform.python_version(),
            "output_dir": rel_posix(self.dir), "artifacts": {},
        }
        self._write()

    @property
    def manifest_path(self) -> Path:
        return self.dir / "manifest.json"

    def _write(self) -> None:
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2, default=str), encoding="utf-8")
        _index_row(self.db_path, self.manifest)

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
            p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
        self.manifest.setdefault("artifacts", {})[name] = rel_posix(p)
        self._write()
        return p

    def finish(self, metrics: dict | None = None, artifacts: dict | None = None, status: str = "completed") -> None:
        self.manifest.update({"status": status, "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                              "runtime_s": round(time.perf_counter() - self._t0, 2), "metrics": metrics or {}})
        if artifacts:
            self.manifest.setdefault("artifacts", {}).update(artifacts)
        self._write()


REQUIRED_MANIFEST_FIELDS = ["run_id", "kind", "status", "started_at", "command", "config", "seed", "git_commit",
                            "datasets", "models", "hardware", "device", "platform", "python", "output_dir", "artifacts"]


def missing_fields(manifest: dict) -> list[str]:
    """Required provenance fields that are absent (V4 1.6). Finished runs must also have runtime and metrics."""
    req = REQUIRED_MANIFEST_FIELDS + (["finished_at", "runtime_s", "metrics"] if manifest.get("status") == "completed" else [])
    return [f for f in req if f not in manifest]


def _row(m: dict, manifest_path: Path) -> dict:
    return {"run_id": m["run_id"], "kind": m["kind"], "status": m.get("status"), "started_at": m.get("started_at"),
            "finished_at": m.get("finished_at"), "runtime_s": m.get("runtime_s"), "seed": m.get("seed"),
            "git_commit": m.get("git_commit"), "command": m.get("command"), "manifest_path": str(manifest_path),
            "metrics": m.get("metrics", {}), "device": m.get("device") or m.get("hardware", {}).get("device")}


def _index_row(db_path: Path, m: dict) -> None:
    """Best-effort update of the SQLite index (a cache; the run folders are the source of truth)."""
    try:
        with _connect(db_path) as con:
            con.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
                m["run_id"], m["kind"], m.get("status"), m.get("started_at"), m.get("finished_at"), m.get("runtime_s"),
                m.get("seed"), m.get("git_commit"), m.get("command"), m["run_id"], json.dumps(m.get("metrics", {}), default=str)))
    except sqlite3.Error:
        pass


def list_runs(kind: str | None = None, db_path: Path | None = None) -> list[dict]:
    """All runs, read from outputs/runs/*/manifest.json (folders copied from a teammate appear automatically)."""
    runs_dir = (db_path or _db_path()).parent / "runs"
    rows = []
    for mp in runs_dir.glob("*/manifest.json"):
        try:
            m = json.loads(mp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if kind is None or m.get("kind") == kind:
            rows.append(_row(m, mp))
    return sorted(rows, key=lambda r: r["started_at"] or "", reverse=True)


def rebuild_index(db_path: Path | None = None) -> int:
    """Rebuild runs.sqlite from the run folders; returns the number of runs indexed."""
    p = db_path or _db_path()
    if p.exists():
        p.unlink()
    n = 0
    for r in list_runs(db_path=p):
        _index_row(p, load_manifest(r))
        n += 1
    return n


def latest_run(kind: str, status: str = "completed", db_path: Path | None = None) -> dict | None:
    for r in list_runs(kind, db_path):
        if r["status"] == status:
            return r
    return None


def load_manifest(run: dict | str) -> dict:
    path = run["manifest_path"] if isinstance(run, dict) else run
    return json.loads(Path(path).read_text(encoding="utf-8"))
