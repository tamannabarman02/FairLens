"""Read real run outputs (never placeholders) for the dashboard."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..config import PROJECT_ROOT
from ..experiments.tracking import latest_run, list_runs, load_manifest


def run_dir(run: dict) -> Path:
    return Path(run["manifest_path"]).parent


def latest_summary(kind: str, filename: str) -> dict | None:
    r = latest_run(kind)
    if not r:
        return None
    p = run_dir(r) / filename
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    d.setdefault("run_id", r["run_id"])
    d["_finished_at"] = r.get("finished_at")
    return d


def latest_frame(kind: str, filename: str) -> pd.DataFrame | None:
    r = latest_run(kind)
    if not r:
        return None
    p = run_dir(r) / filename
    return pd.read_parquet(p) if p.exists() else None


def runs(kind: str | None = None) -> list[dict]:
    return list_runs(kind)


def manifest(run: dict) -> dict:
    return load_manifest(run)


def model_verification() -> dict | None:
    p = PROJECT_ROOT / "outputs" / "model_verification" / "verification.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def dataset_manifest() -> dict | None:
    p = PROJECT_ROOT / "datasets" / "processed" / "dataset_manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def use_case_results() -> dict[str, dict]:
    d = PROJECT_ROOT / "outputs" / "use_cases"
    if not d.exists():
        return {}
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in d.glob("*.json")}


def latest_with_manifest(kind: str, filename: str) -> tuple[dict | None, dict | None]:
    """(summary, manifest) of the latest completed run of `kind`, or (None, None)."""
    r = latest_run(kind)
    if not r:
        return None, None
    s = latest_summary(kind, filename)
    return s, (load_manifest(r) if s else None)


def failed_runs(kind: str) -> list[dict]:
    return [r for r in list_runs(kind) if r["status"] in {"failed", "stopped"}]
