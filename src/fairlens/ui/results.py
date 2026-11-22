"""Read real run outputs (never placeholders) for the dashboard.

Every research number shown in the UI must come through this module, and this module only reads files under
`outputs/` (run summaries, manifests, use-case results) plus the dataset/model manifests used for provenance.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..config import PROJECT_ROOT, outputs_dir
from ..experiments.tracking import abs_path, latest_run, list_runs, load_manifest

COMPLETED, PENDING, FAILED, ILLUSTRATIVE = "Completed run", "Pending", "Failed run", "Illustrative example"
FAILED_STATUSES = {"failed", "stopped", "cancelled", "error", "killed"}
JOB_OF_KIND = {"train_detector": "train", "benchmark_detection": "detection", "benchmark_mitigation": "mitigation",
               "benchmark_gate": "gate", "benchmark_image": "image", "use_cases": "use_cases"}


def _read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def run_dir(run: dict) -> Path:
    """Folder of a run, robust to manifests indexed on another OS (falls back to outputs/runs/<run_id>)."""
    p = abs_path(run["manifest_path"]).parent
    if p.exists():
        return p
    return outputs_dir("runs") / run["run_id"]


def _job_alive(kind: str) -> bool:
    try:
        from ..experiments import jobs

        job = JOB_OF_KIND.get(kind)
        return bool(job and jobs.running(job))
    except Exception:  # noqa: BLE001
        return False


def _process_alive(run: dict) -> bool:
    """True if a live process runs the script recorded in the run's command and started no later than the run."""
    import psutil

    script = (run.get("command") or "").split(" ")[0]
    if not script.endswith(".py"):
        return False
    try:
        started = datetime.fromisoformat(run["started_at"]).timestamp()
    except (KeyError, TypeError, ValueError):
        started = None
    for proc in psutil.process_iter(["cmdline", "create_time"]):
        try:
            cmd = proc.info["cmdline"] or []
        except (psutil.Error, KeyError):
            continue
        if any(Path(c).name == script for c in cmd[1:]):
            if started is None or proc.info["create_time"] <= started + 60:
                return True
    return False


def run_alive(run: dict) -> bool:
    return _job_alive(run.get("kind", "")) or _process_alive(run)


def run_status(run: dict | None) -> str:
    """Map a run's manifest status to one UI label: Completed run / Pending / Failed run.

    A run still marked "running" is Pending while its process is alive, otherwise it was stopped: a failed run.
    """
    if not run:
        return PENDING
    st = (run.get("status") or "").lower()
    if st == "completed":
        return COMPLETED
    if st in FAILED_STATUSES:
        return FAILED
    if st == "running":
        return PENDING if run_alive(run) else FAILED
    return PENDING


def status_note(run: dict) -> str:
    st = (run.get("status") or "").lower()
    if st == "running" and run_status(run) == FAILED:
        return "Marked 'running' in its manifest but no process is active: the run was stopped and never finished."
    if st == "running":
        return "Still running; its outputs are not shown until it completes."
    return ""


def latest_summary(kind: str, filename: str) -> dict | None:
    r = latest_run(kind)
    if not r:
        return None
    p = run_dir(r) / filename
    if not p.exists():
        return None
    d = _read_json(p)
    d.setdefault("run_id", r["run_id"])
    d["_finished_at"] = r.get("finished_at")
    d["_run"] = r
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
    try:
        return load_manifest(run)
    except FileNotFoundError:
        p = run_dir(run) / "manifest.json"
        return _read_json(p) if p.exists() else {"run_id": run["run_id"], "kind": run.get("kind"), "status": run.get("status")}


def _model_ref_text(name: str, ref: dict) -> str:
    """One model as 'name: id@revision', with local checkpoints shown project-relative and rule baselines named."""
    model_id = ref.get("model_id") or ref.get("id")
    if not model_id or model_id == name:
        return f"{name}: no model (rule-based)"
    local = Path(model_id)
    if local.is_absolute() or model_id.startswith("checkpoints"):
        parts = local.parts
        shown = Path(*parts[parts.index("checkpoints"):]).as_posix() if "checkpoints" in parts else local.name
        return f"{name}: {shown} (fine-tuned in this project)"
    return f"{name}: {model_id}@{str(ref.get('revision') or 'unpinned')[:10]}"


def local_time(value) -> str:
    """ISO timestamp (stored in UTC) shown in this computer's local time, e.g. '2026-10-24 04:32'."""
    if not value:
        return "-"
    try:
        return datetime.fromisoformat(str(value)).astimezone().strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return str(value)


def _models_text(m: dict) -> str:
    revs = m.get("model_revisions") or {}
    if revs:
        return "; ".join(_model_ref_text(k, v) if isinstance(v, dict) else f"{k}: {v}" for k, v in revs.items())
    models = m.get("models") or {}
    if models:
        return "; ".join(f"{k}: {v}" for k, v in models.items())
    cfg = m.get("config") or {}
    if cfg.get("base_model"):
        return f"{cfg['base_model']}@{str(cfg.get('revision') or 'unpinned')[:10]}"
    return "not recorded"


def _datasets_text(m: dict) -> str:
    dv = m.get("dataset_versions") or {}
    if dv:
        files = ", ".join(dv.get("files") or {}) or "processed datasets"
        return f"{files} (dataset manifest {dv.get('manifest_sha') or '?'}, built {dv.get('built_at') or '?'})"
    ds = m.get("datasets") or {}
    return "; ".join(f"{Path(str(k).replace(chr(92), '/')).name}: {v}" for k, v in ds.items()) or "not recorded"


KEY_CONFIG = ("detector", "task", "epochs", "lr", "batch_size", "max_length", "n_train", "n_val", "n_test", "seed", "seeds",
              "threshold", "lite", "n_inputs")


def provenance_pairs(run: dict) -> list[tuple[str, str]]:
    """Run ID, models + revisions, dataset version and key config for a result header."""
    m = manifest(run)
    cfg = m.get("config") or {}
    key_cfg = ", ".join(f"{k}={cfg[k]}" for k in KEY_CONFIG if k in cfg) or "-"
    return [("Run ID", run["run_id"]), ("Models", _models_text(m)), ("Dataset version", _datasets_text(m)),
            ("Key config", key_cfg), ("Git commit", str(m.get("git_commit") or "-")),
            ("Finished", local_time(m.get("finished_at") or run.get("finished_at")))]


def latest_any(kind: str) -> dict | None:
    """Most recent run of a kind regardless of status (for showing Pending / Failed honestly)."""
    rs = list_runs(kind)
    return rs[0] if rs else None


def model_verification() -> dict | None:
    p = PROJECT_ROOT / "outputs" / "model_verification" / "verification.json"
    return _read_json(p) if p.exists() else None


def dataset_manifest() -> dict | None:
    p = PROJECT_ROOT / "datasets" / "processed" / "dataset_manifest.json"
    return _read_json(p) if p.exists() else None


def use_case_results() -> dict[str, dict]:
    d = PROJECT_ROOT / "outputs" / "use_cases"
    if not d.exists():
        return {}
    return {p.stem: _read_json(p) for p in d.glob("*.json")}


def benchmark_state(kind: str) -> tuple[str, dict | None]:
    """(status label, run) for a benchmark kind: the latest completed run wins; otherwise the latest attempt's status."""
    done = latest_run(kind)
    if done:
        return COMPLETED, done
    last = latest_any(kind)
    return run_status(last), last
