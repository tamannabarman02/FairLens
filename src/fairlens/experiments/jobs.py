"""Background-safe long jobs launched from the dashboard: subprocess + log file + pid, cancellable."""
from __future__ import annotations

import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

import psutil

from ..config import PROJECT_ROOT, outputs_dir

SCRIPTS = {
    "detection": ["scripts/run_benchmarks.py", "--task", "detection"],
    "mitigation": ["scripts/run_benchmarks.py", "--task", "mitigation"],
    "gate": ["scripts/run_ablations.py"],
    "image": ["scripts/run_image_experiment.py"],
    "train": ["scripts/train_detectors.py", "--model", "all"],
    "use_cases": ["scripts/run_use_cases.py"],
    "results_md": ["scripts/generate_results_md.py"],
}


def _dir() -> Path:
    return outputs_dir("jobs")


def start(name: str, args: list[str] | None = None) -> dict:
    jid = f"{time.strftime('%Y%m%d-%H%M%S')}-{name}-{uuid.uuid4().hex[:4]}"
    log = _dir() / f"{jid}.log"
    cmd = [sys.executable, *(args or SCRIPTS[name])]
    fh = open(log, "w", encoding="utf-8")
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    p = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=fh, stderr=subprocess.STDOUT, creationflags=flags,
                         env={**__import__("os").environ, "PYTHONUTF8": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
    rec = {"id": jid, "name": name, "cmd": cmd, "pid": p.pid, "log": str(log), "started": time.time()}
    (_dir() / f"{jid}.json").write_text(json.dumps(rec), encoding="utf-8")
    return rec


def status(rec: dict) -> str:
    try:
        proc = psutil.Process(rec["pid"])
        if proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE and abs(proc.create_time() - rec["started"]) < 30:
            return "running"
    except psutil.NoSuchProcess:
        pass
    tail = log_tail(rec, 5)
    if "done:" in tail or "wrote" in tail:
        return "completed"
    if rec.get("cancelled"):
        return "cancelled"
    return "finished" if "Traceback" not in log_tail(rec, 60) else "failed"


def cancel(rec: dict) -> None:
    try:
        parent = psutil.Process(rec["pid"])
        for c in parent.children(recursive=True):
            c.kill()
        parent.kill()
    except psutil.NoSuchProcess:
        pass
    rec["cancelled"] = True
    (_dir() / f"{rec['id']}.json").write_text(json.dumps(rec), encoding="utf-8")


def log_tail(rec: dict, n: int = 20) -> str:
    p = Path(rec["log"])
    if not p.exists():
        return ""
    lines = [l for l in p.read_text(encoding="utf-8", errors="replace").splitlines()
             if not any(w in l for w in ("Warning", "warn(", "symlink", "Developer Mode", "unauthenticated"))]
    return "\n".join(lines[-n:])


def all_jobs() -> list[dict]:
    out = []
    for p in sorted(_dir().glob("*.json"), reverse=True):
        rec = json.loads(p.read_text(encoding="utf-8"))
        rec["status"] = status(rec)
        out.append(rec)
    return out


def running(name: str) -> dict | None:
    return next((j for j in all_jobs() if j["name"] == name and j["status"] == "running"), None)
