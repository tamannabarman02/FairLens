"""Result status labels and provenance: stopped runs are never shown as results."""
import json

from fairlens.experiments import tracking
from fairlens.ui import results as R


def _run(tmp_path, status, kind="benchmark_detection", run_id="r1"):
    d = tmp_path / "runs" / run_id
    d.mkdir(parents=True)
    m = {"run_id": run_id, "kind": kind, "status": status, "config": {"detector": "bert", "epochs": 3, "seed": 42},
         "git_commit": "abc", "model_revisions": {"bert": {"model_id": "distilbert-base-uncased", "revision": "12040accade4"}},
         "dataset_versions": {"manifest_sha": "deadbeef", "built_at": "2026-10-01", "files": {"detection.parquet": "x"}}}
    (d / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    return {"run_id": run_id, "kind": kind, "status": status, "manifest_path": str(d / "manifest.json"), "finished_at": None}


def test_status_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "run_alive", lambda run: False)
    assert R.run_status(None) == R.PENDING
    assert R.run_status({"status": "completed"}) == R.COMPLETED
    for s in ("failed", "stopped", "cancelled"):
        assert R.run_status({"status": s}) == R.FAILED
    stale = {"status": "running", "kind": "train_detector"}
    assert R.run_status(stale) == R.FAILED and "stopped" in R.status_note(stale)
    monkeypatch.setattr(R, "run_alive", lambda run: True)
    assert R.run_status(stale) == R.PENDING and "Still running" in R.status_note(stale)


def test_running_run_is_live_when_its_script_process_exists(monkeypatch):
    import psutil

    class P:
        info = {"cmdline": ["python", "-u", "scripts/train_detectors.py", "--model", "deberta"], "create_time": 0.0}

    monkeypatch.setattr(R, "_job_alive", lambda kind: False)
    monkeypatch.setattr(psutil, "process_iter", lambda attrs=None: [P()])
    run = {"status": "running", "kind": "train_detector", "command": "train_detectors.py --model deberta",
           "started_at": "2026-10-24T01:00:00+00:00"}
    assert R.run_status(run) == R.PENDING
    monkeypatch.setattr(psutil, "process_iter", lambda attrs=None: [])
    assert R.run_status(run) == R.FAILED


def test_provenance_has_run_models_dataset_and_config(tmp_path):
    pairs = dict(R.provenance_pairs(_run(tmp_path, "completed")))
    assert pairs["Run ID"] == "r1"
    assert "distilbert-base-uncased@12040accad" in pairs["Models"]
    assert "deadbeef" in pairs["Dataset version"]
    assert "epochs=3" in pairs["Key config"] and "seed=42" in pairs["Key config"]


def test_run_dir_survives_manifests_indexed_on_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "outputs_dir", lambda *p: tmp_path.joinpath(*p))
    _run(tmp_path, "completed", run_id="r9")
    stale = {"run_id": "r9", "manifest_path": "D:" + "\\FairLens\\outputs\\runs\\r9\\manifest.json"}
    assert R.run_dir(stale) == tmp_path / "runs" / "r9"


def test_benchmark_state_prefers_completed_and_otherwise_reports_attempt(tmp_path, monkeypatch):
    db = tmp_path / "runs.sqlite"
    monkeypatch.setattr(tracking, "_db_path", lambda: db)
    monkeypatch.setattr(R, "run_alive", lambda run: False)
    assert R.benchmark_state("benchmark_gate") == (R.PENDING, None)


def test_current_outputs_show_stopped_deberta_runs_as_failed(monkeypatch):
    monkeypatch.setattr(R, "run_alive", lambda run: False)
    for r in R.runs("train_detector"):
        m = R.manifest(r)
        if m.get("status") == "running":
            assert R.run_status(r) == R.FAILED
