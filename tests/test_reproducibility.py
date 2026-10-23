"""V4 1.2 (device order), 1.2d (runs shared as folders) and 1.6 (complete manifests)."""
import json
import shutil

import pytest

from fairlens import hardware
from fairlens.experiments import tracking


def test_device_order_and_override(monkeypatch):
    monkeypatch.setattr(hardware, "available_devices", lambda: ["cuda", "mps", "cpu"])
    monkeypatch.delenv("FAIRLENS_DEVICE", raising=False)
    assert hardware.device() == "cuda"
    monkeypatch.setattr(hardware, "available_devices", lambda: ["mps", "cpu"])
    assert hardware.device() == "mps"
    monkeypatch.setenv("FAIRLENS_DEVICE", "cpu")
    assert hardware.device() == "cpu"
    monkeypatch.setenv("FAIRLENS_DEVICE", "cuda")  # not available -> warn and fall back
    with pytest.warns(UserWarning):
        assert hardware.device() == "mps"


def test_mps_fallback_env_set_in_code():
    import os

    import fairlens.config  # noqa: F401

    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"


def _run(tmp_path, status="completed"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    data = tmp_path / "d.csv"
    data.write_text("a\n1\n", encoding="utf-8")
    t = tracking.RunTracker("unit", config={"x": 1}, seed=3, dataset_files=[data], models={"m": "org/model@abc123"},
                            db_path=tmp_path / "outputs" / "runs.sqlite")
    t.save_artifact("summary.json", {"ok": True})
    if status == "completed":
        t.finish(metrics={"f1": 0.5})
    return t


def test_manifest_has_all_required_fields(tmp_path):
    t = _run(tmp_path)
    m = json.loads(t.manifest_path.read_text(encoding="utf-8"))
    assert tracking.missing_fields(m) == []
    assert "\\" not in json.dumps({k: m[k] for k in ["datasets", "output_dir", "artifacts"]})  # portable paths
    assert m["device"] in {"cpu", "cuda", "mps"}


def test_missing_field_is_detected(tmp_path):
    m = json.loads(_run(tmp_path).manifest_path.read_text(encoding="utf-8"))
    for f in ["device", "seed", "models", "datasets", "config"]:
        broken = {k: v for k, v in m.items() if k != f}
        assert f in tracking.missing_fields(broken)


def test_copied_run_folder_appears_without_database(tmp_path):
    t = _run(tmp_path / "laptop_a")
    dest = tmp_path / "laptop_b" / "outputs"
    (dest / "runs").mkdir(parents=True)
    shutil.copytree(t.dir, dest / "runs" / t.run_id)
    runs = tracking.list_runs(db_path=dest / "runs.sqlite")  # no sqlite file exists on laptop B
    assert [r["run_id"] for r in runs] == [t.run_id] and runs[0]["status"] == "completed"
    assert tracking.rebuild_index(dest / "runs.sqlite") == 1
