import json
import shutil
import sqlite3
from contextlib import closing

from fairlens.config import PROJECT_ROOT
from fairlens.experiments import tracking


def _run(tmp_path, kind="unit", **kw):
    data = tmp_path / "d.csv"
    data.write_text("a,b\n1,2\n", encoding="utf-8")
    t = tracking.RunTracker(kind, config={"x": 1, "seeds": [7, 8]}, seed=7, dataset_files=[data],
                            models={"det": "bert|distilbert-base-uncased|abc123|t0", "rw": "google/flan-t5-base"},
                            db_path=tmp_path / "runs.sqlite", **kw)
    t.save_artifact("res.json", {"ok": True})
    t.finish(metrics={"f1": 0.5}, artifacts={"checkpoint": PROJECT_ROOT / "checkpoints" / "x"})
    return t


def test_manifest_has_every_required_field(tmp_path):
    man = _run(tmp_path).manifest
    assert tracking.missing_fields(man) == []
    on_disk = json.loads((tmp_path / "runs" / man["run_id"] / "manifest.json").read_text(encoding="utf-8"))
    assert tracking.missing_fields(on_disk) == []
    for field in ("finished_at", "runtime_s", "device", "hardware", "run_id", "task"):
        assert on_disk[field] is not None, field
    assert on_disk["device"] in {"cpu", "cuda", "mps"}
    assert on_disk["seeds"] == [7, 8]


def test_manifest_records_model_revisions_and_portable_paths(tmp_path):
    man = _run(tmp_path).manifest
    assert man["model_revisions"]["det"] == {"model_id": "distilbert-base-uncased", "revision": "abc123"}
    assert man["model_revisions"]["rw"]["model_id"] == "google/flan-t5-base"
    assert man["model_revisions"]["rw"]["revision"] == tracking.pinned_revision("google/flan-t5-base")
    assert man["artifacts"]["checkpoint"] == "checkpoints/x"
    for p in [*man["datasets"], *man["outputs"].values(), man["artifacts"]["res.json"]]:
        assert "\\" not in p


def test_rebuild_index_reproduces_rows(tmp_path):
    db = tmp_path / "runs.sqlite"
    _run(tmp_path, "a")
    _run(tmp_path, "b")
    with closing(sqlite3.connect(db)) as con:
        before = sorted(con.execute("SELECT * FROM runs"))
    db.unlink()
    assert tracking.rebuild_index(db_path=db) == 2
    with closing(sqlite3.connect(db)) as con:
        after = sorted(con.execute("SELECT * FROM runs"))
    assert after == before and len(after) == 2


def test_runs_folders_can_move_between_machines(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    t = _run(src)
    dst = tmp_path / "dst"
    shutil.copytree(src / "runs", dst / "runs")
    tracking.rebuild_index(db_path=dst / "runs.sqlite")
    run = tracking.list_runs(db_path=dst / "runs.sqlite")[0]
    assert run["run_id"] == t.run_id and run["metrics"] == {"f1": 0.5}
    assert tracking.load_manifest(run)["run_id"] == t.run_id


def test_stale_windows_manifest_path_falls_back_to_local_run_folder(tmp_path):
    t = _run(tmp_path)
    with closing(sqlite3.connect(tmp_path / "runs.sqlite")) as con, con:
        con.execute("UPDATE runs SET manifest_path=?", (f"D:\\FairLens\\outputs\\runs\\{t.run_id}\\manifest.json",))
    run = tracking.list_runs(db_path=tmp_path / "runs.sqlite")[0]
    assert tracking.load_manifest(run)["run_id"] == t.run_id
