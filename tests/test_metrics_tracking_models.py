import json

import numpy as np

from fairlens.evaluation import metrics
from fairlens.experiments import cache, tracking
from fairlens.models import ModelManager


def test_classification_metrics():
    m = metrics.classification([0, 0, 1, 1], [0.1, 0.6, 0.7, 0.9])
    assert m["confusion"] == [[1, 1], [0, 2]]
    assert abs(m["precision"] - 2 / 3) < 1e-9 and m["recall"] == 1.0 and m["roc_auc"] == 1.0
    assert 0 <= m["ece"] <= 1 and len(m["reliability"]) == 10


def test_ece_perfect_and_bad():
    y = np.array([0] * 50 + [1] * 50)
    assert metrics.calibration(y, y.astype(float))["ece"] == 0
    assert metrics.calibration(y, 1 - y.astype(float))["ece"] == 1


def test_bootstrap():
    ci = metrics.bootstrap_ci(np.ones(20))
    assert ci["mean"] == ci["lo"] == ci["hi"] == 1
    d = metrics.paired_bootstrap_diff([1, 1, 1, 1], [0, 0, 0, 0])
    assert d["diff"] == 1 and d["p_boot"] == 0


def test_tracking_roundtrip(tmp_path):
    data = tmp_path / "d.csv"
    data.write_text("a,b\n1,2\n", encoding="utf-8")
    t = tracking.RunTracker("unit", config={"x": 1}, seed=7, dataset_files=[data], db_path=tmp_path / "runs.sqlite")
    t.save_artifact("res.json", {"ok": True})
    t.finish(metrics={"f1": 0.5})
    runs = tracking.list_runs(db_path=tmp_path / "runs.sqlite")
    assert runs[0]["run_id"] == t.run_id and runs[0]["metrics"] == {"f1": 0.5}
    man = tracking.load_manifest(runs[0])
    assert man["seed"] == 7 and man["status"] == "completed" and man["hardware"]["cpu_threads"] >= 1
    assert list(man["datasets"].values())[0] == tracking.file_sha(data)
    assert json.loads((t.dir / "res.json").read_text(encoding="utf-8")) == {"ok": True}


class _Det:
    hyp_biased, hyp_neutral = "biased", "neutral"

    def __init__(self, revision=None, model="facebook/bart-large-mnli"):
        self.revision, self.model = revision, model

    def info(self):
        return {"name": "zs", "model": self.model, "revision": self.revision}


def test_score_cache_key_includes_revision_and_device(monkeypatch):
    monkeypatch.setattr(cache, "device", lambda: "cpu")
    pinned = cache.detector_identity(_Det())
    assert tracking.pinned_revision("facebook/bart-large-mnli") in pinned and pinned.endswith("|device=cpu")
    assert cache.detector_identity(_Det(revision="r1")) != cache.detector_identity(_Det(revision="r2"))
    monkeypatch.setattr(cache, "device", lambda: "mps")
    assert cache.detector_identity(_Det()) != pinned


def test_rule_based_identity_has_no_device(monkeypatch):
    monkeypatch.setattr(cache, "device", lambda: "mps")
    assert "device=" not in cache.detector_identity(_Det(model=""))


def test_score_cache_roundtrip_unicode(tmp_path):
    c = cache.ScoreCache(tmp_path / "scores.sqlite")
    c.put_many("m", {"Ça va — naïve café": 0.25})
    assert c.get_many("m", ["Ça va — naïve café", "missing"]) == {"Ça va — naïve café": 0.25}
    c.put_gen("m", "prompt é", {"text": "réécrit"})
    assert c.get_gen("m", "prompt é") == {"text": "réécrit"}


class _Blob:
    def __init__(self, mb):
        self.mb = mb


def test_model_manager_keeps_one_large_model():
    m = ModelManager(max_resident_mb=1000)
    size = lambda o: o.mb  # noqa: E731
    m.get("a", lambda: _Blob(700), size)
    m.get("b", lambda: _Blob(800), size)
    assert list(m.resident()) == ["b"]
    m.get("c", lambda: _Blob(100), size)
    m.get("d", lambda: _Blob(50), size)
    assert "b" in m.resident() or m.resident_mb() <= 1000
    loads = []
    m.get("d", lambda: loads.append(1) or _Blob(50), size)
    assert loads == []  # cached
