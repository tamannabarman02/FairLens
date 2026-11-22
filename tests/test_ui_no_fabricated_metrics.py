"""Guard: the dashboard never renders a research metric that was not read from a run file in outputs/.

1. Static: no page passes a literal value to metric_tile / st.metric, and no page source hard-codes a metric-looking
   number ("0.87", "93%") in a string.
2. Dynamic: with an empty outputs folder every research page renders headlessly with no metric tiles and no
   metric-looking numbers, and shows the Pending label instead.
3. Dynamic: with one completed run, the tile on Benchmarks shows exactly the value from that run's summary file.
4. The results helpers only open files under outputs/ (plus the dataset manifest used for provenance).
"""
import ast
import builtins
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGES = sorted((ROOT / "src" / "fairlens" / "ui" / "pages").glob("*.py")) + [ROOT / "app.py"]
METRIC_LIKE = re.compile(r"(?<![\w.#-])(?:0\.\d{2,}|[1-9]\d?(?:\.\d+)?%)(?![\w-])")
CI_LABEL = re.compile(r"95% (?:bootstrap )?CI")
RESEARCH = ["home", "benchmarks", "model_comparison", "error_analysis", "experiments"]


def _consts(node) -> list[str]:
    return [n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def _is_literal(node) -> bool:
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.JoinedStr):
        return all(isinstance(v, ast.Constant) for v in node.values)
    return False


@pytest.mark.parametrize("path", PAGES, ids=lambda p: p.name)
def test_no_literal_metric_values_in_page_source(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if fn in {"metric_tile", "metric"} and len(node.args) >= 2 and _is_literal(node.args[1]):
                bad.append(f"line {node.lineno}: literal value passed to {fn}")
    for s in _consts(tree):
        hit = METRIC_LIKE.search(CI_LABEL.sub("", s))
        if hit:
            bad.append(f"metric-like literal {hit.group(0)!r} in {s[:80]!r}")
    assert bad == []


def _isolate_outputs(monkeypatch, tmp_path):
    from fairlens.experiments import jobs, tracking
    from fairlens.ui import results

    (tmp_path / "runs").mkdir(exist_ok=True)
    monkeypatch.setattr(tracking, "_db_path", lambda: tmp_path / "runs.sqlite")
    monkeypatch.setattr(results, "outputs_dir", lambda *p: tmp_path.joinpath(*p))
    monkeypatch.setattr(results, "use_case_results", lambda: {})
    jd = tmp_path / "jobs"
    jd.mkdir(exist_ok=True)
    monkeypatch.setattr(jobs, "_dir", lambda: jd)


def _render(module: str):
    from streamlit.testing.v1 import AppTest

    script = f"""
import streamlit as st
from fairlens.ui import state
from fairlens.ui.pages import {module} as page
state.init()
st.session_state["_pages"] = {{}}
page.render()
"""
    at = AppTest.from_string(script, default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return "\n".join(m.value for m in at.markdown) + "\n".join(c.value for c in at.caption)


@pytest.mark.parametrize("module", RESEARCH)
def test_research_pages_without_runs_show_pending_and_no_numbers(module, monkeypatch, tmp_path):
    _isolate_outputs(monkeypatch, tmp_path)
    html = _render(module)
    assert "fl-tile" not in html
    assert METRIC_LIKE.search(CI_LABEL.sub("", html)) is None, METRIC_LIKE.search(CI_LABEL.sub("", html))
    if module != "experiments":
        assert "Pending" in html


def test_benchmark_tile_shows_the_value_from_the_run_file(monkeypatch, tmp_path):
    from fairlens.experiments.tracking import RunTracker

    _isolate_outputs(monkeypatch, tmp_path)
    tr = RunTracker("benchmark_detection", config={"n_test": 10}, seed=1, db_path=tmp_path / "runs.sqlite")
    summary = {"n_test": 10, "detectors": {"bert": {"status": "ok", "display": "BERT", "metrics": {
        "macro_f1": 0.6173, "precision": 0.6, "recall": 0.6, "f1": 0.6, "fpr": 0.2, "ece": 0.1, "brier": 0.2, "roc_auc": 0.7,
        "pr_auc": 0.7, "confusion": [[4, 1], [1, 4]], "reliability": []},
        "ci": {"macro_f1": {"lo": 0.5, "hi": 0.7}, "f1": {"value": 0.6, "lo": 0.5, "hi": 0.7}}, "by_source": {},
        "curves": {"roc": {"fpr": [0, 1], "tpr": [0, 1]}}, "ms_per_sample": 1.0, "load_s": 1.0}}}
    (tr.dir / "detection_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    tr.finish({"macro_f1": 0.6173})
    html = _render("benchmarks")
    assert "0.617" in html and tr.run_id in html and "Completed run" in html


def test_results_helpers_only_read_outputs(monkeypatch, tmp_path):
    from fairlens.ui import results

    _isolate_outputs(monkeypatch, tmp_path)
    opened = []
    real_read = Path.read_text
    real_open = builtins.open

    def spy_read(self, *a, **k):
        opened.append(Path(self))
        return real_read(self, *a, **k)

    def spy_open(file, *a, **k):
        opened.append(Path(file))
        return real_open(file, *a, **k)

    monkeypatch.setattr(Path, "read_text", spy_read)
    monkeypatch.setattr(builtins, "open", spy_open)
    for kind, fn in [("benchmark_detection", "detection_summary.json"), ("benchmark_gate", "gate_summary.json")]:
        results.latest_summary(kind, fn)
        results.latest_frame(kind, "x.parquet")
        results.benchmark_state(kind)
    results.dataset_manifest()
    allowed = (tmp_path.resolve(), (ROOT / "outputs").resolve(), (ROOT / "configs").resolve(),
               (ROOT / "datasets" / "processed" / "dataset_manifest.json").resolve())
    stray = [p for p in opened if not any(p.resolve() == a or a in p.resolve().parents for a in allowed)]
    assert stray == []
