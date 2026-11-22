import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "datasets" / "processed"

pytestmark = pytest.mark.skipif(not (PROC / "dataset_manifest.json").exists(), reason="processed datasets not built")


def _verifier():
    spec = importlib.util.spec_from_file_location("verify_data_manifest", ROOT / "scripts" / "verify_data_manifest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_processed_files_match_manifest_hashes_and_counts():
    v = _verifier()
    manifest = json.loads((PROC / "dataset_manifest.json").read_text(encoding="utf-8"))
    files, ok = v.check_files(manifest)
    assert ok, [f for f in files if f["status"] != "ok" and f["status"] != "not in manifest"]
    diffs, ok = v.check_counts(manifest, pd.read_parquet(PROC / "detection.parquet"))
    assert ok, diffs


def test_no_group_or_text_crosses_splits():
    v = _verifier()
    leak = v.check_leakage(pd.read_parquet(PROC / "detection.parquet"), pd.read_parquet(PROC / "crows_pairs.parquet"),
                           pd.read_parquet(PROC / "stereoset_pairs.parquet"))
    assert leak["groups_spanning_splits"] == 0
    assert leak["texts_in_multiple_splits"] == 0
    assert leak["probe_vs_detection_split_mismatches"] == 0


def test_mitigation_eval_uses_only_test_split_benchmark_items():
    assert _verifier().check_mitigation_eval(pd.read_parquet(PROC / "detection.parquet")).get("non_test_items", 0) == 0
