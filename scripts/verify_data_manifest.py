"""Stage-0 data audit: recompute row counts and hashes of datasets/processed/*.parquet, compare them with
datasets/processed/dataset_manifest.json, and check that no group (or identical text) crosses splits.

Usage:  python scripts/verify_data_manifest.py [--json]
Exit code 0 when every check passes, 1 otherwise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "datasets" / "processed"
MANIFEST = PROC / "dataset_manifest.json"
SPLITS = ("train", "val", "test")


def short_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def check_files(manifest: dict) -> tuple[list[dict], bool]:
    rows, ok = [], True
    listed = manifest.get("files", {})
    for p in sorted(PROC.glob("*.parquet")):
        n, sha = len(pd.read_parquet(p)), short_sha(p)
        exp = listed.get(p.name)
        if exp is None:
            rows.append({"file": p.name, "rows": n, "sha256": sha, "status": "not in manifest"})
            continue
        good = exp["rows"] == n and exp["sha256"] == sha
        ok &= good
        rows.append({"file": p.name, "rows": n, "expected_rows": exp["rows"], "sha256": sha,
                     "expected_sha256": exp["sha256"], "status": "ok" if good else "MISMATCH"})
    for name in sorted(set(listed) - {p.name for p in PROC.glob("*.parquet")}):
        ok = False
        rows.append({"file": name, "status": "MISSING (listed in manifest)"})
    return rows, ok


def check_counts(manifest: dict, det: pd.DataFrame) -> tuple[list[dict], bool]:
    actual = det.groupby(["source", "split", "label"]).size().to_dict()
    expected = {(r["source"], r["split"], r["label"]): r["n"] for r in manifest.get("detection_counts", [])}
    diffs = [{"key": list(k), "expected": expected.get(k), "actual": actual.get(k)}
             for k in sorted(set(actual) | set(expected)) if expected.get(k) != actual.get(k)]
    return diffs, not diffs


def check_leakage(det: pd.DataFrame, crows: pd.DataFrame, ss: pd.DataFrame) -> dict:
    groups = det.groupby("group_id")["split"].nunique()
    texts = det.groupby("text")["split"].nunique()
    det_split = det.drop_duplicates("group_id").set_index("group_id")["split"]
    cp_split = crows.set_index("pair_id")["split"]
    ss_split = ("ss:" + ss["id"]).to_frame("g").assign(split=ss["split"].values).set_index("g")["split"]
    probe_split = pd.concat([cp_split, ss_split])
    shared = det_split.index.intersection(probe_split.index)
    mismatch = int((det_split.loc[shared] != probe_split.loc[shared]).sum())
    return {
        "groups_spanning_splits": int((groups > 1).sum()),
        "texts_in_multiple_splits": int((texts > 1).sum()),
        "probe_vs_detection_split_mismatches": mismatch,
        "probe_groups_checked": int(len(shared)),
    }


def check_mitigation_eval(det: pd.DataFrame) -> dict:
    p = PROC / "mitigation_eval.parquet"
    if not p.exists():
        return {"status": "absent"}
    mit = pd.read_parquet(p)
    bench = mit[mit.source != "authored"]
    split_of = det.drop_duplicates("text").set_index("text")["split"]
    found = bench.text.map(split_of)
    return {"benchmark_items": int(len(bench)), "found_in_detection": int(found.notna().sum()),
            "non_test_items": int((found.notna() & (found != "test")).sum())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="print the full report as JSON")
    a = ap.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    det = pd.read_parquet(PROC / "detection.parquet")
    crows = pd.read_parquet(PROC / "crows_pairs.parquet")
    ss = pd.read_parquet(PROC / "stereoset_pairs.parquet")

    files, files_ok = check_files(manifest)
    count_diffs, counts_ok = check_counts(manifest, det)
    leak = check_leakage(det, crows, ss)
    mit = check_mitigation_eval(det)
    leak_ok = (leak["groups_spanning_splits"] == 0 and leak["texts_in_multiple_splits"] == 0
               and leak["probe_vs_detection_split_mismatches"] == 0)
    manifest_leak_ok = manifest.get("groups_spanning_splits") == leak["groups_spanning_splits"]
    mit_ok = mit.get("non_test_items", 0) == 0
    ok = files_ok and counts_ok and leak_ok and manifest_leak_ok and mit_ok
    report = {"files": files, "detection_count_diffs": count_diffs, "leakage": leak,
              "manifest_groups_spanning_splits": manifest.get("groups_spanning_splits"),
              "mitigation_eval": mit, "ok": ok}
    if a.json:
        print(json.dumps(report, indent=2))
        return 0 if ok else 1

    print(f"Manifest: {MANIFEST.relative_to(ROOT).as_posix()} (built {manifest.get('built_at')}, seed {manifest.get('seed')})")
    for f in files:
        extra = f" (expected {f.get('expected_rows')} rows / {f.get('expected_sha256')})" if f["status"] == "MISMATCH" else ""
        print(f"  {f['file']:<26} rows={f.get('rows', '-'):<6} sha256[:16]={f.get('sha256', '-')}  {f['status']}{extra}")
    print(f"Detection per-(source, split, label) counts: {'match manifest' if counts_ok else f'{len(count_diffs)} differ'}")
    for d in count_diffs:
        print(f"  {d}")
    split_tab = det.groupby(["split", "label"]).size().unstack().reindex(list(SPLITS))
    print(f"Detection rows: {len(det)}; per split/label:\n{split_tab.to_string()}")
    print(f"Groups spanning splits: {leak['groups_spanning_splits']} (manifest says {manifest.get('groups_spanning_splits')})")
    print(f"Identical texts in more than one split: {leak['texts_in_multiple_splits']}")
    print(f"Probe pairs whose split differs from detection.parquet: {leak['probe_vs_detection_split_mismatches']} "
          f"of {leak['probe_groups_checked']} shared groups")
    if mit.get("status") != "absent":
        print(f"mitigation_eval benchmark items: {mit['benchmark_items']}, found in detection: {mit['found_in_detection']}, "
              f"from train/val: {mit['non_test_items']}")
    print("RESULT:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
