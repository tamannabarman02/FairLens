"""Recreate outputs/runs.sqlite from the manifests in outputs/runs/<run_id>/manifest.json.

Teammates exchange run folders (copy them into outputs/runs/) and then run this, instead of merging SQLite files.
Usage:  python scripts/rebuild_runs_index.py [--runs-dir DIR] [--db FILE]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fairlens.experiments.tracking import list_runs, rebuild_index  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, default=None, help="folder holding <run_id>/manifest.json (default: outputs/runs)")
    ap.add_argument("--db", type=Path, default=None, help="index file to (re)create (default: outputs/runs.sqlite)")
    a = ap.parse_args()
    n = rebuild_index(runs_dir=a.runs_dir, db_path=a.db)
    print(f"indexed {n} runs")
    for r in list_runs(db_path=a.db):
        print(f"  {r['run_id']}  {r['kind']:<22} {r['status']}")


if __name__ == "__main__":
    main()
