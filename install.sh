#!/usr/bin/env bash
# One-command install for FairLens (Linux/macOS):  bash install.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3.11}
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python scripts/check_env.py > /dev/null
echo "FairLens installed. Start the dashboard with: .venv/bin/streamlit run app.py"
