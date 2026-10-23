#!/usr/bin/env bash
# One-command install for FairLens (Linux/macOS):  bash install.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-$(command -v python3.11 || command -v python3)}
"$PY" -c "import sys; assert sys.version_info[:2] == (3, 11), f'Python 3.11 required, found {sys.version}'"
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python scripts/check_env.py > /dev/null
echo "FairLens installed. Start the dashboard with: .venv/bin/streamlit run app.py"
