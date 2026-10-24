#!/usr/bin/env bash
# One-command install for FairLens (Linux/macOS):
#   bash install.sh [--dev] [--torch-index URL]
#   --dev               also install requirements-dev.txt and the Playwright Chromium browser (UI smoke tests)
#   --torch-index URL   reinstall torch + torchvision from this PyTorch wheel index after the requirements,
#                       e.g. a CUDA index from https://pytorch.org/get-started/locally/ (env: FAIRLENS_TORCH_INDEX)
# Python 3.11 is required; set PYTHON=/path/to/python3.11 if it is not on PATH as python3.11.
set -euo pipefail
cd "$(dirname "$0")"

DEV=0
TORCH_INDEX="${FAIRLENS_TORCH_INDEX:-}"
while [ $# -gt 0 ]; do
    case "$1" in
        --dev) DEV=1 ;;
        --torch-index) shift; TORCH_INDEX="${1:?--torch-index needs a URL}" ;;
        --torch-index=*) TORCH_INDEX="${1#*=}" ;;
        -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

is_py311() { "$1" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 11))' 2>/dev/null; }

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
    for cand in python3.11 python3 python; do
        if command -v "$cand" >/dev/null 2>&1 && is_py311 "$cand"; then PY="$cand"; break; fi
    done
fi
if [ -z "$PY" ] || ! is_py311 "$PY"; then
    echo "Python 3.11 not found. Install it (e.g. brew install python@3.11 / apt install python3.11 python3.11-venv) or set PYTHON." >&2
    exit 1
fi

[ -d .venv ] || "$PY" -m venv .venv
VPY=.venv/bin/python
if ! is_py311 "$VPY"; then
    echo "The existing .venv is not Python 3.11. Delete the .venv folder and run this script again." >&2
    exit 1
fi

"$VPY" -m pip install --upgrade pip
"$VPY" -m pip install -r requirements.txt
"$VPY" -m pip install -e . --no-deps

if [ -n "$TORCH_INDEX" ]; then
    TORCH_V=$("$VPY" -c 'import importlib.metadata as m; print(m.version("torch").split("+")[0])')
    VISION_V=$("$VPY" -c 'import importlib.metadata as m; print(m.version("torchvision").split("+")[0])')
    echo "Reinstalling torch==$TORCH_V and torchvision==$VISION_V from $TORCH_INDEX"
    "$VPY" -m pip install --force-reinstall --no-deps --index-url "$TORCH_INDEX" "torch==$TORCH_V" "torchvision==$VISION_V"
fi

if [ "$DEV" = 1 ]; then
    "$VPY" -m pip install -r requirements-dev.txt
    "$VPY" -m playwright install chromium
fi

"$VPY" scripts/check_env.py --no-write > /dev/null
"$VPY" -c 'import fairlens.hardware as h; print("Compute device:", h.device_label())'
echo "FairLens installed. Start the dashboard with: .venv/bin/python -m streamlit run app.py"
