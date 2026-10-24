# FairLens

Bias detection, mitigation and a proposed **Reliability Gate** that decides whether a rewrite can be trusted
(ACCEPT / REJECT / RETRY / HUMAN_REVIEW). Final-year research project with a Streamlit dashboard.

FairLens is a review assistant and research framework. It cannot guarantee that any output is unbiased, and it is
not a legal or compliance certification tool.

Runs on **Windows, macOS and Linux** with the same code and commands. It picks the best device it finds:
an NVIDIA GPU (CUDA), then an Apple GPU (MPS), then the CPU.

## Requirements

- Python **3.11** (Windows: install from python.org and tick "Add to PATH"; the `py` launcher is used if present)
- About 15 GB free disk for models, 8 GB RAM minimum (16 GB recommended for training and benchmarks)
- Internet on the first run, to download the pinned models

## Install

Windows (PowerShell, in the project folder):

```powershell
powershell -ExecutionPolicy Bypass -File install.ps1
```

macOS / Linux:

```bash
bash install.sh
```

Options (both installers):

| Option | Windows | macOS / Linux |
|---|---|---|
| Developer tools (browser UI tests) | `-Dev` | `--dev` |
| NVIDIA GPU: install CUDA PyTorch from a PyTorch index URL | `-TorchIndex https://download.pytorch.org/whl/cu128` | `--torch-index https://download.pytorch.org/whl/cu128` |

Pick the CUDA index that matches your driver from <https://pytorch.org/get-started/locally/>. Without it, Windows
and Linux get the CPU build; macOS always gets the build with Apple GPU support.

## Run the app

Windows:

```powershell
.venv\Scripts\streamlit run app.py
```

macOS / Linux:

```bash
.venv/bin/streamlit run app.py
```

Then open <http://localhost:8501>. Turn on **Lite mode** in the sidebar on laptops with 8 GB RAM or no GPU.

## Run the tests

```powershell
.venv\Scripts\python -m pytest -q -m "not slow"      # Windows
```

```bash
.venv/bin/python -m pytest -q -m "not slow"          # macOS / Linux
```

GitHub Actions runs the same tests on Windows, macOS and Linux on every push (`.github/workflows/tests.yml`).

## Run the experiments

One command runs every step in order, one heavy job at a time, and writes each step's log to `outputs/logs/`:

```powershell
.venv\Scripts\python scripts\run_all.py --list        # Windows: show the steps
.venv\Scripts\python scripts\run_all.py               # run everything
.venv\Scripts\python scripts\run_all.py --from gate   # resume from a step
```

```bash
.venv/bin/python scripts/run_all.py --list            # macOS / Linux
.venv/bin/python scripts/run_all.py
```

Steps: `train_bert`, `train_deberta`, `prompts`, `detection`, `judge`, `gate`, `mitigation`, `image`,
`use_cases`, `results`. Every run writes a manifest (run ID, models and pinned revisions, device, seeds, dataset
hashes, config, outputs) to `outputs/runs/<run_id>/`. `docs/RESULTS.md` is generated from those runs only.

## Settings (environment variables)

| Variable | Purpose | Example |
|---|---|---|
| `FAIRLENS_DEVICE` | Force a device: `cpu`, `cuda` or `mps` | `cpu` if a GPU run misbehaves |
| `FAIRLENS_CACHE` | Model cache folder. Default: `%LOCALAPPDATA%\FairLens\hf` (Windows), `~/Library/Caches/FairLens/hf` (macOS), `~/.cache/fairlens/hf` (Linux) | `D:\FairLens_cache\hf` to reuse models already downloaded on a D: drive |
| `FAIRLENS_PROFILE` | Force a hardware profile: `lite`, `standard` or `gpu` | `lite` |

Set one for the current terminal:

```powershell
$env:FAIRLENS_DEVICE = "cpu"       # Windows PowerShell
```

```bash
export FAIRLENS_DEVICE=cpu         # macOS / Linux
```

## Sharing runs between teammates

Copy a run folder from `outputs/runs/` to the same place on another machine, then rebuild the index:

```bash
python scripts/rebuild_runs_index.py
```

The final reported numbers for each experiment come from one named device (recorded in its manifest).

## Documentation

- `docs/METHODOLOGY.md`: hypotheses, experiment design, metrics
- `docs/PROTOCOL_CHANGES.md`: every change to the evaluation protocol, with reasons
- `docs/AUDIT.md`: verification of the project's stated status
- `docs/DATASETS.md`, `docs/MODEL_CARD_REGISTRY.md`, `docs/RESULTS.md`
- `docs/PROGRESS.md`: current status and next steps
