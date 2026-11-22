# Run FairLens on a new Mac or Windows PC

Use **one** of these:

| File | What it is | When to use it |
|---|---|---|
| `FairLens_Full.zip` (about 620 MB) | Everything: code, datasets, results **and** the two fine-tuned detectors already in place | **Recommended.** Unzip and go to step 4 |
| `FairLens.zip` (16 MB) + `FairLens-checkpoints.zip` (about 600 MB) | The same content split in two, for sharing over slow links | Only if the full zip is too big to send. Needs the extra move in step 3 |

## 1. Check your computer

| | Mac | Windows |
|---|---|---|
| Supported | Apple silicon (M1, M2, M3, M4). **Intel Macs are not supported** (no matching PyTorch build) | 64-bit Windows 10/11 on Intel/AMD. Windows on ARM is not supported |
| RAM | 8 GB minimum, 16 GB recommended | 8 GB minimum, 16 GB recommended |
| Disk | About 15 GB free (Python packages ~2 GB, models ~6 GB) | Same |
| Internet | Needed for the install and the first run (models download once) | Same |

## 2. Install Python 3.11

Use exactly **3.11** (not 3.12 or 3.13).

- **Mac:** download "macOS 64-bit universal2 installer" for Python 3.11 from <https://www.python.org/downloads/macos/>, or run `brew install python@3.11` if you use Homebrew.
- **Windows:** download "Windows installer (64-bit)" for Python 3.11 from <https://www.python.org/downloads/windows/>. In the installer, tick **"Add python.exe to PATH"** and keep **"py launcher"** ticked.

Check it worked:

```bash
python3.11 --version        # Mac  -> Python 3.11.x
```

```powershell
py -3.11 --version          # Windows -> Python 3.11.x
```

## 3. Unzip

**With `FairLens_Full.zip`:** double-click it (Mac) or right-click → Extract All (Windows). You get one `FairLens` folder that is already complete.

**With the two-part download:**
1. Unzip `FairLens.zip`. You get a folder called `FairLens`.
2. Unzip `FairLens-checkpoints.zip`. You get a folder called `checkpoints`.
3. Move that `checkpoints` folder **inside** the `FairLens` folder.

Either way, the result must look like this (`checkpoints` sits inside `FairLens`, next to `app.py`):

```text
FairLens/                      <- the project folder; open the terminal here
├── app.py
├── install.sh                 (Mac installer)
├── install.ps1                (Windows installer)
├── RUN_ON_NEW_MACHINE.md
├── README.md
├── requirements.txt
├── configs/
├── datasets/
├── docs/
├── outputs/
├── scripts/
├── src/
├── tests/
└── checkpoints/               <- must be INSIDE FairLens
    ├── distilbert_bias/
    │   ├── model.safetensors
    │   └── ...
    └── deberta_bias/
        ├── model.safetensors
        └── ...
```

Windows tip: put the folder somewhere with a short path, such as `C:\FairLens`, to avoid Windows' long-path limit.

## 4. Install

Open a terminal **inside the `FairLens` folder**:
- **Mac:** open Terminal, type `cd ` (with a space), drag the `FairLens` folder onto the window, press Enter.
- **Windows:** open the folder in File Explorer, click the address bar, type `powershell`, press Enter.

Then run one command. It creates a private Python environment in `.venv` and installs everything (5–15 minutes).

**Mac:**
```bash
bash install.sh
```

**Windows:**
```powershell
powershell -ExecutionPolicy Bypass -File install.ps1
```

Optional extras:
- Windows PC with an **NVIDIA GPU**: add the CUDA PyTorch build. Pick the index matching your driver at <https://pytorch.org/get-started/locally/>, for example:
  ```powershell
  powershell -ExecutionPolicy Bypass -File install.ps1 -TorchIndex https://download.pytorch.org/whl/cu128
  ```
- Browser UI tests: add `--dev` (Mac) or `-Dev` (Windows).

## 5. Only if you did not get the checkpoints zip

Train the two detectors yourself (needs internet the first time):

```bash
.venv/bin/python scripts/train_detectors.py --model bert      # Mac
.venv/bin/python scripts/train_detectors.py --model deberta
```

```powershell
.venv\Scripts\python scripts\train_detectors.py --model bert      # Windows
.venv\Scripts\python scripts\train_detectors.py --model deberta
```

About 20 minutes in total on an Apple silicon Mac or NVIDIA GPU. On a CPU-only laptop it can take several hours.

## 6. Start the app

**Mac:**
```bash
.venv/bin/streamlit run app.py
```

**Windows:**
```powershell
.venv\Scripts\streamlit run app.py
```

Your browser opens <http://localhost:8501>. If it does not, open that address yourself. Stop the app with `Ctrl + C` in the terminal.

**First run:** the first analysis downloads the pretrained models (about 5 GB) and loads them, so it can take several minutes. Later runs take seconds. Models are stored once per computer in:
- Mac: `~/Library/Caches/FairLens/hf`
- Windows: `%LOCALAPPDATA%\FairLens\hf`

**Slow laptop or 8 GB RAM:** switch on **Lite mode** in the sidebar (smaller models). Restart the app between a long text demo and the image demo to free memory.

## 7. What to try

- **FairLens Review:** upload a PDF or DOCX job advert, or click "Use the sample job description", then "Run review". A ready-made test file is in `outputs/demo/demo_job_advert.pdf`.
- **Text Analysis:** paste a sentence and click "Analyze & mitigate".
- **Image Analysis:** upload a JPG/PNG/GIF, or choose "COCO sample".
- **Benchmarks / Model Comparison:** the stored results from our runs (they load without any model download).

## 8. Check that everything works (optional)

```bash
.venv/bin/python -m pytest -q -m "not slow"        # Mac
```

```powershell
.venv\Scripts\python -m pytest -q -m "not slow"    # Windows
```

Expect all tests to pass (243 at the time of packaging).

## 9. Re-run the experiments (optional)

```bash
.venv/bin/python scripts/run_all.py --list      # see the steps
.venv/bin/python scripts/run_all.py --from detection
```

Use `.venv\Scripts\python scripts\run_all.py ...` on Windows. Each step's log goes to `outputs/logs/`; every run gets a new run ID in `outputs/runs/`. The full run takes about 40 minutes on an Apple silicon Mac.

## Troubleshooting

| Problem | Fix |
|---|---|
| "Python 3.11 not found" | Install 3.11 (step 2). On Mac you can also run `PYTHON=/path/to/python3.11 bash install.sh` |
| Windows: "running scripts is disabled" | Use the exact command in step 4 (it includes `-ExecutionPolicy Bypass`) |
| `pip` fails to find `torch==2.14.1` on a Mac | You are on an Intel Mac or a non-3.11 Python. Use an Apple silicon Mac and Python 3.11 |
| App says a detector "could not run" / checkpoint missing | Unzip the checkpoints zip into the project (step 3) or train them (step 5) |
| Very slow, or the computer freezes | Turn on Lite mode; close other apps; force CPU with `FAIRLENS_DEVICE=cpu` if the GPU misbehaves |
| Models already downloaded somewhere else | Point to them: `export FAIRLENS_CACHE=/path/to/hf` (Mac) or `$env:FAIRLENS_CACHE = "D:\FairLens_cache\hf"` (Windows) before starting the app |
| Port 8501 already in use | `streamlit run app.py --server.port 8502` and open <http://localhost:8502> |

More detail: `README.md` (setup and settings), `docs/PROGRESS.md` (status), `docs/RESULTS.md` (all results).
