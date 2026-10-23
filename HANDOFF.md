# FairLens handoff

For anyone continuing this project in their own Claude Code (or by hand).

## What this is
FairLens is a final-year research project: bias detection (BERT, DeBERTa, BART-MNLI, rules), mitigation
(rule rewriter, FLAN-T5, Qwen2.5), and the **Reliability Gate** that decides ACCEPT / REJECT / RETRY / HUMAN_REVIEW
for each rewrite, plus an 11-page Streamlit dashboard.

- **The brief (source of truth):** `docs/brief/FairLens_Master_Build_Brief.pdf` (copy of the original at
  `D:\Downloads\FairLens Claude Code Prompt.pdf`). Read it fully first.
- **Current status and exact next steps:** `docs/PROGRESS.md` (always kept up to date).
- **Ground rules from the brief:** no fake results (every number from a run in `outputs/`), verify models before use,
  commit after each step, CPU-friendly, document failures in `docs/LIMITATIONS.md`.
- The user's later instructions: project lives at `D:\FairLens`; keep synthetic demo inputs (`datasets/synthetic_demo/`)
  separate from benchmark data; the UI reference image `FairLens_UI_Reference.png` has **not** been provided yet
  (plan in `docs/design/UI_PLAN.md`; copy it to `docs/design/reference_mockup.png` when it arrives).

## Setup on a new machine (Windows)
```powershell
git clone https://github.com/tamannabarman02/FairLens.git   # private repo; the owner must add you as a collaborator
cd FairLens
powershell -ExecutionPolicy Bypass -File install.ps1     # creates .venv (Python 3.11), installs pinned deps
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -m pytest -q                    # expect all tests to pass
```
- Needs Python 3.11 (`py -3.11`), ~25 GB free disk for models. Linux/macOS: `bash install.sh`.
- Model cache path is `paths.cache_dir` in `configs/models.yaml` (`D:/FairLens_cache/hf`). If that drive does not exist,
  the code falls back to `.cache/hf` in the project, or set `FAIRLENS_CACHE`. Always import `fairlens` before
  `transformers` so the cache env var applies.
- Regenerate everything not in git:
  1. `python scripts/build_datasets.py` and `python datasets/curated/authored_cases.py` and `python scripts/build_mitigation_sets.py`
  2. `python scripts/verify_models.py` (downloads + verifies all models, ~12 GB, ~40 min)
  3. Then follow "Exact next steps" in `docs/PROGRESS.md`.

## Map of the code
| Path | Purpose |
|---|---|
| `configs/models.yaml` | every model ID, gate thresholds, hardware profiles |
| `configs/model_lock.json` | pinned Hub commit per verified model |
| `src/fairlens/detection/` | detector interface, rule lexicon, HF classifiers, zero-shot NLI, registry |
| `src/fairlens/mitigation/` | prompts (one shared family), rule rewriter, FLAN-T5 / Qwen adapters |
| `src/fairlens/evaluation/` | facts checker, similarity/NLI/fluency, metrics (ECE, bootstrap) |
| `src/fairlens/reliability_gate/` | signals, transparent rules, learned gate, retry, baselines |
| `src/fairlens/experiments/` | run tracking (SQLite + manifests), score cache, batched mitigation runner, jobs |
| `src/fairlens/ui/` | components, state, pages (11) ; `app.py` is the entry |
| `scripts/` | train_detectors, run_benchmarks, run_ablations, run_image_experiment, verify_models, build_* |

## Resume here (state when the owner's usage limit was reached)
No jobs are running. DistilBERT is trained (`checkpoints/distilbert_bias`, local only: re-run
`scripts/train_detectors.py --model bert` on a new machine, ~36 min). DeBERTa is NOT trained: fix the RAM-swapping
problem first (freeze embeddings), exactly as described in `docs/PROGRESS.md` → "DeBERTa problem", then follow the
numbered command list in "Exact next steps" there. No benchmark results exist yet; never fill numbers by hand.

## Pending (summary; details in docs/PROGRESS.md)
Detector training (running at handoff time), prompt selection, all four benchmark runs, use-case runs, browser check
of the UI, integration tests, and all research docs (METHODOLOGY, MODEL_CARD_REGISTRY, DATASETS, RESULTS, LIMITATIONS,
PROJECT_REPORT_OUTLINE, APPLICATIONS_AND_VIVA, README). Final summary to the user must state what works, headline
results, what failed, and what to say in the viva, using only real run numbers.
