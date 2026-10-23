# Progress log

Resumable status for anyone continuing this build. **Update this file after every finished step and commit + push.**
Last updated: 2026-10-24 ~01:25 local, when the owner's usage limit was reached. **No jobs are running.**
Repo: https://github.com/tamannabarman02/FairLens (private).

## Status at a glance

| Item | State | Evidence |
|---|---|---|
| Phase 0 environment + setup | DONE | `docs/ENVIRONMENT.md`, `install.ps1`, `requirements.txt` (pinned) |
| Model verification (16 models, real inference) | DONE, all verified | `outputs/model_verification/verification.json`, `configs/model_lock.json` |
| Datasets (BABE, StereoSet, CrowS-Pairs, WinoBias probe) | DONE | `datasets/processed/*.parquet`, `dataset_manifest.json` (15,136 rows, 0 leaking groups) |
| Curated authored set + mitigation set + controlled gate set | DONE | `datasets/curated/authored_cases.jsonl` (91 biased + 45 neutral), `mitigation_eval.parquet` (276), `gate_controlled.parquet` |
| DistilBERT fine-tune | DONE: best val macro-F1 **0.761** (epochs 0.722 / 0.744 / 0.761), 36 min | `checkpoints/distilbert_bias/` (weights local only, gitignored), `fairlens_training.json` tracked |
| DeBERTa-v3-small fine-tune | **NOT DONE**: stopped twice, no checkpoint saved | see "DeBERTa problem" below |
| FLAN-T5 copy-failure investigation | DONE (negative result) | see Findings |
| Prompt selection (train-split dev sample) | NOT RUN | `scripts/select_prompts.py` (now includes few-shot variants) |
| Detection / mitigation / gate+ablation / image benchmarks | **NOT RUN** (no results exist yet) | scripts ready |
| Use-case runs | NOT RUN | `scripts/run_use_cases.py` |
| Dashboard (11 pages) | WRITTEN; browser smoke test **passed 44/44** (11 pages x light/dark x 1366/1920, no Streamlit exceptions, no horizontal overflow) with empty states only | `scripts/ui_smoke.py`, `outputs/ui_screenshots/report.json` (screenshots local) |
| Dashboard with real results / interactive flow (`ui_smoke.py --flow`) | NOT YET VERIFIED | needs trained models + benchmark runs |
| Tests | **44 passed** (`pytest -m "not slow"`); 2 slow real-model tests written, not yet run | `tests/` |
| Docs written | ENVIRONMENT, LEGACY_REVIEW, DATASETS, MODEL_CARD_REGISTRY (measured table filled by generator), METHODOLOGY, ARCHITECTURE, UI_PLAN | `docs/` |
| Docs NOT written | LIMITATIONS, RESEARCH_REPORT, PROJECT_REPORT_OUTLINE, APPLICATIONS_AND_VIVA, final README; RESULTS.md is a placeholder until runs exist | |

## Findings so far (real)
- CPU-only i7-1355U, 15.7 GB RAM but typically ~2 GB free -> Windows swaps (pagefile 2 GB in use observed).
- Florence-2 runs via `florence-community/Florence-2-base` **without trust_remote_code**; ViT-GPT2 fallback verified.
- **FLAN-T5 base and large copy biased input unchanged** in zero-shot (4 prompt formats) and 3-shot tests on train-split
  sentences. Real capability limit; kept as required by the brief and reported as a negative result.
- Qwen2.5-0.5B rewrote the sample correctly (2.4 s); 1.5B slower (9.3 s) with slight meaning drift.
- BART-MNLI original pair hypothesis scored a neutral sentence 0.90 -> 5 templates, chosen on val inside the detection benchmark.
- WNC blocked (gated mirror, unreachable host) -> BABE used.

## DeBERTa problem (must fix first)
DeBERTa-v3-small training ran at >30 s/step (vs DistilBERT ~1 s/step) because of **RAM swapping**: its 128k-token
embedding matrix (~98M of 141M params) plus AdamW state needs ~2.3 GB while only ~1.8 GB was free.
Planned fix (not yet implemented): freeze the embedding layer for DeBERTa in `scripts/train_detectors.py`
(line ~104: build AdamW over `p for p in model.parameters() if p.requires_grad` after setting
`requires_grad=False` on `model.base_model.embeddings`), record `"freeze_embeddings": true` in HPARAMS/metadata,
close other apps, then time 20 steps with `--max-train 320 --epochs 1 --log-every 4 --out-suffix _speedtest`
(run with `python -u` and redirect to a file; piping through grep buffered output). If still too slow, use
`--max-length 64` (p95 token length is 51) and/or `--epochs 2`, and document the change.

## Exact next steps (run from `D:\FairLens`, ONE heavy job at a time)
```powershell
$env:PYTHONUTF8="1"; $env:HF_HUB_DISABLE_PROGRESS_BARS="1"
.venv\Scripts\python.exe -m pytest -q -m "not slow"                         # sanity: expect 44 passed
# 1. fix + train DeBERTa (see above), then:
.venv\Scripts\python.exe -u scripts\train_detectors.py --model deberta > outputs\logs\train_deberta.log 2>&1
# 2. prompt variant on train-split dev sample (~20-30 min)
.venv\Scripts\python.exe scripts\select_prompts.py
# 3. detection benchmark (~1 h, BART-MNLI slowest)
.venv\Scripts\python.exe scripts\run_benchmarks.py --task detection
# 4. gate benchmark + ablations (tunes thresholds on val) (~30 min)
.venv\Scripts\python.exe scripts\run_ablations.py
# 5. mitigation benchmark (uses tuned thresholds from step 4) (~1 h)
.venv\Scripts\python.exe scripts\run_benchmarks.py --task mitigation
# 6. image experiment (downloads COCO annotations 241 MB + 80 images) (~45 min)
.venv\Scripts\python.exe scripts\run_image_experiment.py
# 7. demo use cases, results doc, slow tests, browser flow test
.venv\Scripts\python.exe scripts\run_use_cases.py
.venv\Scripts\python.exe scripts\generate_results_md.py
.venv\Scripts\python.exe -m pytest -q -m slow
.venv\Scripts\python.exe scripts\ui_smoke.py --flow
```
Then write LIMITATIONS, RESEARCH_REPORT, PROJECT_REPORT_OUTLINE, APPLICATIONS_AND_VIVA (real numbers only),
finish README; commit and `git push` after every step.

## Known caveats
- `checkpoints/*/` weights, `datasets/raw/`, caches, logs and screenshots are gitignored (regenerable).
- Avoid Python heredocs with backslashes in Git Bash (they were mangled once); use files or the Edit tool.
- `outputs/runs/` contains 3 `train_detector` manifests: one completed (DistilBERT), two DeBERTa runs that were
  stopped and remain marked `running`; they are real records of stopped runs, not results.

## History
- Phase 0: env check, skeleton, pinned requirements, installer; legacy Codex projects archived to `D:\FairLens_archive`.
- Core library + tests; 16 models verified; datasets built; curated set authored; 11 UI pages; integration tests;
  DistilBERT trained; FLAN-T5 diagnosed; UI browser-tested (empty states); private GitHub repo created and pushed.
