# Status audit

Independent check of each status claim in `docs/PROGRESS.md` ("Status at a glance", last updated 2026-10-24
~01:25 on the original Windows machine) against the files in this snapshot. Audited 2026-10-24 on macOS
(Apple M4, 16 GB, `.venv` Python 3.11.14). Nothing was re-trained or re-downloaded for this audit.

| # | Claim in PROGRESS.md | How it was verified | Actual result | Verdict |
|---|---|---|---|---|
| 1 | Tests: 44 passed (`-m "not slow"`), 2 slow tests not run | `.venv/bin/python -m pytest -q -m "not slow"` on the unmodified snapshot | 44 passed, 2 deselected (2.7 s). The 2 slow real-model tests were not run (they load models) | Confirmed (fast suite only) |
| 2 | 16 models verified with real inference and pinned | Read `outputs/model_verification/verification.json` and `configs/model_lock.json`; compared model IDs and SHAs with each other and with the `model_id` entries in `configs/models.yaml` | 16 entries, all `status: verified`, each with a latency and sample output, all dated 2026-10-23 (original Windows machine). `model_lock.json` pins the same 16 IDs with identical SHAs. **Gap:** `configs/models.yaml` also names `google/flan-t5-small` (`lite_model_id` of the FLAN-T5 rewriter), which is neither verified nor pinned. Verification was not re-run on macOS | Confirmed for the 16; lite FLAN-T5 unverified |
| 3 | Datasets: 15,136 rows, 0 leaking groups | `python scripts/verify_data_manifest.py` (recomputes row counts and SHA-256 of every `datasets/processed/*.parquet`, compares with `dataset_manifest.json`, checks group and text leakage) | All 4 manifest files match rows and hashes. 15,136 is the row count of `detection.parquet` only (other files: CrowS 1,508, StereoSet pairs 4,229, WinoBias 1,584). 0 groups and 0 identical texts span splits; 0 of 5,512 probe pairs disagree with the detection split. `mitigation_eval.parquet` (276) and `gate_controlled.parquet` (2,481) are not listed in the manifest (no recorded hash) | Confirmed; two curated files unhashed |
| 4 | Curated sets: 91 biased + 45 neutral authored; `mitigation_eval` 276 | Counted `datasets/curated/authored_cases.jsonl` and `mitigation_eval.parquet` by source/label; benchmark items cross-checked against `detection.parquet` splits | 136 authored cases (91 label 1, 45 label 0). `mitigation_eval.parquet` has 276 rows: 136 authored + BABE 50/20 + StereoSet 40 + CrowS 30; its 140 benchmark sentences are all in the **test** split (0 from train/val) | Confirmed |
| 5 | DistilBERT fine-tuned, best val macro-F1 0.761 | Read `checkpoints/distilbert_bias/` and run manifest `outputs/runs/20261024-002803-train_detector-691400/manifest.json` | Both record `best_val_macro_f1 = 0.7610638` (epochs 0.722 / 0.744 / 0.761, 2,182 s, Windows i7-1355U, CPU). `checkpoints/distilbert_bias/` contains **only** `fairlens_training.json`: no weights, so the 0.761 cannot be reproduced or used here without re-training. It is a **validation** score, not a test result | Number confirmed as recorded; model absent locally |
| 6 | DeBERTa-v3-small not trained | Listed `checkpoints/`, read the two other run manifests | No `deberta` checkpoint folder. Runs `...-010429-...-107de1` and `...-011003-...-27bede` are `detector: deberta`, `status: running`, no metrics, no artifacts (stopped runs) | Confirmed |
| 7 | No benchmark results exist | Listed `outputs/runs/`, queried `outputs/runs.sqlite`, read `docs/RESULTS.md` | Only 3 `train_detector` runs; no `benchmark_*` runs; `RESULTS.md` says "No completed run" for every task | Confirmed |
| 8 | Dashboard browser smoke test 44/44 | Read `outputs/ui_screenshots/report.json` | 44 page entries (11 pages x light/dark x 1366/1920), 0 with Streamlit errors, 0 with horizontal overflow; 40 entries log browser-console 404s (resource not found). Screenshots are gitignored, so not inspectable here. `flow` is null: the interactive flow was never run | Confirmed as recorded (empty states only) |
| 9 | File uploads work | - | In progress (ingestion / UI agents) | Pending |
| 10 | Git history / private repo pushed | Lead checked the original download for `.git` and tried to open the repo | The download had no `.git` folder, so commit history is unavailable; `tamannabarman02/FairLens` is private and not accessible from this account. The DistilBERT manifest records `git_commit: 04c768c-dirty`, so history existed on the original machine. The current `.git` is a fresh snapshot (index only, no commits) | Not verifiable |

## Other findings

- `outputs/runs.sqlite` stored absolute Windows paths (`D:\FairLens\outputs\runs\...\manifest.json`), so the
  dashboard could not open run manifests on any other machine. Manifests now store project-relative paths, the
  index falls back to the local run folder, and `python scripts/rebuild_runs_index.py` recreates the index from
  the run folders.
- Old manifests record dataset keys with backslashes (`datasets\processed\detection.parquet`) and the
  checkpoint as `D:\FairLens\checkpoints\distilbert_bias`; they are kept unchanged as historical records.
- The hardware of every recorded run is the original Windows laptop (CPU-only); no run has been made on macOS.
