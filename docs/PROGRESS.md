# Progress log

Working specification: **V4** (`docs/spec/FAIRLENS_V4_SPEC.md`, copied from the owner's artifact).
Protocol decisions: `docs/PROTOCOL_CHANGES.md`. Update this file after every step; commit and push.
Repo: https://github.com/tamannabarman02/FairLens (private).

## Stage 0 audit (2026-10-24, Windows 11 laptop, i7-1355U, CPU only, Python 3.11.9)

Evidence file: `outputs/audit/stage0_audit.json` (from `scripts/audit_stage0.py`), plus the files named below.
The owner's Mac was **not available** in this session, so every macOS item is PENDING.

| # | Claim | What was checked / command | Actual outcome | Evidence | Status | Next action |
|---|---|---|---|---|---|---|
| 1 | 44 fast tests pass | `.venv\Scripts\python -m pytest -q -m "not slow"` | 44 passed, 2 deselected (46 collected) on Windows | `outputs/audit/stage0_pytest_fast.txt` | **PARTIAL** (Windows VERIFIED; macOS PENDING) | Run on the Mac; CI (1.2c) adds GitHub-hosted macOS/Linux |
| 2 | Installs from scratch on Windows and macOS/Linux | Read `install.ps1`, `install.sh`, `requirements.txt` | Not yet run on a clean machine. Found a blocker for macOS: `requirements.txt` pins `torch==2.14.1+cpu` from the PyTorch CPU index; no `+cpu` wheels exist for macOS, so `install.sh` would fail there | `requirements.txt` lines 5-6 | **FAILED** (macOS by inspection) / **PENDING** (clean Windows) | Platform markers for torch (1.1), clean-venv install on Windows, CI install on all three OSes |
| 3 | 16 models verified, revisions pinned | Compared `verification.json` with `configs/model_lock.json`; checked pinned snapshot exists in the HF cache | 16/16 `verified`, 16 locked, every lock SHA equals the verified SHA, every pinned snapshot is cached. `google/flan-t5-small` (Lite-mode rewriter) is configured but **not verified or locked**. Inference was not re-run in this audit (records dated 2026-10-23) | `outputs/audit/stage0_audit.json` → `models` | **PARTIAL** | Re-run one inference per model in profiling (1.4) and verify flan-t5-small |
| 4 | 15,136 rows, no leaking groups | `python scripts/build_datasets.py --out D:/FairLens_cache/tmp/rebuild_check`, compared manifests; leakage queries | Rebuild is **byte-identical**: detection 15,136 rows sha `7cbacdedf983ba1c`, CrowS 1,508, StereoSet 4,229, WinoBias 1,584, all hashes match; 0 groups and 0 exact texts span splits; **4 texts span splits after lower-casing/punctuation normalisation** (near-duplicates); mitigation set has 0 benchmark items from train/val; gate set has 0 inputs spanning val/test. `mitigation_eval.parquet` (276 rows) and `gate_controlled.parquet` (2,481 rows) are **not in the manifest** | `stage0_audit.json` → `datasets` | **VERIFIED** (counts, hashes, group leakage) with 2 findings | Log the 4 near-duplicates (D.1) and add both derived files to the manifest (1.3) |
| 5 | DistilBERT trained, val macro-F1 0.761 | `python scripts/eval_checkpoint.py --model bert` on the local checkpoint (weights are present on this laptop, absent from Git) | Re-evaluated val macro-F1 **0.7610638 = recorded 0.7610638** (abs diff 0.0), n = 2,119, 176 s on CPU | `outputs/audit/eval_bert_val.json`, `checkpoints/distilbert_bias/fairlens_training.json`, run `20261024-002803-train_detector-691400` | **VERIFIED** (this laptop) | Other machines must retrain (`train_detectors.py --model bert`) because weights are not in Git |
| 6 | DeBERTa not trained; two stopped runs marked running | Listed `checkpoints/`; read both manifests | No DeBERTa checkpoint. Runs `20261024-010429-…-107de1` (full) and `…-011003-…-27bede` (320-sample speed test) were killed; both were still `running`. **Corrected** to `stopped` with a `stop_note` (cause: RAM swapping) | `outputs/runs/<id>/manifest.json`, `outputs/runs.sqlite` | **VERIFIED** (status corrected) | Profile and apply the DeBERTa fix ladder (1.4) |
| 7 | No benchmark results exist | `runs.sqlite` + folder listing | 3 runs, all `train_detector`; **0 benchmark runs**. Manifest paths are stored as absolute Windows paths (`D:\FairLens\outputs\runs\…`) | `stage0_audit.json` → `runs` | **VERIFIED** (no results) with finding | Relative POSIX paths + index rebuilt from folders (1.2b, 1.2d) |
| 8 | Uploads work for TXT, PDF, DOCX, CSV and images | Function-level calls to `extract_text`/`load_image` on generated files incl. Windows encodings; browser upload **not** tested | TXT, PDF, DOCX (paragraphs), quoted CSV, JPG, PNG, GIF extract correctly; UTF-16+CRLF and Windows-1252 TXT work; corrupt/empty/unsupported files are rejected with messages. **DOCX tables are not extracted** ("No text found"). (An unquoted-comma CSV in the first audit attempt split into columns; that was a malformed test file, not a loader bug.) | `stage0_audit.json` → `uploads` | **PARTIAL** (function level only; 1 bug) | Fix DOCX tables; browser test per format (1.5) |
| 9 | Git history intact | `git fsck`, `git log`, `git ls-remote origin main` | fsck clean, 12 commits from `753455a` (Phase 0) to `df97294`, local = `origin/main`, private repo | terminal output, GitHub | **VERIFIED** | — |
| 10 | Hardware selection | Read `hardware.py`, adapters, `app.py` | Checks CUDA only; **no MPS, no `FAIRLENS_DEVICE` override; models are never moved to a GPU**; MiniLM hard-coded to `device="cpu"`; top bar always shows "CPU" | `stage0_audit.json` → `hardware` | **FAILED** against V4 1.2 | Implement CUDA → MPS → CPU (1.2) |
| 11 | Scripts, locks, manifests, UI plan, tests, run storage exist | Inventory | Present: 13 scripts (`audit_stage0`, `build_datasets`, `build_mitigation_sets`, `check_env`, `eval_checkpoint`, `generate_results_md`, `run_ablations`, `run_benchmarks`, `run_image_experiment`, `run_use_cases`, `select_prompts`, `train_detectors`, `ui_smoke`, `verify_models`), `configs/model_lock.json`, `dataset_manifest.json`, `docs/design/UI_PLAN.md`, 46 tests, `outputs/runs/` + `runs.sqlite`. OS assumptions found: `D:/FairLens_cache/hf` default in `models.yaml`, Windows paths in manifests and `ui_screenshots/report.json` | repo tree | **VERIFIED** (exists) with findings | 1.2b |

Earlier UI smoke test (empty states only): 44/44 page renders without Streamlit exceptions, but each page logged two
browser "404 Not Found" console errors (V4 UI audit finding), not yet investigated.

## Stage status (V4 order)
| Stage | Status |
|---|---|
| 0 Audit | DONE (table above); macOS items PENDING (no Mac in this session) |
| 1 Cross-platform setup and reproducibility | IN PROGRESS |
| 2–7 | NOT STARTED; **no benchmark results exist** |

## Exact next steps
1.1 torch platform markers + clean-venv install on Windows → 1.2 device order → 1.2b OS assumptions → 1.2c CI →
1.2d run index from folders → 1.3 manifest for derived sets → 1.4 profiling + DeBERTa fix ladder → 1.5 uploads e2e →
1.6 manifest fields test → D.1 label mapping → train DeBERTa → D.2 detection run → M.1/M.2 → G.1/G.2 → A.1–A.5 → I.1/I.2 → P.1–P.3, with UI (U.1–U.7) in separate commits.
