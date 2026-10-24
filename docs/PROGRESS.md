# Progress log

Resumable status for anyone continuing this build. **Update this file after every finished step.**
Last updated: 2026-10-24 ~04:50, on macOS (Apple M4, 16 GB, MPS). Cross-platform work done; first full set of
benchmark runs completed. See `docs/PROTOCOL_CHANGES.md` for every protocol change and `docs/AUDIT.md` for the
verification of the earlier status claims.

## Status at a glance (2026-10-24)

| Item | State | Evidence |
|---|---|---|
| Runs on Windows, macOS, Linux | Code done; tests pass on macOS; CI workflow for all three added, **not yet run on GitHub** and **not yet tried on a Windows laptop** | `.github/workflows/tests.yml`, README |
| Device selection CUDA → MPS → CPU | Done | `hardware.py`, `models.py`, `tests/test_device.py` |
| Upload formats (TXT, PDF, DOCX, CSV, JPG, PNG, GIF) | Verified at library level (87 tests) and in the browser flow (54/54, model calls stubbed) | `tests/test_ingestion_formats.py`, `docs/design/UI_AUDIT.md` |
| DistilBERT fine-tune | Done on MPS: best **val** macro-F1 0.768 | run `20261024-032822-train_detector-ceb08f` |
| DeBERTa-v3-small fine-tune | Done on MPS: best **val** macro-F1 0.818 (frozen embeddings, float32: P1, P2) | run `*-034605-train_detector-6acf4e` |
| Prompt selection | Done: `V2_inclusive`, no demos (train-split sample) | `configs/prompt_selection.json` (no run manifest; script does not use the tracker) |
| Detection benchmark | Done (first attempt failed: pandas 3 bug, fixed) | run `20261024-043227-benchmark_detection-a4a61f` |
| LLM-judge prompt (val only) | Done: `checker_system` | `configs/llm_judge.json` |
| Gate vs baselines + ablations | Done | run `20261024-040545-benchmark_gate-7565de` |
| Mitigation benchmark | Done | run `20261024-040853-benchmark_mitigation-000f03` |
| Image experiment | Done | run `20261024-041855-benchmark_image-4acb2c` |
| Six use cases | Done (no run manifest; the script writes `outputs/use_cases/`). **Failure case:** the reliability demo sentence "Women are not suited to engineering, …" scored ensemble 0.48 (< 0.5) and was left unchanged as NEUTRAL_UNCHANGED | `outputs/use_cases/`, `outputs/logs/use_cases.log` |
| Tests | 243 fast + 2 slow (real models) pass on macOS | `pytest -q -m "not slow"`, `pytest -q -m slow` |
| UI | Audited and redesigned; FairLens Review page added; 48/48 page screenshots clean with real results | `docs/design/UI_AUDIT.md` |
| Results doc | Generated from runs | `docs/RESULTS.md` |

## Open decisions (need the team)

1. **BART-MNLI template downstream.** The detection benchmark picks the zero-shot template on validation
   (`T2_single_biased`), but the Gate, mitigation and the app use the default pair template from
   `configs/models.yaml` (`T5_pair_original`, val macro-F1 0.333). Options: keep as is and document it, or write the
   chosen template into the config and re-run `gate`, `mitigation`, `use_cases`, `results`. Not changed silently.
2. **"Harmful choice" for keeping a biased original** (P5 in `PROTOCOL_CHANGES.md`).
3. Human evaluation, red-team corruption tests on real accepted rewrites, risk–coverage curves, threshold
   sensitivity, prior-work review: **not started** (plan stage 4 and 6).
4. Final runs for the paper should be made on one named device; current runs are all MPS.

## Next steps

```text
Windows teammate: install.ps1, run the fast tests, report results
Push to GitHub so the CI matrix runs on Windows/macOS/Linux
Decide open decision 1, then: python scripts/run_all.py --from gate
Stage 4: risk-coverage, red-team, human evaluation, sensitivity
Write LIMITATIONS, related work, paper
```

## Earlier log (Windows, before 2026-10-24)

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
