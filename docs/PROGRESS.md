# Progress log

Resumable status for anyone continuing this build. **Update this file after every finished step and commit.**
Last updated: 2026-10-24 (during Phase 1/2: detector fine-tuning running).

## Status at a glance

| Phase | State | Evidence |
|---|---|---|
| 0 Environment + setup | DONE | `docs/ENVIRONMENT.md`, `install.ps1`, 3 tests pass |
| 1 Model verification | DONE (16/16 verified) | `outputs/model_verification/verification.json`, `configs/model_lock.json` |
| 1 Detector fine-tuning | RUNNING | `outputs/logs/train_detectors.log` -> `checkpoints/distilbert_bias`, `checkpoints/deberta_bias` |
| 1 Prompt selection (dev split) | NOT RUN | `scripts/select_prompts.py` -> `configs/prompt_selection.json` |
| 2 Datasets | DONE | `datasets/processed/*.parquet`, `dataset_manifest.json` (15,136 rows, 0 leaking groups) |
| 2 Mitigation + controlled gate sets | DONE | `mitigation_eval.parquet` (276), `gate_controlled.parquet` (3 seeds x ~827) |
| 3 Gate code (rules, learned, retry, baselines) | DONE, unit-tested | `src/fairlens/reliability_gate/` |
| 3 Gate benchmark + ablations | NOT RUN (needs trained detectors) | `scripts/run_ablations.py` |
| Detection benchmark | NOT RUN (needs trained detectors) | `scripts/run_benchmarks.py --task detection` |
| Mitigation benchmark | NOT RUN | `scripts/run_benchmarks.py --task mitigation` |
| Image experiment | NOT RUN | `scripts/run_image_experiment.py` |
| 4 Architecture | DONE | `src/fairlens/*` per brief |
| 5 UI (11 pages) | WRITTEN, imports OK, NOT yet visually checked | `app.py`, `src/fairlens/ui/` |
| 6 Tests | 40 unit tests pass; integration tests NOT written | `tests/` |
| Docs (METHODOLOGY, MODEL_CARD_REGISTRY, DATASETS, RESULTS, LIMITATIONS, REPORT_OUTLINE, APPLICATIONS_AND_VIVA) | NOT WRITTEN | `docs/` |

## Key findings so far (real, from runs)
- Machine: i7-1355U, 15.7 GB RAM (~2 GB free typically), CPU-only. Venv + HF cache on D: (`D:\FairLens_cache\hf`, ~19 GB).
- Florence-2 runs via the native `florence-community/Florence-2-base` port -> **no `trust_remote_code` needed**. Fallback ViT-GPT2 also verified.
- FLAN-T5 base and large returned the sample sentence **unchanged** with the first prompt; Qwen2.5-0.5B rewrote it correctly.
  -> run `select_prompts.py` (train-split dev sample) before the mitigation benchmark.
- Qwen2.5-1.5B: 9.3 s/rewrite and added "at least" (meaning drift) -> 0.5B chosen as standard; 1.5B is the `large` option.
- BART-MNLI with the original pair hypothesis scored a neutral sentence 0.90 -> template is chosen on the val split inside the detection benchmark (5 documented templates).
- WNC (Wiki Neutrality Corpus) blocked: HF mirror is gated, Stanford host unreachable -> BABE used as the sentence-level biased-language corpus. WinoBias used only as a neutral probe (FPR + pro/anti gap).
- NLI paraphrase test: strict bidirectional entailment was 0.005 for a valid paraphrase -> `entailment_mode` (both/forward) is tuned on the gate val split.

## Exact next steps (run from `D:\FairLens`, in this order)
Use `.venv\Scripts\python.exe` and set `PYTHONUTF8=1`. Run ONE heavy job at a time (RAM).

1. Wait for / re-run training: `python scripts/train_detectors.py --model all` (~35 min BERT, ~1.5 h DeBERTa). Check `checkpoints/*/fairlens_training.json`.
2. `python scripts/select_prompts.py` (~15 min) -> `configs/prompt_selection.json`.
3. `python scripts/run_benchmarks.py --task detection` (~1 h; BART-MNLI is slowest).
4. `python scripts/run_ablations.py` (~30 min) -> gate benchmark, tuned thresholds, ablations.
5. `python scripts/run_benchmarks.py --task mitigation` (~1 h). Uses the val-tuned gate thresholds from step 4 automatically, so run step 4 first.
6. `python scripts/run_image_experiment.py` (~45 min; downloads 241 MB COCO annotations + 80 images).
7. `python scripts/run_use_cases.py` -> `outputs/use_cases/*.json`.
8. Start UI: `.venv\Scripts\streamlit run app.py`; click through all 11 pages in light + dark, at 1366 and 1920 px; fix layout bugs.
9. Write integration tests (`tests/test_integration.py`: full text pipeline + image pipeline on tiny fixtures with fakes; slow-marked real-model test).
10. Write docs: METHODOLOGY, MODEL_CARD_REGISTRY (from verification.json), DATASETS (from dataset_manifest.json), LIMITATIONS, PROJECT_REPORT_OUTLINE, APPLICATIONS_AND_VIVA (real numbers from runs), README (setup/run/screenshots), and `scripts/generate_results_md.py` -> docs/RESULTS.md.
11. Commit after each step.

## Known TODOs / caveats
- `scripts/generate_results_md.py` is referenced by `fairlens.experiments.jobs` but not written yet.
- UI has not been run in a browser yet; expect small Streamlit API/CSS fixes (Streamlit 1.66).
- `datasets/raw/` and `checkpoints/` are gitignored (regenerable): `build_datasets.py` rebuilds data, `train_detectors.py` rebuilds checkpoints.
- Avoid Python heredocs with backslashes in Git Bash (they were mangled once); use files or the Edit tool.

## History
- Phase 0: env check, skeleton, pinned requirements, installer; legacy Codex projects archived to `D:\FairLens_archive` and reviewed (`docs/LEGACY_REVIEW.md`).
- Core library + 40 unit tests; 16 models verified; datasets built; authored curated set (91 biased + 45 neutral); UI pages written.
