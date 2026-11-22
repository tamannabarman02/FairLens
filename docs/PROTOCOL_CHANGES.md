# Protocol changes

Every change to data, models, training, metrics, thresholds, baselines or splits is recorded here, with its
reason, before the run that uses it. Pure engineering changes (device support, paths, encodings, UI) that do not
alter what is measured are listed separately at the end for traceability.

| # | Date | Change | Reason | Affects |
|---|---|---|---|---|
| P1 | 2026-10-24 | DeBERTa-v3-small fine-tuning freezes the embedding layer by default (`freeze_embeddings: true`; `--no-freeze-embeddings` restores full fine-tuning). DistilBERT unchanged. Recorded in each checkpoint's `fairlens_training.json` and run manifest. | The 128k-token embedding matrix (about 98M of 141M parameters) plus AdamW state made training swap at >30 s/step on 8–16 GB laptops (docs/PROGRESS.md, "DeBERTa problem"). Freezing keeps training feasible on every teammate's machine. | Detection benchmark (DeBERTa row) |
| P2 | 2026-10-24 | Sequence classifiers (fine-tuning, fine-tuned/Hub detectors, NLI cross-encoder) are loaded explicitly in float32. | transformers 5.x loads weights in the checkpoint's stored dtype by default. The pinned `microsoft/deberta-v3-small` checkpoint is stored in float16, so training ran in float16 without loss scaling and the loss became NaN after a few steps on both CPU and MPS (verified 2026-10-24). float32 is the precision the methodology assumes. Generation models keep their existing loading (Qwen: `dtype="auto"`). | Detection, preservation signals |
| P3 | 2026-10-24 | New Gate baseline (e) **no-change**: always keeps the original text. Scored with the existing gold definitions. | Required by the revised plan: a policy that never rewrites is the natural lower bound for unnecessary changes. | Gate benchmark |
| P4 | 2026-10-24 | New Gate baseline **LLM-as-judge** (Qwen2.5-0.5B-Instruct, yes/no faithfulness question, greedy, seed 13; abstain = defer). Prompt chosen among 3 documented variants on the **validation split only** (`scripts/validate_llm_judge.py`, highest Cohen's κ, ties → lower unsafe-yes rate, then lower abstain rate), frozen in `configs/llm_judge.json` before any test evaluation. `run_ablations.py` skips the judge if that file is missing. | Required by the revised plan; answers "why not ask an LLM whether the rewrite kept the meaning?" | Gate benchmark |
| P5 | 2026-10-24 | Each policy's selection output also reports `chosen_gold_counts` (ACCEPT / RETRY / REJECT / DEFER). Descriptive only; no metric changed. | `harmful_selection` is defined as gold ≠ ACCEPT, so keeping a biased original counts as harmful just like choosing a fact-breaking rewrite. The counts separate the two without redefining the metric. **Open decision:** whether to report a separate "kept biased original" rate in the paper. | Gate benchmark reporting |

## Known open questions (not yet decided)

- The LLM judge is only asked about changed candidates (as the highest-similarity baseline does), so it never
  selects the original and will make unnecessary changes on neutral inputs when it approves a paraphrase.
  This is reported as a property of a faithfulness-only judge, not tuned away.
- `google/flan-t5-small` (Lite-mode rewriter) is not in `configs/model_lock.json`. Lite mode is for the demo
  only; benchmarks use the standard models.
- `gate_controlled.parquet` and `mitigation_eval.parquet` have no hash in `dataset_manifest.json`
  (see `docs/AUDIT.md`).

## Engineering changes (no change to what is measured)

- Device selection CUDA → MPS → CPU with `FAIRLENS_DEVICE` override; CPU fallback per model; Florence-2 stays
  on CPU on macOS. Numerical differences between devices are possible at floating-point level; the final runs of
  each experiment are made on one named device, recorded in the manifest.
- Per-user model cache folder on every OS (`FAIRLENS_CACHE` override); no drive letters.
- UTF-8 text IO everywhere; Windows no longer needs `PYTHONUTF8=1`.
- Run manifests record run ID, task, models and pinned revisions, device, seeds, dataset hashes, config, outputs,
  git commit, hardware, runtime and status; `outputs/runs.sqlite` is a rebuildable index.
- Score and generation caches are keyed by model, revision and device.
- Ingestion loaders fixed for Windows encodings, DOCX tables/headers, multi-column CSVs, scanned PDFs and image
  edge cases (`tests/test_ingestion_formats.py`).
- The pinned `distilbert-base-uncased` revision could not be downloaded under its legacy repo name (Hub 404 on
  the weights file). The identical file from the same commit (`12040acc…`) under the current name
  `distilbert/distilbert-base-uncased` was placed in the cache for the first training run. `models.yaml` and
  `model_lock.json` now use the canonical name `distilbert/distilbert-base-uncased` at the same commit, so every
  teammate downloads the same weights without the workaround.
