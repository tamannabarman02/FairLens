# Architecture

```
Input (text / TXT / PDF / DOCX / CSV / image)
   │  ingestion/loaders.py (limits, typed errors) · ingestion/segment.py (offset-preserving sentences)
   ▼
Bias Detection ── detection/{rule_based, hf_classifier (BERT, DeBERTa, Hub), ZeroShotNLI (BART-MNLI)}
   │  common interface: predict · predict_batch · explain · info ; built from configs/models.yaml (registry.py)
   ▼
Mitigation ── mitigation/{rule_based, llm_adapters (FLAN-T5 seq2seq, Qwen chat)} · prompts.py (one family)
   ▼
Reliability Gate ── reliability_gate/{signals, rules, retry, learned, baselines}
   │  signals use evaluation/{preservation (MiniLM, NLI cross-encoder, fluency), facts (spaCy + regex)}
   ▼
Output: decision (ACCEPT / REJECT / RETRY / HUMAN_REVIEW) + chosen text + reason code + evidence

Image: multimodal/captioners (BLIP, Florence-2 native, ViT-GPT2 fallback) → caption_evaluation (CLIP, coverage,
flags) → the same text detectors.
```

## Modules
| Package | Responsibility |
|---|---|
| `config.py` | loads `configs/models.yaml`, pins revisions from `configs/model_lock.json`, sets HF cache before any model import |
| `hardware.py` | detects CPU/RAM/GPU, picks `lite` / `standard` / `gpu` profile (`FAIRLENS_PROFILE` overrides) |
| `models.py` | `ModelManager`: lazy loading with a memory budget; evicts least-recently-used models so one large model is resident at a time |
| `pipeline.py` | `analyze_text`, `mitigate_text`, `analyze_image`: the end-to-end flows used by the UI, use cases and tests |
| `experiments/` | `tracking.py` (SQLite index + JSON manifests), `cache.py` (score + generation cache), `mitigation_runner.py` (batched, stage-by-stage), `jobs.py` (background subprocess jobs with cancel), `use_cases.py` |
| `reporting/` | `charts.py` (Plotly, theme-aware), `export.py` (JSON / CSV / HTML / PDF) |
| `ui/` | `components.py` (card, tile, badge, stepper, score bar, decision banner, check grid, tables, diff), `state.py` (cached resources), `results.py` (reads run files only), `pages/` (11 pages) |

## Data flow for results
`scripts/*.py` → `outputs/runs/<run_id>/{manifest.json, *_summary.json, *.parquet}` + `outputs/runs.sqlite`
→ dashboard (`ui/results.py`) and `docs/RESULTS.md` (`scripts/generate_results_md.py`). Nothing else writes metrics.

## Resource strategy (CPU laptop)
- Experiments run stage by stage: all generations with one model, then each detector over all texts, then the
  preservation models, so each model is loaded once per run.
- Detector scores and generations are cached in `outputs/cache/scores.sqlite`, keyed by model identity
  (incl. checkpoint training timestamp and hypothesis template) and text hash.
- Lite mode swaps BART-large-MNLI → DistilBART-MNLI and FLAN-T5-base → FLAN-T5-small.
- Long jobs launched from the dashboard run as separate processes with logs in `outputs/jobs/` and can be cancelled.
