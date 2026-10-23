# Review of the earlier Codex prototype

Two earlier copies existed before this build:

| Location | State |
|---|---|
| `C:\Users\taman\Documents\FairLens` | Newer copy (README says "permanent location"); 373 lines of Python |
| `C:\Users\taman\AppData\Local\Temp\FairLens` | Older subset (no `captioning.py`), plus a `pilot_results.json` |

Both were **left untouched** and an archival copy of each was made at
`D:\FairLens_archive\codex_documents_FairLens` and `D:\FairLens_archive\codex_temp_FairLens`
(only the unreadable `.pytest_cache` folder of the Temp copy was skipped).

## What it contained

- `core.py`: four regex patterns, a count-based "bias score", hard-coded rule rewrites, a token-Jaccard
  "preservation" check and a threshold gate.
- `ingest.py`: TXT/CSV/PDF/DOCX extraction with a 10 MiB limit.
- `captioning.py`: BLIP loader whose weights were never downloaded (inference unverified, per its README).
- `app.py`: plain Streamlit page; no ML detectors, no LLM rewriters, no datasets, no benchmark runs.

## What was reused in `D:\FairLens`

| Reused idea | Where | Change |
|---|---|---|
| File-size-limited TXT/CSV/PDF/DOCX extraction | `src/fairlens/ingestion/loaders.py` | Rewritten: image validation, sentence segmentation, page tags, typed errors |
| Transparent pattern + explanation tuple | `src/fairlens/detection/rule_based.py` | Expanded into a documented lexicon with categories, severities and calibrated scores |
| Honest "unavailable" result instead of substituted output | throughout | Kept as a design principle |

Nothing else (code, numbers or results) was carried over. Its `pilot_results.json` came from authored
cases scored by regex counts and is **not** used anywhere in this project.
