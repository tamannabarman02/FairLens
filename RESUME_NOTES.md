# Resume notes (paused 2026-11-15, 19:25) - COMPLETED 22:40, see docs/PROGRESS.md

Controlled-set leakage fix and re-run on this Windows laptop (CPU). Full details: `docs/PROGRESS.md`,
section "Paused 2026-11-15".

## Done
- Split fixed in `scripts/build_mitigation_sets.py` (one val/test assignment per sentence, seed 42, stratified,
  leakage check). `gate_controlled.parquet` re-split; texts unchanged. Test: 204 inputs (69 neutral), 1,218 candidates.
  Old file: `outputs/backup_before_resplit/`.
- Reuse of stored Mac signals/rewrites added and verified exact (0 differences) for the gate run and Table 7.
- Manuscript builder ready for v3 (Parts 2–3 done; gate numbers read from run JSON).
- Validation LLM-judge calls: **804 of 2,019 saved** in `outputs/cache/scores.sqlite`
  (variant `plain` complete, `checker_system` partly). Every call is committed as it finishes; a restart skips them.

## Left, in order
1. Validation judge (resume, below) → then run it once more (all cached) so a tracked manifest records the environment
2. Prompt choice on validation
3. Test judge
4. Gate run on the new split
5. Table 7 re-decision
6. RESULTS.md and Fig. 8
7. Show the old → new headline table to the user **before** changing any conclusions
8. Build `paper/FairLens_manuscript_anonymised_v3.docx` (final) with the old → new list

## Exact command to resume (Git Bash, from the project folder)
```bash
cd /d/FairLens_Full/FairLens
export HF_HOME="D:\\FairLens_cache\\hf" HF_HUB_OFFLINE=1 PYTHONUTF8=1 HF_HUB_DISABLE_PROGRESS_BARS=1
.venv/Scripts/python.exe -u scripts/prefill_llm_judge_cache.py --split val --variants plain checker_system question_first > outputs/logs/prefill_judge_val_resume.log 2>&1
```
About 1,215 calls remain (~50 min at ~2.5 s/call). Commands for steps 2–8 are in `docs/PROGRESS.md`.

## Environment reminders
- Windows Application Control blocks pandas, scikit-learn and spaCy in `.venv`: run analysis steps with
  `C:\Users\taman\Anaconda3\python.exe scripts/no_torch.py <script>`; run Qwen in `.venv`.
- Models cache: `D:\FairLens_cache\hf` (drive C: is almost full).
- Do not upload `_v3` until step 8; `_v2` is the latest complete manuscript.
