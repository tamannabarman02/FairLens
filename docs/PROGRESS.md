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

## Journal manuscript (2026-11-15)

Target: **AI and Ethics** (Springer Nature, double-anonymous review). Built on Windows from the stored runs above.

| File | What |
|---|---|
| `paper/FairLens_manuscript_anonymised.docx` | Anonymised manuscript, title "FairLens: Automated Framework for Bias Detection, Mitigation and Fairness Evaluation in Large Language and Multimodal AI Systems" (19 pages, abstract 247 words, 9 tables, 7 figures incl. high-level design, Gate flow and 2 app screenshots, 37 numbered refs, declarations; no line numbers, justified, single spacing, layout modelled on a friend's Springer submission from another team). Upload as the main manuscript. |
| `paper/screenshots/` | Dashboard screenshots (FairLens Review, Benchmarks) taken from the Second Review document; copied to Fig3/Fig4. |
| `paper/FairLens_title_page.docx` | Title page with author names, affiliations, acknowledgements, contributions. Yellow placeholders must be filled. Upload as "Title page". |
| `paper/figures/Fig1-7` | 600 dpi figures (Fig3-4 are screenshots), drawn from run JSONs by `scripts/make_paper_figures.py` (needs matplotlib; on this laptop the venv's matplotlib is blocked by Windows Application Control, so use Anaconda Python). |
| `paper/build_manuscript.py`, `paper/build_title_page.py` | Regenerate the .docx files (venv Python). Tables are filled from run JSONs. |

Open decision P5 resolved for the paper: harmful choice is reported split into "kept biased original" and "damaging rewrite" (tuned Gate: 51 vs 1 of 204).
Open decision 1 (BART-MNLI template) is disclosed as a limitation, not re-run.
Before submitting: fill the title page, confirm funding / competing interests, check reference details (DEPEN, LGSA, ParaDeHate author lists), remove personal info in Word (File > Info > Inspect Document).
- 2026-11-15: added Fig. 4 (two frames from the demo video, ~93 s Mitigation and ~102 s Reliability Gate, job-advert example only; the resume and personal-photo scenes in the video were deliberately NOT used because they identify people). Figures renumbered: Fig5 Benchmarks screenshot, Fig6-8 result charts. Manuscript is now 20 pages, 8 figures, 9 tables. `paper/figures/Fig5.eps` is a stale leftover (old detection chart) and can be deleted; it is not used.

## Paused 2026-11-15 (controlled-set leakage fix, re-run on Windows CPU) - resume here
Done:
- Split fix in scripts/build_mitigation_sets.py (sentence-level, seed 42, stratified; leakage check). gate_controlled.parquet resplit with `--resplit-only` (texts unchanged; old file in outputs/backup_before_resplit/). Test: 204 inputs (69 neutral), 1,218 candidates.
- Reuse paths added and verified exact (0 differences vs Mac runs): run_ablations.py --signals-from (check run 20261115-184829-...-b32c19), run_benchmarks.py --task mitigation --candidates-from/--gate-run (check run 20261115-184949-...-66afc9). These two are verification runs on the OLD split, not results.
- Manifests now record environment versions; scripts/no_torch.py hides torch for analysis runs in Anaconda base.
- Manuscript builder (paper/build_manuscript.py -> _v3) has Parts 2-3 done and gate numbers parameterised; 7.4 text needs ENV/JENV + new run IDs.
Running / next (Windows: venv blocks pandas/sklearn/spaCy; use Anaconda base via scripts/no_torch.py for analysis, .venv for Qwen; HF_HOME=D:\FairLens_cache\hf, HF_HUB_OFFLINE=1):
1. .venv: scripts/prefill_llm_judge_cache.py --split val --variants plain checker_system question_first   (answers cached as they go; re-running resumes; log outputs/logs/prefill_judge_val.log)
2. Re-run step 1 once more (all cached) so a tracked manifest records the environment.
3. base: scripts/no_torch.py scripts/validate_llm_judge.py  -> configs/llm_judge.json (check prompts log: 0 uncached rows)
4. .venv: prefill --split test --variants <chosen>
5. base: no_torch.py run_ablations.py --signals-from 20261024-040545-benchmark_gate-7565de
6. base: no_torch.py run_benchmarks.py --task mitigation --candidates-from 20261024-040853-benchmark_mitigation-000f03 --gate-run <step 5 run>
7. generate_results_md.py; Fig. 8; show the user the old->new headline table BEFORE changing conclusions; then build _v3.
- 2026-11-15 22:40 DONE: re-run complete. Runs: llm_judge_prefill f37323 (val) / 30cb0c (test), llm_judge_validation 8ad9d4 (checker_system), benchmark_gate ea9824 (signals from 7565de), benchmark_mitigation 495a89 (candidates from 000f03). docs/RESULTS.md regenerated (Mac version in outputs/backup_before_resplit/RESULTS_mac.md). Manuscript: paper/FairLens_manuscript_anonymised_v3_final.docx.
- Open: Fig. 5 (Benchmarks screenshot) still shows the old leaky headline tiles (5% unsafe-accept, 25% / 98% harmful); retake from the app once it reads the new runs.
- 2026-11-15 23:00: Δ unsafe-accept in run_ablations.py now uses the gold != ACCEPT denominator (same as the column). Re-run: benchmark_gate 794a76 (identical to ea9824 except 4 Δ unsafe-accept entries). Manuscript, Fig. 8 and RESULTS.md now cite 794a76. Table 7 run 495a89 used thresholds 0.60/forward 0.70/0.05 (from ea9824, identical tuned rules): 11 of 1,592 candidate verdicts and 33 reason strings changed, 0 input-level decisions.
- configs/models.yaml: new `gate_app` block pins the app thresholds (0.60 / forward 0.70 / 0.05 / std 1.0, source run 794a76); src/fairlens/ui/state.py tuned_gate_overrides() uses it first. `gate` defaults unchanged. Test: tests/test_gate.py::test_app_uses_pinned_gate_thresholds_without_stored_runs.

## College-format research paper (2026-11-17)
- `paper/FairLens_Research_Paper.docx`: non-anonymised paper in the same format as the reference college paper
  (`D:\Downloads\Context-Aware Multilingual Legal Assistant for Indian Law – Research Paper (1).docx`): US Letter,
  Times New Roman, Roman-numeral sections, IEEE [n] references; authors Muthurani N H, Suryansh Shahi, Sushma B T,
  Tamanna Barman, Trisha L Salian. 21 pages, 9 tables, 8 figures, 37 refs.
- Built by `paper/build_research_paper.py` (Anaconda Python). Tables are copied cell by cell from
  `paper/FairLens_manuscript_anonymised_final.docx`, so numbers match the journal manuscript.
- Still to fill (yellow): college name, city, guide name/designation (affiliation line and Acknowledgment).

## PAUSED 2026-11-17 (resume here)
Done:
- `paper/FairLens_Research_Paper.docx` rebuilt; Fig. 5 is now the full Benchmarks page (`paper/figures/Fig5_benchmarks_full.png`,
  current post-re-split run files, teammate's /Users/... path shortened to checkpoints/...).
- `report/main.tex` + `report/figures/`: full major-project report in the friend's format (MAJOR_PROJECT_final_v3.pdf).
  Red \FILL{} = USNs, college, guide/HoD/principal, dates, Appendix B status, C/D, PSO1, F. Not compiled (no LaTeX here; use Overleaf).
Pending:
1. Live demo screenshots (optional). Script: `scripts/paper_demo_screenshots.py` (CDP-driven headless Chrome against the app on
   port 8599; Anaconda python + websocket-client). Start app with --server.port 8599 --server.headless true, then
   run the script with an output dir and "demo", env DEMO_TEXT = the waitress advert. Model loading on CPU takes 10+ min;
   close other app copies first. Placeholders waiting: `#%%DEMO%%` in paper/build_research_paper.py,
   `%%DEMO_RESULTS%%` and `%%APPENDIX_SNAPSHOTS%%` in report/main.tex.
2. PowerPoint from `D:\Downloads\PP2-Presentation-Format (1) (1).pptx` (15-slide BMSCE template). Plan: fill via PowerPoint COM;
   title/outline/SDGs/collaboration/problem/objectives/HLD/detailed design/implementation/results/snapshots/conclusion/future/
   paper/references/thank you; drop the "Note" slide. Template slide 3 forbids AI-generated content in PPT/report.

## RESUMED + DONE 2026-11-17
- Team corrected: guide Prof. Muthurani N H (Assistant Professor); students Suryansh Shahi 1BM23CS349, Sushma B T
  1WA23CS002, Tamanna Barman 1WA23CS006, Trisha L Salian 1WA23CS022. College details as in the friend's report
  (BMSCE, Bengaluru-560019; HoD Dr. Kavitha Sooda; Principal Dr. Bheemsha Arya).
- Live demo run (illustrative, not a benchmark): waitress advert -> overall bias 0.72; rule rewrite ACCEPT
  (0.561 -> 0.300, facts 1.00, sim 0.77, entail 0.98); FLAN-T5 RETRY (copy); Qwen REJECT (added "21-30 years old",
  changed 6 days -> Mon-Fri, facts 0.67). Screens: paper/figures/demo_*.png, report/figures/demo_*.png.
- Paper: demo added as Figs 6-7 (results figs renumbered 8-10). Report: Section 5.7 Live Demonstration + Appendix A snapshots.
- PPT: paper/FairLens_PP2_Presentation.pptx (25 slides) built by paper/build_presentation.py (Anaconda + PowerPoint COM).
  Red [FILL] left: collaboration details, paper status/ID.

## SN Computer Science manuscript (2026-11-20)
- Target switched to **SN Computer Science** (Springer, single-blind). `paper/FairLens_SNCS_manuscript.docx` (+ `.pdf`, 19 pages)
  built by `paper/build_sncs_manuscript.py` (Anaconda Python) from `paper/FairLens_manuscript_anonymised_final.docx`
  (left unchanged; it is the v3 build with the post-re-split numbers, which match the research paper and report).
- Changes: author block + affiliation + corresponding author under the title; Statements and Declarations rewritten
  (no anonymisation wording; author contributions; "Use of generative AI" points to Sect. 7.5); Author property set; no tracked changes.
- 2026-11-20 (later): applied SN Computer Science guidelines: structured abstract (Purpose / Methods / Results / Conclusion,
  249 words incl. labels), "Consent to publish" heading. Fig. 5 checked: it is the cropped Benchmarks screenshot (charts only,
  no headline tiles), consistent with Table 7, so it was kept. Captions, numbered [n] references, Times font, no field codes
  other than the page number were already compliant. PDF re-exported (19 pages).
- 2026-11-20 (later): SNCS manuscript gained Sect. 6.3 "Live demonstration" (input -> output, waitress advert) with
  Fig. 6 = figures/demo_detection.png and Fig. 7 = figures/demo_mitigation_core.png (same text as the research paper).
  Old Figs 6-8 renumbered to 8-10; Testing is now 6.4. Anonymised manuscript untouched.
- 2026-11-20 (fixes): build_sncs_manuscript.py bug fixed (renumbering rewrote picture runs and deleted Figs 8-10).
  Old Fig. 4 (salesman example) removed; figures now 1-9 (demo = Figs 5-6). Added the score-definition and
  forward/backward entailment sentences to Sect. 6.3. check_figures() asserts every caption has its image.
  Live demo re-run as tracked run outputs/runs/20261120-010638-live_demo-052b5f (scripts/run_live_demo.py, .venv, CPU,
  spaCy en_core_web_sm 3.8.0): reproduces every value in the demo screenshots exactly. PDF: 20 pages.
- 2026-11-20 (Comp-SIF 2027): new 6-page IEEE conference paper, double-blind, in `paper/compsif/`
  (`FairLens_CompSIF_6page.docx` / `.pdf`, `abstract.txt`, `fig1_gate_flow.png`). Built by
  `paper/build_compsif_paper.py` (Anaconda Python); figure by `paper/build_compsif_figure.py`; Fig. 2 reuses
  `paper/figures/Fig7.png`. Every number comes from `FairLens_SNCS_manuscript.docx` (post re-split results; a script
  confirmed all numeric tokens and table rows match). IEEE Word template could not be downloaded (site blocks
  scripts), so its layout is reproduced (Letter, 2x3.5in columns, Times 10pt). PDF exported with Word, then
  `python build_compsif_paper.py --scrub-pdf <pdf>` strips dates/timezone/XMP. Existing manuscripts untouched.

### 2026-11-20 — Comp-SIF paper rebuilt in official IEEE Word template
- `python paper/build_compsif_paper.py --ieee` pours the same content into `paper/compsif/conference-template-a4.docx`
  (official IEEE A4 template; Strict OOXML, converted via Word COM) using its styles and auto-numbering.
- Output: `paper/compsif/FairLens_CompSIF_6page_IEEE.docx` / `.pdf` — 6 pages (A4), old letter-size files kept.
- Template placeholder text, author boxes, copyright footer and app.xml "Company: IEEE" removed; PDF scrubbed with
  `--scrub-pdf`. Text identical to the previous version apart from template-generated numbering.
