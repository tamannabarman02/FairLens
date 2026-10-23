<!-- Text extracted from the owner's V4 specification artifact (https://claude.ai/artifact/TrwXxxfyX6gaoS2HFSL4bF), fetched 2026-10-24. Reference copy; the artifact is authoritative. -->


FairLens Solution Plan
 FairLens · final-year project · working specification v4
 Completing FairLens with evidence behind every claim
 This is the working implementation and research specification. It starts with an audit of what the repository actually contains, then runs the existing experiments in a fixed priority order. Everything must run on Windows, macOS and Linux, because the team uses both Windows laptops and a Mac. The Reliability Gate is treated as a proposed contribution until prior work and results support a stronger claim. A UI plan for a demo-ready app and a product concept follow, both kept secondary to the evidence the paper needs.
 Revised 24 Oct 2026Based on Revised Plan V2, the implementation instructions, the repo docs and the code in src/fairlens/No results exist yet; this page contains none
 Working rules
 0 · Audit
 1 · Cross-platform setup
 Detection
 Mitigation
 Gate
 Analysis
 Images and use cases
 Results and paper
 UI and demo
 Product idea
 Done means
 First week
 What changed
 Objective
 Complete, don't rebuild
 Keep the existing architecture and Git history. Spend the effort on verifying the current state, running the experiments properly, comparing methods fairly, and reporting what the runs show.
 Completion criterion
 FairLens is complete only when
 the required workflows run, verified end to end
 the experiments produce reproducible outputs
 the baselines and the Gate have been compared fairly
 the limitations are documented
 the paper's claims match the evidence
 Apply to every stage
 Working rules
 Audit before codingVerify each stated status item against the repository before building on it.
 Preserve what existsKeep the architecture, module layout and Git history. Change code through normal commits.
 Runs on every teammate's laptopNo feature may depend on one operating system. Windows, macOS and Linux all run the same code, tests and commands.
 UI is separate from protocolUI changes never alter data, metrics, thresholds or results. They go in separate commits from research changes.
 No invented resultsEvery figure comes from a stored run in outputs/ with its run ID.
 No silent protocol changesAny change to data, metrics, thresholds or splits is written to a protocol log, with its reason, before the run that uses it.
 Test data stays sealedThresholds, prompts and judge settings are chosen on validation only. Validation and test results are reported separately.
 Proposed, not provenThe Gate is a proposed contribution. No novelty, safety or superiority claim without results and a prior-work check.
 When an experiment fails or is infeasible
 Report the failure with its evidence→
 Propose a justified alternative→
 Get agreement→
 Log it in the protocol log→
 Then change scope
 Record for every run: the change made, the configuration, seeds, model revisions, the run ID and the actual outputs. Optional features never take time away from evidence the paper needs.
 0Audit the stated status
Before any code
 The project docs make specific claims. Each one is checked against the repository and the result goes into docs/PROGRESS.md. Nothing below has been verified yet.
 Stated in the docs | How to verify | Result | 
 44 fast tests pass | Run pytest -m "not slow" on a Windows laptop and on the Mac | Pending | 
 Installs from scratch on Windows and macOS/Linux | Run install.ps1 on a clean Windows laptop and install.sh on the Mac; both exist today but only Windows has been used | Pending | 
 16 models verified, revisions pinned | Compare verification.json with model_lock.json; re-run one inference per model | Pending | 
 15,136 rows, no leaking groups | Rebuild the datasets and compare hashes and counts with dataset_manifest.json | Pending | 
 DistilBERT trained, val macro-F1 0.761 | Weights are gitignored and absent from this copy; retrain and compare with fairlens_training.json | Pending | 
 DeBERTa not trained | Confirm no checkpoint; two stopped runs are marked "running" in outputs/runs/ | Pending | 
 No benchmark results exist | Query runs.sqlite and list manifests by task | Pending | 
 Uploads work for TXT, PDF, DOCX, CSV and images | Uploaders exist on two pages; this proves nothing yet. Verify end to end in step 1.5 | Pending | 
 Git history is intact | The downloaded folder has no .git; clone the private GitHub repo and work there | Pending | 
 1Cross-platform setup and reproducibility
Start here
 One codebase, one set of commands, any teammate's laptop. The code picks the best device it finds and falls back to CPU, so nobody needs a particular machine to contribute.
 Teammate's machine | Device used | What it can run | 
 Windows laptop with an NVIDIA GPU | CUDA | Everything, including training and the full benchmarks | 
 Windows laptop, no GPU (8–16 GB RAM) | CPU | Tests, UI, demo in Lite mode, small runs; heavy runs only after profiling (1.4) | 
 Mac with Apple silicon | MPS, with CPU fallback for unsupported operations | Training and benchmarks, subject to the MPS/CPU match check (1.2) | 
 Linux, or a free cloud GPU as a fallback | CUDA or CPU | Same as above; used only if no laptop can finish a run | 
 1.1
 Install and test on Windows and macOS
 Clone the repo with history. Install with install.ps1 on Windows and install.sh on macOS/Linux; keep the two scripts doing the same steps with the same pinned Python 3.11 and requirements. Run the fast tests on both.
 Done whenThe same tests pass on one Windows laptop and the Mac, recorded in the audit table.
 1.2
 Automatic device choice: CUDA, then MPS, then CPU
 hardware.py only checks for CUDA today. Make it try CUDA, then Apple MPS, then CPU, with a FAIRLENS_DEVICE environment variable to force one. Set PYTORCH_ENABLE_MPS_FALLBACK=1 inside the code, not in a shell script, so it works the same everywhere.
 Done whenOne detector and one rewriter give matching outputs on CPU, CUDA (if a teammate has one) and MPS, within a stated tolerance.
 1.2b
 Remove operating-system assumptions
 Replace the D:/FairLens_cache/hf default in models.yaml with a per-user cache folder (still overridable with FAIRLENS_CACHE). Use pathlib for every path, so manifests never store \ or /-specific paths; the UI smoke report currently stores Windows-style paths. Open every text file with encoding="utf-8", so Windows no longer needs PYTHONUTF8=1. Add a .gitattributes to stop line-ending churn between Windows and Mac.
 Done whenA grep finds no drive letters, hard-coded separators or encoding-less open() calls, and the tests pass on both systems.
 1.2c
 Automatic checks on all three systems
 A GitHub Actions workflow runs the fast tests on windows-latest, macos-latest and ubuntu-latest on every push, so nobody breaks a teammate's machine without noticing.
 Done whenThe workflow is green on all three.
 1.2d
 Share runs without merge conflicts
 Each run already writes a self-contained folder with its manifest. Treat runs.sqlite as an index rebuilt from those folders, so teammates exchange run folders instead of fighting over one database file. The final reported runs for each experiment come from one named device; cross-device differences are reported, not mixed.
 Done whenA run made on Windows appears correctly in the Mac's dashboard after copying its folder, and the reverse.
 1.3
 Checkpoints and data manifests
 Rebuild datasets and confirm their hashes. List which checkpoints must be regenerated (DistilBERT, DeBERTa).
 Done whenHashes match the manifest, or any mismatch is logged and explained.
 1.4
 Runtime and memory profiling before full runs
 Time 20 training steps for each fine-tuned model and 100 inference samples for each detector, rewriter and captioner, recording peak memory. If DeBERTa is too slow, report it and propose the fix ladder in order: freeze embeddings, max_length 64 (p95 is 51 tokens), LoRA, then deberta-v3-xsmall.
 Done whenA stored profiling run, and agreed training settings recorded in the methodology and the protocol log.
 1.5
 Verify every upload format end to end
 For each of TXT, PDF, DOCX, CSV, JPG, PNG and GIF: upload a real file through the browser UI and check that the extracted text or image reaches the pipeline and produces output. Include a scanned PDF with no text layer, a DOCX with tables, a multi-column CSV, an oversized image, and files saved on Windows (UTF-16 or Windows-1252 text, CRLF line endings). Having an uploader on the page is not proof that it works.
 Done whenA browser flow test per format passes on Windows and macOS, plus a unit test per loader. Any format that fails is reported with its error.
 1.6
 Complete run manifests and a protocol log
 Make each manifest record model IDs and revisions, device, seeds, dataset versions, the full config, output paths and the run ID. Make the score cache key include the model revision and device. Start docs/PROTOCOL_CHANGES.md.
 Done whenA test fails if any required manifest field is missing.
 Evidence stages, in priority order
 Run the existing experiments
 Each stage needs the one before it. Stages 1–4 are the core evidence for the paper.
 1Detection evaluation
Priority 1
 D.1
 Check that labels are compatible
 BABE has sentence-level biased or neutral labels. StereoSet and CrowS-Pairs are minimal pairs, where the comparison between two sentences carries the meaning. Document how each source maps to the shared label, and keep the pair-preference probes separate from the classification metrics.
 Done whenA label-mapping table in docs/DATASETS.md, reviewed before the run.
 D.2
 Held-out benchmark with per-dataset metrics
 Run DistilBERT, DeBERTa, BART-MNLI zero-shot (template chosen on validation), the Hub RoBERTa reference and the rule baseline on the same held-out test sentences. Report precision, recall, F1, macro-F1, ROC-AUC, ECE and runtime, overall and for each dataset.
 Done whenA detection run with bootstrap 95% CIs, per-dataset results, and the WinoBias probe reported separately.
 2Mitigation and critical-fact preservation
Priority 2
 M.1
 Mitigation benchmark
 Rule rewriter, FLAN-T5 and Qwen2.5 against leaving the text unchanged. Same prompt family, greedy decoding, fixed seed, and a prompt variant chosen on a train-split sample. FLAN-T5's copying is reported as measured.
 Done whenA mitigation run with every prompt and output logged.
 M.2
 Critical-fact preservation
 For every rewrite: semantic similarity, entailment in both directions, contradiction, and fact retention for each fact type (names, numbers, dates, negation, roles, quotes). Check the fact extractor itself on a hand-labelled sample before trusting it.
 Done whenA failure rate per fact type for each rewriter with examples, and the extractor's measured accuracy.
 3Reliability Gate vs all baselines
Priority 3
 G.1
 Add the missing baselines
 baselines.py has lowest-bias, highest-similarity, always-first and random. Add no-change, which keeps the original text, and LLM-as-judge, which asks a model whether the rewrite keeps every fact. Tune the judge's prompt on validation and measure its agreement with gold labels before using it as a baseline.
 Done whenBoth baselines have unit tests, and the judge's validation agreement is stored as a run.
 G.2
 Gate comparison
 Tune thresholds on the validation split of the controlled gold set, freeze them, then evaluate every policy on test. Run "always first" on the real mitigation outputs, where it has meaning.
 Done whenUnsafe-accept, acceptable-choice, deferral and unnecessary-change rates for every policy, with paired bootstrap differences against the Gate.
 4Risk–coverage, red-team, human evaluation, ablations
Priority 4
 A.1
 Risk–coverage analysis
 Sweep each policy on validation and plot unsafe-accept rate against human-review rate. Mark the chosen operating point and report its test result separately.
 Done whenCurves stored as a run and shown on the Benchmarks page.
 A.2
 Red-team corruption tests
 Apply controlled corruptions to rewrites the Gate accepted: change a number, swap a name, swap roles, drop or add a negation. Reuse the perturbation code behind gate_controlled.parquet.
 Done whenA catch rate per corruption type, with CIs across seeds.
 A.3
 Blinded human evaluation
 Two or three annotators label a sample of real rewrites independently, without seeing the model or the verdict. Write and pilot the guideline first. Compare the Gate and the LLM judge against the human labels.
 Done whenLabels, guideline and inter-annotator agreement (κ) are stored, with a Gate vs judge vs human table.
 A.4
 Ablations and threshold sensitivity
 Turn off one signal group at a time and vary each threshold around its chosen value. A signal counts as useful only if its CI excludes zero.
 Done whenAn ablation table and sensitivity plots; signals with no measurable effect reported as negative results.
 A.5
 Calibrated or conformal risk control
 Only if there are enough labelled calibration items and the calibration and test data are exchangeable. Otherwise document why it was not done.
 Conditional
 Done whenAchieved test coverage against a target, or a documented reason for skipping.
 5Image captions and the six use cases
Priority 5
 I.1
 Image-caption evaluation
 BLIP and Florence-2 on the same 80 COCO val2017 images, with measures fixed before the run: CLIP alignment, detail coverage, gendered-term rate, unverifiable-attribute rate, and detector scores on captions. The scope is caption bias only.
 Done whenAn image run with per-captioner CIs.
 I.2
 Six use cases
 Run each use case through the full pipeline and store the output, including the Gate verdict and its evidence.
 Done whenSix stored outputs, shown on the Use Cases page from the run files.
 6Results, error analysis, limitations, paper
Priority 6
 P.1
 Reproducible results generation
 generate_results_md.py builds RESULTS.md from completed runs only. Add a command-line runner and a clean-machine README.
 Done whenThe report regenerates identically, and another person reproduces one run from the README.
 P.2
 Error analysis and limitations
 False positives and negatives, preservation failures, Gate mistakes, and negative results such as FLAN-T5 copying and WNC being unavailable.
 Done whendocs/LIMITATIONS.md and a written error analysis with real examples.
 P.3
 Prior work and the paper
 Survey selective rewriting, fairness mitigation, semantic preservation and selective prediction. Then write related work, methodology, setup, results, limitations and conclusion, with a claims table linking every number to its run ID.
 Done whenA full draft whose claims all appear in the claims table, and any novelty statement cites what it differs from.
 7Presentable, demo-ready UI
Runs alongside stages 1–6
 Polish the existing Streamlit app into a coherent research application without rebuilding it. The building blocks already exist: a design-token plan in docs/design/UI_PLAN.md, one stylesheet in assets/styles.css, and reusable helpers in ui/components.py (cards, metric tiles, stepper, score bars, decision banner, empty states, check grid, diff view). The job is consistency, clear workflows and honest states, not a new look for its own sake.
 UI audit · first findings from reading the code
 Finding | Where | Fix | 
 The top bar always says "CPU", even on a GPU or Mac | app.py, hardware chip | Show the device actually chosen in 1.2 | 
 Every analysis page logged two "404 Not Found" browser errors in the last smoke test | outputs/ui_screenshots/report.json | Find the missing resource and remove the errors | 
 The job-description review is spread across Text Analysis, Mitigation and Reliability Gate | three separate pages | One FairLens Review page that reuses them (U.3) | 
 The smoke test only covered empty states | scripts/ui_smoke.py | Run the --flow test once real runs exist | 
 The full visual audit has not been done yet | all 11 pages | Screenshot every page in both themes at 1366 and 1920 px before changing anything (U.1) | 
 U.1
 Audit the current UI before changing it
 Screenshot all 11 pages in light and dark at 1366 and 1920 px wide on both Windows and macOS. List every button and navigation item with what it should do, and check that it does it.
 Done whenAn audit list in docs/design/ with each issue, its page and its screenshot.
 U.2
 One consistent design system
 Keep the FairLens branding and the tokens from UI_PLAN.md. Apply one type scale, spacing scale, card style, button style, table style and chart template everywhere; replace one-off inline styles with the shared helpers. Keep animation minimal and keep the sidebar order fixed. Check readability on a projector.
 Done whenNo page uses colours or spacing outside the tokens, and the audit issues are closed.
 U.3
 FairLens Review page for job descriptions
 A guided six-step page built from existing components and backend calls: 1 upload or paste, 2 flagged phrases with evidence, 3 original vs rewrites, 4 fact-preservation checks, 5 Gate verdict and reason codes, 6 export PDF or JSON. The stepper shows where the user is.
 Done whenA browser flow test completes all six steps on a real PDF and a real DOCX job post, and the exported files contain the evidence.
 U.4
 Home dashboard with clear entry points
 A short statement of what FairLens does, then one entry card each for FairLens Review, Text Analysis, Image Analysis, Mitigation, Reliability Gate, Benchmarks and Use Cases. Each upload control lists its accepted formats, size limit and known limitations, such as scanned PDFs without a text layer.
 Done whenA new user reaches each workflow in one click from Home.
 U.5
 Every state designed
 Progress while models load and run, plain-language errors that say what to do next, empty states with a way forward, and a clear completion state. The Lite-mode toggle says what it changes.
 Done whenEach workflow has been triggered into its loading, error, empty and success states in a test or a recorded check.
 U.6
 Honest research dashboard
 Benchmarks and Model Comparison show readable charts and tables with metric definitions in tooltips, confidence intervals where they exist, and for each result its run ID, models and revisions, dataset version and key configuration. Every result carries one of four labels:
 Completed runPendingFailed runIllustrative example
 All numbers are read from run files; none are typed into the UI.
 Done whenA test fails if a page renders a metric that is not traceable to a run file.
 U.7
 Verify end to end
 Navigation, every upload format, analysis, Gate decisions and every export button, on Windows and macOS. Run the existing tests and add tests for each changed function.
 Done whenui_smoke.py and --flow pass on both systems, with no browser console errors.
 UI acceptance criterion
 In a live demo the app looks coherent and professional, completes the supported workflows, shows real experiment outputs accurately, and says plainly when a result is unavailable or a run failed.
 Product concept · secondary to the paper
 Product idea
 FairLens Review
 A second check for automatic rewrites: before a "more inclusive" version replaces the original, make sure nothing important changed.
 Inclusive-language tools and LLM rewriters can remove biased wording while also dropping a requirement, flipping a negation or changing who did what. FairLens Review sits between the rewrite and the reader. It shows what was flagged, which rewrite it recommends, what it checked, and when a person should decide instead.
 First user
 Recruiters and HR teams
 They write and review job posts. Biased wording costs applicants; a rewrite that drops "5+ years" or a licence requirement causes real hiring problems.
 Second user
 Teams building AI writing tools
 They need to check that their rewrite feature keeps facts. The Gate's verdict and reason codes can be called as a service.
 Research user
 Fairness researchers
 They can compare detectors and rewriters on one dataset with stored, reproducible runs.
 The job-description workflow
 UploadPDF, DOCX or pasted text
 FlagBiased phrases, with the detectors that agree
 RewriteCandidates from the rule rewriter, FLAN-T5 and Qwen
 CheckRequirements, numbers, names and negations kept
 DecideAccept, retry, reject or send to a person, with reasons
 ExportA review report as PDF or JSON
 How it looks · placeholder example
 InputLooking for a young, aggressive salesman with 5+ years of B2B experience.
 Rewrite ALooking for a motivated sales professional with 5+ years of B2B experience.Reject · fact loss
 Rewrite BLooking for a driven sales professional with 5+ years of B2B experience.Accept
 Written for this page to show the idea. Both rewrites remove the biased words; only one keeps the requirement. Replace with real output from the use-case run.
 MVP, from existing parts
 Ingestion loaders verified in step 1.5
 Detectors, rewriters and the Gate as they are
 The FairLens Review page from U.3, joining the Text, Mitigation and Gate pages
 The existing PDF and JSON export
 What it must not claim
 That any output is unbiased
 Legal or compliance certification
 Production validation
 Better accuracy than people, unless the human evaluation shows it
 Evidence it needs first
 Fact-preservation rates on job-post text (stage 2)
 The Gate's unsafe-accept rate against baselines (stage 3)
 Red-team catch rates and human agreement (stage 4)
 Later, as future work only
 A Gate API for writing tools · a bias check that runs on every chatbot release · an alt-text checker that flags captions claiming things the image cannot show · a browser extension for job boards · an India-context detector pack, if stage 4 work shows a documented gap.
 Completion criterion, made checkable
 What "done" looks like
 Criterion | Evidence | Where | 
 Stated status verified | Audit table filled in with results | Stage 0 | 
 Runs on every teammate's laptop | Install and tests pass on Windows and macOS; CI green on Windows, macOS and Linux | 1.1–1.2d | 
 Required workflows run | Browser flow test per upload format; FairLens Review flow test on PDF and DOCX | 1.5, U.3 | 
 Demo-ready UI | UI audit closed; every state designed; result labels on every number; no console errors | U.1–U.7 | 
 Experiments are reproducible | Complete manifests, protocol log, CLI runner, one run reproduced on a clean machine | 1.6, P.1 | 
 Fair comparison | Compatible labels; same data and settings for all methods; validation-only tuning; separate test reporting | D.1–G.2 | 
 Gate evaluated beyond synthetic data | Risk–coverage curves, red-team catch rates, human agreement, LLM-judge comparison | A.1–A.3 | 
 Limitations documented | LIMITATIONS.md with negative results and failures | P.2 | 
 Claims match evidence | Claims table; prior-work review; the Gate still called "proposed" unless results support more | P.3 | 
 Suggested order
 The first week
 Day 1Audit and cross-platform setup
Clone with history on one Windows laptop and the Mac, install, run the tests, fill in the audit table. Add the CUDA → MPS → CPU device choice and remove OS assumptions (stage 0, 1.1–1.2b).
# Windows (PowerShell)
powershell -ExecutionPolicy Bypass -File install.ps1
.venv\Scripts\python -m pytest -q -m "not slow"
# macOS / Linux
bash install.sh
.venv/bin/python -m pytest -q -m "not slow"
 Day 2CI, data, uploads, profiling, UI audit
Add the three-system CI workflow, check dataset hashes, verify every upload format end to end, profile runtime and memory on each machine, start the protocol log, and screenshot the current UI (1.2c–1.6, U.1).
 Day 3Label mapping, training, detection
Agree the label mapping, retrain DistilBERT, train DeBERTa, run detection (D.1–D.2).
python scripts/train_detectors.py --model bert
python scripts/train_detectors.py --model deberta
python scripts/run_benchmarks.py --task detection
 Day 4Mitigation and preservation
Select the prompt on the train-split sample, run mitigation, check the fact extractor (M.1–M.2).
python scripts/select_prompts.py
python scripts/run_benchmarks.py --task mitigation
 Day 5Baselines and the Gate
Add no-change and the validated LLM judge, then tune on validation and evaluate on test (G.1–G.2).
python scripts/run_ablations.py
 Revision log
 What changed
 Version 4 · this update
 Setup is cross-platform: Windows, macOS and Linux, not Mac-only
 Device order CUDA → MPS → CPU with an override, plus a machine-to-device table
 Removal of OS assumptions: drive letters, path separators, file encodings, line endings
 CI on all three systems; run sharing without database conflicts
 New stage 7: UI audit, design system, FairLens Review page, designed states, honest result labels
 Upload and UI checks run on Windows and macOS
 Version 3
 Stage 0 audit table and working rules
 Protocol log and failure procedure
 Label-compatibility and fact-extractor checks
 Six-priority experiment order
 Product concept: FairLens Review
 Gate described as a proposed contribution
 Version 2
 Plan restructured into phases from the revised priorities
 No-change and validated LLM-judge baselines added
 Conformal risk control and India-context work made conditional
 FLAN-T5 infill and several product extras dropped to future work
 Based on Revised Plan V2, the implementation instructions, HANDOFF.md, docs/PROGRESS.md, docs/METHODOLOGY.md, the master build brief, and the code under src/fairlens/. This plan contains no benchmark results; none have been run yet.
