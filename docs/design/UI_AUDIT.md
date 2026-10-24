# FairLens UI audit (2026-10-24)

How to reproduce: `python scripts/ui_smoke.py --flow` (Windows/macOS/Linux; browsers via `python -m playwright install chromium`).
Screenshots go to `outputs/ui_screenshots/` (git-ignored): `<page>_<theme>_<width>.png`. The **before** set is in
`outputs/ui_screenshots/before/` (taken with the original app, viewport-only), the **after** set at the top level (full page).
`report.json` next to them has the machine-readable result.

## Results

| | Before | After |
|---|---|---|
| Pages × themes × widths | 44 (11 pages) | 48 (12 pages, + FairLens Review) |
| Streamlit exceptions / error alerts | 0 | 0 |
| Console errors | 2 × "404 Not Found" on every page except Home (40 total) | 0 |
| Horizontal overflow at 1366 / 1920 | none | none |
| `--flow` end-to-end checks | real-model flow only, no assertions | 54 / 54 passed (stand-in models, see below) |

## Issues found and status

| # | Page | Problem | Evidence (before) | Status |
|---|---|---|---|---|
| 1 | Top bar (all) | Device chip always said "CPU", hard-coded in `app.py`; "FL" avatar meant nothing | `home_light_1366.png` | Fixed: `fairlens.hardware.device_label()` (e.g. "Apple GPU (MPS) · standard"), CPU icon |
| 2 | All except Home | 2 console 404s per page: `/<page>/_stcore/health` and `/<page>/_stcore/host-config`. The Streamlit frontend probes for a base path when a deep URL like `/text` is opened directly, then falls back to `/` | `before/report.json` | Fixed in the smoke test: it now navigates the way a user does (load `/` once, click sidebar links), so 0 console errors. Opening a bookmarked deep link still makes the same harmless probe; this is Streamlit behaviour, not an app bug |
| 3 | Smoke script | `PLAYWRIGHT_BROWSERS_PATH` forced to `D:\FairLens_cache\playwright`; screenshot paths saved with backslashes; text IO without `encoding` | `scripts/ui_smoke.py` | Fixed: default Playwright location, POSIX paths, utf-8 |
| 4 | Smoke script | "full_page" screenshots only captured 900 px, because Streamlit scrolls inside its own container | all `before/*.png` | Fixed: viewport grows to content height before capture |
| 5 | Text Analysis (dark) | Example pills white-on-white (invisible); textarea placeholder invisible | `before/text_dark_1366.png` | Fixed: CSS for 1.66's `button[data-variant=pills]`, placeholder colour |
| 6 | Sidebar (dark) | Lite-mode toggle track invisible (only the knob showed) | `before/text_dark_1366.png` | Fixed |
| 7 | Image Analysis (dark) | Upload button white-on-white | `before/image_dark_1366.png` | Fixed (`stBaseButton-secondary`) |
| 8 | Experiments (dark) | Filter multiselect rendered as a white bar; Config JSON viewer light theme, unreadable | `before/experiments_dark_1920.png` | Fixed: multiselect styled; JSON shown in a themed `<pre>` |
| 9 | Experiments | `runtime_s` and `git` showed `nan`; two stopped DeBERTa runs listed as "running" like live runs | `before/experiments_dark_1920.png` | Fixed: "-" for missing; status label column (see #14) |
| 10 | Experiments | `load_manifest` used the absolute Windows path stored in `runs.sqlite` (crash risk on macOS/Linux) | code | Fixed: `results.run_dir` / `manifest` fall back to `outputs/runs/<run_id>` (infra agent also fixed `tracking.list_runs`) |
| 11 | Theme toggle | Tooltip "Switch to light mode" stayed visible in every dark screenshot | `before/*_dark_*.png` | Fixed in the smoke script (moves the mouse away); tooltip contrast themed |
| 12 | Text Analysis | "Use extracted text" set `text_input` after the text area had rendered (Streamlit raises "cannot be modified after the widget is instantiated") | code | Fixed: `on_click` callback |
| 13 | Text / Image upload | Labels hard-coded formats ("max 10 MB"); no limitations stated; CSV column not selectable; ingestion warnings dropped | `before/image_dark_1366.png` | Fixed: one shared uploader built from `TEXT_EXTS`, `IMAGE_EXTS`, `MAX_TEXT_BYTES`, `MAX_IMAGE_BYTES`, `MAX_CHARS`; states accepted formats, size and known limitations (no OCR for scanned PDFs, password PDFs, legacy .doc, first GIF frame); CSV column picker from `csv_columns`; every `doc.warnings` shown; 200-sentence scoring cap announced |
| 14 | Benchmarks, Model Comparison, Error Analysis, Experiments | No result status; no run provenance next to results; an empty tab showed only a button | `before/benchmarks_light_1366.png` | Fixed: every result block starts with one label (Completed run / Pending / Failed run / Illustrative example) and its run ID, models@revision, dataset version, key config, git commit and finish time. A "running" manifest with no live process (found by command line) is shown as **Failed run**, with a note saying it was stopped |
| 15 | Research tables | Metric names without definitions | all research pages | Fixed: header tooltips from `components.METRIC_HELP` (Macro-F1, ECE, ROC-AUC, unsafe accept, …); section tooltips on charts; CIs kept where the summary has them |
| 16 | Use Cases | "Open in Text Analysis" button actually opened the Reliability Gate; uneven card heights; raw `ExceptionName: msg` errors | `before/use-cases_light_1366.png` | Fixed: label says "Open in Reliability Gate"; equal-height cards with a Pending / Illustrative example label; plain-language error |
| 17 | All analysis pages | Errors shown as `TypeName: message` | code | Fixed: `common.show_error` gives a plain sentence plus a next step (Lite mode, verify_models.py); details in an expander |
| 18 | Design system | ~40 one-off inline styles; font sizes 11 to 50 px with no scale; `##### ` markdown headings mixed with HTML headings | page sources | Fixed: type scale `--fs-xs…--fs-hero` and spacing scale `--sp-1…--sp-7` in `styles.css`; `C.section`, `C.caption`, `C.gap`, `.fl-row`, `.fl-card.compact`; 0 `style=` attributes left in pages (only dynamic bar widths in components) |
| 19 | Home | No single "how to start"; entries for only Text/Image | `before/home_light_1366.png` | Fixed: 3-step "how to start" in the hero; 7 entry cards (Review, Text, Image, Mitigation, Gate, Benchmarks, Use Cases), each verified to navigate; honest research status labels |
| 20 | Sidebar | Flat list mixing workflow and research pages | `before/*.png` | Fixed: Workflow / Research / More groups; Review second; Lite mode explains what it changes |
| 21 | Downloads | Each download click re-ran the app; the third export of a row was cancelled in the browser | flow run | Fixed: `on_click="ignore"` on every download button |
| 22 | Page icon | `page_icon="assets/logo.svg"` relative to the working directory | `app.py` | Fixed: absolute path from `app.py` |

Not fixed (outside UI scope or by design): Streamlit's own deep-link probe (#2) still logs 404s if a user opens
`/benchmarks` directly. Use-case result cards show "Illustrative example" because they run synthetic demo inputs.

## Buttons and navigation inventory

Verified = exercised by `scripts/ui_smoke.py --flow` (F) or the AppTest suite `tests/test_ui_pages.py` (T); C = checked by reading the code.

| Page | Control | Should do | Does it |
|---|---|---|---|
| Top bar | Theme toggle | Switch light/dark | Yes (F, every screenshot) |
| Top bar | Device chip | Show compute device + profile | Yes, `device_label()` |
| Sidebar | 12 page links | Open the page | Yes (F: 12/12 crumbs match) |
| Sidebar | Lite mode toggle | Smaller models for live analysis; says what changes | Yes (C) |
| Home | Start FairLens Review / Try Text Analysis | Navigate | Yes (C, same `go()` as cards) |
| Home | 7 entry-card buttons | Navigate to Review, Text, Image, Mitigation, Gate, Benchmarks, Use Cases | Yes (F: 7/7) |
| Review 1 | Use the sample job description | Fill the text area | Yes (C) |
| Review 1 | Paste text / Upload file tabs, uploader, Text column picker, Use this document | Load text from TXT/PDF/DOCX/CSV | Yes (F: real PDF + DOCX; scanned PDF refused with plain message) |
| Review 1 | Run review | Detection + rewrites + Gate, then step 2 | Yes (F, T) |
| Review 2–6 | Back / Next | Move between steps; text kept on Back | Yes (F, T) |
| Review 3 | Version to review (segmented) | Pick the rewrite used in steps 4–5 | Yes (F) |
| Review 6 | PDF report / JSON / HTML report | Download the review | Yes (F: valid `%PDF`, JSON with disclaimer, HTML) |
| Review 6 | Start a new review | Reset to step 1 | Yes (F) |
| Text Analysis | Example pills, text area, Enter/Upload tabs, Use extracted text | Fill the input | Yes (F: TXT + CSV upload) |
| Text Analysis | Analyze / Analyze & mitigate | Run; the latter goes to Mitigation | Yes (F) |
| Text Analysis | JSON / HTML report / PDF report | Download | Yes (F: all three) |
| Text Analysis | Continue to Mitigation | Navigate | Yes (C) |
| Mitigation | Generate rewrites (empty state) | Run mitigation | Yes (C) |
| Mitigation | Use this version (per card) | Select and open the Gate | Yes (C) |
| Mitigation | Diff segmented control, Prompts expander | Show diff / prompts | Yes (C) |
| Gate | Selected rewrite, tabs, Compare with lowest-bias toggle | Inspect the decision | Yes (F: banner + 7-item preservation grid) |
| Gate | Go to Mitigation / Go to Text Analysis (empty states) | Navigate | Yes (C) |
| Image Analysis | Upload / COCO sample radio, uploader, Generate captions & analyse | Caption + analyse | Yes (F: JPG, PNG, GIF) |
| Benchmarks | 4 tabs; Run … benchmark buttons | Start background jobs | Yes (C; not started during audit to keep RAM low) |
| Model Comparison / Error Analysis | Run … buttons, selectors, radios | Start job / filter | Yes (C, T render) |
| Experiments | Filter kinds, Inspect run, Reproduce this run | Show provenance / re-launch | Yes (F: status labels, C) |
| Experiments | Launch tab Run / Cancel / Refresh progress | Manage background jobs | Yes (C) |
| Experiments | Exports PDF / HTML / JSON / CSV | Export a completed run | Hidden until a completed benchmark run exists (correct empty state) |
| Use Cases | Run / Re-run, Open in Reliability Gate, Open Model Comparison | Run the demo / navigate | Yes (C; not run, they load real models) |
| About | 6 tabs | Show docs | Yes (T render) |

## Smoke-test mode

`--flow` starts the app with `FAIRLENS_UI_STUBS=1` (`src/fairlens/ui/smoke_stubs.py`). The real rule-based detector,
rule rewriter, fact checker, fluency check and Reliability Gate run. Only the neural parts are replaced: word overlap
for similarity/entailment, fixed captions and a constant CLIP score. Every analysis page shows a "Smoke-test mode"
banner while it is on, and the screenshot pass runs without it. Nothing from this mode is a research result.

## Guard against invented numbers

`tests/test_ui_no_fabricated_metrics.py` fails if a page passes a literal value to `metric_tile`/`st.metric`, or a page
source contains a metric-like literal ("0.87", "93%"; "95% CI" labels excepted). It renders every research page against
an empty `outputs/` and requires no tiles, no metric-like numbers and a Pending label. It renders Benchmarks with one
completed run and requires the tile to show that run's value and run ID. It also asserts the `results` helpers open
only files under `outputs/` (plus the dataset manifest). A mutation check, injecting `metric_tile("Gate accuracy",
"0.91")` into Home, makes two of these tests fail.
