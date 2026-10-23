# FairLens UI/UX implementation plan

Status: the reference mockup (`FairLens_UI_Reference.png`) has **not been provided yet**. This plan
is derived from the written brief (Section 6). When the image arrives it will be copied to
`docs/design/reference_mockup.png` and this plan revised against it.

## 1. Design tokens

| Token | Light | Dark | Use |
|---|---|---|---|
| `--bg` | `#F6F5FD` (soft lavender-white) | `#0F1020` | app background |
| `--surface` | `#FFFFFF` | `#1A1B2E` | cards |
| `--surface-2` | `#F0EEFC` | `#23243A` | inputs, table stripes |
| `--border` | `#E6E3F5` | `#2E3050` | card borders |
| `--text` | `#1B1A2E` | `#ECEBFA` | primary text |
| `--muted` | `#6B6A80` | `#A3A2BC` | secondary text |
| `--accent` | `#6D4AFF` | `#8B70FF` | buttons, active nav, stepper, highlights |
| `--accent-soft` | `#EEE9FF` | `#2C2552` | active-item background, chips |
| `--red` | `#E5484D` | `#FF6369` | high bias / REJECT |
| `--amber` | `#F5A524` | `#FFB84D` | medium / RETRY |
| `--green` | `#2BA66A` | `#3DD68C` | accept / pass / low |
| `--blue` | `#3B82F6` | `#60A5FA` | informational / HUMAN_REVIEW |

Typography: Inter (Google Fonts), 15px base, 44px/800 hero, 22px/700 section titles, tabular numbers.
Shape: 14px card radius, 999px pills, shadow `0 1px 2px rgba(20,16,60,.04), 0 8px 24px rgba(20,16,60,.06)`.
All semantic colours are paired with a text label or icon (never colour alone) for accessibility.

## 2. Mechanics

- `.streamlit/config.toml`: light base theme with the accent as `primaryColor`, Inter font.
- `assets/styles.css`: one stylesheet injected once per run; it styles Streamlit's own widgets
  (buttons, tabs, file uploader, sidebar) and defines the component classes below.
- Dark mode: a sun/moon toggle (top right) stores `theme` in `st.session_state`; the CSS is
  injected with a `data-theme` scope so both palettes come from the same token set.
- `src/fairlens/ui/components.py`: `card`, `metric_tile`, `badge`, `stepper`, `score_bar`,
  `decision_banner`, `pipeline_cards`, `empty_state`, `highlighted_text`, `diff_html`,
  `page_header`. All return HTML rendered through `st.markdown(..., unsafe_allow_html=True)`.
- Charts: Plotly with a shared template (transparent background, theme-aware text, the accent
  for FairLens, greys for baselines).
- Navigation: `st.navigation` with a custom sidebar (logo, "FairLens", icon + label items,
  active highlight); main group = Home, Text Analysis, Mitigation, Reliability Gate,
  Image Analysis, Model Comparison, Benchmarks, About; a "More" group = Error Analysis,
  Experiments & Exports, Use Cases.

## 3. Page plan (functional standard = brief, visual standard = mockup)

| Page | Key elements | Data source |
|---|---|---|
| Home | hero "Detect. Reduce. Verify.", Analyze Text / Analyze Image buttons, 4 feature cards, Quick Start (Text / Image tabs) with a working Run Analysis | live pipeline |
| Text Analysis | Enter Text / Upload File tabs, char counter, example chips, stepper, highlighted spans, overall score bar (High/Med/Low), per-detector bars + badges + runtime, categories with trigger phrase + explanation | live pipeline |
| Mitigation | 4 colour-coded cards (Original, Rule, FLAN-T5, Qwen) with "Use this version", diff view, comparison table with best-value highlight, metric badges (bias delta, similarity, facts kept) | live pipeline |
| Reliability Gate | selected rewrite, decision banner, tabs Evaluation Metrics / Detailed Analysis / Explanation, 4 metric tiles, preservation-checks grid, signal bar/radar, toggle vs lowest-bias selection | live pipeline |
| Image Analysis | drag-drop (JPG/PNG/GIF, size limit), preview, BLIP + Florence-2 caption cards, caption analysis table, scope caveat | live pipeline |
| Model Comparison | tabs Performance / Runtime & Resources / Calibration / Confusion Matrices; F1 bars; metrics table | `outputs/benchmarks/` |
| Benchmarks | KPI tiles, bias-score comparison, gate decision distribution, tabs Text / Mitigation / Gate / Image, ablations, gate vs baselines, ROC/PR | `outputs/*` run files |
| Error Analysis | FP/FN browser, preservation failures, example cards | `outputs/benchmarks/`, `outputs/gate/` |
| Experiments & Exports | run history, manifest/provenance viewer, reproduce-run, PDF/CSV/JSON/HTML export | `outputs/runs.sqlite` + manifests |
| Use Cases | 6 one-click demo cards (problem, input, output, gate decision) + disclaimer | live pipeline / cached |
| About | tabs Overview, Methodology, Models, Datasets, Limitations, Citation; architecture flow | docs/*.md |

## 4. States and rules

- **No placeholder numbers.** Every number comes from a live run or a file under `outputs/`.
  Missing results render `empty_state(...)` with a button that launches the relevant script.
- Loading: skeleton blocks while models load, `st.status` for the pipeline with per-stage updates.
- Errors: clear message cards (model failed to load / file unreadable), never a silent fallback.
- Lite mode toggle in the sidebar switches to the lite checkpoints.
- Tooltips (`help=`) on every metric explaining what it measures.
- Layout checked at 1366px and 1920px; columns collapse below 1100px.
