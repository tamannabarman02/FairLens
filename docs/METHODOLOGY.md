# Methodology

## 1. Research question
Bias-mitigation rewrites are usually judged by one number: did a bias detector's score go down? A rewrite can
lower that score while silently changing meaning, numbers, negations or who did what. **Can a transparent,
multi-signal Reliability Gate decide more safely than simple selection rules which rewrite (if any) to accept?**

### Hypotheses
- **H1 (detection).** Fine-tuned encoders (DistilBERT, DeBERTa-v3-small) outperform zero-shot NLI and the rule
  baseline on the shared test split, but no single detector is reliable across all sources (media framing vs
  stereotype minimal pairs).
- **H2 (mitigation).** LLM rewriters reduce detector bias scores more than the rule baseline but also introduce
  more preservation failures (lost facts, contradictions).
- **H3 (gate).** Selecting by lowest bias score accepts more unacceptable rewrites than the Gate; the Gate trades
  this for deferrals to human review.
- **H4 (ablation).** Each signal group (semantic similarity, entailment, facts, detector agreement, uncertainty,
  fluency, bias) contributes measurably to the Gate's safety; groups with no measurable contribution are reported
  as negative results.

## 2. Pipeline
Input → sentence segmentation (offset-preserving, quote/abbreviation aware) → four detectors → three rewriters
(rule, FLAN-T5-base, Qwen2.5-0.5B-Instruct) → Gate signals → decision (+ strict-prompt retry) → output.

## 3. Detection experiment
- **Data:** unified schema from BABE, StereoSet and CrowS-Pairs (`docs/DATASETS.md`); grouped, seeded
  70/15/15 splits; no group crosses splits.
- **Detectors:** DistilBERT and DeBERTa-v3-small fine-tuned on train (AdamW, lr 5e-5, linear warm-up 10%,
  batch 16, length-bucketed batches, best epoch by val macro-F1, seed 42); BART-large-MNLI zero-shot with a
  hypothesis template chosen on 300 val sentences out of 5 documented templates; Hub RoBERTa (BABE) as an
  external pretrained reference; rule baseline.
- **Same data, same metrics:** every detector scores the same 2,652 test sentences; threshold 0.5 for all.
- **Metrics:** precision, recall, F1 and macro-F1 (bootstrap 95% CI, 1,000/300 resamples), ROC-AUC, PR-AUC,
  false-positive rate, confusion matrix, ECE (10 equal-width bins) + Brier score + reliability table,
  ms/sample (fixed 100-sentence subset, batch 16, CPU), resident model MB, load time; per-source macro-F1.
- **Probes:** WinoBias FPR and pro−anti score gap (all sentences are neutral); CrowS-Pairs and StereoSet pairwise
  preference.

## 4. Mitigation experiment
- **Inputs:** `mitigation_eval.parquet` (276: authored hard cases + test-split benchmark sentences).
- **Fairness controls:** one prompt-template family for both LLMs (same instruction and, if selected, the same
  three demonstrations; only the wrapper differs: seq2seq prompt vs chat template), greedy decoding,
  `max_new_tokens=96`, `repetition_penalty=1.1`, seed 13 set before each generation. Prompt variant chosen on a
  30-sentence **train-split** development sample (`configs/prompt_selection.json`). All prompts and outputs are
  logged (`prompts_log.jsonl`) and cached.
- **Retry:** inputs whose best candidate is RETRY are re-prompted with a strict instruction that lists the
  extracted facts, up to `max_retries = 2`, then escalated to HUMAN_REVIEW.
- **Metrics (automatic; no gold labels exist for model outputs):** ensemble bias before/after and reduction (CI),
  changed rate, semantic similarity, bidirectional entailment, fact retention and failure rate, fluency,
  length change, unnecessary-change rate on gold-neutral inputs, similarity to the author's reference rewrite
  (authored subset), Gate verdict distribution.

## 5. Reliability Gate
### Signals (per candidate)
| Signal | Definition |
|---|---|
| `bias_reduction` | ensemble(original) − ensemble(candidate); ensemble = mean of the 4 detectors |
| `semantic_similarity` | cosine of MiniLM embeddings |
| `entailment_both_ways` | min(P(orig ⇒ cand), P(cand ⇒ orig)) from `cross-encoder/nli-deberta-v3-small`; `entail_forward` = P(orig ⇒ cand) |
| `contradiction` | max contradiction probability over both directions |
| `fact_retention` | kept facts / extracted facts (entities, numbers, dates, roles, quotes, + 1 for negation polarity) |
| `detector_agreement` | standard deviation of the 4 detector scores on the candidate |
| `uncertainty` | 0.5·std/0.5 + 0.5·(1 − margin), margin = abs(ensemble − 0.5)·2 |
| `fluency`, `length_ratio` | heuristic: repetition, prompt leakage, non-alphabetic share; word-count ratio |

### Transparent rules
Verdict precedence REJECT > RETRY > HUMAN_REVIEW > ACCEPT. Non-biased input (ensemble < 0.5): unchanged → ACCEPT
(NO CHANGE), changed → REJECT (UNNECESSARY_CHANGE). Biased input: any preservation failure (facts, entailment,
contradiction, similarity, length, fluency) → REJECT; insufficient bias reduction / residual bias → RETRY;
detector disagreement or low confidence → HUMAN_REVIEW; otherwise ACCEPT. Among accepted candidates the one with
the highest bias_reduction + similarity + fact_retention is chosen. Every decision carries reason codes.

### Threshold choice
Defaults in `configs/models.yaml` are a-priori values. A small grid (similarity, entailment mode/threshold, bias
reduction, agreement) is searched on the **validation split of the controlled gate set only**; the chosen values
are then fixed and evaluated on the test split, and used by the dashboard and the mitigation benchmark.

### Learned variant
Logistic regression (standardised, balanced classes) and a depth-4 decision tree over 14 signal features, trained
on the val split to predict ACCEPT/REJECT/RETRY; abstains (HUMAN_REVIEW) when the top probability < 0.6.

## 6. Gate evaluation (controlled gold set)
Candidates with known gold labels (`docs/DATASETS.md` §4), 3 perturbation seeds, val/test split by input.
- **Candidate level:** 3-class macro-F1, accept precision, accept recall, **unsafe-accept rate**
  (P(accept | gold ≠ ACCEPT)), human-review rate, catch rate per perturbation type; also an *oracle-input* variant
  that uses the gold input label to separate detection errors from gate logic.
- **Input level vs baselines:** each policy picks one candidate or defers: FairLens Gate (default and val-tuned),
  learned gates, (a) lowest bias score, (b) highest similarity among changed candidates, (d) random (5 seeds).
  Baseline (c) *always accept the first model* has no meaning on synthetic candidates and is evaluated on the real
  mitigation run instead. Reported: acceptable-choice rate, harmful-choice rate, deferral rate, unnecessary-change
  rate on neutral inputs, with bootstrap CIs and paired bootstrap differences vs the tuned gate.
- **Ablations:** the tuned rule gate with one signal group disabled at a time; paired bootstrap CI of the change in
  unsafe-accept rate and accuracy; per-seed unsafe-accept; and the learned gate retrained without that group.
  A signal "adds measurable value" only if its CI excludes zero.

## 7. Image experiment
BLIP-base vs Florence-2-base on the same 80 COCO val2017 images (60 with the person category, 20 without).
Metrics: CLIP ViT-B/32 image-text cosine (CIs), caption length, detail coverage (share of content words of the 5
human references that the caption mentions), rate of gendered terms on person images (compared with the human
references), rate of attributes pixels cannot normally verify (nationality, wealth, religion...), and the text
detectors' bias scores on the captions; agreement between captioners. Scope: bias in generated captions only.

## 8. Statistics and reproducibility
Bootstrap 95% CIs (percentile, 1,000 resamples unless stated); paired bootstrap for differences; fixed seeds
(data 42, training 42, generation 13, perturbations 11/22/33, random baselines 100–104). Every run writes a
manifest (config, seed, dataset hashes, model revisions, hardware, git commit, runtime) to
`outputs/runs/<run_id>/manifest.json` and a row in `outputs/runs.sqlite`; the dashboard and `docs/RESULTS.md`
read only these files.
