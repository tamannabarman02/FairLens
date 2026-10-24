# Datasets

Built by `scripts/build_datasets.py` (seed 42). Revisions and file hashes are in
`datasets/processed/dataset_manifest.json`. The metrics of these datasets are **not interchangeable**:
each measures a different thing, so results are always reported per source as well as pooled.

## 1. Sources

| Dataset | Licence | What it measures | How FairLens converts it to labels | Known limits |
|---|---|---|---|---|
| **BABE** (Spinde et al., 2021), HF `mediabiasgroup/BABE` | CC BY-NC-SA 4.0 (per paper; HF card has no licence field) | Expert-annotated **media bias / loaded framing** at sentence level in US news, plus biased-word spans | `label` 1 = biased, 0 = non-biased; `bias_type = media_framing`. Official train/test split kept; 15% of train -> val (seeded) | US political news only; framing bias, not social stereotypes; non-commercial licence |
| **StereoSet** (Nadeem et al., 2021), HF `McGill-NLP/stereoset` (validation = public labelled split) | CC BY-SA 4.0 | Stereotype vs anti-stereotype vs unrelated continuations (intra- and inter-sentence) for gender, race, religion, profession | stereotype -> 1, anti-stereotype -> 0; *unrelated* excluded from training (often nonsensical, e.g. "The chess player was fox."), kept for a probe. Inter-sentence text = context + sentence. Both members of an item stay in the same split | Anti-stereotypes are often still generalisations ("The people are fat and unathletic"); some "stereotypes" are mild; known annotation noise (Blodgett et al., 2021) |
| **CrowS-Pairs** (Nangia et al., 2020), GitHub `nyu-mll/crows-pairs` | CC BY-SA 4.0 | Minimal pairs: a sentence that is more stereotypical vs a less stereotypical counterpart, 9 bias types | Only `stereo`-direction pairs (1,290) go into the detection set: `sent_more` -> 1, `sent_less` -> 0 (same pair, same split). All 1,508 pairs kept as a pairwise-preference probe | Originally an MLM-likelihood benchmark; using it as classification labels is an adaptation. `sent_less` often repeats the same generalisation about a different group; documented validity problems (Blodgett et al., 2021) |
| **WinoBias** (Zhao et al., 2018), HF `uclanlp/wino_bias` | MIT | Gender-occupation coreference with pro- and anti-stereotypical pronouns | **Probe only, never training data.** All 1,584 sentences are neutral event descriptions, so any flag is a false positive; we also report the score gap pro − anti | Coreference benchmark, not a biased-language corpus |
| **WNC** (Wiki Neutrality Corpus, Pryzant et al., 2020) | (CC BY-SA) | Biased -> neutralised Wikipedia edits (parallel) | **Not used: blocked.** The Stanford download host did not respond and the only HF mirror (`reza-alipour/WNC`) is gated (manual approval). BABE substitutes as the sentence-level biased-language corpus | Its absence means no large human-written parallel rewrite corpus is available; see references below |
| **COCO 2017 val** (Lin et al., 2014) | Annotations CC BY 4.0; images under per-image Flickr licences (recorded per image) | Image captioning; 5 human reference captions per image | 60 images with the `person` category + 20 without (seeded), used for the captioning experiment. Metadata is used only to select/describe the set; no demographic inference | Western-centric photo distribution; reference captions themselves contain gendered wording |

## 1b. Label mapping (V4 D.1, reviewed 2026-10-24 before the first detection run)

Shared binary label: **1 = the sentence expresses or relies on a stereotype or biased framing; 0 = it does not**, as
judged by each source's own annotation. The sources are not equivalent, so classification metrics are always
reported **per source** as well as pooled, and pair-preference results are never mixed into classification metrics.

| Source | Unit in the source | Source label | FairLens label | Used for |
|---|---|---|---|---|
| BABE | single sentence | `label` 1 biased / 0 non-biased (expert majority) | 1 / 0 unchanged | classification (train/val/test) |
| StereoSet intra/inter | item = context + 3 options | stereotype / anti-stereotype / unrelated | stereotype → 1, anti-stereotype → 0, unrelated → not labelled | classification; **pair probe** stereotype > anti, stereotype > unrelated |
| CrowS-Pairs | minimal pair | `stereo` / `antistereo` direction | `stereo` pairs only: `sent_more` → 1, `sent_less` → 0 | classification; **pair probe** (both directions) |
| WinoBias | single sentence (pro / anti) | coreference condition | none (all neutral) | **probe only**: FPR and pro−anti gap |

Known validity limits of the mapping: StereoSet anti-stereotypes are often still generalisations, and CrowS
`sent_less` often repeats a generalisation about another group (Blodgett et al., 2021). The minimal-pair labels
therefore test whether a detector separates the stereotype-consistent member of a pair, which is harder and noisier
than BABE's sentence-level labels. This is why per-source results are reported separately.

Test-set exclusion PC-08: 4 test items whose normalised text also appears in train/val are dropped from test metrics.

## 2. Unified detection schema (`datasets/processed/detection.parquet`)

Columns: `text, label, bias_type, source, split, group_id` (+ `biased_words` for BABE, `subset`).
Splits are made **by group** (`group_id` = StereoSet item / CrowS pair / BABE uuid), so minimal pairs never
straddle splits; identical texts are de-duplicated with priority test > val > train (23 duplicates removed).
The manifest records `groups_spanning_splits = 0`.

| Source | Train (0/1) | Val (0/1) | Test (0/1) |
|---|---|---|---|
| BABE | 1,193 / 1,460 | 188 / 279 | 441 / 557 |
| CrowS-Pairs | 904 / 904 | 193 / 193 | 192 / 192 |
| StereoSet inter | 1,479 / 1,479 | 324 / 324 | 320 / 320 |
| StereoSet intra | 1,476 / 1,470 | 309 / 309 | 315 / 315 |
| **Total** | 5,052 / 5,313 | 1,014 / 1,105 | 1,268 / 1,384 |

Preprocessing: whitespace normalisation only (no lower-casing; tokenisers handle case). The binary label
means "the sentence expresses or relies on a stereotype or biased framing" under each source's own annotation.

## 3. Probes (`winobias_probe.parquet`, `crows_pairs.parquet`, `stereoset_pairs.parquet`)
Evaluated on test-split items only (WinoBias: its official test files). Reported: WinoBias false-positive
rate per condition and pro−anti score gap; CrowS share of pairs where `sent_more` scores higher (stereo and
antistereo directions separately); StereoSet stereotype > anti-stereotype and stereotype > unrelated preference,
and how often *unrelated* sentences are flagged.

## 4. Curated mitigation and gate sets (`datasets/curated/`, `datasets/processed/`)

- **`authored_cases.jsonl`**: 91 biased + 45 neutral sentences **hand-written during development** (fictional names)
  covering recruitment text, gender/age/race/religion/disability/socioeconomic stereotypes, media framing,
  profession-gender outputs, and hard cases tagged `names, numbers, dates, negation, roles, quotes`. Each biased case
  has one author-written reference rewrite (one acceptable answer, not the only one). This is not crowd-sourced
  data and has a single annotator.
- **`mitigation_eval.parquet`** (276 inputs): the 136 authored cases + **test-split** benchmark sentences
  (50 BABE biased, 20 BABE non-biased, 40 StereoSet stereotypes, 30 CrowS-Pairs stereotypes). Only test-split
  benchmark items are used, so the fine-tuned detectors never saw them.
- **`gate_controlled.parquet`**: for each authored input, candidates with **gold labels known by construction**
  (3 perturbation seeds): biased input -> reference rewrite = ACCEPT, unchanged original = RETRY, seeded
  perturbations of the reference = REJECT (number changed, negation flipped, entity swapped, role swapped,
  antonym, hallucinated addition, truncation, quote altered); neutral input -> unchanged = ACCEPT, needless
  paraphrase or any perturbation = REJECT. Inputs are split 50/50 into val/test per seed; thresholds and the
  learned gate are fit on val only.
  *Limit:* perturbations are synthetic and several target exactly what the fact checker extracts; this
  circularity is discussed in `LIMITATIONS.md` and is why entailment/similarity-only perturbations
  (antonym, addition, truncation) are reported separately.

## 5. Synthetic demonstration inputs (`datasets/synthetic_demo/`)
Example chips and Use-Case inputs for the dashboard. **Never used for any metric.**

## 6. Prompt-development sample
`scripts/select_prompts.py` uses 30 biased sentences from the **train** split to choose the shared prompt
variant; the evaluation set is never used for prompt choice.

## References
Spinde et al. 2021 (BABE, Findings of EMNLP); Nadeem et al. 2021 (StereoSet, ACL); Nangia et al. 2020 (CrowS-Pairs, EMNLP);
Zhao et al. 2018 (WinoBias, NAACL); Pryzant et al. 2020 (Automatically Neutralizing Subjective Bias, AAAI);
Lin et al. 2014 (COCO, ECCV); Blodgett et al. 2021 (Stereotyping Norwegian Salmon, ACL).
