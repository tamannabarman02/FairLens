# Prior work (V4 P.3)

Each entry was checked against its publisher or arXiv page on 2026-10-24 (title, authors, venue). Where only the
abstract was available, the description below is limited to what the abstract states.

## 1. Bias detection and benchmarks
- **StereoSet** — Nadeem, Bethke, Reddy. *StereoSet: Measuring stereotypical bias in pretrained language models.* ACL 2021.
- **CrowS-Pairs** — Nangia, Vania, Bhalerao, Bowman. *CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models.* EMNLP 2020.
- **WinoBias** — Zhao, Wang, Yatskar, Ordonez, Chang. *Gender Bias in Coreference Resolution: Evaluation and Debiasing Methods.* NAACL 2018.
- **BOLD** — Dhamala et al. *BOLD: Dataset and Metrics for Measuring Biases in Open-Ended Language Generation.* FAccT 2021, doi:10.1145/3442188.3445924.
- **BABE** — Spinde, Plank, Krieger, Ruas, Gipp, Aizawa. *Neural Media Bias Detection Using Distant Supervision With BABE — Bias Annotations By Experts.* Findings of EMNLP 2021, pp. 1166–1177.
- **Validity critique** — Blodgett, Lopez, Olteanu, Sim, Wallach. *Stereotyping Norwegian Salmon: An Inventory of Pitfalls in Fairness Benchmark Datasets.* ACL 2021, pp. 1004–1015. (Basis for our caveats on StereoSet/CrowS-Pairs labels.)
- **Survey** — Gallegos et al. *Bias and Fairness in Large Language Models: A Survey.* Computational Linguistics 50(3), 2024, doi:10.1162/coli_a_00524.
- **Job adverts** — Gaucher, Friesen, Kay. *Evidence That Gendered Wording in Job Advertisements Exists and Sustains Gender Inequality.* J. Personality and Social Psychology 101(1), 2011, pp. 109–128. (Source of gender-coded word lists in the rule baseline.)

## 2. Mitigation by rewriting
- **Neutralising subjective bias (WNC)** — Pryzant, Diehl Martinez, Dass, Kurohashi, Jurafsky, Yang. *Automatically Neutralizing Subjective Bias in Text.* AAAI 2020. 180k Wikipedia neutrality edits (the corpus we could not obtain, PC-01).
- **DEPEN** — He, Majumder, McAuley. *Detect and Perturb: Neutral Rewriting of Biased and Sensitive Text via Gradient-based Decoding.* Findings of EMNLP 2021, pp. 4173–4181. Detects sensitive spans, masks and regenerates them.
- **Gender-neutral rewriting** — Sun et al. *They, Them, Theirs: Rewriting with Gender-Neutral English.* arXiv:2102.06788 (2021).
- **Detoxification with parallel data** — Logacheva et al. *ParaDetox: Detoxification with Parallel Data.* ACL 2022, pp. 6804–6818. Crowdsourced paraphrases with a separate content-preservation step.
- **LLM-in-the-loop detoxification** — Yuan et al. *LLM in the Loop: Creating the ParaDeHate Dataset for Hate Speech Detoxification.* arXiv:2506.01484 (2025). Replaces human annotators in the ParaDetox pipeline with GPT-4o-mini; evaluates style accuracy, content preservation and fluency.

## 3. Meaning and fact preservation
- **Style-transfer evaluation** — Mir, Felbo, Obradovich, Rahwan. *Evaluating Style Transfer for Text.* NAACL 2019; Briakou et al. *Evaluating the Evaluation Metrics for Style Transfer.* EMNLP 2021 (arXiv:2110.10668). Content preservation is hard to measure automatically; recommended metrics differ.
- **NLI for consistency** — Laban, Schnabel, Bennett, Hearst. *SummaC: Re-Visiting NLI-based Models for Inconsistency Detection in Summarization.* TACL 10, 2022, pp. 163–177. Sentence-level NLI aggregation for factual consistency (our entailment signal is the sentence-level case).
- **Sentence embeddings** — Reimers, Gurevych. *Sentence-BERT.* EMNLP 2019 (basis of the MiniLM similarity signal).

## 4. Judges, abstention and risk control
- **LLM-as-a-judge** — Zheng et al. *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS 2023 (Datasets and Benchmarks). Documents position, verbosity and self-enhancement biases of LLM judges; motivates validating our judge baseline on gold labels first.
- **Selective prediction** — Geifman, El-Yaniv. *Selective Classification for Deep Neural Networks.* NeurIPS 2017. Risk–coverage curves and abstaining to meet a target risk (our A.1 framing).
- **Abstention in NLP** — Kamath, Jia, Liang. *Selective Question Answering under Domain Shift.* ACL 2020. Raw model confidence is a poor abstention signal under shift; a calibrator helps.
- **Conformal risk control** — Angelopoulos, Bates, Fisch, Lei, Schuster. *Conformal Risk Control.* ICLR 2024 (arXiv:2208.02814). Controls the expected value of a monotone loss; basis of A.5.

## 5. Closest work to the Reliability Gate
- **LGSA** — Karri et al. *LLM-Guided Synthetic Augmentation (LGSA) for Mitigating Bias in AI Systems.* arXiv:2510.13202 (2025). Gender-swapped LLM rewrites screened with automated checks for semantic similarity, attribute verification and toxicity, plus human spot checks.
- **ParaDetox / ParaDeHate** (above): content-preservation checks decide whether a detoxified paraphrase is kept.
- **Persona-augmented benchmarking** — Truong, Fogliato, Heidari, Wu. *Persona-Augmented Benchmarking: Evaluating LLMs Across Diverse Writing Styles.* EMNLP 2025 (arXiv:2507.22168). Rewrites benchmark items in different personas while aiming to keep semantic content; we did not verify its validation procedure in the full text.

## Positioning (what FairLens may and may not claim)
Automatically **filtering generated rewrites with preservation checks is established practice** (ParaDetox-style
content-preservation steps, LGSA's automated screening). FairLens therefore does **not** claim that gating rewrites is new.
What the Reliability Gate combines is: hard per-fact-type preservation checks (numbers, dates, names, roles, negation,
quotations) together with similarity and two-way entailment; detector agreement and confidence as reasons to defer to a
human; a four-way decision (accept / retry with a stricter prompt / reject / human review) with reason codes; and an
evaluation against simpler selection policies (lowest bias score, highest similarity, no change, a validated LLM judge,
random) with risk–coverage, red-team corruption, ablations and conformal risk control. We found no prior work that
evaluates this particular combination for bias-mitigation rewrites, but our search was limited to the queries below, so
any novelty statement in the paper must be phrased as "to our knowledge" and must cite the closest work above.
Whether the combination **improves** on the baselines is decided only by `docs/RESULTS.md`.

### Searches run (2026-10-24)
neutral rewriting of biased text; ParaDetox; selective classification risk–coverage; conformal risk control;
LLM-as-a-judge; NLI inconsistency detection; "verification filter accept or reject LLM rewrite fact preservation
entailment … abstain human review" (extended search); style-transfer content preservation metrics; inclusive-language
job-advert rewriting; BABE; BOLD; Blodgett 2021; Gallegos 2024; Gaucher 2011; Kamath 2020; DEPEN.
