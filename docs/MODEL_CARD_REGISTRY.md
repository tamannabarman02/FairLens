# Model card registry

Every model was checked for task fit, licence, download size and memory, and given **one real inference on a
sample on this machine** by `scripts/verify_models.py` (each model in its own process). Commits are pinned in
`configs/model_lock.json`; adapters load exactly those revisions. Raw records:
`outputs/model_verification/verification.json`.

## Measured on this machine
<!-- MEASURED:START -->
_(filled by `scripts/generate_results_md.py`)_
<!-- MEASURED:END -->

## Decisions per slot

| Slot | Proposed in brief | Selected | Why / alternatives |
|---|---|---|---|
| BERT family | bert-base-uncased or distilbert-base-uncased, fine-tuned | **distilbert-base-uncased**, fine-tuned (`checkpoints/distilbert_bias`) | Both verified. DistilBERT is ~40% smaller (256 vs 420 MB) and trains in ~12 min/epoch on this CPU; bert-base would roughly double training time. Apache-2.0. |
| Hub pretrained bias classifier | e.g. DistilBERT/RoBERTa bias checkpoints | **himel7/bias-detector** (RoBERTa, BABE) | Apache-2.0, binary BABE labels, verified (biased sample 0.49 vs neutral 0.02). Rejected: `d4data/bias-detection-model` (no PyTorch weights, no licence), `valurank/distilroberta-bias` (licence "other"), `mediabiasgroup/da-roberta-babe-ft` (CC BY-NC 4.0; usable for research but stricter), `newsmediabias/UnBIAS-classifier` (OpenRAIL; not needed). Caveat: trained on BABE, possibly including sentences similar to our BABE test split; its BABE scores may be optimistic. |
| DeBERTa family | deberta-v3-small or -base, fine-tuned | **microsoft/deberta-v3-small** | SentencePiece tokenizer verified (`DebertaV2Tokenizer`, ▁ tokens). MIT. -base not attempted: -small was already the slowest model to train on this CPU (see PROGRESS.md for the training configuration actually used). |
| BART-MNLI | facebook/bart-large-mnli; distilled fallback | **facebook/bart-large-mnli** (standard), **valhalla/distilbart-mnli-12-1** (lite) | Large fits (1.55 GB, 2.2 GB peak RAM, ~300 ms/sentence/hypothesis). Hypothesis templates: 5 documented in `scripts/run_benchmarks.py`, chosen on 300 validation sentences. Initial pair template scored a neutral sentence 0.90, which is why selection is needed. DistilBART card has no licence field (derived from MIT BART). |
| Rule-based | transparent lexicon | `fairlens.detection.rule_based` | Word lists from Gaucher et al. 2011 (gender-coded job ads), age-coded job ad guidance, BABE loaded-language examples; group-generalisation patterns. Weights are hand-set priors, not learned. |
| Mitigation LLM A | google/flan-t5-base (large if it fits) | **google/flan-t5-base** | Large fits (3.7 GB peak) but was 2.4x slower and **both copied the biased input unchanged** in zero-shot (4 prompt formats) and 3-shot tests. This is a real capability limit, reported as a negative result; FLAN-T5 is kept because the brief requires it, with the best prompt variant chosen on train data. |
| Mitigation LLM B | Qwen2.5-0.5B or 1.5B-Instruct | **Qwen/Qwen2.5-0.5B-Instruct** (standard); 1.5B as `large_model_id` | 0.5B: 2.4 s/rewrite, rewrote the sample correctly ("an energetic salesperson with five years of experience"). 1.5B: 9.3 s/rewrite and added "at least" (meaning drift). Apache-2.0. |
| Semantic similarity | all-MiniLM-L6-v2 | **sentence-transformers/all-MiniLM-L6-v2** | 87 MB, Apache-2.0. Note: cosine 0.65 for a valid paraphrase of the sample, so the similarity threshold is tuned on validation data. |
| Entailment (NLI) | an NLI model | **cross-encoder/nli-deberta-v3-small** | Apache-2.0; separate from the detection NLI model so the Gate's evidence is not produced by a detector. Strict on paraphrases (both-ways 0.005 on the sample), which motivated the `entailment_mode` option. |
| Facts | spaCy + regex | **en_core_web_sm 3.8.0** + regex | Entities, numbers (incl. number words), dates, roles (gendered role words normalised), quotes, negation polarity. |
| Captioner A | Salesforce/blip-image-captioning-base | **BLIP base** | BSD-3-Clause; "two cats sleeping on a couch" on the sample; ~1.4 s/image. |
| Captioner B | microsoft/Florence-2-base (needs trust_remote_code) | **florence-community/Florence-2-base** | The transformers-native port (`Florence2ForConditionalGeneration`) loads **without `trust_remote_code`**, so no remote code runs. MIT. "Two cats laying on a pink couch next to a remote control." First call was slow (21 s, includes warm-up). Fallback verified: **nlpconnect/vit-gpt2-image-captioning** (Apache-2.0), `--fallback` flag. |
| Alignment | openai/clip-vit-base-patch32 | **CLIP ViT-B/32** | HF card has no licence field; the upstream OpenAI CLIP repository is MIT. Matching caption 0.316 vs wrong caption 0.153 on the sample. |

## Licence summary
Apache-2.0: DistilBERT, BERT, FLAN-T5, Qwen2.5, MiniLM, NLI cross-encoder, himel7/bias-detector, ViT-GPT2.
MIT: DeBERTa-v3, BART-MNLI, Florence-2, CLIP (upstream). BSD-3-Clause: BLIP. BABE data: CC BY-NC-SA (non-commercial).
