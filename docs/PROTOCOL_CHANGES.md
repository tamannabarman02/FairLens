# Protocol changes

Every change to data, labels, splits, metrics, thresholds, prompts or evaluation protocol is recorded here **before**
the run that uses it, with its reason. Approval: the owner gave standing instructions on 2026-10-24 to proceed without
waiting for confirmation while unavailable; entries approved under that instruction are marked *(standing approval)*
and should be reviewed by the team.

| ID | Date | Change | Reason / evidence | Affects | Status |
|---|---|---|---|---|---|
| PC-01 | 2026-10-23 | WNC (Wiki Neutrality Corpus) replaced by **BABE** as the sentence-level biased-language corpus | Stanford host unreachable; HF mirror `reza-alipour/WNC` is gated (manual approval) | detection data | Applied before any benchmark (pre-V4) |
| PC-02 | 2026-10-23 | WinoBias used only as a **neutral probe** (FPR, pro−anti gap), never as training labels | Its sentences are neutral event descriptions; labelling pro=1 would be invalid | detection evaluation | Applied (pre-V4) |
| PC-03 | 2026-10-23 | StereoSet `unrelated` sentences excluded from classification labels; CrowS-Pairs only `stereo`-direction pairs as labels; all pairs kept as preference probes | Unrelated sentences are often nonsensical; antistereo direction is ambiguous as a binary label | detection data | Applied (pre-V4); to be re-reviewed in D.1 |
| PC-04 | 2026-10-23 | BART-MNLI hypothesis template chosen on 300 **validation** sentences from 5 documented templates | First template scored a neutral sentence 0.90 in verification | detection protocol | Implemented, not yet run |
| PC-05 | 2026-10-24 | Gate `entailment_mode` (both / forward) added as a tunable, chosen on the gate **val** split | NLI both-ways = 0.005 for a valid paraphrase in verification; removing a stereotyped clause breaks backward entailment | gate thresholds | Implemented, not yet run |
| PC-06 | 2026-10-24 | Prompt variants extended with a 3-demonstration (few-shot) option, same demos for both LLMs; choice on a 30-sentence **train-split** sample | FLAN-T5 base/large copied biased inputs in zero-shot (4 formats) and 3-shot diagnostics | mitigation prompts | Implemented, not yet run |
| PC-07 | 2026-10-24 | Two killed DeBERTa runs relabelled `running` → `stopped` with a cause note | Stage 0 audit item 6; processes were killed, no checkpoint exists | run records only | Applied |
