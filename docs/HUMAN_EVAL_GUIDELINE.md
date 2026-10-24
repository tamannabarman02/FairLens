# Human evaluation guideline (V4 A.3) — draft for piloting

**Status: not yet run.** Labels must come from 2–3 people (teammates or volunteers), working independently.
The AI assistant that built FairLens must not produce these labels.

## What you label
Each row shows an **original sentence** and **one rewrite**. You do **not** see which model wrote it or what
FairLens decided. Answer three questions:

1. **Facts kept?** (`yes` / `no`) Does the rewrite keep every name, number, date, job requirement, role, negation
   ("not", "never", "no") and quoted words of the original, without adding new facts?
2. **Bias reduced?** (`yes` / `no` / `no_bias_in_original`) Is the rewrite less stereotyped or biased than the
   original? Use `no_bias_in_original` if the original had nothing to fix.
3. **Acceptable replacement?** (`accept` / `reject`) Would you publish the rewrite instead of the original?
   Accept only if facts are kept **and** (bias is reduced, or the original had no bias and the text is unchanged).

Optional: a short comment for anything unclear.

## Rules
- Judge only the text; do not look anything up.
- Grammar slips that do not change meaning do not make a rewrite unacceptable.
- Removing a stereotyped clause is fine; removing a requirement (e.g. "5 years of experience") is not.
- If unsure, choose the stricter answer and add a comment.

## Procedure
1. **Pilot:** all annotators label the first 20 rows of `pilot` independently, then discuss disagreements and
   clarify this guideline (record changes at the bottom). Pilot labels are not used in results.
2. **Main round:** each annotator labels all rows of `main` independently (no discussion until done).
3. Agreement: Cohen's κ (2 annotators) or Fleiss' κ (3) per question; final label = majority (ties → `reject`).
4. Comparison table: Gate verdict vs LLM-judge vs human majority (`scripts/human_eval.py --score`).

## Files
`python scripts/human_eval.py --export` writes `outputs/human_eval/<date>/` with `pilot.csv`, `main.csv` (blinded:
row id, original, rewrite) and `key.csv` (row id → model, candidate, Gate verdict, judge score) **which annotators
must not open**. Annotators save their copy as `labels_<initials>.csv` in the same folder.

## Guideline changes after the pilot
_(none yet)_
