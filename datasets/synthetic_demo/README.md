# Synthetic demonstration inputs

**These files are synthetic.** They were written for the FairLens dashboard demo and the Use Cases page.
They are not benchmark data, not real job adverts, social posts or model outputs, and they are never
used to compute any reported metric. Names and organisations are fictional.

- `examples.json`: the example-prompt chips on the Text Analysis page.
- `use_cases.json`: the six demo scenarios (problem, input, what must be preserved).

Real benchmark data lives in `datasets/processed/` (built by `scripts/build_datasets.py`); the curated
hand-written evaluation set lives in `datasets/curated/` and is documented in `docs/DATASETS.md`.
