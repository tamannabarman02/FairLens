"""Run a FairLens script with PyTorch hidden, for analysis-only steps.

Some environments ship a torch build whose OpenMP runtime clashes with numpy's (e.g. Anaconda base on Windows,
where Application Control allows only the preinstalled binaries). Steps that need no model inference can run
with torch hidden: device detection then reports 'cpu', and the manifest records environment.torch_hidden = true.

  python scripts/no_torch.py scripts/run_ablations.py --signals-from <run_id>
"""
import runpy
import sys

sys.modules["torch"] = None  # any `import torch` now raises ImportError

if len(sys.argv) < 2:
    raise SystemExit(__doc__)
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
