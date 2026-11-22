"""Simplified single-column Gate decision-flow figure for the Comp-SIF paper."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

out = sys.argv[1]
plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42

W, H = 3.45, 3.55
fig = plt.figure(figsize=(W, H), dpi=600)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

LX, LW = 0.05, 1.55     # left (checks) column
RX, RW = 1.85, 1.55     # right (outcome) column
BH = 0.48

def box(x, y, w, h, title, sub, fc, ec, bold=True, fs=6.6, sfs=5.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.04",
                                fc=fc, ec=ec, lw=0.8))
    if sub:
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal")
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=sfs, linespacing=1.1)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", linespacing=1.15)

def arrow(x0, y0, x1, y1, label=None, lx=0, ly=0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color="black", mutation_scale=6))
    if label:
        ax.text((x0 + x1) / 2 + lx, (y0 + y1) / 2 + ly, label, fontsize=5.6, ha="center", va="center")

CHK_FC, CHK_EC = "#FFFFFF", "#333333"
# input
top = H - 0.05 - 0.28
box(LX, top, LW, 0.28, "Original + candidate rewrite", None, "#F2F2F2", "#333333", fs=6.4)

ys = [top - 0.17 - BH - i * (BH + 0.17) for i in range(4)]
checks = [
    ("1. Input biased?", "mean detector score ≥ 0.5"),
    ("2. Facts and meaning kept?", "fact retention, entailment,\nno contradiction, similarity, fluency"),
    ("3. Bias fell enough?", "bias reduction ≥ threshold"),
    ("4. Detectors agree, confident?", "detector spread, distance from 0.5"),
]
outcomes = [
    ("ACCEPT unchanged text", "NEUTRAL_UNCHANGED; any change:\nREJECT (UNNECESSARY_CHANGE)", "#E8E8E8", "#555555"),
    ("REJECT", "FACT_LOSS, ENTAILMENT_FAIL,\nCONTRADICTION, LOW_SIMILARITY", "#D0D0D0", "#333333"),
    ("RETRY (strict prompt, ≤ 2)", "then HUMAN_REVIEW\n(RETRIES_EXHAUSTED)", "#F2F2F2", "#333333"),
    ("HUMAN_REVIEW", "DETECTOR_DISAGREEMENT,\nLOW_CONFIDENCE", "#FFFFFF", "#333333"),
]
arrow(LX + LW / 2, top, LX + LW / 2, ys[0] + BH)
for i, (y, (t, s), (ot, os_, ofc, oec)) in enumerate(zip(ys, checks, outcomes)):
    box(LX, y, LW, BH, t, s, CHK_FC, CHK_EC)
    box(RX, y, RW, BH, ot, os_, ofc, oec, sfs=5.2)
    arrow(LX + LW, y + BH / 2, RX, y + BH / 2, "No", ly=0.06)
    if i < 3:
        arrow(LX + LW / 2, y, LX + LW / 2, ys[i + 1] + BH, "Yes", lx=0.12)

acc_y = 0.05
box(LX, acc_y, LW, 0.36, "ACCEPT (ALL_CHECKS_PASSED)", "best accepted candidate returned", "#FFFFFF", "#000000", fs=6.0, sfs=5.4)
ax.patches[-1].set_linewidth(1.4)
arrow(LX + LW / 2, ys[3], LX + LW / 2, acc_y + 0.36, "Yes", lx=0.12)
ax.text(RX + RW / 2, acc_y + 0.18, "Precedence across candidates:\nREJECT > RETRY >\nHUMAN_REVIEW > ACCEPT",
        fontsize=5.3, ha="center", va="center", style="italic")

fig.savefig(out, dpi=600)
print("saved", out)
