"""Draw the manuscript figures from stored run summaries (no hand-typed numbers).

Run with any Python that has matplotlib, from the project root:
    python scripts/make_paper_figures.py
Writes paper/figures/Fig1..Fig8 (Fig3-5 are app screenshots) as PNG (600 dpi) and EPS.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "outputs" / "runs"
OUT = ROOT / "paper" / "figures"
DETECTION = RUNS / "20261024-043227-benchmark_detection-a4a61f" / "detection_summary.json"
GATE = RUNS / "20261115-225313-benchmark_gate-794a76" / "gate_summary.json"
MITIGATION = RUNS / "20261024-040853-benchmark_mitigation-000f03" / "mitigation_summary.json"

plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "hatch.linewidth": 0.6})
# Colour-blind-safe (Okabe-Ito) plus hatching, so figures also read in greyscale.
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
     "grey": "#999999", "purple": "#CC79A7", "sky": "#56B4E9"}
MM = 1 / 25.4


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.eps", bbox_inches="tight")
    plt.close(fig)


def box(ax, x, y, w, h, title, body="", fc="white", ec=None, lw=0.8, tsize=7.5, bsize=6.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.06",
                                fc=fc, ec=ec or C["blue"], lw=lw))
    if body:
        ax.text(x + w / 2, y + h - 0.17, title, ha="center", va="top", fontweight="bold", fontsize=tsize)
        ax.text(x + w / 2, y + (h - 0.38) / 2, body, ha="center", va="center", fontsize=bsize, linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontweight="bold", fontsize=tsize)


def arrow(ax, x0, y0, x1, y1, ls="-", rad=0.0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", lw=0.8, ls=ls, connectionstyle=f"arc3,rad={rad}"))


def layer(ax, y, h, label, fc):
    ax.add_patch(FancyBboxPatch((0.0, y), 10.0, h, boxstyle="round,pad=0.01,rounding_size=0.08",
                                fc=fc, ec="#B0B0B0", lw=0.5))
    ax.text(0.12, y + h / 2, label, rotation=90, ha="center", va="center", fontsize=6.8, fontweight="bold",
            color="#333333")


def fig1_architecture():
    """High-level design: five layers, with the analysis flow in the core-services layer."""
    fig, ax = plt.subplots(figsize=(174 * MM, 132 * MM))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11.0)
    ax.axis("off")
    # 1 Presentation
    layer(ax, 9.4, 1.5, "Presentation", "#F3F0FA")
    ui = [("FairLens Review", "guided 6-step check\nof a job advert"),
          ("Analysis pages", "text, mitigation,\ngate, image"),
          ("Research pages", "benchmarks, model\ncomparison, errors"),
          ("Experiments", "run history,\nbackground jobs"),
          ("Reports", "PDF, JSON,\nCSV, HTML export")]
    for i, (t, b) in enumerate(ui):
        box(ax, 0.35 + i * 1.93, 9.52, 1.78, 1.25, t, b)
    # 2 Application
    layer(ax, 8.5, 0.75, "App", "#EEF4FA")
    box(ax, 0.35, 8.6, 9.5, 0.55, "Pipeline API: analyze_text · mitigate_text · analyze_image  "
        "(shared by the UI, experiments and tests)", tsize=6.5)
    # 3 Core services with the analysis flow
    layer(ax, 3.55, 4.8, "Core services", "#EAF5EF")
    core = [("Ingestion", "TXT, PDF, DOCX,\nCSV, JPG, PNG, GIF\nsize checks\nsentence splitting\n(offsets kept)"),
            ("Bias detection", "rule lexicon\nDistilBERT\nDeBERTa-v3-small\nBART-MNLI (0-shot)\nscores + evidence"),
            ("Mitigation", "rule rewriter\nFLAN-T5-base\nQwen2.5-0.5B\none prompt family\ngreedy, fixed seed"),
            ("Reliability Gate", "signals → rules\nstrict retry (≤ 2)\nlearned gate\n(alternative)"),
            ("Decision", "ACCEPT / RETRY /\nREJECT /\nHUMAN_REVIEW\n+ reason codes\n+ chosen text")]
    w, gap, y0, h = 1.62, 0.33, 5.75, 2.05
    for i, (t, b) in enumerate(core):
        x = 0.35 + i * (w + gap)
        gate = t == "Reliability Gate"
        box(ax, x, y0, w, h, t, b, fc="#DCEBF7" if gate else "white", lw=1.3 if gate else 0.8)
        if i < len(core) - 1:
            arrow(ax, x + w + 0.02, y0 + h / 2, x + w + gap - 0.02, y0 + h / 2)
    gx = 0.35 + 3 * (w + gap)
    mx = 0.35 + 2 * (w + gap)
    arrow(ax, gx + w / 2, y0 + h + 0.02, mx + w / 2, y0 + h + 0.02, ls="--", rad=0.35)
    ax.text((gx + mx + w) / 2, y0 + h + 0.38, "retry with the facts to keep", ha="center", fontsize=5.8,
            bbox=dict(boxstyle="round,pad=0.1", fc="#EAF5EF", ec="none"))
    box(ax, 0.35, 3.7, 3.5, 1.7, "Multimodal",
        "BLIP-base, Florence-2-base captions\nCLIP image–text alignment\ncaption checks (gendered terms,\n"
        "unverifiable attributes) → text detectors")
    box(ax, 4.6, 3.7, 5.25, 1.7, "Preservation and evaluation services",
        "fact extractor (spaCy + rules): names, numbers,\ndates, roles, quotes, negation\n"
        "similarity (MiniLM) · entailment (NLI cross-encoder)\n"
        "fluency, length · metrics (F1, ECE, bootstrap CIs)")
    arrow(ax, 0.35 + 1 * (w + gap) + w / 2 - 0.4, 5.4, 0.35 + 1 * (w + gap) + w / 2 - 0.4, y0 - 0.02)
    arrow(ax, gx + w / 2, 5.4, gx + w / 2, y0 - 0.02)
    # 4 Model and resource management
    layer(ax, 1.95, 1.4, "Models", "#FBF3E6")
    mm = [("Configuration", "model IDs, thresholds\npinned Hub revisions"),
          ("Model manager", "lazy loading, one large\nmodel resident (LRU)"),
          ("Hardware profiles", "lite / standard / GPU\nCUDA → MPS → CPU"),
          ("Local execution", "no text sent to\nexternal services")]
    for i, (t, b) in enumerate(mm):
        box(ax, 0.35 + i * 2.4, 2.05, 2.25, 1.15, t, b, ec="#C08A2E")
    # 5 Data and experiments
    layer(ax, 0.0, 1.8, "Data", "#F2F2F2")
    dd = [("Datasets", "BABE, StereoSet,\nCrowS-Pairs, WinoBias,\nCOCO, authored sets"),
          ("Run tracking", "JSON manifest per run\n(config, seeds, hashes,\nrevisions, device) + SQLite"),
          ("Caches", "detector scores and\ngenerations keyed by\nmodel, revision, device"),
          ("Results", "dashboard and paper read\nonly stored run files")]
    for i, (t, b) in enumerate(dd):
        box(ax, 0.35 + i * 2.4, 0.12, 2.25, 1.55, t, b, ec="#777777")
    save(fig, "Fig1")


def fig2_gate_flow():
    """Decision flow of the transparent rule Gate for one candidate rewrite."""
    fig, ax = plt.subplots(figsize=(174 * MM, 118 * MM))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    cx, w = 0.3, 4.2
    q = "#FFF6DD"
    box(ax, cx, 9.0, w, 0.8, "Original sentence + candidate rewrite", tsize=7.5)
    steps = [(7.45, "Is the input biased?", "mean detector score ≥ 0.5"),
             (5.75, "Are meaning and facts preserved?", "facts kept, entailment, no contradiction,\nsimilarity, length, fluency"),
             (4.05, "Did bias fall enough?", "bias reduction ≥ threshold, no residual bias"),
             (2.35, "Detectors agree and confident?", "detector spread, distance from 0.5")]
    outs = [("No", "ACCEPT the unchanged text (NEUTRAL_UNCHANGED)\nREJECT any change (UNNECESSARY_CHANGE)",
             "#E3F2EA", C["green"]),
            ("No", "REJECT\nFACT_LOSS · ENTAILMENT_FAIL · CONTRADICTION ·\nLOW_SIMILARITY · LENGTH · FLUENCY_FAIL",
             "#FBE5DA", C["red"]),
            ("No", "RETRY\nINSUFFICIENT_BIAS_REDUCTION · RESIDUAL_BIAS\nre-prompt with the facts to keep (max 2),\n"
                   "then HUMAN_REVIEW (RETRIES_EXHAUSTED)", "#FDF0D5", C["orange"]),
            ("No", "HUMAN_REVIEW\nDETECTOR_DISAGREEMENT · LOW_CONFIDENCE", "#ECECEC", "#666666")]
    arrow(ax, cx + w / 2, 9.0, cx + w / 2, 7.45 + 1.15 + 0.02)
    for (y, t, b), (lab, txt, fc, ec) in zip(steps, outs):
        box(ax, cx, y, w, 1.15, t, b, fc=q, ec="#B8860B")
        ox = 5.35
        box(ax, ox, y + 0.05, 4.4, 1.05, txt, fc=fc, ec=ec, tsize=6.3)
        arrow(ax, cx + w + 0.02, y + 0.57, ox - 0.02, y + 0.57)
        ax.text((cx + w + ox) / 2, y + 0.67, lab, ha="center", fontsize=6.5)
        ax.text(cx + w / 2 + 0.1, y - 0.25, "Yes", fontsize=6.5)
        arrow(ax, cx + w / 2, y - 0.02, cx + w / 2, y - 0.45 if y > 2.4 else 1.32)
    box(ax, cx, 0.3, w, 1.0, "ACCEPT (ALL_CHECKS_PASSED)\nbest accepted candidate is returned",
        fc="#E3F2EA", ec=C["green"], tsize=7)
    ax.text(5.35, 0.75, "Precedence when candidates disagree:\nREJECT > RETRY > HUMAN_REVIEW > ACCEPT",
            fontsize=6.5, va="center")
    save(fig, "Fig2")


def fig6_detection():
    d = load(DETECTION)["detectors"]
    order = ["rule_based", "bert", "deberta", "bart_mnli", "hub_pretrained"]
    names = {"rule_based": "Rule-based", "bert": "DistilBERT (fine-tuned)", "deberta": "DeBERTa-v3-small (fine-tuned)",
             "bart_mnli": "BART-MNLI (zero-shot)", "hub_pretrained": "Public RoBERTa (trained on BABE)"}
    sources = [("babe", "BABE"), ("crows_pairs", "CrowS-Pairs"), ("stereoset_inter", "StereoSet inter"),
               ("stereoset_intra", "StereoSet intra")]
    colours = [C["grey"], C["sky"], C["blue"], C["orange"], C["purple"]]
    hatches = ["", "//", "", "..", "xx"]
    fig, ax = plt.subplots(figsize=(174 * MM, 70 * MM))
    bw = 0.16
    for j, det in enumerate(order):
        vals = [d[det]["by_source"][s]["macro_f1"] for s, _ in sources]
        xs = [i + (j - 2) * bw for i in range(len(sources))]
        ax.bar(xs, vals, bw, label=names[det], color=colours[j], hatch=hatches[j], edgecolor="black", lw=0.4)
    ax.set_xticks(range(len(sources)), [s for _, s in sources])
    ax.set_ylabel("Macro-F1 (test split)")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="black", lw=0.5, ls=":")
    ax.legend(ncol=3, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    save(fig, "Fig6")


def fig8_selection():
    sel = load(GATE)["selection"]
    order = [("lowest_bias", "Lowest bias score"), ("highest_similarity", "Highest similarity"),
             ("no_change", "Keep original"), ("llm_judge", "LLM-as-judge"),
             ("fairlens_gate_tuned", "Rule Gate (val-tuned)"), ("learned_tree", "Learned gate (tree)")]
    parts = [("ACCEPT", "Acceptable choice", C["green"], ""),
             ("RETRY", "Kept biased original", C["orange"], "//"),
             ("REJECT", "Damaging rewrite chosen", C["red"], "xx"),
             ("DEFER", "Deferred to a person", C["grey"], "..")]
    fig, ax = plt.subplots(figsize=(174 * MM, 75 * MM))
    for i, (key, _) in enumerate(order):
        counts = sel[key]["chosen_gold_counts"]
        n = sum(counts.values())
        left = 0.0
        for gold, label, colour, hatch in parts:
            v = counts.get(gold, 0) / n
            ax.barh(i, v, left=left, color=colour, hatch=hatch, edgecolor="black", lw=0.4,
                    label=label if i == 0 else None)
            if v >= 0.06:
                ax.text(left + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=7,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
            left += v
    ax.set_yticks(range(len(order)), [n for _, n in order])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of test inputs (n = 204)")
    ax.legend(ncol=4, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.45, 1.16))
    save(fig, "Fig8")


def fig7_mitigation():
    r = load(MITIGATION)["rewriters"]
    names = {"rule_based": "Rule rewriter", "flan_t5": "FLAN-T5-base", "qwen": "Qwen2.5-0.5B", "original": "No change"}
    marks = {"rule_based": ("s", C["green"]), "flan_t5": ("^", C["orange"]), "qwen": ("o", C["red"]),
             "original": ("D", C["grey"])}
    fig, ax = plt.subplots(figsize=(84 * MM, 70 * MM))
    for k, label in names.items():
        x, y = r[k]["mean_bias_reduction_biased"], r[k]["fact_failure_rate"]
        m, c = marks[k]
        ax.scatter(x, y, marker=m, s=40, color=c, edgecolor="black", lw=0.5, zorder=3)
        ax.annotate(label, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Mean bias-score reduction (biased inputs)")
    ax.set_ylabel("Share of rewrites losing a fact")
    ax.set_xlim(-0.005, 0.11)
    ax.set_ylim(-0.02, 0.5)
    ax.grid(lw=0.3, alpha=0.5)
    save(fig, "Fig7")

def copy_screenshots():
    """Fig3-5 are screenshots of the running dashboard (paper/screenshots/).

    Fig4 stacks two frames from the demo recording: (a) Mitigation, (b) Reliability Gate.
    """
    import shutil

    from PIL import Image, ImageDraw, ImageFont

    shots = ROOT / "paper" / "screenshots"
    shutil.copyfile(shots / "fairlens_review_page.png", OUT / "Fig3.png")
    # Headline-tile row removed (pixels shifted, nothing edited): the tiles showed superseded gate numbers.
    shutil.copyfile(shots / "benchmarks_page_no_tiles.png", OUT / "Fig5.png")
    # Keep only the informative part of each frame (rewrite cards; verdict and metrics), not the page header.
    parts = [Image.open(shots / "mitigation_page.png").convert("RGB").crop((0, 285, 1040, 680)),
             Image.open(shots / "reliability_gate_page.png").convert("RGB").crop((0, 285, 1040, 775))]
    w, gap, pad = max(p.width for p in parts), 24, 44
    sheet = Image.new("RGB", (w, sum(p.height for p in parts) + gap + 2 * pad), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arialbd.ttf", 30)
    except OSError:
        font = ImageFont.load_default()
    y = 0
    for label, p in zip("ab", parts):
        draw.text((4, y + 4), label, fill="black", font=font)
        sheet.paste(p, (0, y + pad))
        y += pad + p.height + gap
    sheet.save(OUT / "Fig4.png", dpi=(300, 300))
    for n in (3, 4, 5):
        Image.open(OUT / f"Fig{n}.png").convert("RGB").save(OUT / f"Fig{n}.tif", compression="tiff_lzw")


if __name__ == "__main__":
    fig1_architecture()
    fig2_gate_flow()
    copy_screenshots()
    fig6_detection()
    fig7_mitigation()
    fig8_selection()
    print("figures written to", OUT)
