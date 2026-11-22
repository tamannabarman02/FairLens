"""Build the PP2 presentation from the department template with PowerPoint (COM automation).

Template: D:\\Downloads\\PP2-Presentation-Format (1) (1).pptx   ->   paper/FairLens_PP2_Presentation.pptx
Numbers are the verified ones used in the paper/report; demo values come from the live run (paper/figures/demo_*).
Red text = still to be filled by the team. Run with the Anaconda interpreter (needs pywin32 and PowerPoint):
    C:\\Users\\taman\\Anaconda3\\python.exe paper\\build_presentation.py
"""
import os
import shutil
from pathlib import Path

import win32com.client
from PIL import Image

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
TEMPLATE = Path(r"D:\Downloads\PP2-Presentation-Format (1) (1).pptx")
OUT = HERE / "FairLens_PP2_Presentation.pptx"
TMP = HERE / "_ppt_images"
TMP.mkdir(exist_ok=True)

RED = 0x0000FF          # BGR for COM
DARK = 0x3A2A1F
ACCENT = 0xD9466F       # purple-ish (BGR of 6F46D9)
GREEN = 0x5E9A1F
FOOTER = "FairLens                         Department of CSE, BMSCE"

# ---------- image preparation ----------
def crop(src, box, name):
    im = Image.open(src)
    w, h = im.size
    l, t, r, b = box
    im.crop((l, t, r if r > 0 else w + r, b if b > 0 else h + b)).save(TMP / name)
    return TMP / name

bench_top = crop(FIG / "Fig5_benchmarks_full.png", (0, 268, 0, 1230), "bench_top.png")
bench_table = crop(FIG / "Fig5_benchmarks_full.png", (0, 1230, 0, 2180), "bench_table.png")
demo_cards = crop(FIG / "demo_mitigation.png", (0, 300, 0, 1210), "demo_cards.png")
demo_det = FIG / "demo_detection.png"
demo_gate = crop(FIG / "demo_gate.png", (0, 420, 0, 1180), "demo_gate_top.png")

# ---------- open template ----------
shutil.copyfile(TEMPLATE, OUT)
app = win32com.client.Dispatch("PowerPoint.Application")
pres = app.Presentations.Open(str(OUT), False, False, False)
S = pres.Slides
base = S(6)                          # "Title and Content" slide used as the pattern
thank_you_id = S(15).SlideID
title_id, outline_id = S(1).SlideID, S(2).SlideID
TEMPLATE_IDS = [S(i).SlideID for i in range(3, 15)]


def shape(slide, prefix):
    for sh in slide.Shapes:
        if sh.Name.startswith(prefix):
            return sh
    return None


def new_slide(title, title_size=36):
    s = base.Duplicate()(1)
    s.MoveTo(pres.Slides.Count)
    t = shape(s, "Title").TextFrame.TextRange
    t.Text = title
    t.Font.Size = title_size
    t.Font.Bold = True
    return s


def bullets(slide, items, size=18, box=None):
    """items: str or (str, level). '**Label:** rest' makes Label bold. Empty str = spacer."""
    ph = shape(slide, "Content")
    if box:
        ph.Left, ph.Top, ph.Width, ph.Height = box
    tr = ph.TextFrame.TextRange
    norm = [(i, 1) if isinstance(i, str) else i for i in items]
    tr.Text = "\r".join(t.replace("**", "") for t, _ in norm)
    for k, (t, lvl) in enumerate(norm, 1):
        para = tr.Paragraphs(k)
        para.IndentLevel = lvl
        para.Font.Size = size if lvl == 1 else size - 2
        para.ParagraphFormat.SpaceAfter = 4
        if t.startswith("**") and "**" in t[2:]:
            label = t[2:t.index("**", 2)]
            para.Characters(1, len(label)).Font.Bold = True
        if "[FILL" in t:
            a = para.Text.find("[FILL")
            para.Characters(a + 1, para.Text.find("]", a) - a + 1).Font.Color.RGB = RED
    ph.TextFrame.AutoSize = 0
    return ph


def drop_content(slide):
    ph = shape(slide, "Content")
    if ph:
        ph.Delete()


def picture(slide, path, l, t, w, h, border=True):
    iw, ih = Image.open(path).size
    sc = min(w / iw, h / ih)
    pw, ph = iw * sc, ih * sc
    pic = slide.Shapes.AddPicture(str(path), False, True, l + (w - pw) / 2, t + (h - ph) / 2, pw, ph)
    if border:
        pic.Line.Visible = True
        pic.Line.ForeColor.RGB = 0xD8D8D8
        pic.Line.Weight = 0.75
    return pic


def table(slide, rows, l, t, w, size=12, col_w=None, bold_rows=(), red_cells=()):
    nr, nc = len(rows), len(rows[0])
    sh = slide.Shapes.AddTable(nr, nc, l, t, w, 20 * nr)
    tb = sh.Table
    if col_w:
        for j, cw in enumerate(col_w, 1):
            tb.Columns(j).Width = cw
    for i, row in enumerate(rows, 1):
        for j, val in enumerate(row, 1):
            cell = tb.Cell(i, j).Shape.TextFrame
            cell.MarginTop = cell.MarginBottom = 2
            r = cell.TextRange
            r.Text = str(val)
            r.Font.Size = size
            r.Font.Bold = (i == 1) or (i - 1 in bold_rows)
            if (i - 1, j - 1) in red_cells:
                r.Font.Color.RGB = RED
    return sh


def textbox(slide, text, l, t, w, h, size=14, bold=False, color=DARK, fill=None, align=1):
    tb = slide.Shapes.AddTextbox(1, l, t, w, h)
    tf = tb.TextFrame
    tf.WordWrap = True
    r = tf.TextRange
    r.Text = text
    r.Font.Size = size
    r.Font.Bold = bold
    r.Font.Color.RGB = color
    r.ParagraphFormat.Alignment = align
    if fill is not None:
        tb.Fill.Visible = True
        tb.Fill.ForeColor.RGB = fill
        tf.MarginLeft = tf.MarginRight = 8
        tf.MarginTop = tf.MarginBottom = 6
    return tb


def stat(slide, value, label, l, t, w, color):
    box = slide.Shapes.AddShape(5, l, t, w, 92)      # rounded rectangle
    box.Fill.ForeColor.RGB = 0xFBF5F7
    box.Line.ForeColor.RGB = 0xE6D9DE
    textbox(slide, value, l, t + 6, w, 40, size=30, bold=True, color=color, align=2)
    textbox(slide, label, l + 6, t + 46, w - 12, 44, size=12, color=DARK, align=2)


def notes(slide, text):
    slide.NotesPage.Shapes.Placeholders(2).TextFrame.TextRange.Text = text


# ======================= 1. Title =======================
s1 = S(1)
t = shape(s1, "Title")
t.Top, t.Height = 60, 150
tr = t.TextFrame.TextRange
tr.Text = ("FairLens: Automated Framework for Bias Detection, Mitigation and Fairness Evaluation "
           "in Large Language and Multimodal AI Systems")
tr.Font.Size = 30
tr.Font.Bold = True
team = table(s1, [["Suryansh Shahi", "Sushma B T", "Tamanna Barman", "Trisha L Salian"],
                  ["1BM23CS349", "1WA23CS002", "1WA23CS006", "1WA23CS022"]], 120, 222, 720, size=16)
team.Table.ApplyStyle("{2D5ABB26-0587-4C30-8999-92F81FD0307C}")   # "No Style, No Grid"
for j in range(1, 5):
    for i in (1, 2):
        team.Table.Cell(i, j).Shape.TextFrame.TextRange.ParagraphFormat.Alignment = 2
        team.Table.Cell(i, j).Shape.TextFrame.TextRange.Font.Bold = False
sub = shape(s1, "Subtitle")
sub.Top, sub.Height = 300, 150
r = sub.TextFrame.TextRange
r.Text = ("Under the guidance of\rProf. Muthurani N H\rAssistant Professor\r"
          "Department of Computer Science and Engineering\rB.M.S. College of Engineering")
r.Font.Size = 18
r.Paragraphs(1).Font.Size = 14
r.Paragraphs(2).Font.Bold = True
sub.TextFrame.TextRange.ParagraphFormat.Alignment = 2

# ======================= 2. Outline =======================
s2 = S(2)
bullets(s2, ["Problem Statement", "Sustainable Development Goals and Collaboration", "Objectives",
             "High Level Design", "Detailed Design", "Implementation",
             "Datasets and Results (tables, graphs)", "Snapshots and Live Demo",
             "Conclusion and Future Enhancements", "Paper Published", "References"], size=16)

# ======================= Problem statement =======================
s = new_slide("Problem Statement")
bullets(s, [
    "LLM tools are now used to \u201cmake text inclusive\u201d, but a rewrite is judged only by whether a bias "
    "detector's score drops",
    "A rewrite can lower that score while deleting a requirement, changing a number or reversing a negation: "
    "the fairness fix itself causes harm",
    "Recruitment AI is classed as high-risk by the EU AI Act and must support human oversight",
], size=17, box=(66, 130, 828, 190))
textbox(s, "Original:  \u201cLooking for a young, aggressive salesman with 5+ years of B2B experience.\u201d\r"
           "Rewrite:   \u201cLooking for a motivated sales professional with B2B experience.\u201d\r"
           "\u2192 Bias wording removed, but the 5-year requirement is silently lost",
        66, 320, 828, 80, size=14, fill=0xF7F2F4)
textbox(s, "Problem: design and evaluate a framework that detects bias in text and image captions, generates neutral "
           "rewrites, and verifies every rewrite with a transparent Reliability Gate before it is accepted.",
        66, 412, 828, 70, size=15, bold=True, color=ACCENT)

# ======================= SDGs =======================
s = new_slide("Sustainable Development Goals (SDGs) Addressed", 32)
drop_content(s)
table(s, [["SDG", "Target", "Justification"],
          ["SDG 5: Gender Equality", "5.1 End discrimination against women",
           "Detects gender-coded wording (\u201csalesman\u201d, \u201cwaitress\u201d) and proposes checked neutral alternatives"],
          ["SDG 8: Decent Work", "8.5 Equal employment opportunity",
           "FairLens Review targets job ads; the Gate ensures requirements (experience, location) are not lost"],
          ["SDG 10: Reduced Inequalities", "10.3 Eliminate discriminatory practices",
           "Flags age, race, religion, disability and socioeconomic stereotypes; defers uncertain cases to a person"],
          ["SDG 9: Innovation", "9.5 Enhance scientific research",
           "Open, reproducible laptop-scale framework; every number traceable to a stored run"]],
      66, 140, 828, size=14, col_w=[190, 220, 418])

# ======================= Collaboration =======================
s = new_slide("Collaboration")
bullets(s, [
    "**Category:** Community Oriented and Multidisciplinary Project",
    ("Community: fairer job advertisements and news text for job seekers and readers", 2),
    ("Multidisciplinary: NLP and deep learning + AI ethics + recruitment / HR practice + AI regulation", 2),
    "**Field work / collaboration:** [FILL: organisation, contact person and nature of collaboration, if any]",
    "**Planned use:** review of real job descriptions by an HR team, with FairLens as a review assistant "
    "(a person makes the final decision)",
], size=18)

# ======================= Objectives =======================
s = new_slide("Objectives")
bullets(s, [
    "**O1:** Compare rule-based, fine-tuned and zero-shot bias detectors on the same held-out data, "
    "including false alarms on neutral text",
    "**O2:** Compare a rule-based rewriter with two LLM rewriters on bias reduction and on preservation of facts "
    "and meaning",
    "**O3:** Design a transparent Reliability Gate (Accept / Retry / Reject / Human review) and evaluate it against "
    "simpler selection policies with ablations",
    "**O4:** Extend the same detection pipeline to captions generated for images",
    "**O5:** Deliver a reproducible, local application in which every result is traceable to a stored run",
], size=17)

# ======================= High level design =======================
s = new_slide("High Level Design")
drop_content(s)
picture(s, FIG / "Fig1.png", 120, 112, 720, 380)
notes(s, "Five layers: dashboard; three shared pipeline functions; core services (ingestion, detection, mitigation, "
         "Reliability Gate, decision); model layer; data layer. Dashed arrow = strict retry.")

# ======================= Detailed design: gate flow =======================
s = new_slide("Detailed Design: Reliability Gate")
drop_content(s)
picture(s, FIG / "Fig2.png", 66, 112, 560, 380)
bullets_box = textbox(s, "", 640, 120, 260, 360, size=13)
bullets_box.TextFrame.TextRange.Text = (
    "Signals per candidate rewrite\r"
    "\u2022 Bias reduction (mean of 4 detectors)\r"
    "\u2022 Semantic similarity (MiniLM)\r"
    "\u2022 Entailment / contradiction (NLI)\r"
    "\u2022 Fact retention: names, numbers, dates, roles, quotes, negations\r"
    "\u2022 Detector agreement, fluency, length\r"
    "Tuned thresholds\r"
    "sim \u2265 0.60 \u00b7 entail \u2265 0.70 \u00b7 \u0394bias \u2265 0.05 \u00b7 facts = 100%")
tr = bullets_box.TextFrame.TextRange
tr.Paragraphs(1).Font.Bold = True
tr.Paragraphs(7).Font.Bold = True

# ======================= Detailed design: components =======================
s = new_slide("Detailed Design: Components")
drop_content(s)
table(s, [["Component", "Models / method"],
          ["Detection", "Rule lexicon \u00b7 DistilBERT & DeBERTa-v3-small (fine-tuned) \u00b7 BART-large-MNLI (zero-shot)"],
          ["Explanation", "Lexicon matches; leave-one-word-out occlusion for transformer detectors"],
          ["Mitigation", "Rule rewriter \u00b7 FLAN-T5-base \u00b7 Qwen2.5-0.5B-Instruct (shared prompt, greedy, seeded)"],
          ["Preservation", "all-MiniLM-L6-v2 similarity \u00b7 NLI cross-encoder \u00b7 spaCy + rules fact extractor"],
          ["Gate", "Ordered transparent rules with reason codes; strict retry (max 2); learned gates as baselines"],
          ["Images", "BLIP-base & Florence-2 captions \u00b7 CLIP ViT-B/32 alignment \u00b7 caption checks"],
          ["Tracking", "Run manifests (config, seeds, dataset hashes, pinned model revisions) + SQLite index"]],
      66, 140, 828, size=14, col_w=[160, 668])

# ======================= Implementation =======================
s = new_slide("Implementation")
bullets(s, [
    "**Stack:** Python 3.11 \u00b7 PyTorch 2.14.1 \u00b7 Transformers 5.20.2 \u00b7 spaCy 3.8.16 \u00b7 scikit-learn \u00b7 "
    "Streamlit 1.66.1 \u00b7 Plotly",
    "**Runs locally:** CPU, Apple MPS or CUDA; lite / standard / GPU profiles; one large model in memory at a time",
    "**Inputs:** text, TXT, PDF, DOCX, CSV, JPG / PNG / GIF (10 MB, 20,000 characters)",
    "**Dashboard:** 12 pages \u2014 guided 6-step FairLens Review, Text Analysis, Mitigation, Reliability Gate, "
    "Image Analysis, Benchmarks, Model Comparison, Error Analysis, Experiments",
    "**Honesty by design:** research pages read only stored runs (enforced by a test); exports to PDF / JSON",
    "**Testing:** 246 automated tests (245 pass in the project env; 1 passes under Anaconda due to a Windows policy)",
], size=15, box=(66, 120, 470, 370))
picture(s, FIG / "Fig3.png", 548, 140, 360, 300)
textbox(s, "FairLens Review page", 548, 440, 360, 24, size=11, align=2)

# ======================= Datasets =======================
s = new_slide("Datasets Used")
drop_content(s)
table(s, [["Dataset", "Used for", "Size used", "Labels"],
          ["BABE", "Detection", "4,118 sentences", "Expert-labelled media bias"],
          ["StereoSet", "Detection, probe", "8,440 sentences", "Stereotype = 1, anti-stereotype = 0"],
          ["CrowS-Pairs", "Detection, probe", "2,578 sentences", "More stereotypical sentence = 1"],
          ["WinoBias", "False-alarm probe", "1,584 sentences", "All neutral"],
          ["Authored cases (ours)", "Mitigation, Gate", "136 sentences", "91 biased, 45 neutral + reference rewrites"],
          ["Mitigation set", "Mitigation", "276 inputs", "211 biased, 65 neutral"],
          ["Controlled gate set", "Gate", "2,481 candidates", "Gold ACCEPT / RETRY / REJECT"],
          ["COCO val2017 subset", "Captions", "80 images", "60 with people, 5 human captions each"]],
      66, 130, 828, size=13, col_w=[190, 160, 160, 318])
textbox(s, "Detection splits 70/15/15 by group (seed 42): 10,365 train \u00b7 2,119 validation \u00b7 2,652 test",
        66, 450, 828, 26, size=13, color=ACCENT)

# ======================= Results: detection =======================
s = new_slide("Results: Bias Detection")
drop_content(s)
table(s, [["Detector", "Macro-F1 [95% CI]", "ROC-AUC", "WinoBias FPR"],
          ["Rule-based", "0.354 [0.342, 0.369]", "0.515", "0.000"],
          ["DistilBERT (fine-tuned)", "0.777 [0.762, 0.791]", "0.856", "0.441"],
          ["DeBERTa-v3-small (fine-tuned)", "0.814 [0.800, 0.828]", "0.896", "0.396"],
          ["BART-MNLI (zero-shot)", "0.611 [0.593, 0.629]", "0.648", "0.417"],
          ["Public RoBERTa (BABE)", "0.646 [0.627, 0.661]", "0.722", "0.054"]],
      40, 130, 470, size=12, col_w=[170, 140, 70, 90], bold_rows=(3,))
picture(s, FIG / "Fig6.png", 520, 120, 410, 250)
textbox(s, "\u2022 DeBERTa is best and consistent across sources (n = 2,652 test sentences)\r"
           "\u2022 But it flags ~40% of neutral WinoBias sentences \u2192 high accuracy \u2260 few false alarms\r"
           "\u2022 The BABE-trained public model fails on stereotype data (\u2264 0.544)",
        40, 370, 880, 110, size=14)

# ======================= Results: rewriting =======================
s = new_slide("Results: Rewriting \u2014 Bias Reduction vs Preservation", 30)
drop_content(s)
table(s, [["Rewriter", "Bias reduction", "Fact failures", "Unnecessary change"],
          ["Rule rewriter", "0.074", "1.8%", "0.0%"],
          ["FLAN-T5-base", "0.018", "9.4%", "20.0%"],
          ["Qwen2.5-0.5B", "0.087", "41.7%", "98.5%"]],
      40, 135, 480, size=14, col_w=[140, 110, 110, 120])
picture(s, FIG / "Fig7.png", 540, 120, 380, 300)
textbox(s, "\u2022 The rewriter that lowers bias most (Qwen2.5) loses a fact in 41.7% of rewrites\r"
           "\u2022 The rule rewriter gets similar reduction with only 1.8% fact failures\r"
           "\u2022 On real rewrites the Gate accepted 0% with fact loss (lowest-bias selection: 25.4%), "
           "at the cost of 47.5% deferred to a person",
        40, 270, 490, 200, size=14)

# ======================= Results: gate =======================
s = new_slide("Results: Reliability Gate (Known Correct Decisions)", 30)
drop_content(s)
stat(s, "96.1%", "lowest-bias selection chose a damaging rewrite", 66, 118, 250, RED)
stat(s, "0%", "damaging rewrites chosen by the Rule Gate", 355, 118, 250, GREEN)
stat(s, "23.5%", "inputs deferred to human review (the cost)", 644, 118, 250, ACCENT)
picture(s, FIG / "Fig8.png", 66, 222, 828, 210)
textbox(s, "204 test inputs with gold decisions. Learned decision tree: most accurate (76.0%) but chose a damaging "
           "rewrite for 13.2% of inputs and is less transparent.",
        66, 436, 828, 50, size=13)

# ======================= Results: ablation =======================
s = new_slide("Results: Which Gate Signals Matter")
drop_content(s)
table(s, [["Signal removed", "Unsafe-accept rate", "\u0394 vs full Gate [95% CI]"],
          ["None (full Gate)", "0.071", "\u2013"],
          ["Bias reduction", "0.133", "+0.062 [+0.048, +0.076]"],
          ["Entailment", "0.127", "+0.056 [+0.042, +0.072]"],
          ["Fact retention", "0.077", "+0.006 [+0.002, +0.011]"],
          ["Semantic similarity", "0.074", "+0.003 [+0.000, +0.007]"],
          ["Agreement / uncertainty / fluency", "0.071", "+0.000"]],
      66, 135, 520, size=13, col_w=[200, 130, 190])
textbox(s, "\u2022 Bias-reduction and entailment checks matter most\r\r"
           "\u2022 Agreement, uncertainty and fluency: no measurable effect (reported as a negative result)\r\r"
           "\u2022 With gold input labels the unsafe-accept rate falls from 7.1% to 0.3%: remaining errors come from "
           "detection, not from the Gate",
        606, 135, 300, 330, size=14)

# ======================= Image captions =======================
s = new_slide("Results: Image Captions")
bullets(s, [
    "80 COCO validation images (60 showing people), captions by BLIP and Florence-2",
    "**Alignment (CLIP):** Florence-2 0.306 \u2248 human references 0.304; BLIP 0.281",
    "**Gendered terms** in captions of people: BLIP 53.3%, Florence-2 60.0% (human references 73.3%)",
    "**Unverifiable attributes** (nationality, wealth, religion): none found",
    "Text detectors flagged 47.5% / 68.8% of captions \u2014 not read as bias rates, given the ~40% false-alarm "
    "rate on neutral text",
], size=18)

# ======================= Snapshots =======================
s = new_slide("Snapshots: Benchmarks Page")
drop_content(s)
picture(s, bench_top, 66, 110, 440, 380)
picture(s, bench_table, 520, 110, 400, 380)

s = new_slide("Live Demo: Input and Detection")
drop_content(s)
textbox(s, "Input: \u201cWe are looking for a young, energetic waitress to join our Bengaluru cafe. She must be "
           "available 6 days a week and have 2 years of experience.\u201d",
        66, 112, 828, 46, size=14, fill=0xF7F2F4)
picture(s, demo_det, 66, 165, 520, 320)
textbox(s, "Scores: DistilBERT 0.51 \u00b7 DeBERTa 0.86 \u00b7 BART-MNLI 0.98 \u00b7 Rule 0.55\r"
           "Overall: 0.72 (high bias)\r\r"
           "Explanations:\r\u2022 Gender: \u201cwaitress\u201d\r\u2022 Age: \u201cyoung\u201d",
        600, 180, 300, 250, size=14)

s = new_slide("Live Demo: Rewrites and Gate Verdicts")
drop_content(s)
picture(s, demo_cards, 66, 108, 560, 380)
textbox(s, "\u2022 Rule-based \u2192 ACCEPT\r  bias 0.561 \u2192 0.300, facts 100%, sim 0.77, entail 0.98\r\r"
           "\u2022 FLAN-T5 \u2192 RETRY (copied input)\r\r"
           "\u2022 Qwen2.5 \u2192 REJECT\r  added \u201c21-30 years old\u201d, changed \u201c6 days\u201d to "
           "\u201cMonday to Friday\u201d, facts 67%",
        640, 120, 270, 360, size=13)

s = new_slide("Live Demo: Reliability Gate Decision")
drop_content(s)
picture(s, demo_gate, 66, 110, 828, 330)
textbox(s, "Accepted rewrite: \u201cWe are looking for an energetic server to join our Bengaluru cafe. She must be available "
           "6 days a week and have 2 years of experience.\u201d  (reason code ALL_CHECKS_PASSED; illustrative run)",
        66, 444, 828, 46, size=12)

# ======================= Conclusion =======================
s = new_slide("Conclusion")
bullets(s, [
    "FairLens detects bias in text and image captions, proposes neutral rewrites and checks every rewrite before "
    "acceptance, on a laptop, with every number traceable to a stored run",
    "Choosing the least-biased rewrite was almost always harmful (96.1% damaging on the controlled set)",
    "The transparent Reliability Gate never chose a damaging rewrite, at the cost of 23.5% deferrals and 33.8% "
    "biased originals kept",
    "A learned gate was more accurate but less safe and less explainable",
    "Mitigation tools should be judged on what they preserve and when they abstain, not only on bias removed",
], size=17)

s = new_slide("Future Enhancements")
bullets(s, [
    "Blinded human evaluation of rewrites and Gate decisions with several annotators",
    "Repeated training with different seeds; verification on a second device",
    "Larger set of real job advertisements with domain-specific fact definitions",
    "Hybrid gate: keep hard preservation checks and reason codes, learn the thresholds",
    "Red-team tests on real accepted rewrites; larger local rewriters",
    "Multilingual detection, including Indian languages",
], size=18)

s = new_slide("Paper Published")
bullets(s, [
    "**Title:** FairLens: Automated Framework for Bias Detection, Mitigation and Fairness Evaluation in Large "
    "Language and Multimodal AI Systems",
    "**Authors:** Muthurani N H, Suryansh Shahi, Sushma B T, Tamanna Barman, Trisha L Salian",
    "**Journal:** AI and Ethics (Springer Nature)",
    "**Status:** [FILL: Submitted / Under review / Accepted]",
    "**Manuscript ID / date:** [FILL]",
], size=18)

s = new_slide("References")
bullets(s, [
    "[1] D. Gaucher et al., \u201cEvidence that gendered wording in job advertisements exists and sustains gender "
    "inequality,\u201d J. Pers. Soc. Psychol., 2011.",
    "[2] S. L. Blodgett et al., \u201cLanguage (technology) is power: A critical survey of \u2018bias\u2019 in NLP,\u201d ACL, 2020.",
    "[3] T. Spinde et al., \u201cNeural media bias detection using distant supervision with BABE,\u201d Findings of EMNLP, 2021.",
    "[4] M. Nadeem et al., \u201cStereoSet: Measuring stereotypical bias in pretrained language models,\u201d ACL, 2021.",
    "[5] N. Nangia et al., \u201cCrowS-Pairs: A challenge dataset for measuring social biases,\u201d EMNLP, 2020.",
    "[6] R. Pryzant et al., \u201cAutomatically neutralizing subjective bias in text,\u201d AAAI, 2020.",
    "[7] P. Laban et al., \u201cSummaC: NLI-based models for inconsistency detection in summarization,\u201d TACL, 2022.",
    "[8] L. Zheng et al., \u201cJudging LLM-as-a-judge with MT-Bench and Chatbot Arena,\u201d NeurIPS, 2023.",
    "[9] P. He et al., \u201cDeBERTaV3,\u201d ICLR, 2023;  [10] Qwen Team, \u201cQwen2.5 technical report,\u201d arXiv, 2024.",
    "[11] Regulation (EU) 2024/1689 (Artificial Intelligence Act), Official J. Eur. Union, 2024.",
], size=12)

# ======================= cleanup and finishing =======================
# delete the original template slides 3..14 (instruction note and empty pattern slides)
for sid in TEMPLATE_IDS:
    pres.Slides.FindBySlideID(sid).Delete()
pres.Slides.FindBySlideID(thank_you_id).MoveTo(pres.Slides.Count)

for sl in pres.Slides:
    f = shape(sl, "Footer")
    if f:
        f.TextFrame.TextRange.Text = FOOTER
pres.Save()
print("slides:", pres.Slides.Count)
pres.Close()
print("saved", OUT)
