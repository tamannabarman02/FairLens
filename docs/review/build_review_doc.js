const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell, WidthType,
  ShadingType, BorderStyle, LevelFormat, ImageRun, PageBreak, Footer, Header, PageNumber, TableOfContents,
} = require("docx");

const ACCENT = "5B3FD9", GREY = "6B6A80", LIGHT = "F0EEFC", W = 9026; // A4 with 1" margins

const t = (text, o = {}) => new TextRun({ text, ...o });
const p = (runs, o = {}) => new Paragraph({ spacing: { after: 120, line: 300 }, ...o,
  children: (Array.isArray(runs) ? runs : [runs]).map(r => typeof r === "string" ? t(r) : r) });
const h1 = (s) => new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true, spacing: { before: 0, after: 200 }, children: [t(s)] });
const h2 = (s) => new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 280, after: 120 }, children: [t(s)] });
const bullet = (runs, level = 0) => new Paragraph({ numbering: { reference: "bullets", level }, spacing: { after: 80, line: 290 },
  children: (Array.isArray(runs) ? runs : [runs]).map(r => typeof r === "string" ? t(r) : r) });
let LIST = 1;
const newList = () => { LIST += 1; return null; };
const num = (runs) => new Paragraph({ numbering: { reference: "numbers", level: 0, instance: LIST }, spacing: { after: 80, line: 290 },
  children: (Array.isArray(runs) ? runs : [runs]).map(r => typeof r === "string" ? t(r) : r) });
const b = (s) => t(s, { bold: true });
const note = (s) => new Paragraph({ spacing: { before: 60, after: 160 }, shading: { type: ShadingType.CLEAR, fill: LIGHT, color: "auto" },
  border: { left: { style: BorderStyle.SINGLE, size: 18, color: ACCENT, space: 8 } }, indent: { left: 160 },
  children: [t(s, { size: 20 })] });
const caption = (s) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [t(s, { italics: true, size: 18, color: GREY })] });

function table(headers, rows, widths, opts = {}) {
  const total = widths.reduce((a, c) => a + c, 0);
  const scale = W / total;
  const ws = widths.map(x => Math.round(x * scale));
  ws[ws.length - 1] += W - ws.reduce((a, c) => a + c, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: "D9D6EE" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const cell = (text, i, head, hl) => new TableCell({
    width: { size: ws[i], type: WidthType.DXA }, borders,
    shading: head ? { type: ShadingType.CLEAR, fill: ACCENT, color: "auto" } : hl ? { type: ShadingType.CLEAR, fill: "EAF7EF", color: "auto" } : undefined,
    margins: { top: 60, bottom: 60, left: 90, right: 90 },
    children: String(text).split("\n").map(line => new Paragraph({ spacing: { after: 0, line: 260 },
      children: [t(line, { size: opts.size || 18, bold: head || (hl && i === 0), color: head ? "FFFFFF" : undefined })] })),
  });
  return new Table({
    width: { size: W, type: WidthType.DXA }, columnWidths: ws,
    rows: [new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, i, true)) }),
      ...rows.map((r, ri) => new TableRow({ children: r.map((c, i) => cell(c, i, false, (opts.highlight || []).includes(ri))) }))],
  });
}
const gap = () => new Paragraph({ spacing: { after: 80 }, children: [] });
const img = (file, w = 600) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 },
  children: [new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width: w, height: Math.round(w * 1350 / 2160) } })] });

const S = "D:/FairLens_cache/docbuild/shots/";
const children = [];

// ---------- title page
children.push(
  new Paragraph({ spacing: { before: 2600, after: 120 }, children: [t("FINAL-YEAR CSE MAJOR PROJECT · SECOND REVIEW", { bold: true, color: ACCENT, size: 22 })] }),
  new Paragraph({ spacing: { after: 160 }, children: [t("FairLens", { bold: true, size: 72 })] }),
  new Paragraph({ spacing: { after: 400 }, children: [t("Bias detection, bias mitigation and a Reliability Gate that checks every rewrite before it is accepted", { size: 30, color: "333344" })] }),
  p([t("Detect · Reduce · Verify", { italics: true, color: ACCENT, size: 26 })]),
  new Paragraph({ spacing: { before: 1200, after: 60 }, children: [t("Team: ", { bold: true }), t("[names and roll numbers]")] }),
  p([t("Guide: ", { bold: true }), t("[guide name]")]),
  p([t("Date: ", { bold: true }), t("October 2026")]),
  new Paragraph({ spacing: { before: 600 }, children: [t("Every number in this document comes from a stored experiment run (outputs/runs/<run_id>/) and is named with its run ID. Nothing is estimated or typed in by hand.", { italics: true, size: 18, color: GREY })] }),
);

// ---------- contents
children.push(new Paragraph({ pageBreakBefore: true, heading: HeadingLevel.HEADING_1, children: [t("Contents")] }),
  new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
  p([t("(In Word: right-click the table and choose Update Field if page numbers are missing.)", { italics: true, size: 18, color: GREY })]));

// ---------- 1 summary
children.push(h1("1. Project at a glance"),
  p("FairLens is a review assistant for biased language. It finds biased or stereotyped wording in text (and in image captions), proposes neutral rewrites, and then checks every rewrite with a Reliability Gate before anything is accepted. The Gate exists because a rewrite can remove the biased words while silently changing the facts: dropping a requirement such as \"5 years of experience\", flipping a negation, or swapping who did what."),
  table(["Part", "What it does"], [
    ["Detect", "Four detectors score every sentence: fine-tuned DistilBERT, fine-tuned DeBERTa-v3-small, zero-shot BART-MNLI and a transparent rule baseline (plus a public RoBERTa model as an external reference)."],
    ["Reduce", "Three rewriters propose neutral versions: a rule-based rewriter, FLAN-T5-base and Qwen2.5-0.5B-Instruct, with the same prompt family and settings."],
    ["Verify", "The Reliability Gate measures each rewrite (bias drop, meaning, two-way entailment, facts, detector agreement, fluency) and decides Accept, Retry, Reject or Human review, with a reason code."],
    ["Images", "BLIP and Florence-2 caption an image; CLIP checks alignment; the captions go through the same detectors."],
    ["App", "A 12-page Streamlit dashboard, including a six-step FairLens Review workflow for job adverts and research pages that show only stored results."],
  ], [1.4, 7]),
  h2("Headline results (all from stored runs)"),
  bullet([b("Best detector: "), t("fine-tuned DeBERTa-v3-small, macro-F1 0.814 (F1 0.824, 95% CI 0.808–0.838) on 2,652 held-out test sentences, consistent across all four data sources (0.811–0.817). Run a4a61f.")]),
  bullet([b("Picking the least-biased rewrite is unsafe: "), t("on the controlled gold set, lowest-bias selection makes a harmful choice 98.0% of the time, against 25.5% for the tuned Gate. Run 7565de.")]),
  bullet([b("On real LLM rewrites "), t("the Gate accepted 0% of rewrites that lost a fact or contradicted the original, against 25.4% and 8.3% for lowest-bias selection, at the cost of sending 47.5% of inputs to a person. Run 000f03.")]),
  bullet([b("LLM rewriters trade facts for bias reduction: "), t("Qwen2.5 lowers bias the most (0.087) but loses at least one fact in 41.7% of its rewrites and changes 98.5% of neutral inputs that needed no change. FLAN-T5 mostly copies its input.")]),
);

// ---------- 2 problem
children.push(h1("2. Problem statement and motivation"),
  p("Biased wording appears in job adverts, news, reports and AI-generated text. Research shows its effects: gender-coded wording in job adverts makes women less likely to apply to male-dominated roles (Gaucher, Friesen and Kay, 2011). Two kinds of tools exist today:"),
  bullet([b("Detectors "), t("flag biased sentences but do not fix them.")]),
  bullet([b("Rewriters "), t("(word-list tools and large language models) propose neutral wording but do not check what else they changed.")]),
  p("The gap is between the two. When an LLM is asked to \"make this inclusive\", it may also drop a requirement, change a number, invert a negation or reassign a role. If the rewrite is then judged only by whether a bias score went down, these errors pass unnoticed. In recruitment, a rewrite that silently drops \"5+ years of experience\" or a licence requirement causes real hiring problems."),
  note("Illustrative example (not a benchmark result): \"Looking for a young, aggressive salesman with 5+ years of B2B experience.\" Rewrite A: \"…a motivated sales professional with B2B experience\" removes the bias but loses the 5+ years. Rewrite B keeps it. A bias score alone cannot tell them apart; FairLens rejects A and accepts B."),
  h2("Research question"),
  p("Can a transparent, multi-signal Reliability Gate decide more safely than simple selection rules which rewrite of a biased sentence (if any) to accept?"),
  h2("Objectives"),
  newList(),
  num("Compare four bias detectors fairly on the same held-out data and metrics."),
  num("Compare rule-based and LLM rewriters on bias reduction and on fact and meaning preservation."),
  num("Design the Reliability Gate and evaluate it against simpler selection methods, with ablations of each signal."),
  num("Extend the same pipeline to image captions (BLIP vs Florence-2)."),
  num("Deliver a working, cross-platform application in which every displayed result is traceable to a stored run."),
);

// ---------- 3 existing systems
children.push(h1("3. Existing systems and how FairLens differs"),
  p("We reviewed the closest published work and tool categories (citations were checked against the publisher or arXiv pages). The table summarises what each does and the gap FairLens addresses."),
  table(["Category", "Examples", "What they do", "Gap FairLens addresses"], [
    ["Bias / media-bias classifiers", "BABE models (Spinde et al., 2021); public Hub classifiers", "Label a sentence biased or neutral", "Detection only: no rewrite, no check of rewrites; one model's view"],
    ["Inclusive-writing word lists", "Gender-decoder lists based on Gaucher et al. (2011); job-advert writing tools", "Flag listed words and suggest replacements", "Fixed lists miss implicit bias; replacements are not checked for meaning or lost requirements"],
    ["Bias neutralisation research", "Pryzant et al. 2020 (WNC); DEPEN, He et al. 2021; gender-neutral rewriting, Sun et al. 2021", "Generate neutral rewrites, evaluated mainly on bias and fluency", "No decision layer that refuses an unsafe rewrite or routes it to a person"],
    ["Detoxification", "ParaDetox (Logacheva et al., 2022); ParaDeHate (Yuan et al., 2025)", "Rewrite toxic text; content preservation is checked when building datasets", "Toxicity, not stereotypes; checks are part of dataset construction, not a run-time gate with reasons"],
    ["General-purpose LLMs", "Prompting an instruction model to \"make this inclusive\"", "Fluent rewrites", "No guarantee facts survive; our runs show Qwen2.5 loses facts in 41.7% of rewrites"],
    ["Automatic filtering of rewrites", "LGSA (Karri et al., 2025)", "Screen generated rewrites with similarity and attribute checks for data augmentation", "Closest idea; no per-fact checks, four-way decision or comparison against selection baselines"],
    ["LLM-as-a-judge", "Zheng et al. (2023)", "A model rates outputs", "Single opinion with known biases; in our runs it was a weaker selector than the Gate"],
  ], [1.5, 2, 1.9, 2.6], { size: 16 }),
  h2("What is new in FairLens"),
  p("Filtering rewrites with automatic checks is established practice, so FairLens does not claim that idea as new. Its contribution is the specific combination below and its evaluation. To our knowledge (from the search listed in docs/PRIOR_WORK.md), this combination has not been evaluated before for bias-mitigation rewrites."),
  newList(),
  num([b("Per-fact-type preservation as hard checks: "), t("names, numbers, dates, roles, quotations and negation are extracted from the original and must survive the rewrite, alongside semantic similarity and entailment in both directions.")]),
  num([b("A four-way decision with reasons: "), t("Accept, Retry with a stricter prompt that lists the facts to keep, Reject, or Human review, each with a reason code (for example FACT_LOSS, RETRIES_EXHAUSTED).")]),
  num([b("\"No change\" as a correct answer: "), t("if the input is not biased, any rewrite is rejected as an unnecessary change.")]),
  num([b("Detector agreement and confidence as reasons to defer "), t("to a person instead of accepting.")]),
  num([b("Evaluation against simpler selectors "), t("(lowest bias score, highest similarity, no change, always first model, random, a validated LLM judge) on a gold set with known answers, plus ablations of every signal with confidence intervals.")]),
  num([b("One pipeline for text and image captions, "), t("with a demo-ready, cross-platform app in which every number is traceable to a run ID.")]),
  note("What we do not claim: that FairLens guarantees unbiased output, that it is legally or compliance certified, that it is production-validated, or that the Gate beats every baseline. A learned decision-tree gate was more accurate than our rule Gate (Section 8)."),
);

// ---------- 4 architecture
children.push(h1("4. System architecture"),
  table(["Stage", "Input", "Processing", "Output"], [
    ["1. Ingestion", "Text, TXT, PDF, DOCX, CSV; JPG, PNG, GIF", "Format checks, size limits, text extraction (incl. DOCX tables), offset-preserving sentence splitting", "Clean sentences"],
    ["2. Bias detection", "Sentences", "DistilBERT, DeBERTa, BART-MNLI, rule baseline; per-sentence scores, highlighted phrases with explanations", "Bias scores and evidence"],
    ["3. Mitigation", "Biased sentence", "Rule rewriter, FLAN-T5, Qwen2.5 (same prompt family, greedy decoding, fixed seed)", "Candidate rewrites"],
    ["4. Reliability Gate", "Original + candidates", "Signals: bias drop, similarity, two-way entailment, contradiction, facts, agreement, confidence, fluency; transparent rules; strict retry", "Decision, chosen rewrite, reason code"],
    ["5. Output", "Decision", "Review page, diff view, fact-check grid, PDF/JSON/HTML export", "Report for a person"],
  ], [1.4, 1.6, 3.6, 1.7], { size: 17 }),
  gap(),
  table(["Module (src/fairlens/)", "Responsibility"], [
    ["ingestion/", "File loaders and sentence segmentation"],
    ["detection/", "Common detector interface (predict, predict_batch, explain, info); rule lexicon; Hugging Face classifiers; zero-shot NLI"],
    ["mitigation/", "Shared prompts, rule rewriter, FLAN-T5 and Qwen adapters"],
    ["evaluation/", "Fact checker, similarity, entailment, fluency, metrics (F1, ECE, bootstrap CIs)"],
    ["reliability_gate/", "Signals, rules, retry, learned gates, baselines, LLM judge"],
    ["multimodal/", "BLIP, Florence-2, CLIP alignment, caption checks"],
    ["experiments/", "Run tracking with manifests, score cache, batched runners"],
    ["ui/", "Reusable components and the 12 pages"],
  ], [2, 6]),
  h2("Engineering choices"),
  bullet([b("Cross-platform: "), t("one codebase and one set of commands for Windows, macOS and Linux; device chosen automatically (CUDA, then Apple MPS, then CPU) with an override.")]),
  bullet([b("Laptop-friendly: "), t("models load lazily and only one large model is kept in memory; a Lite mode uses smaller models.")]),
  bullet([b("Reproducible: "), t("every Hub model pinned to a commit; datasets rebuilt from scripts with fixed seeds and verified hashes; each run writes a manifest (models and revisions, device, seeds, dataset hashes, configuration).")]),
  bullet([b("Honest UI: "), t("results pages read only stored run files and label each result Completed run, Pending, Failed run or Illustrative example.")]),
  h2("Technology stack"),
  p("Python 3.11 · PyTorch 2.14 · Hugging Face Transformers 5.20 · sentence-transformers · spaCy 3.8 · scikit-learn · Streamlit 1.66 · Plotly · SQLite and JSON run manifests · GitHub Actions CI · Playwright (browser tests)."),
);

// ---------- 5 datasets & models
children.push(h1("5. Datasets and models"),
  p("Datasets were built by scripts/build_datasets.py with seed 42. Splits are made by group, so the two sentences of a minimal pair (or a BABE item) never appear in two splits; 0 groups cross splits."),
  table(["Dataset", "Licence", "Used for", "Size used", "Label meaning"], [
    ["BABE (Spinde et al., 2021)", "CC BY-NC-SA 4.0", "Detection", "4,118", "Expert-labelled media bias in news sentences"],
    ["StereoSet (Nadeem et al., 2021)", "CC BY-SA 4.0", "Detection + probe", "8,440", "Stereotype = 1, anti-stereotype = 0"],
    ["CrowS-Pairs (Nangia et al., 2020)", "CC BY-SA 4.0", "Detection + probe", "2,578", "More stereotypical sentence = 1"],
    ["WinoBias (Zhao et al., 2018)", "MIT", "False-positive probe only", "1,584", "All neutral; any flag is a false alarm"],
    ["Authored cases (this project)", "Project", "Mitigation + Gate gold set", "136", "91 biased + 45 neutral, with reference rewrites; hard cases with names, numbers, dates, negation, roles, quotes"],
    ["Mitigation set", "Derived", "Mitigation benchmark", "276", "Authored + held-out benchmark sentences (211 biased / 65 neutral)"],
    ["Gate controlled set", "Derived", "Gate benchmark", "2,481 candidates", "Gold Accept / Retry / Reject known by construction, 3 seeds"],
    ["COCO val2017 subset (Lin et al., 2014)", "CC BY 4.0 + Flickr", "Image benchmark", "80 images", "60 with people, 20 without; 5 human captions each"],
  ], [2.1, 1.3, 1.6, 1.1, 2.6], { size: 16 }),
  p([t("Detection split (0 / 1 labels): train 5,052 / 5,313; validation 1,014 / 1,105; test 1,268 / 1,384. WNC (Pryzant et al., 2020) could not be used: its host was unreachable and the mirror is gated; BABE substitutes.", { size: 19 })]),
  h2("Models"),
  table(["Role", "Model", "Status"], [
    ["Detector", "DistilBERT-base-uncased", "Fine-tuned by us (10,365 sentences, 3 epochs); best val macro-F1 0.768"],
    ["Detector", "DeBERTa-v3-small", "Fine-tuned by us (embedding layer frozen); best val macro-F1 0.818"],
    ["Detector", "BART-large-MNLI", "Pre-trained, zero-shot; hypothesis template chosen on validation"],
    ["Reference detector", "himel7/bias-detector (RoBERTa)", "Pre-trained by its author on BABE"],
    ["Baseline detector", "Rule lexicon", "Ours"],
    ["Rewriters", "FLAN-T5-base; Qwen2.5-0.5B-Instruct", "Pre-trained, prompted (prompt chosen on a train-split sample)"],
    ["Baseline rewriter", "Rule rewriter", "Ours"],
    ["Gate signals", "all-MiniLM-L6-v2; NLI DeBERTa-v3-small cross-encoder; spaCy + rules", "Pre-trained / ours"],
    ["Gate", "Rule Gate; logistic-regression and decision-tree gates", "Ours (thresholds tuned on validation only)"],
    ["Captioners", "BLIP-base; Florence-2-base (no remote code)", "Pre-trained"],
    ["Alignment", "CLIP ViT-B/32", "Pre-trained"],
  ], [1.6, 3, 3.6], { size: 17 }),
);

// ---------- 6 gate
children.push(h1("6. The Reliability Gate"),
  table(["Signal", "How it is measured"], [
    ["Bias reduction", "Mean of the detector scores on the original minus on the rewrite"],
    ["Semantic similarity", "Cosine similarity of MiniLM sentence embeddings"],
    ["Entailment / contradiction", "NLI cross-encoder in both directions (original ⇒ rewrite and rewrite ⇒ original)"],
    ["Fact retention", "Share of names, numbers, dates, roles, quotes and negation polarity that survive"],
    ["Detector agreement", "Spread of the detector scores on the rewrite"],
    ["Confidence", "Distance of the ensemble score from 0.5"],
    ["Fluency and length", "Repetition, leaked prompt text, unusual characters, length ratio"],
  ], [2.2, 6]),
  h2("Decision rules (in priority order)"),
  newList(),
  num([b("Input not biased "), t("(ensemble score below 0.5): keep the original (Accept, NO CHANGE); any rewrite is an unnecessary change (Reject).")]),
  num([b("Reject "), t("if a fact is lost, the texts contradict, similarity is too low, or fluency or length fail.")]),
  num([b("Retry "), t("if meaning is kept but the bias did not drop enough; the LLMs are re-prompted with a stricter prompt that lists the facts to keep, up to 2 times, then Human review.")]),
  num([b("Human review "), t("if the detectors disagree or are not confident.")]),
  num([b("Accept "), t("if every check passes; among several, the best mix of bias drop, similarity and fact retention wins.")]),
  p("Thresholds were tuned on the validation split of the gold set only and then frozen: similarity 0.70, forward entailment 0.30, minimum bias reduction 0.05."),
);

// ---------- 7 results
children.push(h1("7. Experiments and results"),
  h2("Experiment 1 · Text-bias detection (run 20261024-043227-…-a4a61f)"),
  p("All detectors on the same 2,652 held-out test sentences, threshold 0.5."),
  table(["Detector", "Precision", "Recall", "F1 [95% CI]", "Macro-F1", "ROC-AUC", "ms/sample"], [
    ["Rule-based", "0.557", "0.035", "0.067 [0.049, 0.086]", "0.354", "0.515", "0.1"],
    ["DistilBERT (fine-tuned)", "0.775", "0.810", "0.792 [0.776, 0.807]", "0.777", "0.856", "3.1"],
    ["DeBERTa-v3-small (fine-tuned)", "0.818", "0.829", "0.824 [0.808, 0.838]", "0.814", "0.896", "5.1"],
    ["BART-MNLI (zero-shot)", "0.620", "0.674", "0.646 [0.626, 0.666]", "0.611", "0.648", "47.5"],
    ["Hub RoBERTa (pre-trained)", "0.701", "0.564", "0.625 [0.603, 0.647]", "0.646", "0.722", "5.3"],
  ], [2.4, 1, 1, 1.9, 1, 1, 1], { size: 17, highlight: [2] }),
  p([t("False-alarm probe: WinoBias sentences are all neutral, yet the fine-tuned detectors flag about 40% of them (DeBERTa 0.396 / 0.376, DistilBERT 0.441 / 0.428). High benchmark accuracy does not mean few false alarms on everyday text.", { size: 19 })]),
  h2("Experiment 2 · Mitigation on real rewrites (run 20261024-040853-…-000f03)"),
  p("276 inputs (211 biased, 65 neutral). Automatic metrics; no human labels for model outputs yet."),
  table(["Rewriter", "Bias reduction [95% CI]", "Changed", "Similarity", "Fact retention", "Fact failures", "Changed neutral"], [
    ["Rule-based", "0.074 [0.056, 0.092]", "0.275", "0.899", "0.994", "0.018", "0.000"],
    ["FLAN-T5-base", "0.018 [0.008, 0.031]", "0.312", "0.901", "0.960", "0.094", "0.200"],
    ["Qwen2.5-0.5B", "0.087 [0.064, 0.110]", "0.993", "0.828", "0.801", "0.417", "0.985"],
  ], [1.6, 1.9, 1, 1, 1.2, 1.1, 1.2], { size: 17 }),
  gap(),
  table(["Selection policy", "Deferred", "Accepted with fact loss", "Accepted with contradiction", "Unnecessary change", "Bias reduction"], [
    ["FairLens Gate", "0.475", "0.000", "0.000", "0.000", "0.121"],
    ["FairLens Gate + strict retry", "0.464", "0.000", "0.000", "0.015", "0.128"],
    ["Lowest bias score", "0.000", "0.254", "0.083", "0.569", "0.152"],
    ["Highest similarity", "0.000", "0.297", "0.065", "0.985", "0.088"],
    ["Always FLAN-T5", "0.000", "0.094", "0.007", "0.200", "0.018"],
    ["Random (5 seeds)", "0.000", "0.361", "0.078", "0.985", "0.089"],
  ], [2.2, 1, 1.4, 1.5, 1.3, 1.1], { size: 17, highlight: [0, 1] }),
  p([t("Gate decisions on the 276 inputs: 148 Accept, 128 Human review. Strict retry was tried on 124 inputs and rescued 3.", { size: 19 })]),
  h2("Experiment 3 · Reliability Gate on the controlled gold set (run 20261024-040545-…-7565de)"),
  p("136 inputs × 3 perturbation seeds = 2,481 candidates with known correct answers; results on the test split."),
  table(["Policy", "Acceptable choice [95% CI]", "Harmful choice [95% CI]", "Deferred", "Unnecessary change"], [
    ["FairLens Gate (val-tuned rules)", "0.436 [0.373, 0.505]", "0.255 [0.191, 0.314]", "0.309", "0.000"],
    ["Learned gate (decision tree)", "0.799 [0.740, 0.853]", "0.069 [0.039, 0.103]", "0.132", "0.104"],
    ["Learned gate (logistic regression)", "0.637 [0.578, 0.701]", "0.167 [0.113, 0.216]", "0.196", "0.328"],
    ["Lowest bias score", "0.020 [0.005, 0.039]", "0.980 [0.961, 0.995]", "0.000", "1.000"],
    ["Highest similarity", "0.564 [0.495, 0.632]", "0.436 [0.368, 0.505]", "0.000", "1.000"],
    ["No change", "0.328 [0.265, 0.392]", "0.672 [0.608, 0.735]", "0.000", "0.000"],
    ["LLM-as-judge (validated prompt)", "0.299 [0.240, 0.363]", "0.338 [0.275, 0.407]", "0.363", "0.881"],
    ["Random (5 seeds)", "0.133", "0.867", "0.000", "1.000"],
  ], [2.6, 1.9, 1.9, 0.9, 1.1], { size: 17, highlight: [0] }),
  gap(),
  table(["Ablation (tuned rule Gate)", "Unsafe accept", "Change [95% CI]", "Verdict"], [
    ["Full Gate", "0.053", "—", "—"],
    ["Without bias signal", "0.129", "+0.064 [+0.051, +0.078]", "Useful"],
    ["Without entailment", "0.111", "+0.048 [+0.036, +0.060]", "Useful"],
    ["Without fact checker", "0.063", "+0.008 [+0.004, +0.013]", "Small but measurable"],
    ["Without semantic similarity", "0.054", "+0.001 [+0.000, +0.002]", "Negligible"],
    ["Without agreement / uncertainty / fluency", "0.053", "+0.000", "No measurable effect (negative result)"],
  ], [3, 1.2, 2.2, 2.2], { size: 17 }),
  h2("Experiment 4 · Image captions (run 20261024-041855-…-4acb2c)"),
  table(["Captioner", "CLIP similarity [95% CI]", "Words", "Detail coverage", "Gendered terms (people)", "s / image"], [
    ["BLIP-base", "0.281 [0.272, 0.288]", "7.7", "0.158", "0.533", "0.35"],
    ["Florence-2-base", "0.306 [0.298, 0.313]", "10.9", "0.232", "0.600", "8.0 (CPU)"],
    ["COCO human captions", "0.304 [0.297, 0.310]", "—", "—", "0.733", "—"],
  ], [1.8, 2, 0.8, 1.3, 1.6, 1.1], { size: 17 }),
  p([t("Florence-2 captions are longer, more detailed and as well aligned as human captions. Scope: this measures bias in generated captions only, not all visual bias.", { size: 19 })]),
);

// ---------- 8 findings
children.push(h1("8. Key findings and honest limitations"),
  h2("What the results support"),
  bullet("Fine-tuned DeBERTa-v3-small is the strongest detector and is consistent across sources."),
  bullet("Choosing the rewrite with the lowest bias score is unsafe; a gate that checks facts and meaning prevents accepting rewrites that lose facts, at the cost of more human review."),
  bullet("The bias-reduction and entailment signals each add measurable safety; agreement, uncertainty and fluency showed no measurable effect."),
  bullet("Stronger LLM rewriters reduce more bias but lose more facts."),
  h2("What we report carefully"),
  bullet("The transparent rule Gate is not the most accurate selector: a learned decision-tree gate reached 0.799 acceptable choices against 0.436. We present the rule Gate as interpretable and conservative, and the learned gate as a promising direction."),
  bullet("The 1.00 catch rates on planted corruptions are partly by construction, because several perturbation types target exactly what the fact checker extracts."),
  bullet("Detectors raise many false alarms on neutral text (about 40% on WinoBias), and missed an obvious case in the demo (\"Women are not suited to engineering…\" scored 0.48, just under the 0.5 threshold)."),
  h2("Limitations"),
  bullet("The Gate gold set is synthetic and written by one author; mitigation metrics are automatic. A blinded human evaluation is planned but not done."),
  bullet("Each experiment ran once on one device (Apple MPS). Confidence intervals reflect test-item sampling, not training seeds. Cross-device verification on Windows/CPU is in progress."),
  bullet("Bias labels depend on culture and context; the detectors were trained on news and stereotype sentences, not job adverts or resumes."),
  bullet("The public RoBERTa reference was trained on BABE, so its BABE score may be optimistic."),
  bullet("FLAN-T5 mostly copies its input, a real capability limit that we report rather than hide."),
);

// ---------- 9 app
children.push(h1("9. The application"),
  p("FairLens runs locally as a Streamlit web app on Windows, macOS and Linux. Nothing is sent to an external service. Uploaded text is not stored and never changes the benchmark results."),
  img(S + "benchmarks.png"), caption("Figure 1. Benchmarks page: headline results, each read from a stored run with its run ID."),
  img(S + "fairlens_review.png"), caption("Figure 2. FairLens Review: a guided six-step check of a job advert."),
  h2("FairLens Review workflow"),
  newList(),
  num("Upload a job advert (PDF with a text layer, DOCX including tables, TXT) or paste text."),
  num("See the flagged phrases and which detectors agree."),
  num("Compare the original with the rule, FLAN-T5 and Qwen rewrites (word-level diff)."),
  num("Check that numbers, names, dates, roles, quotes and negations survived."),
  num("Read the Gate's verdict and reason codes."),
  num("Export a PDF or JSON report."),
  h2("Pages"),
  p("Home · FairLens Review · Text Analysis · Mitigation · Reliability Gate · Image Analysis · Benchmarks · Model Comparison · Error Analysis · Experiments & Exports · Use Cases · About. Light and dark themes; Lite mode for slower laptops."),
);

// ---------- 10 applications
children.push(h1("10. Applications and users"),
  table(["Application", "Who uses it", "What FairLens provides"], [
    ["Recruitment and job-advert review", "HR teams, recruiters", "Flags biased wording, proposes neutral rewrites, checks that requirements such as years of experience survive"],
    ["Checking AI writing tools", "Teams building rewrite features", "Gate verdicts and reason codes show when a rewrite changed facts"],
    ["Model testing and benchmarking", "AI/ML researchers", "Detectors and rewriters compared on one dataset with stored, reproducible runs"],
    ["Inclusive writing", "Content and communication teams", "Suggested neutral alternatives with evidence"],
    ["Caption evaluation", "Computer-vision researchers", "BLIP vs Florence-2 alignment and caption wording"],
    ["Reliability research", "Fairness and safety researchers", "Risk of accepting unsafe rewrites, compared across selection methods"],
  ], [2.3, 2, 4]),
  note("FairLens is a review assistant, not a legal or compliance certification tool. It cannot guarantee that any text is unbiased; a person makes the final decision."),
);

// ---------- 11 status
children.push(h1("11. Work completed and remaining"),
  table(["Area", "Status"], [
    ["Datasets, splits, hashes, leakage checks", "Done and verified (rebuilds byte-identically)"],
    ["16 models downloaded, verified and pinned", "Done"],
    ["Detector fine-tuning (DistilBERT, DeBERTa)", "Done"],
    ["Detection, mitigation, Gate and image benchmarks", "Done (one run each, Apple MPS)"],
    ["Gate baselines incl. no-change and LLM judge; ablations", "Done"],
    ["Six use cases; 12-page app; FairLens Review page", "Done"],
    ["Automated tests and three-OS CI", "Done (243 fast tests on macOS); Windows run in progress"],
    ["Cross-device check (MPS vs Windows CPU)", "In progress"],
    ["Risk–coverage curves, red-team tests, threshold sensitivity", "Planned"],
    ["Blinded human evaluation (2–3 annotators)", "Planned; guideline written"],
    ["Open decision: BART-MNLI template used inside the Gate", "To decide with the team; would require re-running Experiments 2 and 3"],
    ["Research paper (IEEE format) and final report", "In progress"],
  ], [4.5, 3.7]),
);

// ---------- 12 Q&A
children.push(h1("12. Likely review questions"),
  ...[
    ["Why is FairLens needed when bias detectors already exist?", "Detectors only flag. Rewriting is where facts get lost: Qwen2.5 lost at least one fact in 41.7% of its rewrites. FairLens adds the check between the rewrite and the reader."],
    ["Why not just pick the rewrite with the lowest bias score?", "On the gold set that choice was harmful 98% of the time, because the least-biased candidate is often one that dropped or changed content. The Gate's harmful-choice rate was 25.5%."],
    ["Is the Gate the best method?", "No. A learned decision tree selected better (0.799 vs 0.436 acceptable choices). The rule Gate is transparent and conservative; we report both."],
    ["Why several detectors?", "They disagree and fail differently; disagreement is a reason to ask a person. DeBERTa was strongest overall, while the public RoBERTa model was strong only on its own training source (BABE)."],
    ["How were thresholds chosen without cheating?", "On the validation split only, then frozen; test results are reported separately."],
    ["Can FairLens guarantee unbiased output?", "No. Detectors make errors (about 40% false alarms on neutral WinoBias sentences) and bias labels depend on context."],
    ["What is new compared with existing work?", "Gating rewrites with checks exists (e.g. LGSA, ParaDetox). Our combination of per-fact checks, four-way decisions with retry and human review, and evaluation against simpler selectors is, to our knowledge, not previously evaluated for bias-mitigation rewrites."],
    ["What would you do with more time?", "Human evaluation, repeated training seeds, a larger real job-advert set, and a learned gate that keeps the rule Gate's explanations."],
  ].flatMap(([q, a]) => [p([b("Q. " + q)], { spacing: { before: 160, after: 40 } }), p(a)]),
);

// ---------- 13 references
children.push(h1("13. References"),
  ...[
    "Angelopoulos, Bates, Fisch, Lei, Schuster. Conformal Risk Control. ICLR 2024.",
    "Blodgett, Lopez, Olteanu, Sim, Wallach. Stereotyping Norwegian Salmon: An Inventory of Pitfalls in Fairness Benchmark Datasets. ACL 2021.",
    "Dhamala et al. BOLD: Dataset and Metrics for Measuring Biases in Open-Ended Language Generation. FAccT 2021.",
    "Gallegos et al. Bias and Fairness in Large Language Models: A Survey. Computational Linguistics 50(3), 2024.",
    "Gaucher, Friesen, Kay. Evidence That Gendered Wording in Job Advertisements Exists and Sustains Gender Inequality. J. Personality and Social Psychology 101(1), 2011.",
    "Geifman, El-Yaniv. Selective Classification for Deep Neural Networks. NeurIPS 2017.",
    "He, Majumder, McAuley. Detect and Perturb: Neutral Rewriting of Biased and Sensitive Text via Gradient-based Decoding. Findings of EMNLP 2021.",
    "Karri et al. LLM-Guided Synthetic Augmentation (LGSA) for Mitigating Bias in AI Systems. arXiv:2510.13202, 2025.",
    "Laban, Schnabel, Bennett, Hearst. SummaC: Re-Visiting NLI-based Models for Inconsistency Detection in Summarization. TACL 2022.",
    "Lin et al. Microsoft COCO: Common Objects in Context. ECCV 2014.",
    "Logacheva et al. ParaDetox: Detoxification with Parallel Data. ACL 2022.",
    "Nadeem, Bethke, Reddy. StereoSet: Measuring Stereotypical Bias in Pretrained Language Models. ACL 2021.",
    "Nangia, Vania, Bhalerao, Bowman. CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models. EMNLP 2020.",
    "Pryzant et al. Automatically Neutralizing Subjective Bias in Text. AAAI 2020.",
    "Spinde et al. Neural Media Bias Detection Using Distant Supervision With BABE. Findings of EMNLP 2021.",
    "Sun et al. They, Them, Theirs: Rewriting with Gender-Neutral English. arXiv:2102.06788, 2021.",
    "Yuan et al. LLM in the Loop: Creating the ParaDeHate Dataset for Hate Speech Detoxification. arXiv:2506.01484, 2025.",
    "Zhao, Wang, Yatskar, Ordonez, Chang. Gender Bias in Coreference Resolution: Evaluation and Debiasing Methods. NAACL 2018.",
    "Zheng et al. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. NeurIPS 2023.",
  ].map(r => new Paragraph({ numbering: { reference: "refs", level: 0 }, spacing: { after: 80 }, children: [t(r, { size: 19 })] })),
);

for (let i = children.length - 1; i >= 0; i--) if (children[i] === null) children.splice(i, 1);
const doc = new Document({
  creator: "FairLens team", title: "FairLens — Second Review", description: "Final-year project second review document",
  styles: {
    default: { document: { run: { font: "Calibri", size: 21 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 34, bold: true, color: "1B1A2E", font: "Calibri" }, paragraph: { spacing: { after: 200 }, outlineLevel: 0,
        border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: ACCENT, space: 4 } } } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 25, bold: true, color: ACCENT, font: "Calibri" }, paragraph: { spacing: { before: 280, after: 120 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 360, hanging: 260 } } } }] },
    { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 400, hanging: 300 } } } }] },
    { reference: "refs", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "[%1]", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 520, hanging: 520 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } },
      titlePage: true },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [t("FairLens · Second Review", { size: 16, color: GREY })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [t("Page ", { size: 16, color: GREY }), new TextRun({ children: [PageNumber.CURRENT], size: 16, color: GREY })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => { fs.writeFileSync(process.argv[2], buf); console.log("written", process.argv[2], buf.length); });
