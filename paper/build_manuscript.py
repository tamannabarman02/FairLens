"""Build the anonymised AI and Ethics manuscript (paper/FairLens_manuscript_anonymised.docx).

Tables are filled from the stored run summaries; figures come from scripts/make_paper_figures.py.
Citations are written as [@key] and numbered in order of first appearance.
Run from the project root:  .venv/Scripts/python.exe paper/build_manuscript.py
"""
import json
import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Mm, Pt

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "outputs" / "runs"
FIG = ROOT / "paper" / "figures"
OUT = ROOT / "paper" / "FairLens_manuscript_anonymised_v3_final.docx"


def load(run, name):
    return json.loads((RUNS / run / name).read_text(encoding="utf-8"))


DET = load("20261024-043227-benchmark_detection-a4a61f", "detection_summary.json")
# Controlled-set and policy results: re-run with a sentence-level split (Windows CPU), reusing the stored signals
# and rewrites of the original Mac runs 20261024-040545-benchmark_gate-7565de / 20261024-040853-...-000f03.
GATE_RUN = "20261115-225313-benchmark_gate-794a76"  # same as ea9824 except corrected Δ unsafe-accept
MIT_RUN = "20261115-223752-benchmark_mitigation-495a89"
JUDGE_RUN = "20261115-215934-llm_judge_prefill-30cb0c"  # test-split judge answers (Qwen on CPU)
GATE = load(GATE_RUN, "gate_summary.json")
MIT = load(MIT_RUN, "mitigation_summary.json")
ENV = json.loads((RUNS / GATE_RUN / "manifest.json").read_text(encoding="utf-8"))["environment"]
JENV = json.loads((RUNS / JUDGE_RUN / "manifest.json").read_text(encoding="utf-8"))["environment"]

# A-priori Gate defaults, read from the config and checked against what the gate run recorded as its baseline.
import yaml  # noqa: E402

GDEF = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))["gate"]
_rec = json.loads((RUNS / GATE_RUN / "manifest.json").read_text(encoding="utf-8"))["config"]["gate_default"]
assert all(_rec[k] == GDEF[k] for k in ["min_semantic_similarity", "min_entailment_prob", "min_bias_reduction",
                                        "max_detector_std", "require_bidirectional_entailment"])
assert (_rec["entailment_mode"] == "both") == GDEF["require_bidirectional_entailment"]

# Test-suite counts from the saved pytest logs of the re-run laptop (outputs/test_logs).
_TL = ROOT / "outputs" / "test_logs"
_venv = (_TL / "pytest_venv_windows.txt").read_text(encoding="utf-8")
T_FAIL, T_PASS = (int(x) for x in re.search(r"(\d+) failed, (\d+) passed", _venv).groups())
T_SLOW, T_ALL = (int(x) for x in re.search(r"(\d+)/(\d+) tests collected", _venv).groups())
T_EXTRA = int(re.search(r"(\d+) passed", (_TL / "pytest_anaconda_learned_gate.txt").read_text(encoding="utf-8")).group(1))
assert T_PASS + T_EXTRA == T_ALL and T_FAIL == T_EXTRA

REFS = {
    "gaucher": "Gaucher, D., Friesen, J., Kay, A.C.: Evidence that gendered wording in job advertisements exists and sustains gender inequality. J. Pers. Soc. Psychol. 101(1), 109–128 (2011). https://doi.org/10.1037/a0022530",
    "blodgett2020": "Blodgett, S.L., Barocas, S., Daumé III, H., Wallach, H.: Language (technology) is power: a critical survey of “bias” in NLP. In: Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pp. 5454–5476 (2020). https://doi.org/10.18653/v1/2020.acl-main.485",
    "gallegos": "Gallegos, I.O., Rossi, R.A., Barrow, J., Tanjim, M.M., Kim, S., Dernoncourt, F., Yu, T., Zhang, R., Ahmed, N.K.: Bias and fairness in large language models: a survey. Comput. Linguist. 50(3), 1097–1179 (2024). https://doi.org/10.1162/coli_a_00524",
    "euaiact": "European Parliament and Council of the European Union: Regulation (EU) 2024/1689 laying down harmonised rules on artificial intelligence (Artificial Intelligence Act). Official Journal of the European Union, L series, 12 July 2024",
    "floridi": "Floridi, L., Cowls, J.: A unified framework of five principles for AI in society. Harv. Data Sci. Rev. 1(1) (2019). https://doi.org/10.1162/99608f92.8cd550d1",
    "jobin": "Jobin, A., Ienca, M., Vayena, E.: The global landscape of AI ethics guidelines. Nat. Mach. Intell. 1, 389–399 (2019). https://doi.org/10.1038/s42256-019-0088-2",
    "babe": "Spinde, T., Plank, M., Krieger, J.-D., Ruas, T., Gipp, B., Aizawa, A.: Neural media bias detection using distant supervision with BABE – Bias Annotations By Experts. In: Findings of the Association for Computational Linguistics: EMNLP 2021, pp. 1166–1177 (2021). https://doi.org/10.18653/v1/2021.findings-emnlp.101",
    "stereoset": "Nadeem, M., Bethke, A., Reddy, S.: StereoSet: measuring stereotypical bias in pretrained language models. In: Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics, pp. 5356–5371 (2021). https://doi.org/10.18653/v1/2021.acl-long.416",
    "crows": "Nangia, N., Vania, C., Bhalerao, R., Bowman, S.R.: CrowS-Pairs: a challenge dataset for measuring social biases in masked language models. In: Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, pp. 1953–1967 (2020). https://doi.org/10.18653/v1/2020.emnlp-main.154",
    "winobias": "Zhao, J., Wang, T., Yatskar, M., Ordonez, V., Chang, K.-W.: Gender bias in coreference resolution: evaluation and debiasing methods. In: Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics, pp. 15–20 (2018). https://doi.org/10.18653/v1/N18-2003",
    "blodgett2021": "Blodgett, S.L., Lopez, G., Olteanu, A., Sim, R., Wallach, H.: Stereotyping Norwegian salmon: an inventory of pitfalls in fairness benchmark datasets. In: Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics, pp. 1004–1015 (2021). https://doi.org/10.18653/v1/2021.acl-long.81",
    "bold": "Dhamala, J., Sun, T., Kumar, V., Krishna, S., Pruksachatkun, Y., Chang, K.-W., Gupta, R.: BOLD: dataset and metrics for measuring biases in open-ended language generation. In: Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency, pp. 862–872 (2021). https://doi.org/10.1145/3442188.3445924",
    "pryzant": "Pryzant, R., Diehl Martinez, R., Dass, N., Kurohashi, S., Jurafsky, D., Yang, D.: Automatically neutralizing subjective bias in text. In: Proceedings of the AAAI Conference on Artificial Intelligence 34(01), pp. 480–489 (2020). https://doi.org/10.1609/aaai.v34i01.5385",
    "depen": "He, Z., Majumder, B.P., McAuley, J.: Detect and perturb: neutral rewriting of biased and sensitive text via gradient-based decoding. In: Findings of the Association for Computational Linguistics: EMNLP 2021, pp. 4173–4181 (2021). https://doi.org/10.18653/v1/2021.findings-emnlp.352",
    "sun": "Sun, T., Webster, K., Shah, A., Wang, W.Y., Johnson, M.: They, them, theirs: rewriting with gender-neutral English. arXiv:2102.06788 (2021)",
    "paradetox": "Logacheva, V., Dementieva, D., Ustyantsev, S., Moskovskiy, D., Dale, D., Krotova, I., Semenov, N., Panchenko, A.: ParaDetox: detoxification with parallel data. In: Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics, pp. 6804–6818 (2022). https://doi.org/10.18653/v1/2022.acl-long.469",
    "paradehate": "Yuan, S., Nie, E., Kouba, L., Schmid, H., Schütze, H., Färber, M.: LLM in the loop: creating the ParaDeHate dataset for hate speech detoxification. In: Findings of the Association for Computational Linguistics: IJCNLP-AACL 2025 (2025). https://doi.org/10.18653/v1/2025.findings-ijcnlp.59",
    "lgsa": "Karri, S.S.R., Nallapuneni, Y.S., Mallireddy, L.N.R., Gopichand, G.: LLM-guided synthetic augmentation (LGSA) for mitigating bias in AI systems. arXiv:2510.13202 (2025)",
    "summac": "Laban, P., Schnabel, T., Bennett, P.N., Hearst, M.A.: SummaC: re-visiting NLI-based models for inconsistency detection in summarization. Trans. Assoc. Comput. Linguist. 10, 163–177 (2022). https://doi.org/10.1162/tacl_a_00453",
    "judge": "Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E.P., Zhang, H., Gonzalez, J.E., Stoica, I.: Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. In: Advances in Neural Information Processing Systems 36 (2023)",
    "geifman": "Geifman, Y., El-Yaniv, R.: Selective classification for deep neural networks. In: Advances in Neural Information Processing Systems 30 (2017)",
    "crc": "Angelopoulos, A.N., Bates, S., Fisch, A., Lei, L., Schuster, T.: Conformal risk control. In: International Conference on Learning Representations (2024)",
    "bender": "Bender, E.M., Gebru, T., McMillan-Major, A., Shmitchell, S.: On the dangers of stochastic parrots: can language models be too big? In: Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency, pp. 610–623 (2021). https://doi.org/10.1145/3442188.3445922",
    "distilbert": "Sanh, V., Debut, L., Chaumond, J., Wolf, T.: DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arXiv:1910.01108 (2019)",
    "deberta": "He, P., Gao, J., Chen, W.: DeBERTaV3: improving DeBERTa using ELECTRA-style pre-training with gradient-disentangled embedding sharing. In: International Conference on Learning Representations (2023)",
    "bart": "Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O., Stoyanov, V., Zettlemoyer, L.: BART: denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. In: Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pp. 7871–7880 (2020). https://doi.org/10.18653/v1/2020.acl-main.703",
    "zeroshot": "Yin, W., Hay, J., Roth, D.: Benchmarking zero-shot text classification: datasets, evaluation and entailment approach. In: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, pp. 3914–3923 (2019). https://doi.org/10.18653/v1/D19-1404",
    "flant5": "Chung, H.W., et al.: Scaling instruction-finetuned language models. J. Mach. Learn. Res. 25(70), 1–53 (2024)",
    "qwen": "Qwen Team: Qwen2.5 technical report. arXiv:2412.15115 (2024)",
    "sbert": "Reimers, N., Gurevych, I.: Sentence-BERT: sentence embeddings using Siamese BERT-networks. In: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, pp. 3982–3992 (2019). https://doi.org/10.18653/v1/D19-1410",
    "minilm": "Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., Zhou, M.: MiniLM: deep self-attention distillation for task-agnostic compression of pre-trained transformers. In: Advances in Neural Information Processing Systems 33 (2020)",
    "guo": "Guo, C., Pleiss, G., Sun, Y., Weinberger, K.Q.: On calibration of modern neural networks. In: Proceedings of the 34th International Conference on Machine Learning, pp. 1321–1330 (2017)",
    "coco": "Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft COCO: common objects in context. In: Computer Vision – ECCV 2014, pp. 740–755. Springer, Cham (2014). https://doi.org/10.1007/978-3-319-10602-1_48",
    "blip": "Li, J., Li, D., Xiong, C., Hoi, S.: BLIP: bootstrapping language-image pre-training for unified vision-language understanding and generation. In: Proceedings of the 39th International Conference on Machine Learning, pp. 12888–12900 (2022)",
    "florence": "Xiao, B., Wu, H., Xu, W., Dai, X., Hu, H., Lu, Y., Zeng, M., Liu, C., Yuan, L.: Florence-2: advancing a unified representation for a variety of vision tasks. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4818–4829 (2024)",
    "clip": "Radford, A., et al.: Learning transferable visual models from natural language supervision. In: Proceedings of the 38th International Conference on Machine Learning, pp. 8748–8763 (2021)",
    "hendricks": "Hendricks, L.A., Burns, K., Saenko, K., Darrell, T., Rohrbach, A.: Women also snowboard: overcoming bias in captioning models. In: Computer Vision – ECCV 2018, pp. 793–811. Springer, Cham (2018)",
}

ORDER: list[str] = []


def cite(m):
    keys = [k.strip().lstrip("@") for k in m.group(1).split(";")]
    nums = []
    for k in keys:
        if k not in REFS:
            raise KeyError(k)
        if k not in ORDER:
            ORDER.append(k)
        nums.append(ORDER.index(k) + 1)
    nums = sorted(set(nums))
    parts, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        parts.append(f"{nums[i]}–{nums[j]}" if j - i >= 2 else ", ".join(str(n) for n in nums[i:j + 1]))
        i = j + 1
    return "[" + ", ".join(parts) + "]"




TITLE = ("FairLens: Automated Framework for Bias Detection, Mitigation and Fairness Evaluation in Large "
         "Language and Multimodal AI Systems")

doc = Document()
cp = doc.core_properties
cp.author = ""
cp.last_modified_by = ""
cp.title = ""
cp.comments = ""
cp.keywords = ""

st = doc.styles["Normal"]
st.font.name = "Times New Roman"
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
st.font.size = Pt(11)
st.paragraph_format.line_spacing = 1.0
st.paragraph_format.space_after = Pt(6)
for name, size in (("Heading 1", 13), ("Heading 2", 11.5)):
    h = doc.styles[name]
    h.font.name = "Times New Roman"
    rf = h.element.rPr.rFonts
    for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        rf.attrib.pop(qn(attr), None)
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rf.set(qn(attr), "Times New Roman")
    h.font.size = Pt(size)
    h.font.bold = True
    h.font.italic = False
    h.font.color.rgb = None
    h.paragraph_format.space_before = Pt(12)
    h.paragraph_format.space_after = Pt(6)
lb = doc.styles["List Bullet"]
lb.paragraph_format.space_after = Pt(3)

sec = doc.sections[0]
sec.page_width, sec.page_height = Mm(210), Mm(297)
for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
    setattr(sec, side, Mm(25.4))
# Automatic page numbers in the footer (no line numbering).
fp = sec.footer.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = fp.add_run()
for tag, text in (("begin", None), (None, "PAGE"), ("end", None)):
    if tag:
        e = OxmlElement("w:fldChar")
        e.set(qn("w:fldCharType"), tag)
    else:
        e = OxmlElement("w:instrText")
        e.set(qn("xml:space"), "preserve")
        e.text = text
    r._r.append(e)

TOKEN = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*)")
JUSTIFY = WD_ALIGN_PARAGRAPH.JUSTIFY


def runs(par, text, size=None):
    text = re.sub(r"\[(@[^\]]+)\]", cite, text)
    for part in TOKEN.split(text):
        if not part:
            continue
        if part.startswith("**"):
            run = par.add_run(part[2:-2])
            run.bold = True
        elif part.startswith("*"):
            run = par.add_run(part[1:-1])
            run.italic = True
        else:
            run = par.add_run(part)
        if size:
            run.font.size = Pt(size)
    return par


def P(text, align=JUSTIFY, style=None):
    par = doc.add_paragraph(style=style)
    par.alignment = align
    return runs(par, text)


def H(level, text):
    doc.add_heading(text, level=level)


def bullets(items):
    for it in items:
        P(it, style="List Bullet")


def figure(num, fname, width_mm, caption):
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.keep_with_next = True
    par.add_run().add_picture(str(FIG / fname), width=Mm(width_mm))
    cap = doc.add_paragraph()
    cap.alignment = JUSTIFY
    b = cap.add_run(f"Fig. {num}")
    b.bold = True
    b.font.size = Pt(10)
    runs(cap, " " + caption, size=10)


def table(num, caption, header, rows, note=None, widths=None):
    cap = doc.add_paragraph()
    cap.alignment = JUSTIFY
    cap.paragraph_format.keep_with_next = True
    b = cap.add_run(f"Table {num}")
    b.bold = True
    b.font.size = Pt(10)
    runs(cap, " " + caption, size=10)
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0]._tr.get_or_add_trPr()
    rep = OxmlElement("w:tblHeader")
    rep.set(qn("w:val"), "true")
    hdr.append(rep)
    for row in t.rows:
        cant = OxmlElement("w:cantSplit")
        cant.set(qn("w:val"), "true")
        row._tr.get_or_add_trPr().append(cant)
    for i, row in enumerate([header] + rows):
        for j, val in enumerate(row):
            cell = t.cell(i, j)
            cell.text = ""
            par = cell.paragraphs[0]
            par.paragraph_format.space_after = Pt(0)
            # Never strand the header row; longer tables may continue on the next page (header repeats).
            par.paragraph_format.keep_with_next = i < 2  # caption + header + first row stay together
            runs(par, f"**{val}**" if i == 0 else str(val), size=8.5)
            if widths:
                cell.width = Mm(widths[j])
    n = doc.add_paragraph()
    n.alignment = JUSTIFY
    if note:
        runs(n, note, size=9)
    else:
        # Thin spacer only: separates the table from the next paragraph without a blank line.
        n.paragraph_format.space_after = Pt(0)
        n.paragraph_format.line_spacing = Pt(6)


def f3(x):
    return f"{x:.3f}"


def pct(x):
    return f"{100 * x:.1f}%"


def ci(d, key="value"):
    return f"{d[key]:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]"


dd = DET["detectors"]["deberta"]
rw = MIT["rewriters"]
sel = GATE["selection"]
tg, tree, lb_ = sel["fairlens_gate_tuned"], sel["learned_tree"], sel["lowest_bias"]
N_TEST = tg["success"]["n"]                                  # test inputs (sentence x seed)
N_TEST_NEUTRAL = tg["unnecessary_change_neutral"]["n"]       # neutral test inputs
N_TEST_CAND = sum(GATE["gold_distribution_test"].values())   # test candidates


def share(policy: dict, gold: str) -> float:
    """Share of test inputs for which a policy's choice had this gold label (or DEFER)."""
    counts = policy["chosen_gold_counts"]
    return counts.get(gold, 0) / sum(counts.values())


GATE_DMG = share(tg, "REJECT")
GATE_KEPT = share(tg, "RETRY")
GATE_DEFER = share(tg, "DEFER")
TREE_DMG = share(tree, "REJECT")
GATE_DMG_PHRASE = "never chose one" if GATE_DMG == 0 else f"chose one for {100 * GATE_DMG:.1f}% of inputs"


# =========================================================================== title, abstract, keywords
t = doc.add_paragraph()
t.alignment = WD_ALIGN_PARAGRAPH.CENTER
tr = t.add_run(TITLE)
tr.bold = True
tr.font.size = Pt(16)

ab = doc.add_paragraph()
abr = ab.add_run("Abstract")
abr.bold = True
abr.font.size = Pt(12)
P("Bias-mitigation tools increasingly use large language models (LLMs) to rewrite biased text, yet they are "
  "usually judged only by whether a bias detector's score falls, even though a rewrite can lower that score while "
  "deleting a requirement, changing a number or reversing a negation. We present FairLens, an automated framework "
  "for bias detection, mitigation and fairness evaluation in text and image captions that runs locally. "
  "FairLens combines four bias detectors, three rewriters (rule-based, FLAN-T5 and Qwen2.5), a "
  "captioning extension (BLIP and Florence-2) and a Reliability Gate: a transparent rule system that accepts, "
  "retries, rejects or defers each rewrite to a person, with reason codes, using bias reduction, semantic "
  "similarity, bidirectional entailment, per-fact preservation and detector agreement. A dashboard "
  "guides users through a review and traces every number to a stored run. A fine-tuned "
  "DeBERTa-v3-small detector reached a macro-F1 of 0.814 on 2,652 held-out sentences but flagged about 40% of "
  "neutral WinoBias sentences. Qwen2.5 reduced bias scores the most yet lost at least one fact in 41.7% of its "
  f"rewrites. On {N_TEST} test inputs with known answers, selecting the least-biased candidate chose a "
  f"damaging rewrite {pct(share(lb_, 'REJECT'))} of the time; the Gate "
  + ("never did so" if GATE_DMG == 0 else f"did so {pct(GATE_DMG)} of the time")
  + f", at the cost of deferring {pct(GATE_DEFER)} of inputs and keeping the biased original for {pct(GATE_KEPT)} "
  "of inputs. A learned gate was more accurate but chose damaging rewrites more often and was less transparent. "
  "We argue that "
  "mitigation tools should be evaluated on what they preserve and when they abstain, not only on how much bias "
  "they remove.")
P("**Keywords:** Bias detection; Bias mitigation; Large language models; Human oversight; Fairness evaluation; "
  "Responsible AI")

# =========================================================================== 1 Introduction
H(1, "1 Introduction")
P("Biased wording has measurable effects. Gender-coded language in job advertisements, for example, makes women "
  "less likely to see a role as one in which they belong and less likely to apply [@gaucher]. As language and "
  "multimodal AI systems write and caption more of the text people read, they can reproduce such bias at scale, "
  "and natural language processing (NLP) research has produced many methods to detect biased language and, "
  "increasingly, to rewrite it [@blodgett2020; @gallegos]. Instruction-tuned LLMs make rewriting easy: a user can "
  "ask a model to “make this inclusive” and receive fluent text in seconds.")
P("The ease of rewriting hides a problem of evaluation. A rewrite is usually judged by one number: whether a "
  "bias detector scores it lower than the original. A rewrite can lower that score while silently changing what "
  "the text says. Consider the illustrative advertisement “Looking for a young, aggressive salesman with 5+ "
  "years of B2B experience.” One rewrite, “Looking for a motivated sales professional with B2B experience,” "
  "removes the biased wording and also removes the five-year requirement; another keeps it. A bias score cannot "
  "tell the two apart. In recruitment the first rewrite misleads applicants and the employer; in news text a "
  "reversed negation or a swapped role misinforms readers. A fairness intervention that alters facts thus "
  "creates a harm of its own.")
P("This is an ethical problem as much as a technical one. Widely shared AI ethics principles include "
  "non-maleficence, respect for the autonomy of the people affected, and explicability [@floridi; @jobin]. A "
  "mitigation tool that rewrites an author's text without checking what else it changed fails all three: it may "
  "cause harm, it overrides the author without saying what was lost, and it gives no reason for its output. "
  "Regulation points the same way: the European Union's AI Act classifies AI systems used for recruitment and "
  "the selection of candidates as high-risk and requires that they be designed for effective human oversight "
  "[@euaiact]. A tool that cannot say when it is unsure gives a human overseer little to work with.")
P("FairLens is our response. It is an end-to-end framework that (i) detects biased sentences in text, documents "
  "and image captions with several complementary detectors, (ii) proposes neutral rewrites with a rule-based "
  "rewriter and two LLMs, (iii) checks every rewrite with a Reliability Gate before anything is accepted, and "
  "(iv) evaluates every component on shared data with stored, reproducible runs. The Gate returns one of four "
  "decisions for each candidate rewrite (accept, retry with a stricter prompt, reject, or defer to human review) "
  "with reason codes such as FACT_LOSS or UNNECESSARY_CHANGE. The research question that guides the evaluation "
  "is: *can a transparent, multi-signal gate decide more safely than simple selection rules which rewrite of a "
  "biased sentence, if any, to accept?*")
P("This paper makes five contributions:")
bullets([
    "A statement of five ethical requirements for automated mitigation tools, each mapped to a measurable "
    "quantity (Sect. 4).",
    "The design of FairLens, a layered framework covering ingestion, detection, mitigation, a Reliability Gate, "
    "a captioning extension, resource management for laptops and experiment tracking (Sect. 5).",
    "A working, cross-platform implementation with a guided review workflow in which every displayed result is "
    "traceable to a stored run (Sect. 6).",
    "An evaluation of detectors, rewriters and the Gate against simpler selection policies on public benchmarks, "
    "real rewrites and a controlled set with known correct decisions, with ablations and confidence intervals "
    "(Sects. 7–8).",
    "Negative results reported as such: the rule Gate is less accurate than a learned gate, several signals add "
    "no measurable safety, one rewriter mostly copies its input, and the Gate's remaining failures are inherited "
    "from detector misses (Sects. 8–10).",
])
P("The rest of the paper is organised as follows. Sect. 2 states the objectives and Sect. 3 reviews related work "
  "and research gaps. Sect. 4 sets out the ethical requirements. Sects. 5 and 6 describe the design and "
  "implementation, Sect. 7 the experimental setup and Sect. 8 the results. Sects. 9–12 discuss implications, "
  "limitations, future work and conclusions.")

# =========================================================================== 2 Objectives
H(1, "2 Objectives")
bullets([
    "To compare rule-based, fine-tuned and zero-shot bias detectors fairly, on the same held-out data with the "
    "same metrics, and to measure their false alarms on neutral text.",
    "To compare a rule-based rewriter with two LLM rewriters on bias reduction and on the preservation of facts "
    "and meaning.",
    "To design a transparent Reliability Gate that decides whether a rewrite is accepted, retried, rejected or "
    "sent to a person, and to evaluate it against simpler selection policies with ablations of each signal.",
    "To extend the same detection pipeline to captions generated for images.",
    "To deliver a reproducible, cross-platform application that runs locally and in which every reported result "
    "is traceable to a stored experiment run.",
])

# =========================================================================== 3 Related work
H(1, "3 Related Work")
H(2, "3.1 Detecting biased language")
P("Sentence-level bias detection is trained and evaluated on benchmark datasets that operationalise bias in "
  "different ways. BABE annotates media bias and loaded framing in news sentences [@babe]; StereoSet and "
  "CrowS-Pairs contrast stereotypical and less stereotypical sentences about social groups [@stereoset; @crows]; "
  "WinoBias tests gender–occupation associations in coreference [@winobias]; BOLD measures bias in open-ended "
  "generation [@bold]. These datasets are not interchangeable, and several contain validity problems: "
  "anti-stereotypical sentences often still generalise about a group, and some items are inconsistent "
  "[@blodgett2021]. Surveys stress that “bias” in NLP is often under-specified and that normative reasoning about "
  "who is harmed, and how, should come first [@blodgett2020; @gallegos]. FairLens therefore reports detection per "
  "source and treats any detector score as fallible evidence, not ground truth.")
H(2, "3.2 Neutralising and detoxifying rewrites")
P("Pryzant et al. introduced the Wiki Neutrality Corpus and models that neutralise subjective bias in "
  "encyclopaedic text [@pryzant]. Later work rewrites biased or sensitive text by detecting and perturbing biased "
  "tokens [@depen] or rewrites gendered language into gender-neutral English [@sun]. In detoxification, ParaDetox "
  "and ParaDeHate build parallel corpora and check content preservation while constructing the data "
  "[@paradetox; @paradehate]. These systems are evaluated mainly on how much bias or toxicity is removed and on "
  "fluency; none includes a run-time decision layer that refuses an unsafe rewrite or routes it to a person. The "
  "closest idea to ours is LLM-guided synthetic augmentation, which screens generated rewrites with similarity "
  "and attribute checks [@lgsa]; it targets data augmentation and has no per-fact checks, multi-way decision or "
  "comparison with selection baselines.")
H(2, "3.3 Checking generated text and abstaining")
P("Natural language inference (NLI) models are used to detect factual inconsistency between a source and a "
  "generated text, notably in summarisation [@summac]. LLMs are also used as judges of other models' outputs, "
  "with documented position and self-preference biases [@judge]. Selective classification lets a model abstain "
  "when its confidence is low, trading coverage for risk [@geifman], and conformal methods give distribution-free "
  "control of such risks [@crc]. FairLens combines these strands for mitigation rewrites: it uses entailment and "
  "explicit fact checks to define what must be preserved, compares against an LLM judge, and treats deferral to "
  "a person as a legitimate outcome whose rate must be reported.")
H(2, "3.4 Ethics of automated language intervention")
P("Critiques of large language models warn that fluent output invites unwarranted trust and that the costs of "
  "errors fall unevenly [@bender]. Principle-based frameworks converge on non-maleficence, autonomy, justice and "
  "explicability [@floridi; @jobin], and the EU AI Act turns human oversight into a legal requirement for "
  "high-risk uses such as recruitment [@euaiact]. Captioning models have also been shown to rely on gendered "
  "context rather than visual evidence [@hendricks], so the same concerns apply to multimodal systems.")
H(2, "3.5 Comparison and research gaps")
P("Table 1 summarises how existing approaches relate to FairLens.")
table(1, "Existing approaches and the gap each leaves that FairLens addresses",
      ["Approach", "Examples", "What it does", "Gap addressed by FairLens"],
      [["Bias detectors and benchmarks", "BABE [@babe]; StereoSet [@stereoset]; CrowS-Pairs [@crows]",
        "Flag biased or stereotyped sentences", "Do not fix the text; scores vary by source and raise false alarms"],
       ["Word-list tools", "Gender-coded word lists for job adverts [@gaucher]",
        "Flag or replace listed words", "Miss implicit bias; replacements are not checked for meaning"],
       ["Bias neutralisation", "WNC [@pryzant]; DEPEN [@depen]; gender-neutral rewriting [@sun]",
        "Generate neutral rewrites, evaluated on bias and fluency",
        "No decision layer that refuses an unsafe rewrite or routes it to a person"],
       ["Detoxification", "ParaDetox [@paradetox]; ParaDeHate [@paradehate]",
        "Rewrite toxic text; content preservation checked when building data",
        "Toxicity rather than stereotypes; checks are not a run-time gate with reasons"],
       ["General-purpose LLMs", "Prompting an instruction model to “make this inclusive”", "Fluent rewrites",
        "No guarantee that facts survive"],
       ["Automatic filtering", "LGSA [@lgsa]", "Screen generated rewrites with similarity and attribute checks",
        "No per-fact checks, four-way decision or comparison with selection baselines"],
       ["LLM-as-a-judge", "MT-Bench judges [@judge]", "A model rates another model's output",
        "A single opinion with known biases; no explicit preservation criteria"]],
      widths=[30, 42, 44, 50])
P("From this review we identify five gaps that FairLens addresses:")
bullets([
    "**G1 Detection without verified correction.** Detectors flag bias but do not propose and check fixes.",
    "**G2 Bias score as the only criterion.** Rewriters are judged by bias reduction and fluency, not by whether "
    "facts, negations and roles survive.",
    "**G3 No abstention.** Rewriting systems always return an output; none defers uncertain cases to a person with "
    "reasons.",
    "**G4 Inconsistent evaluation.** Detectors are often reported on their own training source, with no false-alarm "
    "probe or calibration, and selection policies are rarely compared.",
    "**G5 Text-only scope.** Captions produced by vision–language models are rarely screened with the same tools.",
])

# =========================================================================== 4 Ethical requirements
H(1, "4 Ethical Requirements for an Automated Mitigation Tool")
P("Before choosing metrics we state what a mitigation tool owes the author of the text and its readers. Each "
  "requirement is tied to a quantity measured in Sect. 8.")
bullets([
    "**R1 Fidelity (non-maleficence).** A rewrite must not remove or change facts, reverse negations, alter "
    "quotations or reassign roles. *Measured by* the rate at which accepted rewrites lose a fact or contradict the "
    "original, and the share of selections that are damaging rewrites.",
    "**R2 Necessity (respect for autonomy).** Text that is not biased should be left alone; changing it overrides "
    "the author for no benefit. *Measured by* the unnecessary-change rate on neutral inputs.",
    "**R3 Transparency and contestability (explicability).** Each decision should come with reasons that a person "
    "can check and dispute. *Addressed by* rule-based decisions with reason codes; we report the accuracy cost of "
    "this choice against a learned gate.",
    "**R4 Oversight under uncertainty.** When the evidence is conflicting or weak, the decision should pass to a "
    "person. *Measured by* the deferral rate.",
    "**R5 Honest accounting of costs.** Deferral adds human work, and a cautious system may leave biased text "
    "unchanged. *Measured by* the deferral rate and the rate at which biased originals are kept, reported "
    "alongside the safety gains.",
])
P("These requirements conflict. R1 and R4 favour caution, while the purpose of mitigation favours change; R2 "
  "favours restraint, while detector false alarms push towards unnecessary edits. FairLens is designed to make "
  "these trade-offs visible rather than to collapse them into a single score.")

# =========================================================================== 5 System design
H(1, "5 System Design")
P("FairLens is organised in five layers (Fig. 1). The *presentation layer* is an interactive dashboard. The "
  "*application layer* exposes three pipeline functions (analyse text, mitigate text, analyse image) that are "
  "shared by the dashboard, the experiment scripts and the tests, so that the evaluated system and the deployed "
  "system are the same code. The *core services* implement the analysis flow: ingestion, bias detection, "
  "mitigation, the Reliability Gate and the decision, supported by a multimodal service and by preservation and "
  "evaluation services. The *model layer* handles configuration, model loading and hardware, and the *data layer* "
  "holds datasets, run records and caches. The following subsections describe each component.")
H(2, "5.1 Ingestion and segmentation")
P("Users can paste text or upload TXT, PDF (with a text layer), DOCX (including tables and headers), CSV (one "
  "chosen column) and JPG, PNG or GIF images. Each loader checks file type and size (10 MB per file, 60 "
  "megapixels per image) and analyses at most 20,000 characters, reporting any truncation. Text is split into "
  "sentences by an offset-preserving segmenter that is aware of quotations and abbreviations, so that every "
  "flagged phrase and every rewrite can be mapped back to its exact position in the original document.")
figure(1, "Fig1.png", 135, "High-level design of FairLens. Arrows in the core-services layer show the analysis "
       "flow from an uploaded document to a decision; the dashed arrow is the strict retry, which re-prompts the "
       "LLM rewriters with the facts that must be kept. The multimodal service feeds generated captions to the same "
       "detectors, and the preservation services supply the Gate's signals")
H(2, "5.2 Bias detection")
P("All detectors implement one interface (score a sentence, score a batch, explain a score, describe the model) "
  "and are built from a single configuration file. Four detectors are used in the pipeline: a lexicon-based rule "
  "detector, DistilBERT [@distilbert] and DeBERTa-v3-small [@deberta] fine-tuned for binary bias classification, "
  "and BART-large-MNLI used as a zero-shot classifier that compares the entailment of a “biased” and a “neutral” "
  "hypothesis [@bart; @zeroshot]. The pipeline's bias score is the mean of the four detector scores, and their "
  "spread is used as a measure of disagreement. Explanations are given as highlighted phrases: lexicon matches "
  "for the rule detector and, for the transformer detectors, the words whose removal lowers the bias score most "
  "(leave-one-word-out occlusion). A publicly available RoBERTa classifier trained on BABE is included as an "
  "external reference in the benchmarks only.")
H(2, "5.3 Mitigation")
P("Three rewriters produce candidate rewrites for each biased sentence: a rule rewriter that substitutes terms "
  "from a curated lexicon, FLAN-T5-base [@flant5] and Qwen2.5-0.5B-Instruct [@qwen]. Both LLMs receive the same "
  "instruction from one prompt family; only the wrapper differs (a sequence-to-sequence prompt or a chat "
  "template). Decoding is greedy with at most 96 new tokens, a repetition penalty of 1.1 and a fixed seed, so "
  "results are repeatable, and all prompts and outputs are logged. The unchanged original is always kept as a "
  "candidate, because for a sentence that is not biased the correct action is to change nothing.")
H(2, "5.4 Reliability Gate")
P("The Gate is the core of FairLens. For each candidate it computes the signals in Table 2 and applies "
  "transparent rules in a fixed order (Fig. 2). If the input is not judged biased, the unchanged text is "
  "accepted and any change is rejected as unnecessary. For a biased input, any preservation failure leads to "
  "rejection; if meaning is preserved but bias did not fall enough, the LLMs are re-prompted with a stricter "
  "instruction that lists the extracted facts, at most twice, after which the input goes to human review; if the "
  "detectors disagree or the score is close to the threshold, the input goes to human review; otherwise the "
  "candidate is accepted. Among several accepted candidates the one with the highest sum of bias reduction, "
  "similarity and fact retention is returned. Every verdict carries one or more reason codes, which the "
  "dashboard shows next to the evidence.")
table(2, "Signals computed by the Reliability Gate for each candidate rewrite",
      ["Signal", "Definition"],
      [["Bias reduction", "Mean detector score of the original minus that of the candidate"],
       ["Semantic similarity", "Cosine similarity of all-MiniLM-L6-v2 sentence embeddings [@sbert; @minilm]"],
       ["Entailment", "P(original ⇒ candidate) and the minimum over both directions, from an NLI cross-encoder "
                      "(DeBERTa-v3-small)"],
       ["Contradiction", "Maximum contradiction probability over both directions"],
       ["Fact retention", "Share of facts extracted from the original (named entities, numbers, dates, roles, "
                          "quotations and negation polarity, using spaCy and rules) that survive in the candidate"],
       ["Detector agreement", "Standard deviation of the detector scores on the candidate"],
       ["Uncertainty", "Combination of detector disagreement and closeness of the bias score to 0.5"],
       ["Fluency, length", "Heuristic checks for repetition, prompt leakage and non-alphabetic text; word-count "
                           "ratio"]],
      widths=[38, 122])
figure(2, "Fig2.png", 118, "Decision flow of the transparent rule Gate for one candidate rewrite, with the reason "
       "codes attached to each outcome. When several candidates are assessed, the verdicts are combined with the "
       "precedence REJECT > RETRY > HUMAN_REVIEW > ACCEPT")
_TR = GATE['tuned_rules']
P("The default thresholds were set a priori: minimum similarity "
  f"{GDEF['min_semantic_similarity']:.2f}, "
  f"{'bidirectional' if GDEF['require_bidirectional_entailment'] else 'forward'} entailment probability "
  f"{GDEF['min_entailment_prob']:.2f}, minimum bias reduction {GDEF['min_bias_reduction']:.2f} and maximum "
  f"detector standard deviation {GDEF['max_detector_std']:.2f}. They were then tuned on the validation split of a "
  f"controlled gate set only (Sect. 7.1), giving minimum similarity {_TR['min_semantic_similarity']:.2f}, "
  f"{_TR['entailment_mode']} entailment probability {_TR['min_entailment_prob']:.2f} and minimum bias reduction "
  f"{_TR['min_bias_reduction']:.2f}, with the detector limit raised to {_TR['max_detector_std']:.1f}"
  + (", which switches the disagreement check off because the standard deviation of probabilities cannot exceed 0.5"
     if _TR['max_detector_std'] > 0.5 else "")
  + ". The validation-tuned values were frozen before any test evaluation and are the ones used by the "
  "application; Table 8 reports the Gate with both the default and the validation-tuned thresholds. As a less "
  "transparent alternative, FairLens also includes learned gates (a logistic regression and a depth-4 decision "
  "tree over 14 signal features) that defer when their top class probability is below 0.6.")
H(2, "5.5 Multimodal extension")
P("For images, FairLens generates captions with BLIP-base [@blip] and Florence-2-base [@florence], measures how "
  "well each caption matches the image with CLIP ViT-B/32 [@clip], checks the caption for gendered terms and for "
  "attributes that pixels cannot normally verify (such as nationality, wealth or religion), and passes the "
  "caption to the same text detectors. This covers bias in generated captions, not visual bias in general.")
H(2, "5.6 Model and resource management")
P("FairLens is designed to run on an ordinary laptop without sending text to an external service, which matters "
  "for confidential documents such as job advertisements. A model manager loads models lazily and keeps one large "
  "model in memory at a time, evicting the least recently used. Hardware is detected at start-up and a profile "
  "is chosen: *lite* (DistilBART-MNLI and FLAN-T5-small for slower machines), *standard* or *GPU*; the device is "
  "selected in the order CUDA, Apple Metal Performance Shaders (MPS), CPU, with a manual override. Every model "
  "from the Hugging Face Hub is pinned to a specific commit.")
H(2, "5.7 Experiment tracking and reproducibility")
P("Every experiment writes a manifest that records its configuration, seeds, dataset hashes, pinned model "
  "revisions, device, hardware, software versions and runtime, together with its outputs and a row in a SQLite "
  "index. Detector scores and generations are cached by model, revision, device and text, and experiments run "
  "stage by stage so that each model is loaded once. The dashboard and the results reported here read only these "
  "stored files; the tables in this paper were filled programmatically from them.")

# =========================================================================== 6 Implementation
H(1, "6 Implementation")
H(2, "6.1 Technology and code structure")
P("FairLens is written in Python 3.11 with PyTorch 2.14.1, Hugging Face Transformers 5.20.2, sentence-transformers, "
  "spaCy 3.8.16 and scikit-learn; the dashboard uses Streamlit 1.66.1 and Plotly, and run records use JSON and SQLite. "
  "All dependencies are pinned, and installer scripts set up the same environment on Windows, macOS and Linux. "
  "Table 3 lists the main modules.")
table(3, "Main modules of the FairLens code base",
      ["Module", "Responsibility"],
      [["ingestion", "File loaders with size and format checks; offset-preserving sentence segmentation"],
       ["detection", "Common detector interface; rule lexicon; fine-tuned classifiers; zero-shot NLI; registry"],
       ["mitigation", "Shared prompt family; rule rewriter; FLAN-T5 and Qwen2.5 adapters"],
       ["evaluation", "Fact extractor; similarity, entailment and fluency; metrics (F1, ECE, bootstrap CIs)"],
       ["reliability_gate", "Signals; transparent rules; strict retry; learned gates; baselines; LLM judge"],
       ["multimodal", "BLIP and Florence-2 captioners; CLIP alignment; caption checks"],
       ["experiments", "Run tracking (manifests and SQLite); score and generation caches; batched runners; "
                       "background jobs"],
       ["reporting", "Charts; export of results as PDF, JSON, CSV and HTML"],
       ["ui", "Reusable components and the dashboard pages"]],
      widths=[35, 125])
H(2, "6.2 Application and review workflow")
P("The dashboard has twelve pages grouped into a workflow section (Home, FairLens Review, Text Analysis, "
  "Mitigation, Reliability Gate, Image Analysis), a research section (Benchmarks, Model Comparison, Error "
  "Analysis, Experiments and Exports) and supporting pages (Use Cases, About), with light and dark themes. The "
  "FairLens Review page (Fig. 3) guides a user through a six-step check of a job advertisement: (1) paste or "
  "upload the text; (2) see the flagged phrases and which detectors agree; (3) compare the rule, FLAN-T5 and "
  "Qwen2.5 rewrites with a word-level difference view; (4) check that numbers, names, dates, roles, quotations "
  "and negations survived; (5) read the Gate's verdict and reason codes; and (6) export a PDF or JSON report. The "
  "page states that FairLens is a review assistant and not a legal or compliance certification tool, and that a "
  "person makes the final decision.")
figure(3, "Fig3.png", 120, "The FairLens Review page: a guided six-step check of a job description, from input "
       "to flagged phrases, rewrites, fact checks, the Gate's verdict and export. The notice at the top states that "
       "the tool assists a human reviewer and does not certify text as unbiased")
P("Fig. 4 shows steps 3 and 5 of this workflow in the running application for the advertisement sentence "
  "“Looking for a young, aggressive salesman with 5 years of experience in B2B software sales.” The rule rewrite "
  "(“Looking for a proactive salesperson with 5 years of experience in B2B software sales.”) is accepted with the "
  "reason code ALL_CHECKS_PASSED, a semantic similarity of 0.85, forward entailment of 0.97 and full fact "
  "retention. FLAN-T5 returns the input unchanged and receives RETRY, and Qwen2.5's rewrite receives REJECT. The "
  "Detailed Analysis and Explanation tabs list each preservation check, the detector scores before and after the "
  "rewrite, and the rule that produced the verdict. This is an illustrative example, not a benchmark result.")
figure(4, "Fig4.png", 105, "An example from the running application. **a** The Mitigation page shows the original "
       "sentence beside the rule-based, FLAN-T5 and Qwen2.5 rewrites, each with its Gate verdict, change in bias "
       "score, similarity and fact retention, and the Gate's recommendation. **b** The Reliability Gate page shows the "
       "verdict for the selected rewrite with its reason code and the evaluation metrics against their thresholds")
P("The research pages show results from completed runs only (Fig. 5). Each result is labelled as a completed run, "
  "pending, a failed run or an illustrative example, and shows the identifier of the run it was read from. "
  "Uploaded text is processed locally, is not stored and never changes the benchmark results. Long "
  "experiments can be started from the dashboard as background jobs that can be cancelled.")
figure(5, "Fig5.png", 120, "The Benchmarks page of the dashboard. Its charts, the mean bias score on biased inputs "
       "before and after each rewriter and the Gate's decisions on the real rewrites, are read from stored run files")
H(2, "6.3 Testing")
P("The code base has an automated test suite covering ingestion of every supported format and its edge cases, "
  "segmentation, the fact extractor and Gate rules, metrics, run tracking, the pipeline end to end, and the "
  "dashboard pages, including a test that no page displays a metric that is not read from a run file. On the "
  "original development machine, before the controlled set was re-split (Sect. 7.1), 243 fast tests and 2 slow "
  "tests that load the real models passed, a browser-driven test of the review flow (with model calls stubbed) "
  "passed all 54 checks, and a continuous-integration workflow passed the test suite on Windows, macOS and Linux. "
  f"After the re-split, the suite ({T_ALL - T_SLOW} fast and {T_SLOW} slow tests) was run again on the Windows "
  f"laptop used for the re-runs (Sect. 7.4): {T_PASS} tests passed in the project environment, and the "
  + ("remaining test" if T_EXTRA == 1 else f"remaining {T_EXTRA} tests")
  + ", which fits the learned gates, passed under the Anaconda interpreter, because the device's Application Control "
  "policy blocks a compiled scikit-learn library in the project environment. The browser test and the "
  "continuous-integration workflow were not repeated after the re-split. The script that builds the controlled set "
  "also checks that no source sentence appears in more than one split, and stops if one does; this check runs "
  "whenever the set is built.")

# =========================================================================== 7 Experimental setup
H(1, "7 Experimental Setup")
H(2, "7.1 Datasets")
P("Table 4 lists the data. For detection, BABE [@babe], StereoSet (intra- and inter-sentence) [@stereoset] and "
  "CrowS-Pairs [@crows] were unified into one binary schema (1 = the sentence expresses or relies on a stereotype "
  "or biased framing under the source's own annotation). Splits (70/15/15, seed 42) were made by group, so that "
  "the two members of a minimal pair never fall into different splits, giving 10,365 training, 2,119 validation "
  "and 2,652 test sentences. WinoBias [@winobias] was used only as a false-alarm probe. The Wiki Neutrality "
  "Corpus [@pryzant] could not be obtained (its host was unreachable and the only mirror is gated).")
P("For mitigation, we hand-wrote 136 sentences with fictional names (91 biased, 45 neutral) covering recruitment "
  "text, gender, age, race, religion, disability and socioeconomic stereotypes and media framing, deliberately "
  "including names, numbers, dates, negations, roles and quotations; each biased sentence has one reference "
  "rewrite. Together with 140 sentences from the *test* split of the detection benchmarks they form a mitigation "
  "set of 276 inputs. To evaluate the Gate against known correct answers we built a controlled set: for each "
  "biased input the reference rewrite is ACCEPT, the unchanged original is RETRY and seeded perturbations of the "
  "reference are REJECT (number changed, negation flipped, entity swapped, role swapped, antonym, hallucinated "
  "addition, truncation, altered quotation); for each neutral input the unchanged original is ACCEPT and any "
  f"change is REJECT. Three perturbation seeds give {GATE['n_candidates']:,} candidates. The split into validation "
  "and test is made by source sentence: each of the 136 sentences is assigned once, stratified by label, so all "
  "three perturbation seeds of a sentence, and its reference rewrite, fall in the same half. The test split has "
  f"{N_TEST // 3} sentences, giving {N_TEST} inputs (sentence × seed; {N_TEST_NEUTRAL} of them neutral, from "
  f"{N_TEST_NEUTRAL // 3} neutral sentences) and {N_TEST_CAND:,} candidates.")
table(4, "Datasets used in the evaluation",
      ["Dataset", "Licence", "Used for", "Size used", "Label meaning"],
      [["BABE [@babe]", "CC BY-NC-SA 4.0", "Detection", "4,118 sentences", "Expert-labelled media bias in news"],
       ["StereoSet [@stereoset]", "CC BY-SA 4.0", "Detection, probe", "8,440 sentences",
        "Stereotype = 1, anti-stereotype = 0"],
       ["CrowS-Pairs [@crows]", "CC BY-SA 4.0", "Detection, probe", "2,578 sentences",
        "More stereotypical sentence = 1"],
       ["WinoBias [@winobias]", "MIT", "False-alarm probe", "1,584 sentences", "All neutral; any flag is a false alarm"],
       ["Authored cases (this work)", "–", "Mitigation, Gate", "136 sentences",
        "91 biased, 45 neutral, with reference rewrites"],
       ["Mitigation set", "Derived", "Mitigation", "276 inputs", "211 biased, 65 neutral"],
       ["Controlled gate set", "Derived", "Gate", "2,481 candidates", "Gold ACCEPT / RETRY / REJECT by construction"],
       ["COCO val2017 subset [@coco]", "CC BY 4.0 + Flickr", "Captions", "80 images",
        "60 with people, 20 without; 5 human captions each"]],
      widths=[38, 26, 26, 26, 50])
H(2, "7.2 Training and configuration")
P("DistilBERT and DeBERTa-v3-small were fine-tuned on the training split (AdamW, learning rate 5×10⁻⁵, 10% "
  "linear warm-up, batch size 16, best epoch chosen by validation macro-F1, seed 42). For DeBERTa the embedding "
  "layer was frozen, because its 128k-token embedding matrix with optimiser state did not fit in laptop memory; "
  "all classifiers were loaded in 32-bit floating point. For BART-MNLI, five hypothesis templates were compared "
  "on 300 validation sentences. The LLM prompt variant was chosen on 30 biased sentences from the training split; "
  "the evaluation sets were never used for prompt choice. The learned gates were trained on the validation half "
  "of the controlled set.")
H(2, "7.3 Baselines and metrics")
P("Each selection policy either picks one candidate per input (possibly the unchanged original) or defers. "
  "Besides the rule and learned gates we evaluate choosing the candidate with the *lowest bias score*; choosing "
  "the changed candidate with the *highest similarity*; *keeping the original*; always taking the *first model's* "
  "output (real rewrites only); *random* choice (five seeds); and an *LLM-as-judge* [@judge] in which "
  "Qwen2.5-0.5B-Instruct is asked whether a changed candidate preserves the meaning, with the prompt chosen among "
  "three variants on the validation split by Cohen's κ and abstentions counted as deferrals.")
P("On the controlled set a selection is *acceptable* if its gold label is ACCEPT and *harmful* otherwise; we "
  "report the two kinds of harm separately (keeping a biased original, gold RETRY; choosing a damaging rewrite, "
  "gold REJECT). At the candidate level we report three-class macro-F1 and the *unsafe-accept rate*, P(accept | "
  "gold ≠ ACCEPT). On real rewrites, where no gold labels exist for model outputs, we report the reduction in "
  "mean bias score, the rate at which an accepted rewrite loses an extracted fact or is contradicted, the "
  "unnecessary-change rate on neutral inputs, the deferral rate and the rate at which biased inputs are left "
  "unchanged. Detection is evaluated with precision, recall, F1, macro-F1, ROC-AUC and expected calibration error "
  "(ECE, 10 bins) [@guo], per source and pooled.")
H(2, "7.4 Statistics and compute")
P("Confidence intervals are 95% percentile bootstrap intervals (1,000 resamples unless stated); differences "
  "between policies and ablations use paired bootstrap. Seeds were fixed for data splitting and training (42), "
  "generation (13), perturbations (11, 22, 33) and random baselines (100–104). Detector training, rewrite "
  "generation and all model-based signals (detector scores, similarity, entailment, fact checks) come from single "
  "runs on one laptop (Apple M4, 16 GB memory, MPS backend). After a leakage check showed that the original "
  "validation/test split of the controlled set was made per seed rather than per sentence, the controlled-set and "
  "Gate experiments were re-run on a Windows laptop (CPU): the split, threshold tuning, learned gates, LLM-judge "
  "prompt selection and judge answers, and all Gate and policy decisions (Tables 7–9, Fig. 8) were recomputed, "
  "reusing the stored signals and rewrites of the original runs unchanged. Because of Windows application-control "
  f"restrictions this analysis used Python {ENV['python']}, pandas {ENV['packages']['pandas']} and scikit-learn "
  f"{ENV['packages']['scikit-learn']}, and the judge model ran with PyTorch {JENV['packages']['torch']} and "
  f"Transformers {JENV['packages']['transformers']}; the original pinned environment is unchanged for everything else. "
  "Every run records its configuration, device and package versions.")
H(2, "7.5 Use of generative AI")
P("Two LLMs (FLAN-T5-base and Qwen2.5-0.5B-Instruct) are components of the framework, as described above. The "
  "research idea, including the Reliability Gate, as well as the framework design, choice of datasets, metrics and "
  "experiments, and the interpretation of results are the authors' own work. The authors used an AI coding "
  "assistant (Claude, Anthropic) for support in implementing and testing parts of the software and in drafting and "
  "editing parts of this manuscript. All experimental results come from the stored runs; the authors reviewed all "
  "code and text, checked every reported value against the run outputs, and take full responsibility for the "
  "content.")

# =========================================================================== 8 Results
H(1, "8 Results")
H(2, "8.1 Bias detection")
names = {"rule_based": "Rule-based", "bert": "DistilBERT (fine-tuned)", "deberta": "DeBERTa-v3-small (fine-tuned)",
         "bart_mnli": "BART-MNLI (zero-shot)", "hub_pretrained": "Public RoBERTa (BABE-trained)"}
rows = []
for k, label in names.items():
    d = DET["detectors"][k]
    m, wb = d["metrics"], d["probes"]["winobias"]
    rows.append([label, f3(m["precision"]), f3(m["recall"]), ci(d["ci"]["macro_f1"]), f3(m["roc_auc"]),
                 f3(m["ece"]), f"{wb['fpr_pro']:.3f} / {wb['fpr_anti']:.3f}", f"{d['ms_per_sample']:.1f}"])
P("Table 5 and Fig. 6 summarise detection on the 2,652 test sentences. Fine-tuned DeBERTa-v3-small was the "
  f"strongest detector (macro-F1 {dd['ci']['macro_f1']['value']:.3f}, 95% CI "
  f"{dd['ci']['macro_f1']['lo']:.3f}–{dd['ci']['macro_f1']['hi']:.3f}) and the most consistent across sources "
  "(0.811–0.817), followed by DistilBERT (0.777). The public RoBERTa classifier scored highest on BABE (0.853), "
  "the source it was trained on, but at or below 0.544 on the stereotype datasets, a reminder that a single "
  "benchmark number does not transfer across definitions of bias. Zero-shot BART-MNLI reached 0.611 and the rule "
  "baseline 0.354, with very low recall (0.035).")
table(5, f"Detection on the shared test split (n = {DET['n_test']:,}, threshold 0.5). Macro-F1 with 95% bootstrap "
      "CI. WinoBias FPR is the share of neutral WinoBias sentences flagged as biased in the pro- and "
      "anti-stereotypical conditions",
      ["Detector", "Precision", "Recall", "Macro-F1 [95% CI]", "ROC-AUC", "ECE", "WinoBias FPR (pro / anti)",
       "ms per sentence"], rows, widths=[34, 16, 14, 30, 15, 12, 25, 14])
wb_b = DET["detectors"]["bert"]["probes"]["winobias"]
gap = dd["probes"]["winobias"]["mean_gap_pro_minus_anti"]
P("High benchmark accuracy did not mean few false alarms on everyday text. The fine-tuned detectors flagged "
  f"{pct(dd['probes']['winobias']['fpr_pro'])} and {pct(dd['probes']['winobias']['fpr_anti'])} (DeBERTa) and "
  f"{pct(wb_b['fpr_pro'])} and {pct(wb_b['fpr_anti'])} (DistilBERT) of the neutral WinoBias sentences. DeBERTa also "
  f"scored pro-stereotypical sentences slightly higher than anti-stereotypical ones (mean gap {gap['mean']:.3f}, "
  f"CI {gap['lo']:.3f}–{gap['hi']:.3f}). On minimal pairs, DeBERTa preferred the more stereotypical CrowS-Pairs "
  f"sentence in {pct(dd['probes']['crows_pairs']['pref_more_stereo']['mean'])} of pairs. Calibration errors of "
  "about 0.12 for the fine-tuned models mean their scores should not be read as probabilities. In a qualitative "
  "check, the sentence “Women are not suited to engineering, …” received a mean detector score of 0.48, just "
  "below the threshold, so the pipeline left it unchanged. Both kinds of error matter ethically: false alarms "
  "invite needless rewriting of an author's words (R2), and misses leave biased text in place (R5).")

figure(6, "Fig6.png", 135, "Macro-F1 of each detector by data source on the test split. The public RoBERTa "
       "classifier was trained on BABE and does not generalise to the stereotype datasets; the fine-tuned "
       "DeBERTa-v3-small detector is consistent across all four sources. The dotted line marks 0.5")
H(2, "8.2 Rewriting: bias reduction against preservation")
rnames = {"rule_based": "Rule rewriter", "flan_t5": "FLAN-T5-base", "qwen": "Qwen2.5-0.5B-Instruct",
          "original": "No change (reference)"}
rows = []
for k, label in rnames.items():
    r_ = rw[k]
    c = r_["bias_reduction_ci"]
    # Similarity and entailment are averaged over changed outputs only, so they are undefined for "No change".
    unchanged_only = r_["changed_rate"] == 0
    rows.append([label, f"{r_['mean_bias_reduction_biased']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}]",
                 pct(r_["changed_rate"]), "–" if unchanged_only else f3(r_["semantic_similarity"]),
                 "–" if unchanged_only else f3(r_["entailment_both"]),
                 f3(r_["fact_retention"]), pct(r_["fact_failure_rate"]), pct(r_["unnecessary_change_rate_neutral"])])
P(f"Table 6 and Fig. 7 compare the rewriters on the {MIT['n_inputs']} mitigation inputs. Qwen2.5 reduced the mean "
  f"bias score the most ({rw['qwen']['mean_bias_reduction_biased']:.3f}) but lost at least one extracted fact in "
  f"{pct(rw['qwen']['fact_failure_rate'])} of its rewrites and changed "
  f"{pct(rw['qwen']['unnecessary_change_rate_neutral'])} of the neutral inputs that needed no change. The rule "
  f"rewriter achieved a similar reduction ({rw['rule_based']['mean_bias_reduction_biased']:.3f}) with a fact "
  f"failure rate of {pct(rw['rule_based']['fact_failure_rate'])}, because it only substitutes terms. FLAN-T5 "
  f"changed only {pct(rw['flan_t5']['changed_rate'])} of inputs and reduced bias by "
  f"{rw['flan_t5']['mean_bias_reduction_biased']:.3f}: in development tests with several prompt formats and "
  "few-shot examples it mostly copied its input, a capability limit we report rather than hide. Bias reductions "
  "are small in absolute terms because the mean detector score includes the rule and zero-shot detectors, which "
  "respond weakly to rewording.")
table(6, "Rewriters on the mitigation set (276 inputs; 211 biased, 65 neutral). Bias reduction is the mean drop in "
      "detector score on biased inputs (95% CI). Similarity and entailment are averaged over changed outputs only "
      "(undefined for the unchanged reference); fact retention and fact failures are computed over all outputs, "
      "where an unchanged output keeps every fact. Fact failures: share of outputs losing at least one extracted "
      "fact. Unnecessary change: share of neutral inputs that were changed. All measures are automatic",
      ["Rewriter", "Bias reduction [95% CI]", "Changed", "Similarity", "Entailment (both ways)", "Fact retention",
       "Fact failures", "Unnecessary change"], rows, widths=[32, 30, 15, 17, 20, 16, 15, 17])
figure(7, "Fig7.png", 84, "Bias reduction against fact loss for each rewriter on the mitigation set. The rewriter "
       "that lowers bias scores most also loses facts most often; the rule rewriter achieves a similar reduction "
       "with few fact losses because it only substitutes terms")
pol = MIT["policies"]
pnames = {"fairlens_gate": "Rule Gate", "fairlens_gate_retry": "Rule Gate + strict retry",
          "lowest_bias": "Lowest bias score", "highest_similarity": "Highest similarity",
          "always_flan_t5": "Always first model (FLAN-T5)", "random_mean": "Random (mean of 5 seeds)"}
rows = []
for k, label in pnames.items():
    p_ = pol[k]
    rows.append([label, pct(p_["accepted_fact_failure_rate"]), pct(p_["accepted_contradiction_rate"]),
                 pct(p_["unnecessary_change_rate"]), f3(p_["mean_bias_reduction_biased"]), pct(p_["defer_rate"]),
                 pct(p_["biased_left_unchanged_rate"])])
g, rt, idet = pol["fairlens_gate"], MIT["retry"], MIT["input_detection"]
P("Table 7 applies the selection policies to these real rewrites. The Gate accepted no rewrite that lost a fact or "
  f"was contradicted by the NLI model, against {pct(pol['lowest_bias']['accepted_fact_failure_rate'])} and "
  f"{pct(pol['lowest_bias']['accepted_contradiction_rate'])} for lowest-bias selection, and it made no unnecessary "
  f"changes. The cost was {pct(g['defer_rate'])} of inputs deferred to a person and "
  f"{pct(g['biased_left_unchanged_rate'])} of biased inputs left unchanged. The latter matches the detection rate: "
  f"the detectors flagged only {pct(idet['pred_biased_rate_on_gold_biased'])} of the biased inputs, and the Gate, "
  f"by design, does not rewrite text it does not judge biased. The strict retry rarely helped: of "
  f"{rt['inputs_retried']} inputs retried, {rt['accepted_after_retry']} were accepted after retry, which suggests "
  "that a 0.5-billion-parameter model cannot reliably follow an instruction to keep listed facts.")
table(7, "Selection policies on real rewrites (276 inputs). Fact loss and contradiction are measured on the chosen "
      "output; deferred inputs choose nothing. Bias reduction is the mean drop in detector score over the biased "
      "inputs for which the policy chose an output (deferred inputs are excluded and a kept original counts as 0), "
      "so a policy that picks the best candidate per input or defers difficult inputs can exceed any single "
      "rewriter in Table 6",
      ["Policy", "Accepted with fact loss", "Accepted with contradiction", "Unnecessary change", "Bias reduction",
       "Deferred", "Biased left unchanged"], rows, widths=[40, 20, 22, 20, 18, 17, 22])

H(2, "8.3 The Reliability Gate on known correct decisions")
snames = {"lowest_bias": "Lowest bias score", "highest_similarity": "Highest similarity",
          "no_change": "Keep original", "llm_judge": "LLM-as-judge (Qwen2.5)",
          "fairlens_gate_default": "Rule Gate (default thresholds)",
          "fairlens_gate_tuned": "Rule Gate (validation-tuned)", "learned_logreg": "Learned gate (logistic)",
          "learned_tree": "Learned gate (decision tree)", "random_mean": "Random (mean of 5 seeds)"}
rows = []
for k, label in snames.items():
    s = sel[k]

    def cell(key):
        v = s[key]
        if isinstance(v, dict):
            return f"{v['mean']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]" if v.get("lo") is not None else f"{v['mean']:.3f}"
        return f"{v:.3f}"

    def mean(key):
        v = s[key]
        return f"{v['mean']:.3f}" if isinstance(v, dict) else f"{v:.3f}"

    cg = s.get("chosen_gold_counts")
    n = sum(cg.values()) if cg else None
    rows.append([label, cell("success"), cell("harmful_selection"),
                 f"{cg.get('RETRY', 0) / n:.3f}" if cg else "–", f"{cg.get('REJECT', 0) / n:.3f}" if cg else "–",
                 mean("defer"), mean("unnecessary_change_neutral")])
P(f"Table 8 and Fig. 8 report input-level selection on the {tg['success']['n']} test inputs of the controlled set. "
  f"Choosing the least-biased candidate was almost always wrong: it chose a damaging rewrite for "
  f"{lb_['chosen_gold_counts']['REJECT']} of {lb_['success']['n']} inputs "
  f"({pct(lb_['chosen_gold_counts']['REJECT'] / lb_['success']['n'])}), because the candidate with the lowest bias "
  "score is often one that dropped or changed content. Highest-similarity selection made more acceptable choices "
  f"than the rule Gate ({pct(sel['highest_similarity']['success']['mean'])} against {pct(tg['success']['mean'])}), "
  "but every one of its errors was a damaging rewrite and it changed every neutral input. The LLM judge made "
  f"acceptable choices for {pct(sel['llm_judge']['success']['mean'])} of inputs and changed "
  f"{pct(sel['llm_judge']['unnecessary_change_neutral']['mean'])} of neutral inputs, because it was only asked "
  "about faithfulness.")
P(f"The validation-tuned rule Gate's harmful-choice rate was {pct(tg['harmful_selection']['mean'])}, but "
  + (f"all {tg['chosen_gold_counts'].get('RETRY', 0)} of these cases were biased originals kept unchanged; it never "
     "chose a damaging rewrite. It " if GATE_DMG == 0 else
     f"{tg['chosen_gold_counts'].get('RETRY', 0)} of these "
     f"{tg['chosen_gold_counts'].get('RETRY', 0) + tg['chosen_gold_counts'].get('REJECT', 0)} cases were biased "
     f"originals kept unchanged; it {GATE_DMG_PHRASE}. It ")
  + f"deferred {pct(tg['defer']['mean'])} of inputs and made no unnecessary changes. The learned decision tree was the "
  f"most accurate policy ({pct(tree['success']['mean'])} acceptable choices, {pct(tree['defer']['mean'])} deferred), "
  f"but it chose a damaging rewrite {tree['chosen_gold_counts']['REJECT']} times and changed "
  f"{pct(tree['unnecessary_change_neutral']['mean'])} of neutral inputs.")
table(8, f"Input-level selection on the controlled test split ({N_TEST} inputs). Harmful = gold label not ACCEPT, split "
      f"into keeping a biased original and choosing a damaging rewrite. Unnecessary change is measured on the {N_TEST_NEUTRAL} "
      "neutral test inputs. 95% bootstrap CIs in brackets (none for the random baseline, a mean over seeds)",
      ["Policy", "Acceptable choice", "Harmful choice", "of which: kept biased original",
       "of which: damaging rewrite", "Deferred", "Unnecessary change"], rows, widths=[36, 30, 30, 17, 17, 15, 15])
cl = GATE["candidate_level"]
tun, orc = cl["rules_val_tuned"], cl["rules_val_tuned_oracle_input"]
conf = tun["confusion"]
n_bad = conf["REJECT"].get("ACCEPT", 0)
catch = [v for k, v in tun["catch_rate_by_perturbation"].items() if k != "unchanged"]
catch_lo, catch_hi = 100 * min(catch), 100 * max(catch)
P(f"At the candidate level the tuned rule Gate reached a three-class macro-F1 of {tun['macro_f1_3class']:.3f} with "
  f"an unsafe-accept rate of {pct(tun['unsafe_accept_rate'])}. Of its unsafe accepts, "
  f"{conf['RETRY'].get('ACCEPT', 0)} were unchanged biased originals accepted because the detectors did not judge "
  f"the input biased, and "
  + ("none was a corrupted rewrite. " if n_bad == 0 else
     f"{n_bad} {'was a corrupted rewrite' if n_bad == 1 else 'were corrupted rewrites'}. ")
  + "When the gold input label was "
  f"supplied instead of the detectors' judgement, the unsafe-accept rate fell to {pct(orc['unsafe_accept_rate'])} "
  f"and macro-F1 rose to {orc['macro_f1_3class']:.3f}. The Gate's remaining failures are therefore inherited "
  "mainly from detection, not from its preservation checks. The Gate caught "
  + (f"{catch_hi:.0f}%" if round(catch_lo) == round(catch_hi) else f"{catch_lo:.0f}–{catch_hi:.0f}%")
  + " of every planted corruption "
  "type; this is partly by construction, because several perturbations target exactly what the fact checker "
  "extracts (Sect. 10).")

figure(8, "Fig8.png", 140, f"Outcome of each selection policy on the {N_TEST} test inputs of the controlled set. "
       "Hatching distinguishes outcomes in greyscale. Lowest-bias selection almost always picks a damaging rewrite; "
       f"the rule Gate {'never does' if GATE_DMG == 0 else 'rarely does'}, but defers {pct(GATE_DEFER)} of inputs and "
       f"keeps the biased original for {pct(GATE_KEPT)} of inputs; the learned decision tree is the most accurate but "
       f"picks a damaging rewrite for {pct(TREE_DMG)} of inputs and is less transparent")
H(2, "8.4 Which signals matter")
abl = GATE["ablations"]
anames = {"no_bias": "Bias reduction", "no_entailment": "Entailment", "no_facts": "Fact retention",
          "no_semantic": "Semantic similarity", "no_agreement": "Detector agreement",
          "no_uncertainty": "Uncertainty", "no_fluency": "Fluency"}
rows = [["None (full Gate)", f"{abl['full']['unsafe_accept_rate']:.3f}", "–", f"{abl['full']['macro_f1_3class']:.3f}", "–"]]
for k, label in anames.items():
    a = abl[k]
    du, da = a["delta_unsafe_accept"], a["delta_accuracy"]
    rows.append([label, f"{a['metrics']['unsafe_accept_rate']:.3f}",
                 f"{du['diff']:+.3f} [{du['lo']:+.3f}, {du['hi']:+.3f}]", f"{a['metrics']['macro_f1_3class']:.3f}",
                 f"{da['diff']:+.3f} [{da['lo']:+.3f}, {da['hi']:+.3f}]"])
P("Table 9 removes one signal group at a time from the tuned rule Gate. Removing the bias-reduction check raised "
  f"the unsafe-accept rate by {100 * abl['no_bias']['delta_unsafe_accept']['diff']:.1f} percentage points and "
  f"removing entailment by {100 * abl['no_entailment']['delta_unsafe_accept']['diff']:.1f}; removing fact retention "
  f"had a small but non-zero effect ({100 * abl['no_facts']['delta_unsafe_accept']['diff']:.1f} points), largely "
  "because entailment also catches most planted corruptions. Detector agreement, uncertainty and fluency had no "
  "measurable effect on this set. We report this as a negative result: these signals mainly route inputs to human "
  "review, and the controlled set contains few cases where that matters.")
table(9, f"Ablations of the validation-tuned rule Gate on the controlled test candidates (n = {N_TEST_CAND:,}). Differences "
      "relative to the full Gate with paired bootstrap 95% CIs. Δ unsafe accept uses the same denominator as the "
      "Unsafe-accept column (candidates whose gold label is not ACCEPT); Δ accuracy is the change in three-class "
      "accuracy over all candidates",
      ["Signal removed", "Unsafe accept", "Δ unsafe accept [95% CI]", "Macro-F1", "Δ accuracy [95% CI]"],
      rows, widths=[40, 22, 40, 20, 40])

H(2, "8.5 Image captions")
P("On 80 COCO validation images (60 showing people) [@coco], Florence-2 captions were longer and as well aligned "
  "with the images as the human reference captions (CLIP similarity 0.306 against 0.304 for references and 0.281 "
  "for BLIP). Gendered terms appeared in 53.3% (BLIP) and 60.0% (Florence-2) of captions of images with people, "
  "against 73.3% of images with at least one gendered term in the human references. No caption contained an "
  "attribute that pixels cannot normally verify. The text detectors flagged 47.5% and 68.8% of captions as "
  "biased; given the false-alarm rates in Sect. 8.1 we do not read these as bias rates.")

# =========================================================================== 9 Discussion
H(1, "9 Discussion")
H(2, "9.1 A bias score is not an acceptance criterion")
P("The central finding is that optimising for the bias score selects the wrong rewrite. On the controlled set, "
  f"lowest-bias selection chose a damaging rewrite for {pct(share(lb_, 'REJECT'))} of inputs; on real rewrites it accepted outputs that "
  "lost a fact a quarter of the time. The stronger rewriter was the less faithful one. A mitigation tool evaluated "
  "only on bias reduction can therefore look successful while harming the people who rely on the text, which is a "
  "direct failure of non-maleficence (R1). We recommend that mitigation work report preservation measures and "
  "unnecessary-change rates alongside bias reduction, and that the definition of “what must be preserved” be made "
  "explicit for each domain, as the fact types in FairLens do for recruitment and news text.")
H(2, "9.2 Abstention moves work to people and leaves some bias in place")
P(f"The Gate's safety comes from abstaining. It deferred {pct(GATE_DEFER)} of controlled inputs and "
  f"{pct(MIT['policies']['fairlens_gate']['defer_rate'])} of real ones; it kept the biased original for "
  f"{pct(GATE_KEPT)} of controlled inputs and left {pct(MIT['policies']['fairlens_gate']['biased_left_unchanged_rate'])} "
  "of biased real inputs unchanged. Both are ethical costs (R5). Deferral is only responsible "
  "where a qualified person will actually review the case; otherwise it shifts the burden to whoever receives the "
  "text, and a reviewer facing many deferrals may come to rubber-stamp them. Keeping biased originals is the price "
  "of not rewriting text the detectors do not flag; lowering the threshold would catch more bias but, given "
  "false-alarm rates near 40% on neutral WinoBias sentences, would also rewrite many authors' words for no reason "
  "(R2). Our oracle analysis shows that better detection, not more rewriting freedom, is the main route to fewer "
  "misses. Organisations deploying such tools should decide explicitly how much review capacity they have and "
  "which error they would rather make.")
H(2, "9.3 Transparency has an accuracy cost")
P(f"A learned decision tree made acceptable choices for {pct(tree['success']['mean'])} of inputs against "
  f"{pct(tg['success']['mean'])} for the rule Gate, with fewer deferrals ({pct(share(tree, 'DEFER'))} against "
  f"{pct(GATE_DEFER)}), but it chose a damaging rewrite for {pct(TREE_DMG)} of inputs and changed "
  f"{pct(tree['unnecessary_change_neutral']['mean'])} of neutral inputs, "
  + ("while the rule Gate did neither" if GATE_DMG == 0 and tg['unnecessary_change_neutral']['mean'] == 0 else
     f"against {pct(GATE_DMG)} and {pct(tg['unnecessary_change_neutral']['mean'])} for the rule Gate")
  + ". The rule Gate's advantages are that each "
  "verdict comes with reason codes that an author can check and contest (R3), that it "
  + ("never changed a neutral input" if tg['unnecessary_change_neutral']['mean'] == 0 else "rarely changed a neutral input")
  + ", and that it "
  + ("never chose a damaging rewrite" if GATE_DMG == 0 else "rarely chose a damaging rewrite")
  + ". In "
  "settings classed as high-risk, such as recruitment [@euaiact], we consider these properties worth the lower "
  "accuracy, but this is a value judgement that deployers should make openly rather than a technical result.")
H(2, "9.4 What a tool like this should and should not claim")
P("FairLens is a review assistant. It cannot certify that a text is unbiased: detectors encode particular, "
  "culturally situated definitions of bias, and they make errors in both directions. Presenting such a tool as "
  "proof of compliance would be a form of fairness-washing. Its appropriate role is to show an author what may be "
  "biased, offer alternatives, state what each alternative would change and say when it is unsure, leaving the "
  "final decision with the author. Running the whole pipeline locally also keeps confidential documents away from "
  "external services, at the cost of using smaller models.")

# =========================================================================== 10 Limitations
H(1, "10 Limitations")
bullets([
    "The controlled gold set and the reference rewrites were written by one author, and its perturbations are "
    "synthetic; several target exactly what the fact checker extracts, so the high catch rates are partly by "
    "construction. Perturbations that only entailment or similarity can detect (antonyms, additions, truncation) "
    "are a fairer test and were also caught, but the set is small.",
    "Mitigation quality was measured automatically; there is no human evaluation of the real rewrites yet. "
    "Entailment in both directions is strict for bias rewrites, which legitimately remove content.",
    "Each experiment was run once. Confidence intervals reflect sampling of test items, not "
    "variation across training seeds or hardware.",
    "The controlled-set and Gate experiments were re-run on a different device and software environment (Windows, "
    "CPU; Sect. 7.4) from the original training and signal runs. They reuse the stored signals, so only the "
    "analysis environment differs, except for the LLM-as-judge baseline, whose answers were generated on the new "
    "device.",
    "All data are English. The detectors were trained on news framing and crowd-sourced stereotype sentences, not "
    "on job advertisements; the labels of StereoSet and CrowS-Pairs have known validity problems [@blodgett2021]. "
    "The public RoBERTa reference was trained on BABE, so its BABE score is optimistic.",
    "The rewriters are small (0.5 billion and 250 million parameters). Larger models may preserve facts better, "
    "which would change the trade-off in Fig. 7 but not the need to check it.",
    "The BART-MNLI hypothesis template chosen on validation for the detection benchmark was not propagated to the "
    "Gate, the mitigation runs and the application, which used the default template with lower validation "
    "macro-F1; re-running these experiments with the chosen template may change the Gate's input-level detection.",
    "The image experiment used 80 images and measures only caption wording.",
])

# =========================================================================== 11 Future work
H(1, "11 Future Work")
bullets([
    "A blinded human evaluation of real rewrites and of the Gate's decisions, with several annotators.",
    "Repeated training runs with different seeds, and verification of all results on a second device and on CPU.",
    "A larger set of real job advertisements and other domain texts, with domain-specific definitions of the facts "
    "that must be preserved.",
    "A hybrid gate that keeps the rule Gate's hard preservation checks and reason codes but learns its thresholds, "
    "together with risk–coverage curves and threshold-sensitivity analysis.",
    "Red-team tests in which real accepted rewrites are corrupted, to measure the Gate beyond synthetic "
    "perturbations.",
    "Larger local rewriters and multilingual detectors, and propagation of the validation-selected zero-shot "
    "template to the full pipeline.",
])

# =========================================================================== 12 Conclusion
H(1, "12 Conclusion")
P("FairLens is an automated framework that detects bias in text and image captions, proposes neutral rewrites and "
  "checks each rewrite before it is accepted, in a laptop-scale application in which every number is traceable to "
  "a stored run. Its evaluation shows that bias-mitigation rewrites should be accepted for what they keep as well "
  "as for what they remove. On a controlled set with known answers, choosing the least-biased rewrite was almost "
  "always harmful, while a transparent gate that checks facts and meaning "
  + ("never" if GATE_DMG == 0 else "rarely")
  + f" chose a damaging rewrite, at the price of deferring {pct(GATE_DEFER)} of inputs and keeping the biased "
  f"original for {pct(GATE_KEPT)} of inputs. A learned gate was more accurate but chose damaging rewrites more "
  "often and was less explainable. We hope that the requirements in Sect. 4, together with the reporting of "
  "deferral and unnecessary-change rates, help make the costs of automated language intervention visible to those "
  "who deploy it and to those whose words it changes.")

# =========================================================================== declarations
H(1, "Statements and Declarations")
P("**Funding** The authors did not receive support from any organisation for the submitted work.")
P("**Competing interests** The authors have no relevant financial or non-financial interests to disclose.")
P("**Ethics approval** Not applicable. The study involved no human participants or animals. Only publicly available "
  "datasets were used, under their licences; BABE was used for non-commercial research only, as its licence "
  "requires. The hand-written evaluation sentences use fictional names.")
P("**Consent to participate** Not applicable.")
P("**Consent for publication** Not applicable.")
P("**Data availability** The benchmark datasets are publicly available from their original sources [@babe; "
  "@stereoset; @crows; @winobias; @coco]. The hand-written evaluation set, the derived datasets, the stored run "
  "outputs and manifests will be deposited in a public repository upon acceptance; during review they are "
  "available from the editor on request in anonymised form.")
P("**Code availability** The source code will be released in a public repository upon acceptance and is available "
  "to the editor and reviewers on request in anonymised form during review.")
P("**Author contributions** Provided on the separate title page to preserve anonymity during double-anonymous "
  "review.")

# =========================================================================== references
H(1, "References")
for i, k in enumerate(ORDER, 1):
    par = doc.add_paragraph()
    par.alignment = JUSTIFY
    par.paragraph_format.left_indent = Mm(7)
    par.paragraph_format.first_line_indent = Mm(-7)
    par.paragraph_format.space_after = Pt(3)
    rr = par.add_run(f"{i}. {REFS[k]}")
    rr.font.size = Pt(10)

unused = set(REFS) - set(ORDER)
if unused:
    raise SystemExit(f"references never cited: {sorted(unused)}")
doc.save(OUT)
print("wrote", OUT, "| references:", len(ORDER))
