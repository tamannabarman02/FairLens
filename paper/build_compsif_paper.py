"""Build the anonymous 6-page IEEE conference paper for Comp-SIF 2027.

Source of every number: paper/FairLens_SNCS_manuscript.docx (post re-split results).
Output: paper/compsif/FairLens_CompSIF_6page.docx  (PDF is exported with Word separately).
Formatting follows the IEEE conference template (US Letter): margins top 0.75in, bottom 1in,
left/right 0.625in; two 3.5in columns with 0.25in gap; Times New Roman; 24pt title, 10pt body,
9pt bold abstract, 8pt tables, captions and references.
"""
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "compsif"
# --ieee: pour the same content into the official IEEE Word template (paper/compsif/conference-template-a4.docx)
# using its own styles, sections and auto-numbering (headings, TABLE I, Fig. 1, [1]).
IEEE = "--ieee" in sys.argv
IEEE_TEMPLATE = OUT_DIR / "conference-template-a4.docx"
OUT = OUT_DIR / ("FairLens_CompSIF_6page_IEEE.docx" if IEEE else "FairLens_CompSIF_6page.docx")
COL_W = 3.37 if IEEE else 3.45  # usable column width (A4 template: 4866 twips)
FIG = OUT_DIR / "fig1_gate_flow.png"
FIG2 = HERE / "figures" / "Fig7.png"  # bias reduction vs fact loss (source figure)
FONT = "Times New Roman"

# --------------------------------------------------------------------------- references
REFS = {
    "gaucher": 'D. Gaucher, J. Friesen, and A. C. Kay, "Evidence that gendered wording in job advertisements exists '
               'and sustains gender inequality," *J. Pers. Soc. Psychol.*, vol. 101, no. 1, pp. 109–128, 2011.',
    "blodgett": 'S. L. Blodgett, S. Barocas, H. Daumé III, and H. Wallach, "Language (technology) is power: '
                'A critical survey of “bias” in NLP," in *Proc. 58th Annu. Meeting Assoc. Comput. Linguistics '
                '(ACL)*, 2020, pp. 5454–5476.',
    "gallegos": 'I. O. Gallegos *et al.*, "Bias and fairness in large language models: A survey," *Comput. Linguistics*, '
                'vol. 50, no. 3, pp. 1097–1179, 2024.',
    "euaiact": 'European Parliament and Council of the European Union, "Regulation (EU) 2024/1689 laying down harmonised '
               'rules on artificial intelligence (Artificial Intelligence Act)," *Official Journal of the European Union*, '
               'Jul. 2024.',
    "babe": 'T. Spinde, M. Plank, J.-D. Krieger, T. Ruas, B. Gipp, and A. Aizawa, "Neural media bias detection using '
            'distant supervision with BABE – Bias Annotations By Experts," in *Findings Assoc. Comput. Linguistics: '
            'EMNLP*, 2021, pp. 1166–1177.',
    "stereoset": 'M. Nadeem, A. Bethke, and S. Reddy, "StereoSet: Measuring stereotypical bias in pretrained language '
                 'models," in *Proc. 59th Annu. Meeting Assoc. Comput. Linguistics (ACL)*, 2021, pp. 5356–5371.',
    "crows": 'N. Nangia, C. Vania, R. Bhalerao, and S. R. Bowman, "CrowS-Pairs: A challenge dataset for measuring social '
             'biases in masked language models," in *Proc. Conf. Empirical Methods Natural Lang. Process. (EMNLP)*, 2020, '
             'pp. 1953–1967.',
    "winobias": 'J. Zhao, T. Wang, M. Yatskar, V. Ordonez, and K.-W. Chang, "Gender bias in coreference resolution: '
                'Evaluation and debiasing methods," in *Proc. Conf. North Amer. Chapter Assoc. Comput. Linguistics '
                '(NAACL)*, 2018, pp. 15–20.',
    "pryzant": 'R. Pryzant, R. Diehl Martinez, N. Dass, S. Kurohashi, D. Jurafsky, and D. Yang, "Automatically '
               'neutralizing subjective bias in text," in *Proc. AAAI Conf. Artif. Intell.*, vol. 34, no. 1, 2020, '
               'pp. 480–489.',
    "depen": 'Z. He, B. P. Majumder, and J. McAuley, "Detect and perturb: Neutral rewriting of biased and sensitive text '
             'via gradient-based decoding," in *Findings Assoc. Comput. Linguistics: EMNLP*, 2021, pp. 4173–4181.',
    "jobin": 'A. Jobin, M. Ienca, and E. Vayena, "The global landscape of AI ethics guidelines," *Nature Mach. '
             'Intell.*, vol. 1, pp. 389\u2013399, 2019.',
    "paradetox": 'V. Logacheva *et al.*, "ParaDetox: Detoxification with parallel data," in *Proc. 60th Annu. Meeting '
                 'Assoc. Comput. Linguistics (ACL)*, 2022, pp. 6804–6818.',
    "lgsa": 'S. S. R. Karri, Y. S. Nallapuneni, L. N. R. Mallireddy, and G. Gopichand, "LLM-guided synthetic augmentation '
            '(LGSA) for mitigating bias in AI systems," arXiv:2510.13202, 2025.',
    "summac": 'P. Laban, T. Schnabel, P. N. Bennett, and M. A. Hearst, "SummaC: Re-visiting NLI-based models for '
              'inconsistency detection in summarization," *Trans. Assoc. Comput. Linguistics*, vol. 10, pp. 163–177, '
              '2022.',
    "judge": 'L. Zheng *et al.*, "Judging LLM-as-a-judge with MT-Bench and Chatbot Arena," in *Advances in Neural '
             'Information Processing Systems (NeurIPS)*, vol. 36, 2023.',
    "selective": 'Y. Geifman and R. El-Yaniv, "Selective classification for deep neural networks," in *Advances in '
                 'Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017.',
    "deberta": 'P. He, J. Gao, and W. Chen, "DeBERTaV3: Improving DeBERTa using ELECTRA-style pre-training with '
               'gradient-disentangled embedding sharing," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2023.',
    "qwen": 'Qwen Team, "Qwen2.5 technical report," arXiv:2412.15115, 2024.',
}
ORDER: list[str] = []


def cite(keys: str) -> str:
    nums = []
    for k in keys.split(","):
        k = k.strip()
        if k not in REFS:
            raise KeyError(k)
        if k not in ORDER:
            ORDER.append(k)
        nums.append(ORDER.index(k) + 1)
    return ", ".join(f"[{n}]" for n in nums)


TITLE = "FairLens: Fact-Preserving Bias Mitigation with a Reliability Gate"
# title of the IEEE-template version (FairLens_CompSIF_6page_IEEE.*)
TITLE_IEEE = ("FairLens: Automated Framework for Bias Detection, Mitigation and Fairness Evaluation in Large "
              "Language and Multimodal AI Systems")


def scrub_pdf(path: Path) -> None:
    """Rewrite PDF metadata: keep only the title (no author, dates/timezone or XMP packet)."""
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import DictionaryObject, NameObject, TextStringObject
    w = PdfWriter()
    for page in PdfReader(str(path)).pages:  # copies only page-reachable objects, dropping the XMP stream
        w.add_page(page)
    title = TITLE_IEEE if "_IEEE" in path.name else TITLE
    info = DictionaryObject({NameObject("/Title"): TextStringObject(title)})
    w._info = w._add_object(info)
    tmp = path.with_suffix(".tmp.pdf")
    with open(tmp, "wb") as fh:
        w.write(fh)
    tmp.replace(path)
    print("scrubbed", path)


if len(sys.argv) > 2 and sys.argv[1] == "--scrub-pdf":
    scrub_pdf(Path(sys.argv[2]))
    sys.exit(0)

# --------------------------------------------------------------------------- helpers
def load_ieee_template():
    """Open the official template (Strict OOXML, which python-docx cannot read) via Word, save a transitional
    copy, and return it with its body emptied. Keeps the template's styles, numbering, page size, margins and
    section layout: section 1 (one column: title + author) and the two-column body section."""
    import copy
    import tempfile
    import win32com.client
    tmp = Path(tempfile.mkdtemp()) / "ieee_template.docx"
    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    try:
        d = word.Documents.Open(str(IEEE_TEMPLATE), False, True)
        d.SaveAs2(str(tmp), 12)
        d.Close(False)
    finally:
        word.Quit()
    dd = Document(str(tmp))
    body = dd.element.body
    sects = [p.find(qn("w:pPr") + "/" + qn("w:sectPr")) for p in body.iterchildren(qn("w:p"))]
    sects = [s for s in sects if s is not None]
    title_sect = copy.deepcopy(sects[0])  # one column, holds the title block
    two_col = next(s for s in sects if (c := s.find(qn("w:cols"))) is not None and c.get(qn("w:num")) == "2")
    two_col = copy.deepcopy(two_col)
    tbl_pr = copy.deepcopy(dd.tables[0]._tbl.tblPr)  # template table properties (borders, alignment)
    # the first-page footer only holds the "XXX-X-XXXX-XXXX-X/XX/$XX.00 ©20XX IEEE" placeholder: drop it
    for sp in (title_sect, two_col):
        for el in sp.findall(qn("w:footerReference")) + sp.findall(qn("w:headerReference")) + sp.findall(qn("w:titlePg")):
            sp.remove(el)
    for rid, rel in list(dd.part.rels.items()):
        if rel.reltype.endswith("/footer") or rel.reltype.endswith("/header"):
            dd.part.drop_rel(rid)
    final = body.find(qn("w:sectPr"))
    for el in list(body):
        body.remove(el)
    t = two_col.find(qn("w:type"))
    if t is None:
        t = OxmlElement("w:type")
        two_col.insert(0, t)
    t.set(qn("w:val"), "continuous")
    body.append(two_col)
    return dd, title_sect, tbl_pr


if IEEE:
    doc, TITLE_SECT, TBL_PR = load_ieee_template()
else:
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = FONT
    st.font.size = Pt(10)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    pf = st.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
    pf.widow_control = True

TOKEN = re.compile(r"(\*\*.+?\*\*|\*.+?\*|\{[a-z0-9_,]+\})")


def runs(p, text, size=None, bold=False, italic=False, small_caps=False):
    for part in TOKEN.split(text):
        if not part:
            continue
        b, i = bold, italic
        if part.startswith("**"):
            part, b = part[2:-2], True
        elif part.startswith("*"):
            part, i = part[1:-1], True
        elif part.startswith("{"):
            part = cite(part[1:-1])
        if IEEE:  # keep "Table IV", "Fig. 1", "Section II" on one line
            part = re.sub(r"\b(Table|Fig\.|Section|Sections) (?=[IVX\d])", "\\1\u00a0", part)
        r = p.add_run(part)
        if IEEE:  # fonts and sizes come from the template styles; only inline emphasis is set
            if b and not bold:
                r.bold = True
            if i and not italic:
                r.italic = True
            if bold:
                r.bold = True
            if italic:
                r.italic = True
            continue
        r.bold, r.italic = b, i
        r.font.name = FONT
        if small_caps:
            r.font.small_caps = True
        if size:
            r.font.size = Pt(size)
    return p


def P(text, indent=True, after=0, size=None, align=WD_ALIGN_PARAGRAPH.JUSTIFY):
    if IEEE:
        return runs(doc.add_paragraph(style="Body Text"), text)
    p = doc.add_paragraph()
    p.alignment = align
    if indent:
        p.paragraph_format.first_line_indent = Inches(0.2)
    p.paragraph_format.space_after = Pt(after)
    return runs(p, text, size=size)


ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"]
_h1 = [0]
_h2 = [0]


def H1(text):
    if IEEE:  # template numbers Heading 1 as I., II., ...
        runs(doc.add_paragraph(style="Heading 1"), text)
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    runs(p, f"{ROMAN[_h1[0]]}. {text}", small_caps=True)
    _h1[0] += 1
    _h2[0] = 0


def H1_unnumbered(text):
    if IEEE:  # template uses unnumbered Heading 5 for Acknowledgment / References
        runs(doc.add_paragraph(style="Heading 5"), text)
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    runs(p, text, small_caps=True)


def H2(text):
    if IEEE:  # template numbers Heading 2 as A., B., ...
        runs(doc.add_paragraph(style="Heading 2"), text)
        return
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    runs(p, f"{chr(ord('A') + _h2[0])}. {text}", italic=True)
    _h2[0] += 1


def bullet(text, size=None):
    if IEEE:
        runs(doc.add_paragraph(style="bullet list"), text)
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.left_indent = Inches(0.17)
    p.paragraph_format.first_line_indent = Inches(-0.12)
    p.paragraph_format.tab_stops.add_tab_stop(Inches(0.17))
    runs(p, "\u2022\t" + text, size=size)


def set_cols(section, num, space_twips=360):
    sp = section._sectPr
    cols = sp.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        sp.append(cols)
    cols.set(qn("w:num"), str(num))
    cols.set(qn("w:space"), str(space_twips))


def page_setup(section):
    section.page_width, section.page_height = Inches(8.5), Inches(11)
    section.top_margin, section.bottom_margin = Inches(0.75), Inches(1.0)
    section.left_margin = section.right_margin = Inches(0.625)
    section.header_distance = section.footer_distance = Inches(0.5)


def _border(el, tag, sz):
    b = OxmlElement(f"w:{tag}")
    b.set(qn("w:val"), "single" if sz else "nil")
    if sz:
        b.set(qn("w:sz"), str(sz))
        b.set(qn("w:space"), "0")
        b.set(qn("w:color"), "000000")
    el.append(b)


TAB_N = [0]


def table_ieee(title, header, rows, widths, bold_rows=(), note=None, left_cols=(0,)):
    import copy
    scale = COL_W / sum(widths) if sum(widths) > COL_W else 1.0
    widths = [w * scale for w in widths]
    cap = doc.add_paragraph(style="table head")  # template auto-prefixes "TABLE I."
    cap.paragraph_format.keep_with_next = True
    runs(cap, title)
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    old = t._tbl.tblPr
    new = copy.deepcopy(TBL_PR)
    for el in new.findall(qn("w:tblW")) + new.findall(qn("w:tblLayout")) + new.findall(qn("w:tblCellMar")):
        new.remove(el)
    lay = OxmlElement("w:tblLayout")
    lay.set(qn("w:type"), "fixed")
    new.append(lay)
    mar = OxmlElement("w:tblCellMar")  # narrow cell padding so 6-7 numeric columns fit one 3.37in column
    for side, v in (("left", 40), ("right", 40)):
        m = OxmlElement(f"w:{side}")
        m.set(qn("w:w"), str(v))
        m.set(qn("w:type"), "dxa")
        mar.append(m)
    new.append(mar)
    old.getparent().replace(old, new)
    t.autofit = False
    for ci, w in enumerate(widths):
        t.columns[ci].width = Inches(w)
    allrows = [header] + rows
    for ri, vals in enumerate(allrows):
        row = t.rows[ri]
        row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        for ci, v in enumerate(vals):
            c = row.cells[ci]
            c.width = Inches(widths[ci])
            p = c.paragraphs[0]
            p.style = doc.styles["table col head" if ri == 0 else "table copy"]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if ci in left_cols else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next = ri < len(allrows) - 1
            runs(p, v, bold=ri in bold_rows and ri > 0)
    if note:
        n = doc.add_paragraph(style="table footnote")
        # one explanatory note per table: no "a." footnote letter, justified instead of right-aligned
        numpr = OxmlElement("w:numPr")
        nid = OxmlElement("w:numId")
        nid.set(qn("w:val"), "0")
        numpr.append(nid)
        n._p.get_or_add_pPr().insert(1, numpr)
        n.paragraph_format.left_indent = n.paragraph_format.first_line_indent = Inches(0)
        n.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        runs(n, note)


def table(title, header, rows, widths, bold_rows=(), note=None, left_cols=(0,)):
    if IEEE:
        return table_ieee(title, header, rows, widths, bold_rows, note, left_cols)
    TAB_N[0] += 1
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before = Pt(6)
    cap.paragraph_format.keep_with_next = True
    runs(cap, f"TABLE {ROMAN[TAB_N[0] - 1]}", size=8)
    cap2 = doc.add_paragraph()
    cap2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap2.paragraph_format.space_after = Pt(3)
    cap2.paragraph_format.keep_with_next = True
    runs(cap2, title, size=8, small_caps=True)

    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    tblPr = t._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for tag, sz in (("top", 8), ("left", 0), ("bottom", 8), ("right", 0), ("insideH", 0), ("insideV", 0)):
        _border(borders, tag, sz)
    tblPr.append(borders)
    mar = OxmlElement("w:tblCellMar")
    for side, v in (("left", 40), ("right", 40), ("top", 0), ("bottom", 0)):
        m = OxmlElement(f"w:{side}")
        m.set(qn("w:w"), str(v))
        m.set(qn("w:type"), "dxa")
        mar.append(m)
    tblPr.append(mar)
    lay = OxmlElement("w:tblLayout")
    lay.set(qn("w:type"), "fixed")
    tblPr.append(lay)
    for ci, w in enumerate(widths):
        t.columns[ci].width = Inches(w)
    allrows = [header] + rows
    for ri, vals in enumerate(allrows):
        row = t.rows[ri]
        trPr = row._tr.get_or_add_trPr()
        cs = OxmlElement("w:cantSplit")
        trPr.append(cs)
        for ci, v in enumerate(vals):
            c = row.cells[ci]
            c.width = Inches(widths[ci])
            p = c.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if ci in left_cols else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next = ri < len(allrows) - 1
            runs(p, v, size=8, bold=(ri == 0 or ri in bold_rows))
            if ri == 0:
                tcPr = c._tc.get_or_add_tcPr()
                tb = OxmlElement("w:tcBorders")
                _border(tb, "bottom", 4)
                tcPr.append(tb)
    if note:
        n = doc.add_paragraph()
        n.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        n.paragraph_format.space_before = Pt(2)
        n.paragraph_format.space_after = Pt(6)
        runs(n, note, size=7)
    else:
        doc.add_paragraph().paragraph_format.space_after = Pt(2)


def figure(path, width_in, caption):
    if IEEE:
        p = doc.add_paragraph(style="Normal")  # template Normal is centred
        p.paragraph_format.keep_with_next = True
        p.add_run().add_picture(str(path), width=Inches(min(width_in, COL_W)))
        c = doc.add_paragraph(style="figure caption")  # template auto-prefixes "Fig. 1."
        runs(c, re.sub(r"^Fig\. \d+\.\s*", "", caption))
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Inches(width_in))
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    c.paragraph_format.space_before = Pt(3)
    c.paragraph_format.space_after = Pt(6)
    runs(c, caption, size=8)


# --------------------------------------------------------------------------- title block (1 column)
if IEEE:
    doc.add_paragraph(style="paper title").add_run(TITLE_IEEE)
    a = doc.add_paragraph(style="Author")
    a.add_run("Anonymous Author(s)")
    a.add_run().add_break()
    a.add_run("Anonymous Institution").italic = True
    a._p.get_or_add_pPr().append(TITLE_SECT)  # end of the template's one-column title section
else:
    page_setup(doc.sections[0])
    set_cols(doc.sections[0], 1)
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    t.paragraph_format.space_after = Pt(12)
    r = t.add_run("FairLens: Fact-Preserving Bias Mitigation with a Reliability Gate")
    r.font.size = Pt(24)
    r.font.name = FONT

    a = doc.add_paragraph()
    a.alignment = WD_ALIGN_PARAGRAPH.CENTER
    runs(a, "Anonymous Author(s)", size=11)
    a2 = doc.add_paragraph()
    a2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    a2.paragraph_format.space_after = Pt(14)
    runs(a2, "*Anonymous Institution*", size=10)

    sec = doc.add_section(WD_SECTION.CONTINUOUS)
    page_setup(sec)
    set_cols(sec, 2)

# --------------------------------------------------------------------------- abstract
ABSTRACT = (
    "Bias-mitigation rewrites by large language models are usually judged only by whether a bias "
    "detector's score falls, yet a rewrite can lower that score while deleting a requirement, changing a number or "
    "reversing a negation. We ask whether a transparent gate can choose a rewrite more safely than simple selection "
    "rules. FairLens combines four bias detectors, three rewriters (rule-based, FLAN-T5 and Qwen2.5) and a "
    "Reliability Gate: rules that accept, retry, reject or defer each rewrite to a person, with reason "
    "codes, based on bias reduction, similarity, entailment, fact preservation and detector agreement. A fine-tuned "
    "DeBERTa-v3-small detector reached macro-F1 0.814 on 2,652 held-out sentences but flagged about 40% of neutral "
    "WinoBias sentences. Qwen2.5 reduced bias scores the most yet lost a fact in 41.7% of its rewrites. "
    "On 204 controlled test inputs, selecting the least-biased candidate chose a damaging rewrite 96.1% of the time; "
    "the Gate never did, at the cost of deferring 23.5% of inputs and keeping the biased original for 33.8%. A "
    "learned decision tree was more accurate but chose a damaging rewrite for 13.2% of inputs. Mitigation tools "
    "should be judged on what they preserve and when they abstain, not only on how much bias they remove."
)
KEYWORDS = ("Bias detection, bias mitigation, large language models, human oversight, fairness evaluation, "
            "responsible AI.")
if IEEE:
    p = doc.add_paragraph(style="Abstract")
    runs(p, "*Abstract*—" + ABSTRACT)
    runs(doc.add_paragraph(style="Keywords"), "Index Terms—" + KEYWORDS)
else:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(6)
    runs(p, "*Abstract*", size=9, bold=True); runs(p, "—" + ABSTRACT, size=9, bold=True)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(6)
    runs(p, "*Index Terms*", size=9, bold=True); runs(p, "—" + KEYWORDS, size=9, bold=True)

# --------------------------------------------------------------------------- I. Introduction
H1("Introduction")
P("Biased wording has measurable effects: gender-coded language in job advertisements makes women less likely to "
  "see a role as one in which they belong and less likely to apply {gaucher}. Natural language processing (NLP) "
  "offers many methods to detect biased language and, increasingly, to rewrite it {blodgett,gallegos}, and "
  "instruction-tuned large language models (LLMs) make rewriting easy: a user asks a model to “make this "
  "inclusive” and receives fluent text in seconds.")
P("The ease of rewriting hides a problem of evaluation. A rewrite is usually judged by one number, whether a bias "
  "detector scores it lower than the original, yet it can lower that score while silently changing what the text "
  "says: rewriting “Looking for a young, aggressive salesman with 5+ years of B2B experience” as "
  "“Looking for a motivated sales professional with B2B experience” removes the biased wording and also "
  "the five-year requirement, and a bias score cannot tell it apart from a rewrite that keeps it. A fairness "
  "intervention that alters facts creates a harm of its own.")
P("This is an ethical problem as much as a technical one. Widely shared AI ethics principles include "
  "non-maleficence, respect for the autonomy of the people affected, and explicability {jobin}. A mitigation tool "
  "that rewrites an author's text without checking what else it changed fails all three: it may cause harm, it "
  "overrides the author without saying what was lost, and it gives no reason for its output. Regulation points "
  "the same way: the EU AI Act "
  "classifies AI systems used for recruitment as high-risk and requires that they be designed for effective human "
  "oversight {euaiact}, and a tool that cannot say when it is unsure gives a human overseer little to work with.")
P("This paper presents FairLens, a framework that detects biased sentences, proposes neutral rewrites and checks "
  "every rewrite with a *Reliability Gate* before anything is accepted. The Gate returns one of four decisions "
  "for each candidate (accept, retry with a stricter prompt, reject, or defer to human review) with reason codes "
  "such as FACT_LOSS or UNNECESSARY_CHANGE. The research question is: *can a transparent, multi-signal gate decide "
  "more safely than simple selection rules which rewrite of a biased sentence, if any, to accept?* The "
  "contributions are:", after=2)
bullet("FairLens, a locally running pipeline of four bias detectors, three rewriters and a rule-based Reliability "
       "Gate whose every verdict carries checkable reason codes.")
bullet("A controlled gate set with known correct decisions, split by source sentence to avoid leakage, on which "
       "the Gate is compared with lowest-bias, highest-similarity, keep-original, LLM-judge, learned-gate and "
       "random selection.")
bullet("Evidence that selecting by bias score is unsafe: lowest-bias selection chose a damaging rewrite for 96.1% "
       "of test inputs, whereas the Gate never did, at a measured cost in deferrals and kept biased originals.")
bullet("Negative results reported as such: the rule Gate is less accurate than a learned gate, several signals add "
       "no measurable safety, one rewriter mostly copies its input, and the Gate's remaining failures are inherited "
       "from detector misses.")
P("Section II reviews related work, Section III describes FairLens and the Gate, Section IV the experimental "
  "setup and Section V the results; Sections VI–VIII discuss ethical implications, limitations and "
  "conclusions.")

# --------------------------------------------------------------------------- II. Related work
H1("Related Work")
P("*Bias detection.* Sentence-level detectors are trained and evaluated on benchmarks that operationalise bias "
  "differently: BABE annotates media bias in news sentences {babe}, StereoSet and CrowS-Pairs contrast more and "
  "less stereotypical sentences about social groups {stereoset,crows}, and WinoBias tests gender–occupation "
  "associations {winobias}. Surveys stress that “bias” in NLP is often under-specified "
  "{blodgett,gallegos}; we therefore report detection per source and treat any detector score as fallible "
  "evidence, not ground truth.")
P("*Debiasing and rewriting.* Pryzant et al. neutralise subjective bias in encyclopaedic text {pryzant}; later "
  "work perturbs biased tokens during decoding {depen}, and detoxification corpora such as ParaDetox check content preservation while building the data "
  "{paradetox}. These systems are evaluated mainly on how much bias or toxicity is removed and on fluency; none "
  "has a run-time decision layer that refuses an unsafe rewrite or routes it to a person. LLM-guided synthetic "
  "augmentation screens generated rewrites with similarity and attribute checks {lgsa}, but targets data "
  "augmentation and has no per-fact checks, multi-way decision or comparison with selection baselines.")
P("*Faithfulness checks and human oversight.* Natural language inference (NLI) models detect factual "
  "inconsistency between a source and a generated text {summac}, and LLMs are used as judges of other models' "
  "outputs despite documented position and self-preference biases {judge}. Selective classification lets a model "
  "abstain when unsure, trading coverage for risk {selective}, and human oversight is a legal requirement for "
  "high-risk uses {euaiact}. FairLens combines these strands for mitigation: entailment and explicit fact checks "
  "define what must be preserved, an LLM judge is a baseline, and deferral to a person is a legitimate outcome "
  "whose rate is always reported.")

# --------------------------------------------------------------------------- III. FairLens and the Gate
H1("FairLens and the Reliability Gate")
H2("Requirements")
P("Before choosing metrics we state what a mitigation tool owes the author of a text and its readers, and tie "
  "each requirement to a measured quantity:", after=2)
bullet("*R1 Fidelity (non-maleficence):* a rewrite must not remove or change facts, reverse negations, alter "
       "quotations or reassign roles; measured by accepted rewrites that lose a fact or are contradicted, and by "
       "damaging selections.")
bullet("*R2 Necessity (autonomy):* text that is not biased should be left alone; measured by the "
       "unnecessary-change rate on neutral inputs.")
bullet("*R3 Transparency and contestability:* each decision should come with reasons a person can check and "
       "dispute; addressed by rule-based decisions with reason codes, whose accuracy cost we report against a "
       "learned gate.")
bullet("*R4 Oversight under uncertainty:* when evidence is conflicting or weak, the decision should pass to a "
       "person; measured by the deferral rate.")
bullet("*R5 Honest accounting of costs:* deferral adds human work and a cautious system may leave biased text "
       "unchanged; measured by deferral and kept-biased-original rates, reported alongside the safety gains.")
P("These requirements conflict: R1 and R4 favour caution while the purpose of mitigation favours change, and R2 "
  "favours restraint while detector false alarms push towards unnecessary edits. FairLens is designed to make "
  "these trade-offs visible rather than to collapse them into a single score.")
H2("Pipeline")
P("Four detectors score each sentence: a lexicon-based rule detector, DistilBERT and DeBERTa-v3-small {deberta} "
  "fine-tuned for binary bias classification, and BART-large-MNLI used as a zero-shot classifier; the pipeline's "
  "bias score is the mean of the four scores and their spread measures disagreement. Explanations are given as "
  "highlighted phrases: lexicon matches for the rule detector and, for the transformer detectors, the words whose "
  "removal lowers the bias score most. A public RoBERTa classifier trained on BABE is included as an external "
  "reference in the benchmarks only. Three rewriters propose "
  "candidates: a rule rewriter that substitutes terms from a curated lexicon, FLAN-T5-base and "
  "Qwen2.5-0.5B-Instruct {qwen}, the two LLMs sharing one prompt family with greedy decoding (at most 96 new "
  "tokens, repetition penalty 1.1, fixed seed). The unchanged original is always kept as a candidate, because "
  "for a sentence that is not biased the correct action is to change nothing. An extension passes captions "
  "generated for images (BLIP and Florence-2) to the same detectors; caption results are not reported here. All "
  "models run locally, so confidential text is not sent to an external service.")
H2("Gate Signals")
P("For each candidate the Gate computes the signals in Table I.", after=0)
table("Signals Computed by the Reliability Gate",
      ["Signal", "Definition"],
      [["Bias reduction", "Mean detector score of the original minus that of the candidate"],
       ["Similarity", "Cosine similarity of all-MiniLM-L6-v2 sentence embeddings"],
       ["Entailment", "P(original ⇒ candidate) and minimum over both directions (DeBERTa-v3-small NLI cross-encoder)"],
       ["Contradiction", "Maximum contradiction probability over both directions"],
       ["Fact retention", "Share of extracted facts (entities, numbers, dates, roles, quotations, negation "
                          "polarity; spaCy and rules) that survive in the candidate"],
       ["Detector agreement", "Standard deviation of detector scores on the candidate"],
       ["Uncertainty", "Detector disagreement and closeness of the bias score to 0.5"],
       ["Fluency, length", "Repetition, prompt leakage, non-alphabetic text; word-count ratio"]],
      [0.95, 2.5], left_cols=(0, 1))
H2("Decision Order and Thresholds")
P("The Gate applies transparent rules in a fixed order (Fig. 1). If the input is not judged biased, the "
  "unchanged text is accepted and any change is rejected as unnecessary. For a biased input, any preservation "
  "failure leads to rejection; if meaning is preserved but bias did not fall enough, the LLMs are re-prompted with "
  "a stricter instruction that lists the extracted facts, at most twice, after which the input goes to human "
  "review; if the detectors disagree or the score is close to the threshold, the input goes to human review; "
  "otherwise the candidate is accepted. Among several accepted candidates, the one with the highest sum of bias "
  "reduction, similarity and fact retention is returned.")
P("The default thresholds were set a priori: minimum similarity 0.70, bidirectional entailment probability 0.50, "
  "minimum bias reduction 0.15 and maximum detector standard deviation 0.35. Tuning on the validation split of the "
  "controlled gate set only (Section IV) gave minimum similarity 0.60, forward entailment probability 0.70 and "
  "minimum bias reduction 0.05, with the detector limit raised to 1.0, which switches the disagreement check off "
  "because the standard deviation of probabilities cannot exceed 0.5. The validation-tuned values were frozen "
  "before any test evaluation; results are reported for both settings.")
figure(FIG, 3.4,
       "Fig. 1. Decision flow of the rule-based Reliability Gate for one candidate rewrite, with the reason codes "
       "attached to each outcome. When several candidates are assessed, verdicts are combined with the precedence "
       "REJECT > RETRY > HUMAN_REVIEW > ACCEPT.")

# --------------------------------------------------------------------------- IV. Setup
H1("Experimental Setup")
H2("Datasets and Split")
P("For detection, BABE {babe}, StereoSet (intra- and inter-sentence) {stereoset} and CrowS-Pairs {crows} were "
  "unified into one binary schema (1 = the sentence expresses or relies on a stereotype or biased framing under "
  "the source's own annotation) and split 70/15/15 (seed 42) by group, so that the two members of a minimal "
  "pair never fall into different splits, giving 10,365 training, 2,119 validation and 2,652 test sentences. "
  "WinoBias {winobias} was used only as a false-alarm probe, since all its sentences are neutral; the Wiki "
  "Neutrality Corpus {pryzant} could not be obtained (its host was unreachable and the only mirror is gated). DistilBERT and "
  "DeBERTa-v3-small were fine-tuned on the training split (AdamW, learning rate 5×10⁻⁵, batch size "
  "16, 10% linear warm-up, best epoch by validation macro-F1, seed 42); for DeBERTa the embedding layer was "
  "frozen because its 128k-token embedding matrix with optimiser state did not fit in laptop memory. For "
  "BART-MNLI, five hypothesis templates were compared on 300 validation sentences, and the LLM prompt variant was "
  "chosen on 30 biased sentences from the training split; the evaluation sets were never used for prompt choice. "
  "For mitigation, 136 sentences with fictional names were hand-written "
  "(91 biased, 45 neutral), covering recruitment text, gender, age, race, religion, disability and socioeconomic "
  "stereotypes and media framing, and deliberately including names, numbers, dates, negations, roles and "
  "quotations; each biased sentence has one reference rewrite. With 140 sentences from the detection test split "
  "they form a mitigation set of 276 inputs (211 biased, 65 neutral).")
H2("Controlled Gate Set")
P("To evaluate the Gate against known correct answers, for each biased input the reference rewrite is labelled "
  "ACCEPT, the unchanged original RETRY, and seeded perturbations of the reference REJECT (number changed, "
  "negation flipped, entity or role swapped, antonym, hallucinated addition, truncation, altered quotation); for "
  "each neutral input the unchanged original is ACCEPT and any change is REJECT. Three perturbation seeds give "
  "2,481 candidates. The validation/test split is made *by source sentence*: each of the 136 sentences is "
  "assigned once, stratified by label, so all three seeds of a sentence and its reference rewrite fall in the "
  "same half, and the build script stops if any source sentence appears in more than one split. The test split "
  "has 68 sentences, giving 204 inputs (69 of them neutral) and 1,218 candidates.")
H2("Metrics")
P("On the controlled set a selection is *acceptable* if its gold label is ACCEPT and *harmful* otherwise, split "
  "into keeping a biased original (gold RETRY) and choosing a damaging rewrite (gold REJECT); we also report "
  "deferrals, unnecessary changes on neutral inputs, candidate-level three-class macro-F1 and the unsafe-accept "
  "rate, P(accept | gold ≠ ACCEPT). On real rewrites, which have no gold labels, we report bias reduction, "
  "the rate at which an accepted rewrite loses an extracted fact or is contradicted, unnecessary change, deferral "
  "and the rate at which biased inputs are left unchanged. Detection is evaluated with macro-F1, ROC-AUC and the "
  "WinoBias false-positive rate (FPR). Confidence intervals are 95% percentile bootstrap intervals (1,000 "
  "resamples); ablations use paired bootstrap. Seeds were fixed for data splitting and training (42), generation "
  "(13), perturbations (11, 22, 33) and random baselines (100\u2013104).")
H2("Baselines")
P("Each policy picks one candidate per input (possibly the unchanged original) or defers: the candidate with the "
  "*lowest bias score*; the changed candidate with the *highest similarity*; *keep the original*; an "
  "*LLM-as-judge* {judge} in which Qwen2.5-0.5B-Instruct is asked whether a changed candidate preserves the "
  "meaning (prompt chosen among three variants on validation by Cohen's κ, abstentions counted as "
  "deferrals); *learned gates*, a logistic regression and a depth-4 decision tree over 14 signal features trained "
  "on the validation half, which defer when their top class probability is below 0.6; and *random* choice (five "
  "seeds). Each experiment was run once; after a leakage check showed that an earlier split had been made per "
  "seed rather than per sentence, all controlled-set and Gate experiments were recomputed on a second machine "
  "(CPU), reusing the stored signals and rewrites unchanged.")

# --------------------------------------------------------------------------- V. Results
H1("Results")
H2("Bias Detection")
table("Detection on the Shared Test Split (n = 2,652)",
      ["Detector", "Macro-F1 [95% CI]", "ROC-AUC", "WinoBias FPR pro / anti"],
      [["Rule-based", "0.354 [0.342, 0.369]", "0.515", "0.000 / 0.000"],
       ["DistilBERT (ft.)", "0.777 [0.762, 0.791]", "0.856", "0.441 / 0.428"],
       ["DeBERTa-v3-small (ft.)", "0.814 [0.800, 0.828]", "0.896", "0.396 / 0.376"],
       ["BART-MNLI (zero-shot)", "0.611 [0.593, 0.629]", "0.648", "0.417 / 0.427"],
       ["RoBERTa (public, BABE)", "0.646 [0.627, 0.661]", "0.722", "0.054 / 0.062"]],
      [1.25, 1.0, 0.45, 0.75], bold_rows=(3,),
      note="Threshold 0.5; ft. = fine-tuned. WinoBias FPR is the share of neutral WinoBias sentences flagged as biased in the pro- and "
           "anti-stereotypical conditions. The public RoBERTa classifier is an external reference trained on BABE.")
P("Table II summarises detection. Fine-tuned DeBERTa-v3-small was the strongest detector (macro-F1 0.814) and "
  "the most consistent across sources (0.811–0.817), followed by DistilBERT (0.777). The public RoBERTa "
  "classifier scored highest on BABE (0.853), the source it was trained on, but at or below 0.544 on the "
  "stereotype datasets, a reminder that one benchmark number does not transfer across definitions of bias. "
  "Zero-shot BART-MNLI reached 0.611 and the rule baseline 0.354, with very low recall (0.035). "
  "High benchmark accuracy did not mean few false alarms on everyday text: the fine-tuned detectors flagged 39.6% "
  "and 37.6% (DeBERTa) and 44.1% and 42.8% (DistilBERT) of the neutral WinoBias sentences, and calibration errors "
  "of about 0.12 mean their scores should not be read as probabilities. DeBERTa also scored pro-stereotypical "
  "WinoBias sentences slightly higher than anti-stereotypical ones (mean gap 0.016, CI 0.003\u20130.028) and "
  "preferred the more stereotypical CrowS-Pairs sentence in 90.1% of minimal pairs. Errors also run the other way: the "
  "sentence “Women are not suited to engineering, …” received a mean detector score of 0.48, just "
  "below the threshold, so the pipeline left it unchanged. Both kinds of error matter ethically: false alarms "
  "invite needless rewriting of an author's words, and misses leave biased text in place.")
H2("Rewriters: Bias Reduction Against Preservation")
table("Rewriters on the Mitigation Set (276 Inputs)",
      ["Rewriter", "Bias red. [95% CI]", "Chan-ged", "Fact failures", "Unnec. change"],
      [["Rule rewriter", "0.074 [0.056, 0.092]", "27.5%", "1.8%", "0.0%"],
       ["FLAN-T5-base", "0.018 [0.008, 0.031]", "31.2%", "9.4%", "20.0%"],
       ["Qwen2.5-0.5B-Instr.", "0.087 [0.064, 0.110]", "99.3%", "41.7%", "98.5%"],
       ["No change (ref.)", "0.000 [0.000, 0.000]", "0.0%", "0.0%", "0.0%"]],
      [1.0, 1.0, 0.45, 0.5, 0.5],
      note="Bias reduction: mean drop in detector score on the 211 biased inputs. Fact failures: share of outputs "
           "losing at least one extracted fact. Unnecessary change: share of the 65 neutral inputs that were "
           "changed. All measures are automatic.")
P("Table III compares the rewriters. Qwen2.5 reduced the mean bias score the most (0.087) but lost at least one "
  "fact in 41.7% of its rewrites and changed 98.5% of neutral inputs that needed no change. The rule rewriter "
  "achieved a similar reduction (0.074) with a fact failure rate of 1.8%, because it only substitutes terms. "
  "FLAN-T5 changed only 31.2% of inputs: in development tests with several prompt formats and few-shot examples "
  "it mostly copied its input, a capability limit we report rather than hide. Bias reductions are small in "
  "absolute terms because the mean detector score includes the rule and zero-shot detectors, which respond "
  "weakly to rewording. Fig. 2 shows the trade-off: the rewriter that lowers bias scores most also loses facts "
  "most often.")
figure(FIG2, 2.7,
       "Fig. 2. Bias reduction against fact loss for each rewriter on the mitigation set. The rule rewriter achieves "
       "a reduction similar to Qwen2.5 with few fact losses because it only substitutes terms.")
table("Selection Policies on Real Rewrites (276 Inputs)",
      ["Policy", "Fact loss", "Contra-dicted", "Unnec. change", "Bias red.", "De-ferred", "Biased kept"],
      [["Rule Gate", "0.0%", "0.0%", "0.0%", "0.121", "47.5%", "26.5%"],
       ["Gate + strict retry", "0.0%", "0.0%", "1.5%", "0.128", "46.4%", "26.5%"],
       ["Lowest bias score", "25.4%", "8.3%", "56.9%", "0.152", "0.0%", "15.2%"],
       ["Highest similarity", "29.7%", "6.5%", "98.5%", "0.088", "0.0%", "0.0%"],
       ["Always FLAN-T5", "9.4%", "0.7%", "20.0%", "0.018", "0.0%", "65.4%"],
       ["Random (5 seeds)", "36.1%", "7.8%", "98.5%", "0.089", "0.0%", "0.0%"]],
      [1.05, 0.38, 0.42, 0.42, 0.38, 0.4, 0.4], bold_rows=(1,),
      note="Fact loss and contradiction are measured on the chosen output; deferred inputs choose nothing. Bias "
           "reduction is averaged over biased inputs for which the policy chose an output (a kept original counts "
           "as 0), so a policy can exceed any single rewriter in Table III. Always FLAN-T5: always take the first "
           "model's output. Biased kept: biased inputs left unchanged.")
P("Table IV applies the selection policies to these real rewrites. The "
  "Gate accepted no rewrite that lost a fact or was contradicted by the NLI model, against 25.4% and 8.3% for "
  "lowest-bias selection, and made no unnecessary changes. The cost was 47.5% of inputs deferred to a person and "
  "26.5% of biased inputs left unchanged, which matches detection: the detectors flagged only 73.5% of the "
  "biased inputs, and the Gate by design does not rewrite text it does not judge biased. The strict retry rarely "
  "helped: of 124 inputs retried, 3 were accepted after retry.")
H2("Selection on Known Correct Decisions")
P("Table V reports input-level selection on the 204 controlled test inputs. Choosing the least-biased candidate "
  "was almost always wrong: it chose a damaging rewrite for 196 of 204 inputs (96.1%), because the candidate "
  "with the lowest bias score is often one that dropped or changed content. Highest-similarity selection made "
  "more acceptable choices than the rule Gate (59.3% against 42.6%), but every one of its errors was a damaging "
  "rewrite and it changed every neutral input. The LLM judge made acceptable choices for 23.5% of inputs and "
  "changed 89.9% of neutral inputs, because it was only asked about faithfulness. The validation-tuned rule Gate's "
  "harmful-choice rate was 33.8%, but all 69 of these cases were biased originals kept unchanged; it never chose "
  "a damaging rewrite, deferred 23.5% of inputs and made no unnecessary changes. With the default thresholds "
  "it was more cautious still (Table V: 0.338 deferred, 0.324 acceptable), so tuning on validation reduced "
  "deferrals without adding damaging choices. The learned decision tree was "
  "the most accurate policy (76.0% acceptable choices, 4.9% deferred), but it chose a damaging rewrite 27 times "
  "(13.2%) and changed 26.1% of neutral inputs.")
table("Input-Level Selection on the Controlled Test Split (204 Inputs)",
      ["Policy", "Accept-able", "Kept biased", "Damag-ing", "De-ferred", "Unnec. change"],
      [["Lowest bias score", "0.015", "0.025", "0.961", "0.000", "0.957"],
       ["Highest similarity", "0.593", "0.000", "0.407", "0.000", "1.000"],
       ["Keep original", "0.338", "0.662", "0.000", "0.000", "0.000"],
       ["LLM-as-judge (Qwen2.5)", "0.235", "0.000", "0.319", "0.446", "0.899"],
       ["Rule Gate (default)", "0.324", "0.338", "0.000", "0.338", "0.000"],
       ["Rule Gate (val.-tuned)", "0.426", "0.338", "0.000", "0.235", "0.000"],
       ["Learned gate (logistic)", "0.603", "0.000", "0.216", "0.181", "0.464"],
       ["Learned gate (tree)", "0.760", "0.059", "0.132", "0.049", "0.261"],
       ["Random (5 seeds)", "0.130", "–", "–", "0.000", "1.000"]],
      [1.25, 0.42, 0.45, 0.45, 0.42, 0.46], bold_rows=(6,),
      note="Shares of the 204 inputs. Harmful choices are split into keeping a biased original (gold RETRY) and "
           "choosing a damaging rewrite (gold REJECT); random selection made harmful choices for 0.870 of inputs. "
           "Unnecessary change is measured on the 69 neutral test inputs.")
P("At the candidate level the tuned Gate reached a three-class macro-F1 of 0.665 with an unsafe-accept rate of "
  "7.1%. Of its unsafe accepts, 72 were unchanged biased originals accepted because the detectors did not judge "
  "the input biased, and none was a corrupted rewrite. When the gold input label was supplied instead of the "
  "detectors' judgement, the unsafe-accept rate fell to 0.3% and macro-F1 rose to 0.864: the Gate's remaining "
  "failures are inherited mainly from detection, not from its preservation checks. The Gate caught 100% of every "
  "planted corruption type; this is partly by construction, because several perturbations target exactly what "
  "the fact checker extracts.")
H2("Ablation")
P("Removing one signal group at a time, the bias-reduction check (+6.2 percentage points of unsafe accepts) and "
  "entailment (+5.6 points) mattered most; fact retention had a small but non-zero effect (+0.6 points), largely "
  "because entailment also catches most planted corruptions, and detector agreement, uncertainty and fluency had "
  "no measurable effect on this set. We report this as a negative result: these signals mainly route inputs to "
  "human review, and the controlled set contains few cases where that matters.")

# --------------------------------------------------------------------------- VI. Discussion
H1("Discussion and Ethical Considerations")
P("The central finding is that optimising for the bias score selects the wrong rewrite, and the stronger rewriter "
  "was the less faithful one: a tool evaluated only on bias reduction can look successful while harming the "
  "people who rely on the text; on real rewrites, lowest-bias selection accepted outputs that lost a fact a "
  "quarter of the time. We recommend that mitigation work report preservation measures and "
  "unnecessary-change rates alongside bias reduction, and make explicit for each domain what must be preserved, "
  "as the fact types in FairLens do for recruitment and news text. The Gate's safety comes from abstaining, which has costs of its own. It deferred "
  "23.5% of controlled inputs and 47.5% of real ones, and kept the biased original for 33.8% of controlled inputs. "
  "Deferral is only responsible where a qualified person will actually review the case; otherwise it shifts the "
  "burden to whoever receives the text, and a reviewer facing many deferrals may come to rubber-stamp them. "
  "Lowering the detection threshold would catch more bias but, with false-alarm rates near 40% on neutral "
  "WinoBias sentences, would also rewrite many authors' words for no reason. Keeping biased originals is thus "
  "the price of not rewriting text the detectors do not flag, and the oracle analysis shows that "
  "better detection, not more rewriting freedom, is the main route to fewer misses. Organisations deploying such "
  "tools should decide explicitly how much review capacity they have and which error they would rather make.")
P("Transparency also has an accuracy cost. The learned tree made more acceptable choices (76.0% against 42.6%) "
  "with fewer deferrals, but chose damaging rewrites and changed neutral inputs, while the rule Gate did neither "
  "and gives reasons an author can check and contest. In high-risk settings such as recruitment {euaiact} we "
  "consider these properties worth the lower accuracy, but this is a value judgement that deployers should make "
  "openly. Finally, FairLens is a review assistant, not a certification tool: detectors encode particular, "
  "culturally situated definitions of bias and err in both directions, and presenting such a tool as proof of "
  "fairness or compliance would be a form of fairness-washing. Its appropriate role is to show an author what "
  "may be biased, offer alternatives, state what each would change and say when it is unsure, leaving the final "
  "decision with a person. Running the whole pipeline locally also keeps confidential documents, such as "
  "unpublished job advertisements, away from external services, at the cost of using smaller models.")

# --------------------------------------------------------------------------- VII. Limitations
H1("Limitations")
bullet("The controlled gold set and reference rewrites are small and were written by a single author; the "
       "perturbations are synthetic and several target exactly what the fact checker extracts, so high catch "
       "rates are partly by construction.")
bullet("Mitigation quality was measured automatically; there is no human evaluation of the real rewrites yet. "
       "Entailment in both directions is strict for bias rewrites, which legitimately remove content.")
bullet("All data are English, and the detectors were trained on news framing and crowd-sourced stereotype "
       "sentences rather than job advertisements; the StereoSet and CrowS-Pairs labels have known validity "
       "problems.")
bullet("The rewriters are small (0.5 billion and 250 million parameters); larger models may preserve facts better, "
       "which would change the trade-off but not the need to check it.")
bullet("Each experiment was run once, so confidence intervals reflect sampling of test items, not variation "
       "across training seeds or hardware; the controlled-set and Gate experiments were re-run in a different "
       "device and software environment from the original signal runs; they reuse the stored signals, except the "
       "LLM-judge answers, which were regenerated.")
bullet("The BART-MNLI hypothesis template chosen on validation for the detection benchmark was not propagated to "
       "the Gate, mitigation runs and application, which used the default template; re-running with the chosen "
       "template may change the Gate's input-level detection.")

# --------------------------------------------------------------------------- VIII. Conclusion
H1("Conclusion")
P("FairLens detects bias, proposes neutral rewrites and checks each rewrite before it is accepted. On a "
  "controlled set with known answers, choosing the least-biased rewrite was almost always harmful, while a "
  "transparent gate that checks facts and meaning never chose a damaging rewrite, at the price of deferring "
  "23.5% of inputs and keeping the biased original for 33.8%. A learned gate was more accurate but chose damaging "
  "rewrites more often and was less explainable. Bias-mitigation rewrites should be accepted for what they keep as "
  "well as for what they remove, and deferral and unnecessary-change rates should be reported alongside bias "
  "reduction. Future work includes a blinded human evaluation of real rewrites and Gate decisions with several "
  "annotators, a larger set of real job advertisements with domain-specific definitions of the facts that must "
  "be preserved, a hybrid gate that keeps the rule Gate's hard preservation checks and reason codes but learns "
  "its thresholds, and red-team tests in which real accepted rewrites are corrupted.")

H1_unnumbered("Use of Generative AI")
P("The authors designed the study and the Reliability Gate; an AI assistant (Claude, Anthropic) supported parts "
  "of the implementation and editing, and all results were checked by the authors.", indent=True)

# --------------------------------------------------------------------------- references
H1_unnumbered("References")
for i, k in enumerate(ORDER, 1):
    if IEEE:  # template "references" style auto-numbers [1], [2], ...
        runs(doc.add_paragraph(style="references"), re.sub(r'"([^"]*)"', "“\\1”", REFS[k]))
        continue
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.left_indent = Inches(0.3)
    p.paragraph_format.first_line_indent = Inches(-0.3)
    p.paragraph_format.tab_stops.add_tab_stop(Inches(0.3))
    ref = re.sub(r'"([^"]*)"', "\u201c\\1\u201d", REFS[k])
    runs(p, f"[{i}]\t" + ref, size=8)
unused = [k for k in REFS if k not in ORDER]
if unused:
    sys.exit(f"unused references: {unused}")

# --------------------------------------------------------------------------- metadata (anonymous)
cp = doc.core_properties
cp.author = ""
cp.last_modified_by = ""
cp.title = TITLE_IEEE if IEEE else "FairLens: Fact-Preserving Bias Mitigation with a Reliability Gate"
cp.subject = ""
cp.keywords = ""
cp.comments = ""
cp.category = ""
cp.revision = 1
cp.created = cp.modified = datetime(2026, 1, 1, tzinfo=timezone.utc)
OUT_DIR.mkdir(exist_ok=True)
doc.save(OUT)
if IEEE:  # template's docProps/app.xml still names "Company: IEEE" and the placeholder title: replace it
    import shutil
    import zipfile
    APP = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<Properties xmlns="http://schemas.'
           'openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/'
           'officeDocument/2006/docPropsVTypes"><Application>Microsoft Office Word</Application><DocSecurity>0'
           '</DocSecurity><ScaleCrop>false</ScaleCrop><Company></Company><LinksUpToDate>false</LinksUpToDate>'
           '<SharedDoc>false</SharedDoc><HyperlinksChanged>false</HyperlinksChanged><AppVersion>16.0000'
           '</AppVersion></Properties>')
    tmp = OUT.with_suffix(".tmp.docx")
    with zipfile.ZipFile(OUT) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = APP.encode("utf-8") if item.filename == "docProps/app.xml" else zin.read(item.filename)
            zout.writestr(item, data)
    shutil.move(tmp, OUT)
(OUT_DIR / "abstract.txt").write_text(ABSTRACT + "\n", encoding="utf-8")
print("saved", OUT, "| refs:", len(ORDER), "| abstract words:", len(ABSTRACT.split()), "chars:", len(ABSTRACT))
