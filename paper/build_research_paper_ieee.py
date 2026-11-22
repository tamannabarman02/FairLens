"""Rebuild the FairLens research paper in IEEE two-column A4 conference format.

Source : paper/FairLens_Research_Paper.docx (content and results; left unchanged)
Format : paper/compsif/conference-template-a4.docx (official IEEE A4 Word template, opened via Word)
Output : paper/FairLens_Research_Paper_IEEE.docx  (+ .pdf exported with Word)

Changes besides the layout: presentation title, named author block (students, then guide with designation).
Run with Anaconda:  C:\\Users\\taman\\Anaconda3\\python.exe -X utf8 paper/build_research_paper_ieee.py
"""
import copy
import io
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

HERE = Path(__file__).resolve().parent
SRC = HERE / "FairLens_Research_Paper.docx"
OUT = HERE / "FairLens_Research_Paper_IEEE.docx"
PDF = OUT.with_suffix(".pdf")
TEMPLATE = HERE / "compsif" / "conference-template-a4.docx"

TITLE = "FairLens: Bias Detection and Mitigation in AI Systems with a Reliability Gate"
DEPT, COLLEGE, CITY = "Department of CSE", "B.M.S. College of Engineering", "Bengaluru, India"
AUTHORS = [  # students first, then the guide
    ("Suryansh Shahi", ""),
    ("Sushma B T", ""),
    ("Tamanna Barman", ""),
    ("Trisha L Salian", ""),
    ("Prof. Muthurani N H", "Assistant Professor"),
]
COL_W = 3.37    # one column of the A4 template
FULL_W = 6.9    # both columns
SPAN_TABLES = {1, 5, 6, 7, 8, 9}   # wide tables placed across both columns
SPAN_FIGS = {1}                 # architecture diagram across both columns


# --------------------------------------------------------------------------- template
def load_template():
    import tempfile
    import win32com.client
    tmp = Path(tempfile.mkdtemp()) / "ieee_template.docx"
    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    try:
        d = word.Documents.Open(str(TEMPLATE), False, True)
        d.SaveAs2(str(tmp), 12)  # Strict OOXML -> transitional, readable by python-docx
        d.Close(False)
    finally:
        word.Quit()
    dd = Document(str(tmp))
    body = dd.element.body
    sects = [p.find(qn("w:pPr") + "/" + qn("w:sectPr")) for p in body.iterchildren(qn("w:p"))]
    sects = [s for s in sects if s is not None]
    one_col = copy.deepcopy(sects[0])
    two_col = copy.deepcopy(next(s for s in sects if (c := s.find(qn("w:cols"))) is not None
                                 and c.get(qn("w:num")) == "2"))
    tbl_pr = copy.deepcopy(dd.tables[0]._tbl.tblPr)
    for sp in (one_col, two_col):  # drop the "©20XX IEEE" placeholder footer
        for el in sp.findall(qn("w:footerReference")) + sp.findall(qn("w:headerReference")) + sp.findall(qn("w:titlePg")):
            sp.remove(el)
        t = sp.find(qn("w:type"))
        if t is None:
            t = OxmlElement("w:type")
            sp.insert(0, t)
        t.set(qn("w:val"), "continuous")
    for rid, rel in list(dd.part.rels.items()):
        if rel.reltype.endswith("/footer") or rel.reltype.endswith("/header"):
            dd.part.drop_rel(rid)
    for el in list(body):
        body.remove(el)
    body.append(copy.deepcopy(two_col))
    return dd, one_col, two_col, tbl_pr


doc, ONE_COL, TWO_COL, TBL_PR = load_template()
src = Document(str(SRC))


def nb(text):  # keep "Table IV", "Fig. 1", "Section II" together
    return re.sub(r"\b(Table|Tables|Fig\.|Figs\.|Section|Sections) (?=[IVX\d])", "\\1\u00a0", text)


def add_runs(p, runs, lead_italic=False):
    """Copy source runs, keeping bold/italic/sub/superscript. A bold lead-in becomes an italic run-in heading."""
    for k, r in enumerate(runs):
        if not r.text:
            continue
        nr = p.add_run(nb(r.text))
        if lead_italic and k == 0 and r.bold:
            nr.italic = True
            continue
        if r.bold:
            nr.bold = True
        if r.italic:
            nr.italic = True
        if r.font.superscript:
            nr.font.superscript = True
        if r.font.subscript:
            nr.font.subscript = True
    return p


def section_break(sect):
    p = doc.add_paragraph(style="Normal")
    p.paragraph_format.space_before = p.paragraph_format.space_after = Pt(0)
    pf = p._p.get_or_add_pPr()
    pf.append(copy.deepcopy(sect))
    return p


def set_size(p, pt):
    for r in p.runs:
        r.font.size = Pt(pt)


# --------------------------------------------------------------------------- title and authors (one column)
doc.add_paragraph(style="paper title").add_run(TITLE)


def author_table(people, width):
    t = doc.add_table(rows=1, cols=len(people))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for ci, (name, role) in enumerate(people):
        c = t.rows[0].cells[ci]
        c.width = Inches(width / len(people))
        p = c.paragraphs[0]
        p.style = doc.styles["Author"]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run(name).font.size = Pt(11)
        for line, it in [x for x in ((role, True), (DEPT, True), (COLLEGE, True), (CITY, False)) if x[0]]:
            p.add_run().add_break()
            r = p.add_run(line)
            r.italic = it
            r.font.size = Pt(9)
    return t


author_table(AUTHORS[:3], FULL_W)
sp = doc.add_paragraph(style="Normal")
sp.paragraph_format.space_after = Pt(4)
author_table(AUTHORS[3:], FULL_W * 2 / 3)
end_title = doc.add_paragraph(style="Normal")
end_title.paragraph_format.space_after = Pt(6)
end_title._p.get_or_add_pPr().append(copy.deepcopy(ONE_COL))  # end of the one-column title block


# --------------------------------------------------------------------------- tables
def build_table(src_tbl, title, span):
    width = FULL_W if span else COL_W
    if span:
        section_break(TWO_COL)
    cap = doc.add_paragraph(style="table head")  # template prefixes "TABLE I."
    cap.paragraph_format.keep_with_next = True
    cap.add_run(nb(title))
    nrows, ncols = len(src_tbl.rows), len(src_tbl.columns)
    # column widths: proportional to the longest text in each column (bounded), scaled to the target width
    lens = []
    for ci in range(ncols):
        m = max(max((len(w) for w in src_tbl.cell(ri, ci).text.split()), default=1) * 1.6 for ri in range(nrows))
        avg = sum(len(src_tbl.cell(ri, ci).text) for ri in range(nrows)) / nrows
        lens.append(max(m, min(avg, 60), 6))
    widths = [width * x / sum(lens) for x in lens]
    t = doc.add_table(rows=nrows, cols=ncols)
    old = t._tbl.tblPr
    new = copy.deepcopy(TBL_PR)
    for el in new.findall(qn("w:tblW")) + new.findall(qn("w:tblLayout")) + new.findall(qn("w:tblCellMar")):
        new.remove(el)
    lay = OxmlElement("w:tblLayout")
    lay.set(qn("w:type"), "fixed")
    new.append(lay)
    mar = OxmlElement("w:tblCellMar")
    for side, v in (("left", 40), ("right", 40)):
        m = OxmlElement(f"w:{side}")
        m.set(qn("w:w"), str(v))
        m.set(qn("w:type"), "dxa")
        mar.append(m)
    new.append(mar)
    old.getparent().replace(old, new)
    t.autofit = False
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for ci, w in enumerate(widths):
        t.columns[ci].width = Inches(w)
    for ri in range(nrows):
        row = t.rows[ri]
        row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        for ci in range(ncols):
            c = row.cells[ci]
            c.width = Inches(widths[ci])
            sc = src_tbl.cell(ri, ci)
            p = c.paragraphs[0]
            p.style = doc.styles["table col head" if ri == 0 else "table copy"]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if (ci == 0 or not re.match(r"^[\d.\-–+−% \[\],/()a-z]*$", sc.text)) \
                else WD_ALIGN_PARAGRAPH.CENTER
            if ri == 0:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next = ri < nrows - 1
            first = True
            for sp_ in sc.paragraphs:
                if not first:
                    p.add_run().add_break()
                first = False
                for r in sp_.runs:
                    if not r.text:
                        continue
                    nr = p.add_run(nb(r.text))
                    if r.bold and ri > 0:
                        nr.bold = True
                    if r.italic:
                        nr.italic = True
    gap = doc.add_paragraph(style="Normal")
    gap.paragraph_format.space_after = Pt(4)
    if span:
        gap._p.get_or_add_pPr().append(copy.deepcopy(ONE_COL))


# --------------------------------------------------------------------------- body
def image_blob(p):
    blips = p._p.findall(".//" + qn("a:blip"))
    if not blips:
        return None
    return src.part.related_parts[blips[0].get(qn("r:embed"))].blob


H2_LEAD = re.compile(r"^([A-G])\. (.+?)\.\s*$")
items = list(src.element.body.iterchildren())
paras = {id(p._p): p for p in src.paragraphs}
tables = {id(t._tbl): t for t in src.tables}

state = {"h1": 0, "h2": None, "table": 0, "fig": 0, "pending_table_title": None, "pending_fig": None,
         "in_refs": False, "section": ""}
skip_title = True

for el in items:
    if el.tag == qn("w:tbl"):
        state["table"] += 1
        build_table(tables[id(el)], state["pending_table_title"], state["table"] in SPAN_TABLES)
        state["pending_table_title"] = None
        continue
    if el.tag != qn("w:p"):
        continue
    p = paras[id(el)]
    text = p.text.strip()
    style = p.style.name
    blob = image_blob(p)

    if blob is not None:
        state["fig"] += 1
        state["pending_fig"] = blob
        continue
    if not text:
        continue
    if style == "Heading 1" or (style != "Heading 2" and state["section"] == ""):
        continue  # old title and author lines are replaced above
    if style == "Heading 2":
        skip_title = False
        state["section"] = text
        if text == "Abstract":
            continue
        if text in ("Acknowledgment", "References"):
            doc.add_paragraph(style="Heading 5").add_run(text)
            state["in_refs"] = text == "References"
            continue
        m = re.match(r"^[IVX]+\. (.+)$", text)
        doc.add_paragraph(style="Heading 1").add_run(nb(m.group(1) if m else text))
        state["h1"] += 1
        state["h2"] = None
        continue
    if state["section"] == "Abstract":
        if text.lower().startswith("keywords"):
            kw = re.sub(r"^Keywords\s*[—–-]\s*", "", text)
            doc.add_paragraph(style="Keywords").add_run("Index Terms—" + kw.rstrip(".") + ".")
        else:
            a = doc.add_paragraph(style="Abstract")
            a.add_run("Abstract").italic = True
            a.add_run("—" + nb(text))
        continue
    if text.startswith("TABLE "):
        state["pending_table_title"] = text.split("\n", 1)[1].strip() if "\n" in text else ""
        continue
    if style == "Quote" and re.match(r"^Fig\. \d+\.", text):
        span = state["fig"] in SPAN_FIGS
        if span:
            section_break(TWO_COL)
        fp = doc.add_paragraph(style="Normal")
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        fp.paragraph_format.keep_with_next = True
        fp.add_run().add_picture(io.BytesIO(state["pending_fig"]), width=Inches(5.2 if span else COL_W))
        c = doc.add_paragraph(style="figure caption")  # template prefixes "Fig. 1."
        c.add_run(nb(re.sub(r"^Fig\. \d+\.\s*", "", text)))
        if span:
            c._p.get_or_add_pPr().append(copy.deepcopy(ONE_COL))
        state["pending_fig"] = None
        continue
    if state["in_refs"]:
        doc.add_paragraph(style="references").add_run(re.sub(r"^\[\d+\]\s*", "", text))
        continue
    # sub-heading written as a bold run-in "A. Title." -> IEEE Heading 2 (template letters it A., B., ...)
    first = p.runs[0] if p.runs else None
    m = H2_LEAD.match(first.text) if first is not None and first.bold else None
    if m:
        expect = "A" if state["h2"] is None else chr(ord(state["h2"]) + 1)
        if m.group(1) != expect:
            sys.exit(f"sub-heading order broken at: {text[:60]}")
        state["h2"] = m.group(1)
        doc.add_paragraph(style="Heading 2").add_run(nb(m.group(2)))
        bp = doc.add_paragraph(style="Body Text")
        rest = p.runs[1:]
        if rest:
            rest_text = rest[0].text
            rest[0].text = rest_text.lstrip()
        add_runs(bp, rest)
        continue
    if style == "List Paragraph" and not re.match(r"^(\d\)|[GR]\d )", text):
        add_runs(doc.add_paragraph(style="bullet list"), p.runs)
        continue
    add_runs(doc.add_paragraph(style="Body Text"), p.runs, lead_italic=True)

if state["pending_fig"] is not None or state["pending_table_title"] is not None:
    sys.exit("a figure or table caption was not matched")

# --------------------------------------------------------------------------- metadata and save
cp = doc.core_properties
cp.title = TITLE
cp.author = ", ".join(n for n, _ in AUTHORS)
cp.subject = cp.keywords = cp.comments = cp.category = ""
doc.save(OUT)

import shutil
import zipfile
APP = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<Properties xmlns="http://schemas.'
       'openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/'
       'officeDocument/2006/docPropsVTypes"><Application>Microsoft Office Word</Application><DocSecurity>0'
       '</DocSecurity><ScaleCrop>false</ScaleCrop><Company>B.M.S. College of Engineering</Company><LinksUpToDate>'
       'false</LinksUpToDate><SharedDoc>false</SharedDoc><HyperlinksChanged>false</HyperlinksChanged><AppVersion>'
       '16.0000</AppVersion></Properties>')
tmp = OUT.with_suffix(".tmp.docx")
with zipfile.ZipFile(OUT) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        zout.writestr(item, APP.encode() if item.filename == "docProps/app.xml" else zin.read(item.filename))
shutil.move(tmp, OUT)

# --------------------------------------------------------------------------- PDF via Word
import win32com.client
word = win32com.client.DispatchEx("Word.Application")
word.Visible = False
try:
    d = word.Documents.Open(str(OUT))
    d.Fields.Update()
    pages = d.ComputeStatistics(2)
    d.Save()
    d.ExportAsFixedFormat(str(PDF), 17)
    d.Close(False)
finally:
    word.Quit()
print("saved", OUT.name, "and", PDF.name, "| pages:", pages, "| tables:", state["table"], "| figures:", state["fig"])
