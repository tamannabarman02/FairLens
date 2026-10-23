import io

import pytest

from fairlens.ingestion.loaders import IngestionError, extract_text, load_image
from fairlens.ingestion.segment import split_sentences


def test_txt_and_csv():
    assert extract_text("a.txt", "Hello world.".encode()).text == "Hello world."
    csv_bytes = b"id,text\n1,First row.\n2,Second row.\n"
    doc = extract_text("a.csv", csv_bytes)
    assert doc.text == "First row.\nSecond row."


def test_docx_and_pdf_roundtrip():
    from docx import Document

    buf = io.BytesIO()
    d = Document()
    d.add_paragraph("Looking for a young salesman.")
    d.save(buf)
    assert "young salesman" in extract_text("x.docx", buf.getvalue()).text

    from reportlab.pdfgen import canvas

    pbuf = io.BytesIO()
    c = canvas.Canvas(pbuf)
    c.drawString(72, 720, "Women are bad at maths.")
    c.save()
    doc = extract_text("x.pdf", pbuf.getvalue())
    assert "Women are bad" in doc.text and doc.pages == 1


@pytest.mark.parametrize("name,data,msg", [
    ("a.txt", b"", "empty"),
    ("a.exe", b"abc", "Unsupported"),
    ("a.pdf", b"not a pdf", "PDF"),
    ("a.txt", b"x" * (11 * 1024 * 1024), "limit"),
], ids=["empty", "unsupported", "bad-pdf", "too-large"])
def test_errors(name, data, msg):
    with pytest.raises(IngestionError, match=msg):
        extract_text(name, data)


def test_docx_tables_in_document_order():
    from docx import Document

    buf = io.BytesIO()
    d = Document()
    d.add_paragraph("Senior Sales Role")
    t = d.add_table(rows=2, cols=2)
    t.cell(0, 0).text, t.cell(0, 1).text = "Experience", "5+ years in B2B sales"
    t.cell(1, 0).text, t.cell(1, 1).text = "Licence", "Valid driving licence"
    d.add_paragraph("Apply by 30 June.")
    d.save(buf)
    lines = extract_text("job.docx", buf.getvalue()).text.splitlines()
    assert lines == ["Senior Sales Role", "Experience | 5+ years in B2B sales", "Licence | Valid driving licence", "Apply by 30 June."]


@pytest.mark.parametrize("enc", ["utf-16", "cp1252", "utf-8-sig"])
def test_windows_saved_text(enc):
    raw = "Café résumé line one.\r\nSecond line.\r\n".encode(enc)
    doc = extract_text("w.txt", raw)
    assert "Café résumé line one." in doc.text and "Second line." in doc.text


def test_multicolumn_csv_picks_text_column():
    data = 'id,title,text,salary\n1,Sales,"Looking for a young salesman, 5 years",30000\n2,Ops,"Neutral advert.",25000\n'.encode()
    assert extract_text("jobs.csv", data).text.splitlines() == ["Looking for a young salesman, 5 years", "Neutral advert."]


def test_scanned_pdf_without_text_layer_is_reported():
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.rect(50, 50, 200, 200, fill=1)  # an image-like page with no text layer
    c.save()
    with pytest.raises(IngestionError, match="no extractable text"):
        extract_text("scan.pdf", buf.getvalue())


def test_oversized_image_rejected():
    with pytest.raises(IngestionError, match="limit"):
        load_image("big.png", b"\x89PNG" + b"0" * (11 * 1024 * 1024))


def test_image_validation():
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (64, 48), "red").save(buf, format="PNG")
    assert load_image("x.png", buf.getvalue()).size == (64, 48)
    gif = io.BytesIO()
    Image.new("P", (32, 32)).save(gif, format="GIF")
    assert load_image("x.gif", gif.getvalue()).mode == "RGB"
    with pytest.raises(IngestionError):
        load_image("x.png", b"garbage")
    with pytest.raises(IngestionError, match="Unsupported"):
        load_image("x.bmp", buf.getvalue())


def test_sentence_split_keeps_quotes_and_abbreviations():
    t = 'Dr. Smith said "We hire everyone. Always." Then she left. Mr. Lee stayed.'
    s = [x.text for x in split_sentences(t)]
    assert s == ['Dr. Smith said "We hire everyone. Always."', "Then she left.", "Mr. Lee stayed."]
    for x in split_sentences(t):
        assert t[x.start:x.end] == x.text
