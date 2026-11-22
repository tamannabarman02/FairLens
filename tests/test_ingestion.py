import io

import pytest

from fairlens.ingestion.loaders import IngestionError, extract_text, load_image
from fairlens.ingestion.segment import split_sentences


def test_txt_and_csv():
    assert extract_text("a.txt", "Hello world.".encode()).text == "Hello world."
    csv_bytes = b"id,text\n1,First row.\n2,Second row.\n"
    doc = extract_text("a.csv", csv_bytes)
    assert doc.text == "First row.\n\nSecond row." and doc.column == "text"


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
