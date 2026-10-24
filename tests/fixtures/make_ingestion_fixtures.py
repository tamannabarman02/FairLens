"""Regenerate the upload fixtures in tests/fixtures/ingestion/ (run: python tests/fixtures/make_ingestion_fixtures.py)."""
from __future__ import annotations

import datetime as dt
import io
from pathlib import Path

OUT = Path(__file__).resolve().parent / "ingestion"

JD_PAGE_1 = [
    "Senior Sales Executive – Leeds",
    "We are looking for a young, energetic salesman to join our team.",
    "You must have 5+ years of experience and a record of 120% target attainment.",
    "Salary: £45,000 – £55,000 per year, plus a 10% bonus.",
    "The ideal candidate is a “digital native” who thrives under pressure.",
]
JD_PAGE_2 = [
    "Benefits include 25 days of paid leave and a 6.5% pension match.",
    "Applications close on 31 March 2026. Contact Dr. Patel for details.",
]
TXT_BODY = ("Café owner seeks a “rockstar” barista. You need 5+ years’ experience – pay is £12.50/hour.\n"
            "\n"
            "We’re an equal-opportunity employer. Ask for “Sam” at the counter.\n")
CP1252_BODY = ("Our chairman wants a “strong, aggressive” leader. Budget: €80,000 – £70,000.\n"
               "\n"
               "Candidates must be “young at heart”. Start date: 1 June.\n")


def _pdf(pages: list[list[str]], scanned_pages: frozenset[int] = frozenset(), **canvas_kwargs) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4, invariant=1, **canvas_kwargs)
    for i, lines in enumerate(pages):
        if i in scanned_pages:
            c.drawImage(ImageReader(_scan_image(lines)), 72, 480, width=450, height=300)
        else:
            y = 760
            for line in lines:
                c.drawString(72, y, line)
                y -= 22
        c.showPage()
    c.save()
    return buf.getvalue()


def _scan_image(lines: list[str]):
    from PIL import Image, ImageDraw

    img = Image.new("L", (600, 400), 255)
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((20, 20 + 30 * i), line, fill=0)
    return img


def _docx() -> bytes:
    from docx import Document

    d = Document()
    d.core_properties.created = d.core_properties.modified = dt.datetime(2026, 1, 1)
    d.sections[0].header.paragraphs[0].text = "FairLens Recruitment – Confidential"
    d.sections[0].footer.paragraphs[0].text = "Equal opportunities employer."
    d.add_heading("Warehouse Operative", level=1)
    d.add_paragraph("We need a strong young man for night shifts.")
    d.add_paragraph("The role pays “competitive” rates.")
    table = d.add_table(rows=4, cols=2)
    rows = [("Requirement", "Detail"), ("Experience", "3+ years in logistics"), ("Age", "Must be under 30"),
            ("Salary", "£24,500 per year")]
    for row, (a, b) in zip(table.rows, rows):
        row.cells[0].text, row.cells[1].text = a, b
    merged = table.add_row().cells
    merged[0].merge(merged[1]).text = "Shift pattern: 4 nights on, 3 nights off"
    d.add_paragraph("Apply by 30 June.")
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def _image(fmt: str, size=(320, 240), mode="RGB", **save_kwargs) -> bytes:
    from PIL import Image, ImageDraw

    img = Image.new(mode, size, (30, 120, 200) if mode == "RGB" else (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    w, h = size
    draw.rectangle((w // 4, h // 4, 3 * w // 4, 3 * h // 4), fill=(220, 40, 40) + ((255,) if mode == "RGBA" else ()))
    buf = io.BytesIO()
    img.save(buf, format=fmt, **save_kwargs)
    return buf.getvalue()


def _rotated_jpeg() -> bytes:
    from PIL import Image

    img = Image.new("RGB", (120, 60), (40, 160, 90))
    exif = Image.Exif()
    exif[0x0112] = 6
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif.tobytes())
    return buf.getvalue()


def _animated_gif() -> bytes:
    from PIL import Image

    frames = [Image.new("P", (48, 48), i * 60) for i in range(3)]
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:], duration=100, loop=0, transparency=0)
    return buf.getvalue()


def _huge_dimensions_png() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("1", (10_000, 10_000), 1).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _csv_multi_column() -> bytes:
    lines = [
        "id,title,advert,salary,location",
        '1,Sales lead,"We want a young, hungry salesman with 5+ years of experience.","£40,000",Leeds',
        '2,Nurse,"The ideal nurse is a caring woman who says ""yes"" to every shift.",£32000,York',
        '3,Engineer,"Looking for an engineer.\nMust have a degree and 2+ years of Python.",£55000,Hull',
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def build() -> dict[str, bytes]:
    jd = _pdf([JD_PAGE_1, JD_PAGE_2])
    docx = _docx()
    photo = _image("JPEG", quality=85)
    return {
        "job_description_2_pages.pdf": jd,
        "scanned_no_text_layer.pdf": _pdf([JD_PAGE_1], scanned_pages=frozenset({0})),
        "mixed_text_and_scanned.pdf": _pdf([JD_PAGE_1, JD_PAGE_2], scanned_pages=frozenset({1})),
        "password_protected.pdf": _pdf([JD_PAGE_1], encrypt="secret"),
        "truncated.pdf": jd[: len(jd) // 3],
        "header_paragraphs_table.docx": docx,
        "truncated.docx": docx[: len(docx) // 2],
        "multi_column.csv": _csv_multi_column(),
        "excel_bom_crlf.csv": "Text,Label\r\nShe is too emotional to lead.,biased\r\nThe team meets on Mondays.,neutral\r\n".encode("utf-8-sig"),
        "semicolon_delimited.csv": "id;text;label\n1;Only men need apply.;biased\n2;Parking is free.;neutral\n".encode("utf-8"),
        "utf8.txt": TXT_BODY.encode("utf-8"),
        "utf8_bom.txt": TXT_BODY.encode("utf-8-sig"),
        "utf16_notepad_unicode.txt": TXT_BODY.replace("\n", "\r\n").encode("utf-16"),
        "windows1252_crlf.txt": CP1252_BODY.replace("\n", "\r\n").encode("cp1252"),
        "empty.txt": b"",
        "whitespace_only.txt": b"  \r\n\t \r\n",
        "pdf_saved_as.txt": jd,
        "docx_saved_as.pdf": docx,
        "legacy_word.doc": b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 504,
        "photo.jpg": photo,
        "photo_exif_rotated.jpg": _rotated_jpeg(),
        "chart.png": _image("PNG"),
        "transparent_alpha.png": _image("PNG", size=(64, 64), mode="RGBA"),
        "static.gif": _image("GIF", size=(64, 48)),
        "animated.gif": _animated_gif(),
        "tiny_8x8.png": _image("PNG", size=(8, 8)),
        "huge_dimensions.png": _huge_dimensions_png(),
        "truncated.jpg": photo[: len(photo) // 2],
        "not_an_image.png": b"This is a text file pretending to be a PNG.\n",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in build().items():
        (OUT / name).write_bytes(data)
        print(f"{name:32s} {len(data):>8,d} bytes")


if __name__ == "__main__":
    main()
