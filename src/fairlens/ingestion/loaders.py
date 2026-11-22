"""File ingestion: TXT, PDF, DOCX, CSV text extraction and image validation (size/format limits)."""
from __future__ import annotations

import codecs
import csv
import io
import re
import warnings
import zipfile
from dataclasses import dataclass, field

MAX_TEXT_BYTES = 10 * 1024 * 1024
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 60_000_000
MIN_IMAGE_SIDE = 16
MAX_CHARS = 20_000
TEXT_EXTS = ("txt", "pdf", "docx", "csv")
IMAGE_EXTS = ("jpg", "jpeg", "png", "gif")
TEXT_COLUMN_NAMES = ("text", "sentence", "content", "description")
CSV_DELIMITERS = ",;\t|"

_CHAR_FIXES = str.maketrans({
    " ": " ", " ": " ", " ": " ", "\t": " ", "\f": "\n", "\v": "\n",
    "­": None, "​": None, "﻿": None, "\x00": None,
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st",
})
_IMAGE_MAGIC = (b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"GIF87a", b"GIF89a")
_LEGACY_OFFICE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_KIND_LABELS = {"pdf": "a PDF", "docx": "a Word (DOCX) document"}


class IngestionError(ValueError):
    """Raised with a user-facing message when a file cannot be used."""


@dataclass
class Document:
    name: str
    text: str
    kind: str
    pages: int | None = None
    truncated: bool = False
    column: str | None = None
    warnings: list[str] = field(default_factory=list)


def ext_of(name: str) -> str:
    return name.lower().rsplit(".", 1)[-1] if "." in name else ""


def sniff_kind(data: bytes) -> str | None:
    """Identify a file from its bytes: 'pdf', 'docx', 'zip', 'damaged_zip', 'image', 'legacy_office' or None (plain text)."""
    if data.lstrip(b"\x00\t\n\r ").startswith(b"%PDF-"):
        return "pdf"
    if data.startswith(_IMAGE_MAGIC):
        return "image"
    if data.startswith(_LEGACY_OFFICE_MAGIC):
        return "legacy_office"
    if data.startswith(b"PK\x03\x04"):
        try:
            names = zipfile.ZipFile(io.BytesIO(data)).namelist()
        except zipfile.BadZipFile:
            return "damaged_zip"
        return "docx" if "word/document.xml" in names else "zip"
    return None


def _decode(data: bytes) -> str:
    if data.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
        encodings = ("utf-32",)
    elif data.startswith(codecs.BOM_UTF8):
        encodings = ("utf-8-sig",)
    elif data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        encodings = ("utf-16",)
    elif b"\x00" in data[:4096]:
        encodings = (_bomless_utf16(data[:4096]),)
    else:
        encodings = ("utf-8", "cp1252")
    for enc in encodings:
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    raise IngestionError("Could not decode the text file (tried UTF-8, UTF-16, Windows-1252). Save it as UTF-8 and try again.")


def _bomless_utf16(sample: bytes) -> str:
    even, odd = sample[0::2].count(0), sample[1::2].count(0)
    if odd > 0.3 * len(sample) / 2 and even == 0:
        return "utf-16-le"
    if even > 0.3 * len(sample) / 2 and odd == 0:
        return "utf-16-be"
    raise IngestionError("This does not look like a text file. Upload a TXT, PDF, DOCX or CSV file.")


def _normalise(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").translate(_CHAR_FIXES)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _resolve_kind(name: str, data: bytes) -> tuple[str, list[str]]:
    ext = ext_of(name)
    sniffed = sniff_kind(data)
    if sniffed == "image":
        raise IngestionError("This file is an image, not a document. Upload JPG, PNG or GIF files on the Image Analysis page.")
    if sniffed == "legacy_office" or (sniffed is None and ext in ("doc", "xls")):
        raise IngestionError("Old Word (.doc) and Excel (.xls) files are not supported. Save the file as DOCX or CSV and upload it again.")
    if sniffed == "damaged_zip":
        raise IngestionError("The file is damaged or incomplete and could not be opened. Save it again and re-upload it.")
    if sniffed == "zip":
        raise IngestionError("This file is a ZIP-based document that is not a Word file (for example Excel or PowerPoint). "
                             "Save it as DOCX, PDF, CSV or TXT and upload it again.")
    if sniffed is None and ext == "pdf" and b"%PDF-" in data[:1024]:
        sniffed = "pdf"
    if sniffed in _KIND_LABELS:
        notes = [] if ext == sniffed else [f"The file is named '.{ext}' but is actually {_KIND_LABELS[sniffed]}, so it was read as one."]
        return sniffed, notes
    if ext in _KIND_LABELS:
        raise IngestionError(f"This file is named '.{ext}' but is not {_KIND_LABELS[ext]}. It may be damaged or saved in another format.")
    if ext in ("txt", "csv"):
        return ext, []
    if not ext:
        raise IngestionError("The file has no extension. Upload a TXT, PDF, DOCX or CSV file.")
    raise IngestionError(f"Unsupported file type '.{ext}'. Use TXT, PDF, DOCX or CSV.")


def extract_text(name: str, data: bytes, text_column: str | None = None) -> Document:
    if not data:
        raise IngestionError("The file is empty.")
    if len(data) > MAX_TEXT_BYTES:
        raise IngestionError(f"File exceeds the {MAX_TEXT_BYTES // 2**20} MB limit.")
    kind, notes = _resolve_kind(name, data)
    pages = column = None
    if kind == "txt":
        text = _decode(data)
    elif kind == "csv":
        text, column, csv_notes = _extract_csv(_decode(data), text_column)
        notes += csv_notes
    elif kind == "pdf":
        text, pages, pdf_notes = _extract_pdf(data)
        notes += pdf_notes
    else:
        text = _extract_docx(data)
    text = _normalise(text)
    if not text:
        raise IngestionError("No text found in the file.")
    truncated = len(text) > MAX_CHARS
    if truncated:
        notes.append(f"Only the first {MAX_CHARS:,} characters are used.")
    return Document(name, text[:MAX_CHARS], kind, pages, truncated, column, notes)


def _csv_rows(text: str) -> list[list[str]]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    header = text.split("\n", 1)[0]
    delimiter = max(CSV_DELIMITERS, key=header.count)
    try:
        rows = list(csv.reader(io.StringIO(text, newline=""), delimiter=delimiter if header.count(delimiter) else ","))
    except csv.Error as exc:
        raise IngestionError("Could not parse the CSV file. Check that it is a plain comma-separated file.") from exc
    return [r for r in rows if any(c.strip() for c in r)]


def csv_columns(data: bytes) -> list[str]:
    """Header names of a CSV upload, for letting the user pick the text column."""
    rows = _csv_rows(_decode(data))
    return [h.strip() for h in rows[0]] if rows else []


def _longest_text_column(rows: list[list[str]], width: int) -> int:
    def avg_len(col: int) -> float:
        cells = [r[col].strip() for r in rows if col < len(r)]
        words = [c for c in cells if c and not re.fullmatch(r"[\d\s.,%£$€+-]+", c)]
        return sum(map(len, words)) / max(1, len(cells))

    return max(range(width), key=avg_len)


def _extract_csv(text: str, text_column: str | None) -> tuple[str, str | None, list[str]]:
    rows = _csv_rows(text)
    if not rows:
        raise IngestionError("The CSV file has no rows.")
    header = [h.strip() for h in rows[0]]
    lower = [h.lower() for h in header]
    notes = []
    if text_column:
        if text_column.strip().lower() not in lower:
            raise IngestionError(f"The CSV has no column named '{text_column}'. Columns found: {', '.join(header)}.")
        col = lower.index(text_column.strip().lower())
    elif any(c in lower for c in TEXT_COLUMN_NAMES):
        col = next(lower.index(c) for c in TEXT_COLUMN_NAMES if c in lower)
    elif len(header) == 1:
        return "\n\n".join(r[0] for r in rows), None, notes
    else:
        col = _longest_text_column(rows[1:], len(header))
        notes.append(f"No column named {' / '.join(TEXT_COLUMN_NAMES)} was found, so the column '{header[col]}' "
                     "(longest text) was used.")
    body = [r[col].strip() for r in rows[1:] if col < len(r) and r[col].strip()]
    if not body:
        raise IngestionError(f"The CSV column '{header[col]}' has no text.")
    return "\n\n".join(body), header[col], notes


def _extract_pdf(data: bytes) -> tuple[str, int, list[str]]:
    from pypdf import PdfReader
    from pypdf.errors import DependencyError, FileNotDecryptedError

    try:
        reader = PdfReader(io.BytesIO(data))
        locked = reader.is_encrypted and not reader.decrypt("")
        page_texts = [] if locked else [p.extract_text() or "" for p in reader.pages]
    except (FileNotDecryptedError, DependencyError) as exc:
        raise IngestionError("This PDF is password-protected. Remove the password and upload it again.") from exc
    except Exception as exc:  # noqa: BLE001
        raise IngestionError("Could not read the PDF. The file may be damaged or incomplete.") from exc
    if locked:
        raise IngestionError("This PDF is password-protected. Remove the password and upload it again.")
    if not any(t.strip() for t in page_texts):
        raise IngestionError("This PDF has no selectable text. Scanned PDFs need OCR, which FairLens does not do.")
    blank = [str(i + 1) for i, t in enumerate(page_texts) if not t.strip()]
    notes = [f"Page {', '.join(blank)} of {len(page_texts)} has no selectable text (probably scanned) and was skipped."] if blank else []
    return "\n\n".join(t for t in page_texts if t.strip()), len(page_texts), notes


def _extract_docx(data: bytes) -> str:
    from docx import Document as Docx

    try:
        d = Docx(io.BytesIO(data))
        blocks = _docx_blocks(d)
    except Exception as exc:  # noqa: BLE001
        raise IngestionError("Could not read the Word document. The file may be damaged or incomplete.") from exc
    return "\n\n".join(blocks)


def _docx_blocks(d) -> list[str]:
    headers: list[str] = []
    footers: list[str] = []
    for section in d.sections:
        parts = [(section.header, headers), (section.footer, footers)]
        if section.different_first_page_header_footer:
            parts += [(section.first_page_header, headers), (section.first_page_footer, footers)]
        if d.settings.odd_and_even_pages_header_footer:
            parts += [(section.even_page_header, headers), (section.even_page_footer, footers)]
        for part, target in parts:
            if not part.is_linked_to_previous:
                target += [b for b in _container_blocks(part) if b not in target]
    return headers + _container_blocks(d) + footers


def _container_blocks(container) -> list[str]:
    from docx.table import Table

    blocks = []
    for item in container.iter_inner_content():
        if isinstance(item, Table):
            blocks += _table_rows(item)
        elif item.text.strip():
            blocks.append(item.text)
    return blocks


def _table_rows(table) -> list[str]:
    rows, seen = [], set()
    for row in table.rows:
        cells = []
        for cell in row.cells:
            if cell._tc in seen:
                continue
            seen.add(cell._tc)
            text = " ".join(_container_blocks(cell)).strip()
            if text:
                cells.append(text)
        if cells:
            rows.append(" | ".join(cells))
    return rows


def load_image(name: str, data: bytes):
    """Validate and open an image. Returns an RGB PIL image (first frame for GIF, EXIF rotation applied, transparency on white)."""
    from PIL import Image, ImageOps, UnidentifiedImageError

    if not data:
        raise IngestionError("The image file is empty.")
    if len(data) > MAX_IMAGE_BYTES:
        raise IngestionError(f"Image exceeds the {MAX_IMAGE_BYTES // 2**20} MB limit.")
    if ext_of(name) not in IMAGE_EXTS:
        raise IngestionError(f"Unsupported image type '.{ext_of(name)}'. Use JPG, PNG or GIF.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            img = Image.open(io.BytesIO(data))
            img.verify()
            img = Image.open(io.BytesIO(data))
    except Image.DecompressionBombError as exc:
        raise IngestionError(f"The image has too many pixels; the limit is {MAX_IMAGE_PIXELS // 1_000_000} megapixels.") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise IngestionError("The file is not a valid image or is corrupted.") from exc
    if img.width * img.height > MAX_IMAGE_PIXELS:
        raise IngestionError(f"The image is {img.width}x{img.height} pixels; the limit is {MAX_IMAGE_PIXELS // 1_000_000} megapixels. "
                             "Resize it and try again.")
    try:
        img.seek(0)
        img = _flatten(ImageOps.exif_transpose(img))
    except (OSError, SyntaxError, ValueError) as exc:
        raise IngestionError("The file is not a valid image or is corrupted.") from exc
    if min(img.size) < MIN_IMAGE_SIDE:
        raise IngestionError(f"The image is too small to caption (minimum {MIN_IMAGE_SIDE}x{MIN_IMAGE_SIDE} pixels).")
    return img


def _flatten(img):
    from PIL import Image

    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return img.convert("RGB")
