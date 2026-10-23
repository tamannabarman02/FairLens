"""File ingestion: TXT, PDF, DOCX, CSV text extraction and image validation (size/format limits)."""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass

MAX_TEXT_BYTES = 10 * 1024 * 1024
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_CHARS = 20_000
TEXT_EXTS = {"txt", "pdf", "docx", "csv"}
IMAGE_EXTS = {"jpg", "jpeg", "png", "gif", "webp"}


class IngestionError(ValueError):
    """Raised with a user-facing message when a file cannot be used."""


@dataclass
class Document:
    name: str
    text: str
    kind: str
    pages: int | None = None
    truncated: bool = False


def ext_of(name: str) -> str:
    return name.lower().rsplit(".", 1)[-1] if "." in name else ""


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    raise IngestionError("Could not decode the text file (tried UTF-8, UTF-16, Windows-1252).")


def extract_text(name: str, data: bytes, text_column: str | None = None) -> Document:
    if not data:
        raise IngestionError("The file is empty.")
    if len(data) > MAX_TEXT_BYTES:
        raise IngestionError(f"File exceeds the {MAX_TEXT_BYTES // 2**20} MB limit.")
    ext = ext_of(name)
    pages = None
    if ext == "txt":
        text = _decode(data)
    elif ext == "csv":
        rows = list(csv.reader(io.StringIO(_decode(data))))
        if not rows:
            raise IngestionError("The CSV file has no rows.")
        header = [h.strip().lower() for h in rows[0]]
        col = None
        for cand in ([text_column.lower()] if text_column else []) + ["text", "sentence", "content", "description"]:
            if cand in header:
                col = header.index(cand)
                break
        body = rows[1:] if col is not None else rows
        text = "\n".join(r[col] if col is not None and col < len(r) else " | ".join(r) for r in body if any(c.strip() for c in r))
    elif ext == "pdf":
        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(data))
            pages = len(reader.pages)
            text = "\n".join(p.extract_text() or "" for p in reader.pages)
        except Exception as exc:  # noqa: BLE001
            raise IngestionError(f"Could not read the PDF ({type(exc).__name__}). Is it encrypted or corrupted?") from exc
        if not text.strip():
            raise IngestionError("The PDF has no extractable text (it may be a scanned image).")
    elif ext == "docx":
        try:
            from docx import Document as Docx

            d = Docx(io.BytesIO(data))
            text = "\n".join(p.text for p in d.paragraphs)
        except Exception as exc:  # noqa: BLE001
            raise IngestionError(f"Could not read the DOCX file ({type(exc).__name__}).") from exc
    else:
        raise IngestionError(f"Unsupported file type '.{ext}'. Use TXT, PDF, DOCX or CSV.")
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        raise IngestionError("No text found in the file.")
    truncated = len(text) > MAX_CHARS
    return Document(name, text[:MAX_CHARS], ext, pages, truncated)


def load_image(name: str, data: bytes):
    """Validate and open an image. Returns a PIL image (first frame for GIF)."""
    from PIL import Image, UnidentifiedImageError

    if not data:
        raise IngestionError("The image file is empty.")
    if len(data) > MAX_IMAGE_BYTES:
        raise IngestionError(f"Image exceeds the {MAX_IMAGE_BYTES // 2**20} MB limit.")
    if ext_of(name) not in IMAGE_EXTS:
        raise IngestionError(f"Unsupported image type '.{ext_of(name)}'. Use JPG, PNG or GIF.")
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
        img = Image.open(io.BytesIO(data))
        img.seek(0)
        img = img.convert("RGB")
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise IngestionError("The file is not a valid image or is corrupted.") from exc
    if min(img.size) < 16:
        raise IngestionError("The image is too small to caption (minimum 16x16 pixels).")
    return img
