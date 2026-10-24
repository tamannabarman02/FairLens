"""Every supported upload format, end to end at the library level, using the files in tests/fixtures/ingestion/."""
import io

import pytest
from conftest import INGESTION_FIXTURES, FakeDetector

from fairlens.ingestion import loaders
from fairlens.ingestion.loaders import IngestionError, csv_columns, extract_text, load_image
from fairlens.ingestion.segment import split_sentences

TXT_EXPECTED = ("Café owner seeks a “rockstar” barista. You need 5+ years’ experience – pay is £12.50/hour.\n\n"
                "We’re an equal-opportunity employer. Ask for “Sam” at the counter.")
JD_PHRASES = ["Senior Sales Executive – Leeds", "young, energetic salesman", "5+ years of experience", "120% target",
              "£45,000 – £55,000", "10% bonus", "“digital native”", "25 days of paid leave", "6.5% pension",
              "31 March 2026", "Contact Dr. Patel for details."]
TEXT_FIXTURES = ["job_description_2_pages.pdf", "mixed_text_and_scanned.pdf", "header_paragraphs_table.docx",
                 "multi_column.csv", "excel_bom_crlf.csv", "semicolon_delimited.csv", "utf8.txt", "utf8_bom.txt",
                 "utf16_notepad_unicode.txt", "windows1252_crlf.txt", "pdf_saved_as.txt", "docx_saved_as.pdf"]


def _error(name, data, **kwargs):
    with pytest.raises(IngestionError) as info:
        extract_text(name, data, **kwargs)
    return str(info.value)


def _image_error(name, data):
    with pytest.raises(IngestionError) as info:
        load_image(name, data)
    return str(info.value)


def test_upload_constants_describe_the_supported_formats():
    assert loaders.TEXT_EXTS == ("txt", "pdf", "docx", "csv")
    assert loaders.IMAGE_EXTS == ("jpg", "jpeg", "png", "gif")
    assert loaders.MAX_TEXT_BYTES == loaders.MAX_IMAGE_BYTES == 10 * 2**20
    assert loaders.MAX_CHARS == 20_000 and loaders.MIN_IMAGE_SIDE == 16 and loaders.MAX_IMAGE_PIXELS == 60_000_000


# ----------------------------------------------------------------------------------------- TXT
def test_utf8_txt_keeps_smart_quotes_numbers_and_accents(upload):
    doc = extract_text("utf8.txt", upload("utf8.txt"))
    assert doc.text == TXT_EXPECTED and doc.kind == "txt" and not doc.warnings


def test_utf8_bom_txt_drops_the_bom(upload):
    assert extract_text("utf8_bom.txt", upload("utf8_bom.txt")).text == TXT_EXPECTED


def test_notepad_unicode_utf16_txt_decodes_and_normalises_crlf(upload):
    raw = upload("utf16_notepad_unicode.txt")
    assert raw[:2] == b"\xff\xfe" and "\r\n".encode("utf-16-le") in raw
    assert extract_text("utf16_notepad_unicode.txt", raw).text == TXT_EXPECTED


def test_utf16_txt_without_bom_is_detected():
    assert extract_text("a.txt", TXT_EXPECTED.encode("utf-16-le")).text == TXT_EXPECTED
    assert extract_text("a.txt", TXT_EXPECTED.encode("utf-16-be")).text == TXT_EXPECTED


def test_windows1252_txt_with_even_byte_count_is_not_misread_as_utf16(upload):
    raw = upload("windows1252_crlf.txt")
    assert len(raw) % 2 == 0 and b"\r\n" in raw and b"\x93" in raw
    text = extract_text("windows1252_crlf.txt", raw).text
    assert text == ("Our chairman wants a “strong, aggressive” leader. Budget: €80,000 – £70,000.\n\n"
                    "Candidates must be “young at heart”. Start date: 1 June.")


def test_pdf_ligatures_and_invisible_characters_are_normalised():
    text = extract_text("a.txt", "A ﬁreman and  ­staﬀ​ wanted.\r\n".encode("utf-8")).text
    assert text == "A fireman and staff wanted."


def test_empty_file_is_rejected(upload):
    assert _error("empty.txt", upload("empty.txt")) == "The file is empty."


def test_whitespace_only_file_reports_no_text(upload):
    assert _error("whitespace_only.txt", upload("whitespace_only.txt")) == "No text found in the file."


def test_binary_bytes_named_txt_are_rejected():
    assert "does not look like a text file" in _error("a.txt", bytes(range(256)) * 4)


def test_text_over_the_size_limit_is_rejected():
    assert "10 MB limit" in _error("a.txt", b"x" * (loaders.MAX_TEXT_BYTES + 1))


def test_long_text_is_truncated_to_max_chars_with_a_note():
    doc = extract_text("a.txt", ("word " * 5000).encode("utf-8"))
    assert doc.truncated and len(doc.text) == loaders.MAX_CHARS and "20,000" in doc.warnings[0]


# --------------------------------------------------------------------------- wrong / unsupported types
def test_pdf_saved_with_txt_extension_is_read_as_pdf_with_a_note(upload):
    doc = extract_text("pdf_saved_as.txt", upload("pdf_saved_as.txt"))
    assert doc.kind == "pdf" and doc.pages == 2 and "5+ years" in doc.text
    assert doc.warnings == ["The file is named '.txt' but is actually a PDF, so it was read as one."]


def test_docx_saved_with_pdf_extension_is_read_as_docx_with_a_note(upload):
    doc = extract_text("docx_saved_as.pdf", upload("docx_saved_as.pdf"))
    assert doc.kind == "docx" and "3+ years in logistics" in doc.text and "Word (DOCX)" in doc.warnings[0]


def test_image_uploaded_as_document_points_to_the_image_page(upload):
    assert "Image Analysis page" in _error("photo.txt", upload("photo.jpg"))


def test_legacy_word_doc_asks_for_docx(upload):
    assert "Save the file as DOCX" in _error("legacy_word.doc", upload("legacy_word.doc"))


@pytest.mark.parametrize("name,msg", [("a.exe", "Unsupported file type '.exe'"), ("README", "no extension")])
def test_unsupported_or_missing_extension_is_named(name, msg):
    assert msg in _error(name, b"plain text")


def test_garbage_named_pdf_says_it_is_not_a_pdf():
    assert _error("a.pdf", b"not a pdf") == "This file is named '.pdf' but is not a PDF. It may be damaged or saved in another format."


# ----------------------------------------------------------------------------------------- PDF
def test_two_page_pdf_extracts_both_pages_with_numbers_and_quotes(upload):
    doc = extract_text("job_description_2_pages.pdf", upload("job_description_2_pages.pdf"))
    assert doc.kind == "pdf" and doc.pages == 2 and not doc.warnings
    for phrase in JD_PHRASES:
        assert phrase in doc.text
    assert doc.text.index("10% bonus") < doc.text.index("25 days")


def test_scanned_pdf_explains_that_ocr_is_not_supported(upload):
    assert _error("scan.pdf", upload("scanned_no_text_layer.pdf")) == \
        "This PDF has no selectable text. Scanned PDFs need OCR, which FairLens does not do."


def test_partly_scanned_pdf_keeps_the_text_pages_and_names_the_skipped_page(upload):
    doc = extract_text("mixed.pdf", upload("mixed_text_and_scanned.pdf"))
    assert doc.pages == 2 and "5+ years" in doc.text and "25 days" not in doc.text
    assert doc.warnings == ["Page 2 of 2 has no selectable text (probably scanned) and was skipped."]


def test_password_protected_pdf_asks_to_remove_the_password(upload):
    assert _error("locked.pdf", upload("password_protected.pdf")) == \
        "This PDF is password-protected. Remove the password and upload it again."


def test_truncated_pdf_reports_damage_without_internal_error_names(upload):
    assert _error("truncated.pdf", upload("truncated.pdf")) == "Could not read the PDF. The file may be damaged or incomplete."


# ---------------------------------------------------------------------------------------- DOCX
def test_docx_includes_header_paragraphs_table_rows_and_footer_in_order(upload):
    doc = extract_text("jd.docx", upload("header_paragraphs_table.docx"))
    assert doc.text.split("\n\n") == [
        "FairLens Recruitment – Confidential", "Warehouse Operative", "We need a strong young man for night shifts.",
        "The role pays “competitive” rates.", "Requirement | Detail", "Experience | 3+ years in logistics",
        "Age | Must be under 30", "Salary | £24,500 per year", "Shift pattern: 4 nights on, 3 nights off",
        "Apply by 30 June.", "Equal opportunities employer."]


def test_docx_merged_table_cell_appears_once(upload):
    assert extract_text("jd.docx", upload("header_paragraphs_table.docx")).text.count("Shift pattern") == 1


def test_truncated_docx_reports_damage(upload):
    assert "damaged or incomplete" in _error("truncated.docx", upload("truncated.docx"))


def test_zip_that_is_not_a_word_file_is_explained():
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("xl/workbook.xml", "<workbook/>")
    assert "not a Word file" in _error("sheet.docx", buf.getvalue())


# ----------------------------------------------------------------------------------------- CSV
def test_csv_without_a_known_text_column_uses_the_longest_text_column_and_says_so(upload):
    doc = extract_text("multi_column.csv", upload("multi_column.csv"))
    assert doc.column == "advert" and "'advert'" in doc.warnings[0]
    assert doc.text.split("\n\n") == [
        "We want a young, hungry salesman with 5+ years of experience.",
        "The ideal nurse is a caring woman who says \"yes\" to every shift.",
        "Looking for an engineer.\nMust have a degree and 2+ years of Python."]


def test_csv_text_column_can_be_chosen_case_insensitively(upload):
    doc = extract_text("multi_column.csv", upload("multi_column.csv"), text_column="TITLE")
    assert doc.column == "title" and doc.text == "Sales lead\n\nNurse\n\nEngineer" and not doc.warnings


def test_csv_unknown_text_column_lists_the_available_columns(upload):
    msg = _error("multi_column.csv", upload("multi_column.csv"), text_column="body")
    assert msg == "The CSV has no column named 'body'. Columns found: id, title, advert, salary, location."


def test_csv_columns_lists_the_headers_for_the_ui(upload):
    assert csv_columns(upload("multi_column.csv")) == ["id", "title", "advert", "salary", "location"]
    assert csv_columns(upload("excel_bom_crlf.csv")) == ["Text", "Label"]


def test_excel_csv_with_bom_and_crlf_uses_the_text_column(upload):
    doc = extract_text("excel.csv", upload("excel_bom_crlf.csv"))
    assert doc.column == "Text" and doc.text == "She is too emotional to lead.\n\nThe team meets on Mondays."


def test_semicolon_delimited_csv_is_split_on_semicolons(upload):
    doc = extract_text("semi.csv", upload("semicolon_delimited.csv"))
    assert doc.column == "text" and doc.text == "Only men need apply.\n\nParking is free."


def test_single_column_csv_without_header_keeps_every_row():
    assert extract_text("a.csv", b"young salesman wanted\nolder staff need not apply\n").text == \
        "young salesman wanted\n\nolder staff need not apply"


def test_csv_rows_become_separate_sentences_even_without_punctuation():
    doc = extract_text("a.csv", b"id,text\n1,young salesman wanted\n2,older staff need not apply\n")
    assert [s.text for s in split_sentences(doc.text)] == ["young salesman wanted", "older staff need not apply"]


# -------------------------------------------------------------------------------------- images
@pytest.mark.parametrize("name,size", [("photo.jpg", (320, 240)), ("chart.png", (320, 240)), ("static.gif", (64, 48))])
def test_jpg_png_gif_load_as_rgb(upload, name, size):
    img = load_image(name, upload(name))
    assert img.mode == "RGB" and img.size == size


def test_png_alpha_is_flattened_onto_white_not_black(upload):
    img = load_image("alpha.png", upload("transparent_alpha.png"))
    assert img.getpixel((0, 0)) == (255, 255, 255) and img.getpixel((32, 32)) == (220, 40, 40)


def test_animated_gif_uses_the_first_frame(upload):
    img = load_image("animated.gif", upload("animated.gif"))
    assert img.mode == "RGB" and img.size == (48, 48) and img.getpixel((0, 0)) == (255, 255, 255)


def test_exif_rotated_phone_photo_is_turned_upright(upload):
    assert load_image("phone.jpg", upload("photo_exif_rotated.jpg")).size == (60, 120)


def test_png_named_jpg_still_loads(upload):
    assert load_image("chart.jpg", upload("chart.png")).size == (320, 240)


def test_image_over_the_byte_limit_is_rejected(upload):
    data = upload("photo.jpg") + b"\0" * loaders.MAX_IMAGE_BYTES
    assert _image_error("big.jpg", data) == "Image exceeds the 10 MB limit."


def test_image_over_the_pixel_limit_is_rejected_before_decoding(upload):
    assert _image_error("huge.png", upload("huge_dimensions.png")).startswith("The image is 10000x10000 pixels; the limit is 60 megapixels.")


@pytest.mark.parametrize("name", ["truncated.jpg", "not_an_image.png"])
def test_corrupt_image_is_rejected(upload, name):
    assert _image_error(name, upload(name)) == "The file is not a valid image or is corrupted."


def test_tiny_image_is_rejected(upload):
    assert "minimum 16x16" in _image_error("tiny.png", upload("tiny_8x8.png"))


def test_empty_image_is_rejected():
    assert _image_error("a.png", b"") == "The image file is empty."


@pytest.mark.parametrize("name", ["x.bmp", "x.webp", "x.txt"])
def test_unsupported_image_extension_is_rejected(upload, name):
    assert "Use JPG, PNG or GIF" in _image_error(name, upload("chart.png"))


# --------------------------------------------------------------------- errors are human-readable
@pytest.mark.parametrize("name,fixture", [("scan.pdf", "scanned_no_text_layer.pdf"), ("locked.pdf", "password_protected.pdf"),
                                          ("truncated.pdf", "truncated.pdf"), ("truncated.docx", "truncated.docx"),
                                          ("old.doc", "legacy_word.doc"), ("photo.txt", "photo.jpg")])
def test_error_messages_never_expose_exception_names(upload, name, fixture):
    with pytest.raises(IngestionError) as info:
        extract_text(name, upload(fixture))
    msg = str(info.value)
    assert "Error" not in msg and "Exception" not in msg
    if info.value.__cause__ is not None:
        assert type(info.value.__cause__).__name__ not in msg


# ------------------------------------------------------------------- segmentation and pipeline
@pytest.mark.parametrize("name", TEXT_FIXTURES)
def test_every_text_fixture_segments_with_offsets_into_the_extracted_text(upload, name):
    doc = extract_text(name, upload(name))
    sentences = split_sentences(doc.text)
    assert sentences
    for s in sentences:
        assert doc.text[s.start:s.end] == s.text
    assert [s.start for s in sentences] == sorted(s.start for s in sentences)


def test_paragraph_breaks_with_crlf_or_spaces_split_sentences():
    for text in ("Heading\r\n\r\nnext line here", "Heading\n \nnext line here"):
        assert [s.text for s in split_sentences(text)] == ["Heading", "next line here"]


@pytest.mark.parametrize("name", ["job_description_2_pages.pdf", "header_paragraphs_table.docx", "multi_column.csv",
                                  "windows1252_crlf.txt"])
def test_uploaded_text_runs_through_analyze_text_with_offsets_intact(upload, name):
    from fairlens.pipeline import analyze_text

    table = {"salesman": 0.9, "young": 0.8, "aggressive": 0.7, "woman": 0.6}
    doc = extract_text(name, upload(name))
    a = analyze_text(doc.text, {"a": FakeDetector("a", table), "b": FakeDetector("b", table)}, explain=False)
    assert a.text == doc.text and a.ensemble_score >= 0.5
    assert len(a.sentences) == len(split_sentences(doc.text))
    for s in a.sentences:
        assert doc.text[s["start"]:s["end"]] == s["text"]
    for span in a.spans:
        assert doc.text[span["start"]:span["end"]] == span["text"]


@pytest.mark.parametrize("name", ["photo.jpg", "transparent_alpha.png", "animated.gif"])
def test_uploaded_image_reaches_the_captioners_as_rgb(upload, name):
    from fairlens.multimodal.captioners import Caption
    from fairlens.pipeline import analyze_image

    seen = []

    class Captioner:
        name, display = "fake", "Fake"

        def caption(self, image):
            seen.append(image.mode)
            return Caption("fake", "fake", "a young woman at a desk", 1.0)

    class Aligner:
        def score(self, image, captions):
            return [0.3 for _ in captions]

    r = analyze_image(load_image(name, upload(name)), {"fake": Captioner()},
                      {"a": FakeDetector("a", {"young": 0.8})}, Aligner())
    assert seen == ["RGB"] and r.captions[0]["detector_scores"]["a"] == 0.8


def test_all_fixtures_used_here_exist():
    for name in TEXT_FIXTURES:
        assert (INGESTION_FIXTURES / name).is_file()
