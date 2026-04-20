"""Tests for PDF text extraction (no Vertex / LanceDB)."""
import logging
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from pypdf import PdfWriter

from src.pdf_extract import (
    OcrNotAvailableError,
    PdfExtractError,
    extract_pdf_content,
    extract_tables_as_markdown,
    extract_text_from_pdf_bytes,
)


def _blank_pdf_bytes() -> bytes:
    w = PdfWriter()
    w.add_blank_page(width=612, height=792)
    buf = BytesIO()
    w.write(buf)
    return buf.getvalue()


def test_extract_blank_pdf_returns_empty():
    data = _blank_pdf_bytes()
    assert extract_text_from_pdf_bytes(data) == ""


def test_extract_invalid_bytes_raises():
    with pytest.raises(PdfExtractError, match="Invalid|corrupted|Empty"):
        extract_text_from_pdf_bytes(b"not a pdf")


def test_extract_joins_pages_via_mock():
    mock_page_a = MagicMock()
    mock_page_a.extract_text.return_value = "Hello "
    mock_page_b = MagicMock()
    mock_page_b.extract_text.return_value = "World"
    mock_reader = MagicMock()
    mock_reader.pages = [mock_page_a, mock_page_b]

    with patch("src.pdf_extract.PdfReader", return_value=mock_reader):
        out = extract_text_from_pdf_bytes(b"%PDF-fake")
    assert "Hello" in out and "World" in out
    assert "\n\n" in out


def test_extract_pdf_content_ocr_fallback():
    mock_pdf2image = MagicMock()
    mock_pytesseract = MagicMock()

    mock_pdf2image.convert_from_bytes.return_value = ["img1"]
    mock_pytesseract.image_to_string.return_value = "OCR Success Text"

    with patch("src.pdf_extract.extract_text_from_pdf_bytes", return_value=""), \
         patch("src.pdf_extract.extract_tables_as_markdown", return_value=[]), \
         patch.dict("sys.modules", {"pdf2image": mock_pdf2image, "pytesseract": mock_pytesseract}):

        out = extract_pdf_content(b"fake-pdf")

    assert "OCR Success Text" in out
    mock_pdf2image.convert_from_bytes.assert_called_once()
    mock_pytesseract.image_to_string.assert_called_once_with("img1")


def test_ocr_not_available_error_path(caplog):
    with caplog.at_level(logging.WARNING):
        with patch("src.pdf_extract.extract_text_from_pdf_bytes", return_value=""), \
             patch("src.pdf_extract.extract_tables_as_markdown", return_value=[]), \
             patch.dict("sys.modules", {"pdf2image": MagicMock(), "pytesseract": None}):

            out = extract_pdf_content(b"fake-pdf")

        assert out == ""
        assert "PDF contains no text and OCR is not available" in caplog.text


def test_extract_tables_as_markdown():
    mock_pdfplumber = MagicMock()
    mock_pdf = MagicMock()
    mock_page = MagicMock()

    mock_page.extract_tables.return_value = [
        [["Header 1", "Header 2"], ["Row 1 Col 1", "Row 1 Col 2"]]
    ]
    mock_pdf.pages = [mock_page]
    mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf

    with patch.dict("sys.modules", {"pdfplumber": mock_pdfplumber}):
        tables = extract_tables_as_markdown(b"fake")

    assert len(tables) == 1
    assert "| Header 1 | Header 2 |" in tables[0]
    assert "| --- | --- |" in tables[0]
    assert "| Row 1 Col 1 | Row 1 Col 2 |" in tables[0]


def test_extract_pdf_content_appends_tables():
    with patch("src.pdf_extract.extract_text_from_pdf_bytes", return_value="Normal Main Text"), \
         patch("src.pdf_extract.extract_tables_as_markdown", return_value=["| Table Header |", "| --- |"]):

        out = extract_pdf_content(b"fake-pdf")

    assert "Normal Main Text" in out
    assert "| Table Header |" in out
    assert out == "Normal Main Text\n\n| Table Header |\n\n| --- |"
