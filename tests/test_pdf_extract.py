"""Tests for PDF text extraction (no Vertex / LanceDB)."""
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from pypdf import PdfWriter

from src.pdf_extract import PdfExtractError, extract_text_from_pdf_bytes


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
