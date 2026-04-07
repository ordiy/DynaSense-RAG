"""Unit tests for upload validation helpers (no Vertex / rag_core import)."""

from __future__ import annotations

from src.api.upload_validation import (
    is_allowed_text_upload,
    is_docx_upload,
    is_pdf_upload,
    is_xlsx_upload,
)


def test_is_pdf_upload_by_extension():
    assert is_pdf_upload("doc.PDF", None) is True
    assert is_pdf_upload("x.pdf", "application/octet-stream") is True


def test_is_pdf_upload_by_content_type():
    assert is_pdf_upload("unknown", "application/pdf") is True
    assert is_pdf_upload("unknown", "application/pdf; charset=binary") is True


def test_is_allowed_text_upload():
    assert is_allowed_text_upload("notes.txt") is True
    assert is_allowed_text_upload("README.md") is True
    assert is_allowed_text_upload("x.MARKDOWN") is True
    assert is_allowed_text_upload("blob.bin") is False
    assert is_allowed_text_upload(None) is False
    assert is_allowed_text_upload("") is False


def test_is_docx_upload_by_extension():
    assert is_docx_upload("report.docx", None) is True
    assert is_docx_upload("REPORT.DOCX", None) is True
    assert is_docx_upload("report.doc", None) is False


def test_is_docx_upload_by_content_type():
    ct = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    assert is_docx_upload("unknown", ct) is True
    # MIME params (e.g. charset) are stripped before comparison, matching PDF behaviour
    assert is_docx_upload("unknown", ct + "; charset=binary") is True


def test_is_xlsx_upload_by_extension():
    assert is_xlsx_upload("data.xlsx", None) is True
    assert is_xlsx_upload("DATA.XLSX", None) is True
    assert is_xlsx_upload("data.xls", None) is False


def test_is_xlsx_upload_by_content_type():
    ct = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    assert is_xlsx_upload("unknown", ct) is True


def test_upload_types_are_mutually_exclusive():
    assert is_pdf_upload("file.pdf", None) is True
    assert is_docx_upload("file.pdf", None) is False
    assert is_xlsx_upload("file.pdf", None) is False

    assert is_docx_upload("file.docx", None) is True
    assert is_pdf_upload("file.docx", None) is False
    assert is_xlsx_upload("file.docx", None) is False

    assert is_xlsx_upload("file.xlsx", None) is True
    assert is_pdf_upload("file.xlsx", None) is False
    assert is_docx_upload("file.xlsx", None) is False
