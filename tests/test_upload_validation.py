"""Unit tests for upload validation helpers (no Vertex / rag_core import)."""

from __future__ import annotations

from src.api.upload_validation import is_allowed_text_upload, is_pdf_upload


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
