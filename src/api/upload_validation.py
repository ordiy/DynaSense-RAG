"""Pure helpers for upload filename / content-type checks (no heavy imports)."""

from __future__ import annotations


_TEXT_SUFFIXES = (".txt", ".md", ".markdown")

_DOCX_CONTENT_TYPE = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)
_XLSX_CONTENT_TYPE = (
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)


def is_pdf_upload(filename: str | None, content_type: str | None) -> bool:
    fn = (filename or "").lower()
    if fn.endswith(".pdf"):
        return True
    ct = (content_type or "").lower().split(";")[0].strip()
    return ct in ("application/pdf", "application/x-pdf")


def is_docx_upload(filename: str | None, content_type: str | None) -> bool:
    fn = (filename or "").lower()
    if fn.endswith(".docx"):
        return True
    ct = (content_type or "").lower().split(";")[0].strip()
    return ct == _DOCX_CONTENT_TYPE


def is_xlsx_upload(filename: str | None, content_type: str | None) -> bool:
    fn = (filename or "").lower()
    if fn.endswith(".xlsx"):
        return True
    ct = (content_type or "").lower().split(";")[0].strip()
    return ct == _XLSX_CONTENT_TYPE


def is_allowed_text_upload(filename: str | None) -> bool:
    """Non-PDF text path only accepts known text extensions to avoid decoding binaries as UTF-8."""
    fn = (filename or "").lower()
    return any(fn.endswith(s) for s in _TEXT_SUFFIXES)
