"""Pure helpers for upload filename / content-type checks (no heavy imports)."""

from __future__ import annotations


_TEXT_SUFFIXES = (".txt", ".md", ".markdown")


def is_pdf_upload(filename: str | None, content_type: str | None) -> bool:
    fn = (filename or "").lower()
    if fn.endswith(".pdf"):
        return True
    ct = (content_type or "").lower().split(";")[0].strip()
    return ct in ("application/pdf", "application/x-pdf")


def is_allowed_text_upload(filename: str | None) -> bool:
    """Non-PDF text path only accepts known text extensions to avoid decoding binaries as UTF-8."""
    fn = (filename or "").lower()
    return any(fn.endswith(s) for s in _TEXT_SUFFIXES)
