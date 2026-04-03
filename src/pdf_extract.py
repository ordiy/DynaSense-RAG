"""
Extract plain text from PDF bytes for ingestion.

Design: keep PDF parsing isolated from FastAPI and rag_core so the pipeline always
receives UTF-8 text — the same path as TXT/MD for chunking, embeddings, and Neo4j.
"""
from __future__ import annotations

from io import BytesIO

from pypdf import PdfReader


class PdfExtractError(ValueError):
    """Raised when the file is not a valid PDF or cannot be read."""


def extract_text_from_pdf_bytes(data: bytes) -> str:
    """
    Extract text from all pages, join with blank lines.

    Note: image-only / scanned PDFs without OCR typically yield empty text;
    callers should reject empty results and suggest OCR or TXT upload.
    """
    if not data:
        raise PdfExtractError("Empty file.")

    try:
        reader = PdfReader(BytesIO(data), strict=False)
    except Exception as e:
        raise PdfExtractError(f"Invalid or corrupted PDF: {e}") from e

    parts: list[str] = []
    for page in reader.pages:
        try:
            raw = page.extract_text()
        except Exception:
            raw = None
        t = (raw or "").strip()
        if t:
            parts.append(t)

    return "\n\n".join(parts).strip()
