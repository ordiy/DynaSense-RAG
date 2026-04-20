"""
Extract plain text from PDF bytes for ingestion.

Design: keep PDF parsing isolated from FastAPI and rag_core so the pipeline always
receives UTF-8 text — the same path as TXT/MD for chunking, embeddings, and graph ingest.
"""
from __future__ import annotations

import logging
from io import BytesIO

from pypdf import PdfReader

logger = logging.getLogger(__name__)


class PdfExtractError(ValueError):
    """Raised when the file is not a valid PDF or cannot be read."""


class OcrNotAvailableError(Exception):
    """Raised when OCR is required but dependencies are missing."""


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


def _extract_text_via_ocr(data: bytes) -> str:
    """
    Extract text from a PDF using OCR (pdf2image and pytesseract).
    """
    try:
        from pdf2image import convert_from_bytes
        import pytesseract
    except ImportError as e:
        raise OcrNotAvailableError("OCR dependencies (pdf2image, pytesseract) are missing.") from e

    try:
        images = convert_from_bytes(data)
        text_parts = []
        for image in images:
            text = pytesseract.image_to_string(image)
            text_parts.append(text.strip())
        return "\n\n".join(filter(None, text_parts)).strip()
    except Exception as e:
        logger.warning(f"OCR failed during image conversion or text extraction: {e}")
        return ""


def extract_tables_as_markdown(data: bytes) -> list[str]:
    """
    Extract tables from a PDF and format them as Markdown strings.
    Requires pdfplumber.
    """
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber not installed. Table extraction skipped.")
        return []

    tables_md = []
    try:
        with pdfplumber.open(BytesIO(data)) as pdf:
            for page in pdf.pages:
                extracted_tables = page.extract_tables()
                for table in extracted_tables:
                    if not table:
                        continue

                    md_lines = []
                    for i, row in enumerate(table):
                        # Clean cell text (handle None, replace newlines)
                        cleaned_row = [str(cell).replace('\n', ' ').strip() if cell is not None else "" for cell in row]
                        row_str = "| " + " | ".join(cleaned_row) + " |"
                        md_lines.append(row_str)

                        # Add header separator after the first row
                        if i == 0:
                            sep_row = "| " + " | ".join(["---"] * len(cleaned_row)) + " |"
                            md_lines.append(sep_row)

                    if md_lines:
                        tables_md.append("\n".join(md_lines))
    except Exception as e:
        logger.warning(f"Failed to extract tables: {e}")

    return tables_md


def extract_image_captions_from_pdf(data: bytes, llm=None) -> list[str]:
    """
    Extract images from a PDF, filter by size, and generate captions using a Vision LLM.
    """
    from src.core.config import get_settings

    if not get_settings().image_caption_enabled:
        return []

    import base64
    from pypdf import PdfReader
    from langchain_core.messages import HumanMessage
    from src.core.inference import get_llm

    if llm is None:
        llm = get_llm(get_settings())

    captions = []
    try:
        reader = PdfReader(BytesIO(data), strict=False)
        image_count = 0
        for page in reader.pages:
            if image_count >= 10:
                break
            for image_file_object in page.images:
                if image_count >= 10:
                    break
                image_bytes = image_file_object.data
                if len(image_bytes) < 10 * 1024:
                    continue  # Skip images smaller than 10KB

                try:
                    b64 = base64.b64encode(image_bytes).decode("utf-8")
                    msg = HumanMessage(content=[
                        {'type': 'text', 'text': 'Describe this image in 1-2 sentences focusing on data, labels, and key information relevant for document search. Output only the description.'},
                        {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{b64}'}}
                    ])
                    response = llm.invoke([msg])
                    captions.append(f"[图片描述] {response.content.strip()}")
                    image_count += 1
                except Exception as e:
                    logger.warning(f"Failed to caption image: {e}")
    except Exception as e:
        logger.warning(f"Failed to read PDF for image extraction: {e}")

    return captions


def extract_pdf_content(data: bytes) -> str:
    """
    Extract text from PDF bytes, falling back to OCR if empty, and append any tables.

    This is the primary entry point for extracting content from a PDF.
    """
    # 1. Standard text extraction
    text = extract_text_from_pdf_bytes(data)

    # 2. OCR fallback if no text found
    if not text.strip():
        try:
            text = _extract_text_via_ocr(data)
        except OcrNotAvailableError as e:
            logger.warning(f"PDF contains no text and OCR is not available: {e}")

    # 3. Extract tables
    tables = extract_tables_as_markdown(data)

    # 4. Extract image captions
    image_captions = extract_image_captions_from_pdf(data)

    # 5. Combine text, tables, and image captions
    parts = []
    if text:
        parts.append(text)
    if tables:
        parts.append("\n\n".join(tables))
    if image_captions:
        parts.append("\n".join(image_captions))

    return "\n\n".join(parts).strip()
