"""Document upload and task status."""

from __future__ import annotations

import os
import time
import uuid

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from src.api import state
from src.api.state import cleanup_tasks as _cleanup_tasks
from src.api.upload_validation import is_allowed_text_upload, is_pdf_upload
from src.core.config import get_settings
from src.pdf_extract import PdfExtractError, extract_text_from_pdf_bytes
from src.rag_core import process_document_task

router = APIRouter(prefix="/api", tags=["ingest"])


@router.post("/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    _cleanup_tasks()
    max_b = get_settings().max_upload_bytes
    content = await file.read(max_b + 1)
    if len(content) > max_b:
        raise HTTPException(status_code=413, detail="Uploaded file is too large.")

    safe_filename = os.path.basename(file.filename) if file.filename else "upload.txt"

    if is_pdf_upload(file.filename, file.content_type):
        try:
            text_content = extract_text_from_pdf_bytes(content)
        except PdfExtractError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if not text_content.strip():
            raise HTTPException(
                status_code=400,
                detail=(
                    "No extractable text in PDF (common for scanned-only documents). "
                    "Use OCR first or upload TXT/MD."
                ),
            )
        ingest_format = "pdf"
    else:
        if not is_allowed_text_upload(file.filename):
            raise HTTPException(
                status_code=400,
                detail=(
                    "For non-PDF uploads, use a .txt, .md, or .markdown filename, "
                    "or upload a .pdf file."
                ),
            )
        text_content = content.decode("utf-8", errors="replace")
        ingest_format = "text"

    task_id = str(uuid.uuid4())
    state.tasks[task_id] = {
        "status": "pending",
        "progress": 0,
        "created_at": time.time(),
        "ingest_format": ingest_format,
        "filename": safe_filename,
    }

    background_tasks.add_task(process_document_task, text_content, safe_filename, state.tasks[task_id])

    return {
        "task_id": task_id,
        "message": "Document uploaded and processing started.",
        "ingest_format": ingest_format,
    }


@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    _cleanup_tasks()
    if task_id not in state.tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return state.tasks[task_id]
