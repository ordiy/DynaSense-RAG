"""Controlled tabular profiling (no arbitrary code)."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.core.analytics_profile import is_allowed_tabular_filename, profile_tabular_bytes
from src.core.config import get_settings

router = APIRouter(prefix="/api", tags=["analytics"])


@router.post("/analytics/profile")
async def post_analytics_profile(file: UploadFile = File(...)):
    """
    Upload a **CSV / TSV / TXT (comma) / XLSX** file and receive a **fixed-structure** profile:
    row/column counts, null rates, numeric min/max/mean/std, top value frequencies.

    This is **not** LLM-generated code execution — suitable for governance-sensitive workloads.
    """
    settings = get_settings()
    max_b = settings.max_analytics_upload_bytes
    content = await file.read(max_b + 1)
    if len(content) > max_b:
        raise HTTPException(status_code=413, detail="File too large for analytics profile.")

    name = file.filename or "upload.csv"
    if not is_allowed_tabular_filename(name):
        raise HTTPException(
            status_code=400,
            detail="Use a .csv, .tsv, .txt, or .xlsx filename.",
        )

    try:
        out = profile_tabular_bytes(
            content,
            name,
            max_rows=settings.max_analytics_rows,
            max_cols=settings.max_analytics_cols,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return out
