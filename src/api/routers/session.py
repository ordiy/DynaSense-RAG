"""Multi-turn session chat and A/B memory comparison."""

from __future__ import annotations

import base64
import logging
import os
import time
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.api import state
from src.api.schemas import ChatMessage, ChatSessionABRequest, ChatSessionRequest
from src.api.guardrails import guard_query_or_raise
from src.api.session_memory import build_query_with_history, trim_session_history
from src.api.state import cleanup_chat_sessions
from src.api.upload_validation import (
    is_allowed_text_upload,
    is_docx_upload,
    is_pdf_upload,
    is_xlsx_upload,
)
from src.core.config import get_settings
from src.core.exceptions import QueryGuardrailError
from src.rag_core import run_chat_pipeline, run_chat_pipeline_multimodal

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["session"])


@router.post("/chat/session")
async def chat_session(request: ChatSessionRequest):
    cleanup_chat_sessions()
    s = get_settings()
    conversation_id = request.conversation_id or str(uuid.uuid4())
    now = time.time()

    session = state.chat_sessions.get(conversation_id)
    if session is None:
        session = {"created_at": now, "updated_at": now, "messages": []}
        state.chat_sessions[conversation_id] = session

    messages = session["messages"]
    messages.append({"role": "user", "content": request.message})
    session["updated_at"] = now

    request_query = build_query_with_history(
        messages,
        budget=s.chat_memory_query_budget,
        mode=request.memory_mode,
    )
    request_query = request_query[: s.max_query_len]

    try:
        guard_query_or_raise(request.message)
        result = run_chat_pipeline(request_query)
        answer = result.get("answer", "")
        messages.append({"role": "assistant", "content": answer})
        session["messages"] = trim_session_history(messages)
        session["updated_at"] = time.time()
        return {
            "conversation_id": conversation_id,
            "memory_mode": request.memory_mode,
            "request_query": request_query,
            "answer": answer,
            "logs": result.get("logs", []),
            "context_used": result.get("context_used", []),
            "citations": result.get("citations", []),
            "route": result.get("route"),
            "effective_route": result.get("effective_route"),
            "router_reason": result.get("router_reason"),
            "history": [ChatMessage(**m).model_dump() for m in session["messages"]],
        }
    except QueryGuardrailError:
        raise
    except Exception:
        logger.exception("Session chat pipeline failed.")
        raise HTTPException(status_code=500, detail="Internal server error.")


_IMAGE_MIME_PREFIXES = ("image/jpeg", "image/png", "image/webp", "image/gif", "image/heic", "image/heif")
_MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB per image


@router.post("/chat/session/multimodal")
async def chat_session_multimodal(
    message: str = Form(...),
    conversation_id: str = Form(None),
    memory_mode: str = Form("prioritized"),
    files: list[UploadFile] = File(default=[]),
):
    """
    Multi-turn session chat with optional file/image attachments.

    Accepts ``multipart/form-data``:
    - ``message`` (required): user text
    - ``conversation_id`` (optional): resume an existing session
    - ``memory_mode`` (optional): "prioritized" (default) or "legacy"
    - ``files`` (optional, repeatable): images (JPEG/PNG/WEBP/GIF) and/or
      documents (PDF / DOCX / XLSX / TXT / MD)

    Images are passed directly to Gemini vision; text documents are extracted
    and prepended to the query as additional context.
    """
    cleanup_chat_sessions()
    s = get_settings()
    conversation_id = conversation_id or str(uuid.uuid4())
    now = time.time()

    session = state.chat_sessions.get(conversation_id)
    if session is None:
        session = {"created_at": now, "updated_at": now, "messages": []}
        state.chat_sessions[conversation_id] = session

    # ── Process attachments ──────────────────────────────────────────────────
    image_parts: list[tuple[str, str]] = []   # (mime_type, base64_data)
    doc_texts: list[str] = []

    max_b = s.max_upload_bytes
    for upload in files:
        raw = await upload.read(max(max_b, _MAX_IMAGE_BYTES) + 1)
        mime = (upload.content_type or "").lower().split(";")[0].strip()
        fname = os.path.basename(upload.filename or "upload")

        if any(mime.startswith(p) for p in _IMAGE_MIME_PREFIXES) or fname.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp", ".gif", ".heic", ".heif")
        ):
            if len(raw) > _MAX_IMAGE_BYTES:
                raise HTTPException(status_code=413, detail=f"Image '{fname}' exceeds 10 MB limit.")
            image_parts.append((mime or "image/jpeg", base64.b64encode(raw).decode()))

        elif is_pdf_upload(fname, mime):
            if len(raw) > max_b:
                raise HTTPException(status_code=413, detail=f"File '{fname}' exceeds size limit.")
            from src.pdf_extract import PdfExtractError, extract_text_from_pdf_bytes
            try:
                doc_texts.append(f"[Attached PDF: {fname}]\n{extract_text_from_pdf_bytes(raw)}")
            except PdfExtractError as e:
                raise HTTPException(status_code=400, detail=str(e))

        elif is_docx_upload(fname, mime):
            if len(raw) > max_b:
                raise HTTPException(status_code=413, detail=f"File '{fname}' exceeds size limit.")
            from src.docx_extract import DocxExtractError, extract_text_from_docx_bytes
            try:
                doc_texts.append(f"[Attached DOCX: {fname}]\n{extract_text_from_docx_bytes(raw)}")
            except DocxExtractError as e:
                raise HTTPException(status_code=400, detail=str(e))

        elif is_xlsx_upload(fname, mime):
            if len(raw) > max_b:
                raise HTTPException(status_code=413, detail=f"File '{fname}' exceeds size limit.")
            from src.xlsx_extract import XlsxExtractError, extract_text_from_xlsx_bytes
            try:
                doc_texts.append(f"[Attached XLSX: {fname}]\n{extract_text_from_xlsx_bytes(raw)}")
            except XlsxExtractError as e:
                raise HTTPException(status_code=400, detail=str(e))

        elif is_allowed_text_upload(fname):
            if len(raw) > max_b:
                raise HTTPException(status_code=413, detail=f"File '{fname}' exceeds size limit.")
            doc_texts.append(f"[Attached file: {fname}]\n{raw.decode('utf-8', errors='replace')}")

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: '{fname}'. Allowed: images, PDF, DOCX, XLSX, TXT, MD.",
            )

    # ── Build augmented message for session history ──────────────────────────
    session_messages = session["messages"]
    session_messages.append({"role": "user", "content": message})
    session["updated_at"] = now

    # Prepend extracted document text to the query for RAG context
    augmented_message = message
    if doc_texts:
        augmented_message = "\n\n".join(doc_texts) + f"\n\n[User question]: {message}"

    request_query = build_query_with_history(
        session_messages,
        budget=s.chat_memory_query_budget,
        mode=memory_mode,
    )[: s.max_query_len]

    # Swap the last user turn's content for the augmented version when building query
    if doc_texts:
        probe = session_messages[:-1] + [{"role": "user", "content": augmented_message}]
        request_query = build_query_with_history(
            probe, budget=s.chat_memory_query_budget, mode=memory_mode
        )[: s.max_query_len]

    try:
        guard_query_or_raise(message)

        if image_parts:
            result = run_chat_pipeline_multimodal(request_query, image_parts)
        else:
            result = run_chat_pipeline(request_query)

        answer = result.get("answer", "")
        session_messages.append({"role": "assistant", "content": answer})
        session["messages"] = trim_session_history(session_messages)
        session["updated_at"] = time.time()

        return {
            "conversation_id": conversation_id,
            "memory_mode": memory_mode,
            "answer": answer,
            "logs": result.get("logs", []),
            "context_used": result.get("context_used", []),
            "citations": result.get("citations", []),
            "route": result.get("route"),
            "has_images": bool(image_parts),
            "has_documents": bool(doc_texts),
            "history": [ChatMessage(**m).model_dump() for m in session["messages"]],
        }
    except QueryGuardrailError:
        raise
    except HTTPException:
        raise
    except Exception:
        logger.exception("Multimodal session chat pipeline failed.")
        raise HTTPException(status_code=500, detail="Internal server error.")


@router.post("/chat/session/ab")
async def chat_session_ab(request: ChatSessionABRequest):
    cleanup_chat_sessions()
    s = get_settings()
    conversation_id = request.conversation_id or str(uuid.uuid4())
    sess = state.chat_sessions.get(conversation_id)
    base_messages = list(sess["messages"]) if sess else []
    probe_messages = base_messages + [{"role": "user", "content": request.message}]

    mode_queries = {
        "prioritized": build_query_with_history(
            probe_messages, budget=s.chat_memory_query_budget, mode="prioritized"
        )[: s.max_query_len],
        "legacy": build_query_with_history(
            probe_messages, budget=s.chat_memory_query_budget, mode="legacy"
        )[: s.max_query_len],
    }

    results: dict[str, dict] = {}
    try:
        guard_query_or_raise(request.message)
    except QueryGuardrailError:
        raise

    for mode in ("prioritized", "legacy"):
        try:
            out = run_chat_pipeline(mode_queries[mode])
            answer = out.get("answer", "")
            blocked = "未能找到与您问题高度相关的信息" in answer
            results[mode] = {
                "request_query": mode_queries[mode],
                "answer": answer,
                "logs": out.get("logs", []),
                "context_used": out.get("context_used", []),
                "citations": out.get("citations", []),
                "blocked": blocked,
            }
        except Exception:
            logger.exception("A/B chat pipeline failed for mode=%s", mode)
            results[mode] = {
                "request_query": mode_queries[mode],
                "answer": "Internal server error.",
                "logs": [],
                "context_used": [],
                "citations": [],
                "blocked": True,
            }

    recommended_mode = "prioritized"
    if results["prioritized"]["blocked"] and not results["legacy"]["blocked"]:
        recommended_mode = "legacy"
    elif results["legacy"]["blocked"] and not results["prioritized"]["blocked"]:
        recommended_mode = "prioritized"

    return {
        "conversation_id": conversation_id,
        "message": request.message,
        "recommended_mode": recommended_mode,
        "results": results,
    }


@router.get("/chat/session/{conversation_id}")
async def get_chat_session(conversation_id: str):
    cleanup_chat_sessions()
    session = state.chat_sessions.get(conversation_id)
    if not session:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {
        "conversation_id": conversation_id,
        "history": [ChatMessage(**m).model_dump() for m in session["messages"]],
        "updated_at": session.get("updated_at"),
    }


@router.delete("/chat/session/{conversation_id}")
async def delete_chat_session(conversation_id: str):
    cleanup_chat_sessions()
    if conversation_id in state.chat_sessions:
        state.chat_sessions.pop(conversation_id, None)
        return {"conversation_id": conversation_id, "deleted": True}
    raise HTTPException(status_code=404, detail="Conversation not found")
