"""Single-turn chat."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.api.guardrails import guard_query_or_raise
from src.api.schemas import ChatRequest
from src.core.exceptions import QueryGuardrailError
from src.rag_core import iter_chat_stream_events, run_chat_pipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat")
async def chat(request: ChatRequest):
    try:
        guard_query_or_raise(request.query)
        return run_chat_pipeline(request.query)
    except QueryGuardrailError:
        raise
    except Exception:
        logger.exception("Chat pipeline failed.")
        raise HTTPException(status_code=500, detail="Internal server error.")


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Server-Sent Events (SSE): ``data:`` JSON lines with ``meta`` (citations, route), ``token``, ``done``.

    Uses Vertex ``llm.stream`` for generation; retrieval path matches ``run_chat_pipeline``.
    """
    try:
        guard_query_or_raise(request.query)
    except QueryGuardrailError:
        raise

    def event_generator():
        try:
            for ev in iter_chat_stream_events(request.query):
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception:
            logger.exception("Chat stream failed.")
            err = {"type": "error", "detail": "stream_failed"}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
