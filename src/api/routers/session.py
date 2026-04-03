"""Multi-turn session chat and A/B memory comparison."""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import APIRouter, HTTPException

from src.api import state
from src.api.schemas import ChatMessage, ChatSessionABRequest, ChatSessionRequest
from src.api.guardrails import guard_query_or_raise
from src.api.session_memory import build_query_with_history, trim_session_history
from src.api.state import cleanup_chat_sessions
from src.core.config import get_settings
from src.core.exceptions import QueryGuardrailError
from src.rag_core import run_chat_pipeline

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
            "history": [ChatMessage(**m).model_dump() for m in session["messages"]],
        }
    except QueryGuardrailError:
        raise
    except Exception:
        logger.exception("Session chat pipeline failed.")
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
                "blocked": blocked,
            }
        except Exception:
            logger.exception("A/B chat pipeline failed for mode=%s", mode)
            results[mode] = {
                "request_query": mode_queries[mode],
                "answer": "Internal server error.",
                "logs": [],
                "context_used": [],
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
