"""Human feedback collection (MVP, in-process ring buffer)."""

from __future__ import annotations

import time
import uuid

from fastapi import APIRouter

from src.api import state
from src.api.schemas import FeedbackRequest
from src.infrastructure.persistence.postgres_connection import get_pool
from src.infrastructure.persistence.postgres_feedback import FeedbackStore

router = APIRouter(prefix="/api", tags=["feedback"])


@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest) -> dict:
    """
    Record thumbs up/down/neutral plus optional tags for offline review.

    Design: append-only in-memory list capped at ``MAX_FEEDBACK_ENTRIES``; last entries win.
    """
    entry = {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "conversation_id": request.conversation_id,
        "query": request.query,
        "rating": request.rating,
        "comment": request.comment,
        "tags": list(request.tags)[:10],
        "trace_id": request.trace_id,
    }
    state.feedback_log.append(entry)
    # Ring buffer: keep tail for bounded memory in long-running demos.
    if len(state.feedback_log) > state.MAX_FEEDBACK_ENTRIES:
        state.feedback_log = state.feedback_log[-state.MAX_FEEDBACK_ENTRIES :]
        
    try:
        pool = get_pool()
        FeedbackStore(pool).insert(entry)
    except RuntimeError:
        pass
        
    return {"ok": True, "feedback_id": entry["id"]}


@router.get("/feedback/summary")
async def feedback_summary() -> dict:
    """Lightweight stats for the engineer console (counts by rating)."""
    counts = {-1: 0, 0: 0, 1: 0}
    for row in state.feedback_log:
        r = row.get("rating")
        if r in counts:
            counts[r] += 1
    return {"n": len(state.feedback_log), "by_rating": counts}

@router.get("/feedback/negative")
async def get_negative_feedback() -> list[dict]:
    try:
        pool = get_pool()
        return FeedbackStore(pool).get_negative(limit=50)
    except RuntimeError:
        results = []
        for row in reversed(state.feedback_log):
            if row.get("rating") == -1:
                results.append(row)
                if len(results) >= 50:
                    break
        return results
