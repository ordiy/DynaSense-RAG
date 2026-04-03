"""Pydantic request/response DTOs for the HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.core.config import get_settings

_s = get_settings()
_MAX_Q = _s.max_query_len
_MAX_SUB = _s.max_expected_substring_len


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=_MAX_Q)


class ChatSessionRequest(BaseModel):
    conversation_id: str | None = None
    message: str = Field(min_length=1, max_length=_MAX_Q)
    memory_mode: Literal["prioritized", "legacy"] = "prioritized"


class ChatSessionABRequest(BaseModel):
    conversation_id: str | None = None
    message: str = Field(min_length=1, max_length=_MAX_Q)


class EvalRequest(BaseModel):
    query: str = Field(min_length=1, max_length=_MAX_Q)
    expected_substring: str = Field(min_length=1, max_length=_MAX_SUB)
    use_hybrid: bool = False


class EvalBatchCase(BaseModel):
    id: str | None = None
    query: str = Field(min_length=1, max_length=_MAX_Q)
    expected_substring: str = Field(min_length=1, max_length=_MAX_SUB)


class EvalBatchRequest(BaseModel):
    cases: list[EvalBatchCase] = Field(min_length=1, max_length=500)
    use_hybrid: bool = False


class Neo4jSearchRequest(BaseModel):
    keywords: list[str] = Field(min_length=1, max_length=20)
    limit: int = Field(default=40, ge=1, le=100)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class FeedbackRequest(BaseModel):
    """Optional feedback after a turn; used for quality iteration (MVP, not persisted to DB)."""

    conversation_id: str | None = None
    query: str = Field(min_length=1, max_length=_MAX_Q)
    rating: Literal[-1, 0, 1]
    comment: str | None = Field(default=None, max_length=2000)
    tags: list[str] = Field(default_factory=list, max_length=10)
