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


class GraphSearchRequest(BaseModel):
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


class LoanCompareRequest(BaseModel):
    """What-If: same principal & term, two annual rates (amortizing loan)."""

    principal: float = Field(gt=0, le=1e12, description="Loan principal (same currency unit throughout).")
    annual_rate_percent_before: float = Field(ge=0, le=50, description="Nominal APR before, e.g. 3.5 for 3.5%.")
    annual_rate_percent_after: float = Field(ge=0, le=50, description="Nominal APR after.")
    loan_months: int = Field(ge=1, le=600, description="Amortization horizon in months.")


class LoanSnapshotOut(BaseModel):
    principal: float
    annual_rate_percent: float
    loan_months: int
    monthly_payment: float
    total_paid: float
    total_interest: float


class LoanCompareResponse(BaseModel):
    scenario: Literal["loan_rate_compare"]
    before: LoanSnapshotOut
    after: LoanSnapshotOut
    deltas: dict[str, float]


class ConstrainedGraphRequest(BaseModel):
    """Whitelisted graph templates only — no arbitrary Cypher (see graph_constrained_queries)."""

    template: Literal["edges_from_entity", "multi_keyword_edges", "graph_global_summary"]
    params: dict = Field(default_factory=dict)


class ConstrainedGraphSuggestRequest(BaseModel):
    question: str = Field(min_length=1, max_length=_MAX_Q)


class ConstrainedGraphResponse(BaseModel):
    template_id: str
    rows: list[dict]
    logs: list[str]


class ConstrainedGraphSuggestResponse(BaseModel):
    template_id: str | None
    params: dict
    executed: ConstrainedGraphResponse | None = None
