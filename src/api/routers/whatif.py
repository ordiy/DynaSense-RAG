"""What-If tools API (deterministic math; no RAG / Vertex required)."""

from __future__ import annotations

from fastapi import APIRouter

from src.api.schemas import LoanCompareRequest, LoanCompareResponse, LoanSnapshotOut
from src.api.whatif_pipeline import run_loan_rate_compare_pipeline

router = APIRouter(prefix="/api/whatif", tags=["whatif"])


@router.post("/loan/compare", response_model=LoanCompareResponse)
async def loan_rate_compare(request: LoanCompareRequest) -> LoanCompareResponse:
    """
    Compare two interest-rate scenarios for the same amortizing loan.

    Implements a **fixed DAG** (tool calls only): snapshot @ rate A → snapshot @ rate B → deltas.
    No ReAct loop; suitable for tests and FP&A-style sensitivity.
    """
    raw = run_loan_rate_compare_pipeline(
        request.principal,
        request.annual_rate_percent_before,
        request.annual_rate_percent_after,
        request.loan_months,
    )
    return LoanCompareResponse(
        scenario="loan_rate_compare",
        before=LoanSnapshotOut(**raw["before"]),
        after=LoanSnapshotOut(**raw["after"]),
        deltas=raw["deltas"],
    )
