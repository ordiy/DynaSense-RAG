"""
What-If orchestration (DAG-style, no ReAct).

Design: one entrypoint per scenario family. Each pipeline step is a pure function
or tool call; order is fixed (no LLM-chosen loops). Suitable for compliance and
testing.
"""

from __future__ import annotations

from typing import Any

from src.tools.loan_whatif import compare_rate_scenarios, loan_snapshot


def run_loan_rate_compare_pipeline(
    principal: float,
    annual_rate_percent_before: float,
    annual_rate_percent_after: float,
    loan_months: int,
) -> dict[str, Any]:
    """
    Fixed DAG: validate inputs implicitly via tools → compare before/after → structured dict.

    Extend with more steps (e.g. tax, fees) by chaining here without changing HTTP contract.
    """
    before, after, deltas = compare_rate_scenarios(
        principal,
        annual_rate_percent_before,
        annual_rate_percent_after,
        loan_months,
    )
    return {
        "scenario": "loan_rate_compare",
        "before": before.__dict__,
        "after": after.__dict__,
        "deltas": deltas,
    }


def run_single_loan_snapshot(
    principal: float,
    annual_rate_percent: float,
    loan_months: int,
) -> dict[str, Any]:
    """Single-tool path: one snapshot for dashboards or sanity checks."""
    snap = loan_snapshot(principal, annual_rate_percent, loan_months)
    return {"scenario": "loan_snapshot", "snapshot": snap.__dict__}
