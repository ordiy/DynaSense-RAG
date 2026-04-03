"""
Amortizing loan calculators — pure Python tools for What-If scenarios.

Typical use: compare monthly payment and total interest when the annual rate
changes, holding principal and term fixed (enterprise sensitivity / FP&A style).

Formulas: standard fixed-rate amortization (equal monthly payment).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LoanSnapshot:
    """One scenario: payment, totals, and inputs echoed for audit."""

    principal: float
    annual_rate_percent: float
    loan_months: int
    monthly_payment: float
    total_paid: float
    total_interest: float


def monthly_payment(principal: float, annual_rate_percent: float, loan_months: int) -> float:
    """
    Fixed-rate amortizing loan: equal payment each month.

    ``annual_rate_percent`` is nominal APR in percent (e.g. 3.5 for 3.5%).
    """
    if principal <= 0:
        raise ValueError("principal must be positive")
    if loan_months < 1:
        raise ValueError("loan_months must be >= 1")
    r = annual_rate_percent / 100.0 / 12.0
    if abs(r) < 1e-16:
        return principal / loan_months
    pow_term = (1.0 + r) ** loan_months
    return principal * (r * pow_term) / (pow_term - 1.0)


def loan_snapshot(principal: float, annual_rate_percent: float, loan_months: int) -> LoanSnapshot:
    """Compute monthly payment, total paid, and total interest for one scenario."""
    m = monthly_payment(principal, annual_rate_percent, loan_months)
    total_paid = m * loan_months
    total_interest = total_paid - principal
    return LoanSnapshot(
        principal=principal,
        annual_rate_percent=annual_rate_percent,
        loan_months=loan_months,
        monthly_payment=m,
        total_paid=total_paid,
        total_interest=total_interest,
    )


def compare_rate_scenarios(
    principal: float,
    annual_rate_percent_before: float,
    annual_rate_percent_after: float,
    loan_months: int,
) -> tuple[LoanSnapshot, LoanSnapshot, dict[str, float]]:
    """
    Run two tool calls in sequence (before / after); return snapshots + deltas.

    This is the core "What-If" comparison without any LLM or ReAct loop.
    """
    before = loan_snapshot(principal, annual_rate_percent_before, loan_months)
    after = loan_snapshot(principal, annual_rate_percent_after, loan_months)
    deltas = {
        "delta_monthly_payment": after.monthly_payment - before.monthly_payment,
        "delta_total_interest": after.total_interest - before.total_interest,
        "delta_total_paid": after.total_paid - before.total_paid,
    }
    return before, after, deltas
