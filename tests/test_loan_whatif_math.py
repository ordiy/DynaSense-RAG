"""Unit tests for amortizing loan tools (no FastAPI / Vertex)."""

from __future__ import annotations

import math

import pytest

from src.tools.loan_whatif import compare_rate_scenarios, loan_snapshot, monthly_payment


def test_zero_rate_straight_line():
    assert math.isclose(monthly_payment(120_000, 0, 12), 10_000.0, rel_tol=1e-9)


def test_known_6pct_30yr_100k():
    # Reference computed with same formula (599.55)
    m = monthly_payment(100_000, 6, 360)
    assert math.isclose(m, 599.55, rel_tol=1e-4)


def test_compare_increases_payment_when_rate_rises():
    before, after, d = compare_rate_scenarios(
        principal=1_000_000,
        annual_rate_percent_before=3.5,
        annual_rate_percent_after=4.2,
        loan_months=360,
    )
    assert before.monthly_payment < after.monthly_payment
    assert d["delta_monthly_payment"] > 0
    assert d["delta_total_interest"] > 0
    assert before.total_interest < after.total_interest


def test_loan_snapshot_fields():
    s = loan_snapshot(500_000, 4.0, 240)
    assert s.principal == 500_000
    assert s.loan_months == 240
    assert s.total_paid > s.principal
    assert math.isclose(s.total_interest, s.total_paid - s.principal, rel_tol=1e-9)


def test_invalid_principal():
    with pytest.raises(ValueError):
        monthly_payment(0, 5, 12)
