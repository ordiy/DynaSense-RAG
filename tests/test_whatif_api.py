"""HTTP tests for What-If router only (no rag_core import)."""

from __future__ import annotations

import math

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routers import whatif


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(whatif.router)
    return TestClient(app)


def test_loan_compare_ok():
    c = _client()
    r = c.post(
        "/api/whatif/loan/compare",
        json={
            "principal": 1_000_000,
            "annual_rate_percent_before": 3.5,
            "annual_rate_percent_after": 4.2,
            "loan_months": 360,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["scenario"] == "loan_rate_compare"
    assert "before" in data and "after" in data
    assert math.isclose(
        data["deltas"]["delta_monthly_payment"],
        data["after"]["monthly_payment"] - data["before"]["monthly_payment"],
        rel_tol=1e-6,
    )


def test_loan_compare_validation():
    c = _client()
    r = c.post(
        "/api/whatif/loan/compare",
        json={
            "principal": -1,
            "annual_rate_percent_before": 3,
            "annual_rate_percent_after": 4,
            "loan_months": 360,
        },
    )
    assert r.status_code == 422
