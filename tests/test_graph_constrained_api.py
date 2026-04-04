"""HTTP tests for constrained graph debug endpoints (no rag_core)."""

from __future__ import annotations

from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routers import debug_routes


def _client(monkeypatch) -> TestClient:
    monkeypatch.setattr(
        "src.api.routers.debug_routes.get_settings",
        lambda: type("S", (), {"debug_data_api": True})(),
    )
    app = FastAPI()
    app.include_router(debug_routes.router)
    return TestClient(app)


def test_constrained_run_422_on_bad_template(monkeypatch):
    c = _client(monkeypatch)
    r = c.post(
        "/api/debug/graph/constrained/run",
        json={"template": "edges_from_entity", "params": {"name_substring": ""}},
    )
    assert r.status_code == 422


@patch("src.api.routers.debug_routes.execute_constrained_template")
def test_constrained_run_ok(mock_ex, monkeypatch):
    from src.core.graph_constrained_queries import ConstrainedGraphResult

    mock_ex.return_value = ConstrainedGraphResult("edges_from_entity", [{"a": 1}], ["log"])
    c = _client(monkeypatch)
    r = c.post(
        "/api/debug/graph/constrained/run",
        json={"template": "edges_from_entity", "params": {"name_substring": "x"}},
    )
    assert r.status_code == 200
    assert r.json()["template_id"] == "edges_from_entity"


def test_suggest_without_execute(monkeypatch):
    c = _client(monkeypatch)
    r = c.post(
        "/api/debug/graph/constrained/suggest",
        json={"question": "请给出图谱摘要"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["template_id"] == "graph_global_summary"
    assert data["executed"] is None


def test_debug_disabled_403(monkeypatch):
    monkeypatch.setattr(
        "src.api.routers.debug_routes.get_settings",
        lambda: type("S", (), {"debug_data_api": False})(),
    )
    app = FastAPI()
    app.include_router(debug_routes.router)
    c = TestClient(app)
    r = c.post(
        "/api/debug/graph/constrained/run",
        json={"template": "graph_global_summary", "params": {}},
    )
    assert r.status_code == 403
