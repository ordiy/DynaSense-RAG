"""Feedback router without loading rag_core."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import state
from src.api.routers import feedback


def setup_function() -> None:
    state.feedback_log.clear()


def test_post_feedback_and_summary():
    app = FastAPI()
    app.include_router(feedback.router)
    c = TestClient(app)
    r = c.post(
        "/api/feedback",
        json={"query": "what is MAP-RAG?", "rating": 1, "comment": "ok", "tags": ["demo"]},
    )
    assert r.status_code == 200
    assert r.json().get("ok") is True
    assert "feedback_id" in r.json()
    s = c.get("/api/feedback/summary")
    assert s.status_code == 200
    data = s.json()
    assert data["n"] == 1
    assert data["by_rating"]["1"] == 1
