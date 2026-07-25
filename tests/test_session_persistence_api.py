"""Session list/get/delete/chat with ConversationStore (mocked pipeline)."""
from __future__ import annotations

from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware

from src.api.auth_middleware import AuthGateMiddleware
from src.api.routers import auth, session
from src.infrastructure.persistence.postgres_conversations import (
    MemoryConversationStore,
    reset_memory_conversation_store,
)


def _app() -> FastAPI:
    application = FastAPI()
    application.add_middleware(AuthGateMiddleware)
    application.add_middleware(
        SessionMiddleware, secret_key="test-secret-key", session_cookie="map_rag_session"
    )
    application.include_router(auth.router)
    application.include_router(session.router)
    return application


def _login(client: TestClient) -> None:
    r = client.post(
        "/api/auth/login", json={"username": "demo", "password": "demo123"}
    )
    assert r.status_code == 200


def test_session_list_chat_get_delete(demo_auth_enabled, monkeypatch):
    reset_memory_conversation_store()
    mem = MemoryConversationStore()

    def _store():
        return mem

    monkeypatch.setattr(
        "src.api.routers.session.get_conversation_store", _store
    )

    client = TestClient(_app())
    _login(client)

    empty = client.get("/api/chat/sessions")
    assert empty.status_code == 200
    assert empty.json()["sessions"] == []

    with patch(
        "src.api.routers.session.run_chat_pipeline",
        return_value={
            "answer": "你好",
            "logs": [],
            "context_used": [],
            "citations": [],
            "route": "VECTOR",
            "effective_route": "VECTOR",
            "router_reason": "test",
        },
    ):
        chat = client.post(
            "/api/chat/session",
            json={"message": "关联方有哪些？", "memory_mode": "prioritized"},
        )
    assert chat.status_code == 200
    body = chat.json()
    cid = body["conversation_id"]
    assert body["answer"] == "你好"
    assert len(body["history"]) == 2

    listed = client.get("/api/chat/sessions")
    assert listed.status_code == 200
    sessions = listed.json()["sessions"]
    assert len(sessions) == 1
    assert sessions[0]["conversation_id"] == cid
    assert "关联方" in sessions[0]["title"]

    got = client.get(f"/api/chat/session/{cid}")
    assert got.status_code == 200
    assert len(got.json()["history"]) == 2

    deleted = client.delete(f"/api/chat/session/{cid}")
    assert deleted.status_code == 200
    assert client.get(f"/api/chat/session/{cid}").status_code == 404


def test_session_requires_auth(demo_auth_enabled):
    reset_memory_conversation_store()
    client = TestClient(_app())
    assert client.get("/api/chat/sessions").status_code == 401


def test_session_anonymous_when_auth_disabled(auth_disabled, monkeypatch):
    reset_memory_conversation_store()
    mem = MemoryConversationStore()
    monkeypatch.setattr(
        "src.api.routers.session.get_conversation_store", lambda: mem
    )
    client = TestClient(_app())
    with patch(
        "src.api.routers.session.run_chat_pipeline",
        return_value={
            "answer": "ok",
            "logs": [],
            "context_used": [],
            "citations": [],
        },
    ):
        r = client.post("/api/chat/session", json={"message": "hello"})
    assert r.status_code == 200
    assert r.json()["answer"] == "ok"
    # Owned by anonymous
    assert mem.list_conversations("anonymous")
