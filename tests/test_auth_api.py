"""Demo auth API and middleware tests (no Vertex / no live DB)."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware

from src.api.auth_middleware import AuthGateMiddleware
from src.api.auth_users import authenticate, clear_demo_users_cache, parse_auth_users
from src.api.routers import auth, pages


def _auth_app(secret: str = "test-secret-key") -> FastAPI:
    application = FastAPI()
    application.add_middleware(AuthGateMiddleware)
    application.add_middleware(
        SessionMiddleware,
        secret_key=secret,
        session_cookie="map_rag_session",
    )
    application.include_router(auth.router)
    application.include_router(pages.router)
    return application


def test_parse_auth_users_and_authenticate(demo_auth_enabled):
    users = parse_auth_users(
        '[{"username":"demo","password":"demo123","display_name":"Demo User"}]'
    )
    assert "demo" in users
    assert authenticate("demo", "demo123") is not None
    assert authenticate("demo", "wrong") is None
    assert authenticate("nosuch", "demo123") is None


def test_login_me_logout_flow(demo_auth_enabled):
    client = TestClient(_auth_app())
    assert client.get("/api/auth/me").status_code == 401

    bad = client.post("/api/auth/login", json={"username": "demo", "password": "nope"})
    assert bad.status_code == 401

    ok = client.post(
        "/api/auth/login", json={"username": "demo", "password": "demo123"}
    )
    assert ok.status_code == 200
    assert ok.json()["username"] == "demo"
    assert ok.json()["display_name"] == "Demo User"

    me = client.get("/api/auth/me")
    assert me.status_code == 200
    assert me.json()["username"] == "demo"

    out = client.post("/api/auth/logout")
    assert out.status_code == 200
    assert client.get("/api/auth/me").status_code == 401


def test_auth_gate_blocks_api_and_redirects_pages(demo_auth_enabled, tmp_path, monkeypatch):
    # pages router reads STATIC_DIR; ensure login.html exists via real static
    client = TestClient(_auth_app())
    r = client.get("/api/chat/session/x", follow_redirects=False)
    assert r.status_code == 401

    page = client.get("/", follow_redirects=False)
    assert page.status_code == 302
    assert "/login" in page.headers.get("location", "")


def test_auth_disabled_bypass(auth_disabled):
    clear_demo_users_cache()
    client = TestClient(_auth_app())
    me = client.get("/api/auth/me")
    assert me.status_code == 200
    assert me.json()["username"] == "anonymous"
    # Protected path allowed when auth off
    r = client.get("/api/chat/session/x", follow_redirects=False)
    # No session router mounted → 404, not 401
    assert r.status_code == 404
