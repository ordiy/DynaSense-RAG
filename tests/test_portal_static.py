"""Ensure customer portal static asset exists; login page available for auth gate."""
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware

from src.api.auth_middleware import AuthGateMiddleware
from src.api.routers import pages


def test_portal_html_exists():
    root = Path(__file__).resolve().parent.parent
    p = root / "src" / "static" / "portal.html"
    assert p.is_file()
    text = p.read_text(encoding="utf-8")
    assert "/api/chat/session" in text
    assert "DynaSense" in text


def test_demo_redirects_to_login_when_auth_enabled(demo_auth_enabled):
    application = FastAPI()
    application.add_middleware(AuthGateMiddleware)
    application.add_middleware(
        SessionMiddleware, secret_key="test-secret-key", session_cookie="map_rag_session"
    )
    application.include_router(pages.router)
    client = TestClient(application)
    r = client.get("/demo", follow_redirects=False)
    assert r.status_code == 302
    assert "/login" in r.headers.get("location", "")


def test_login_page_serves_html(auth_disabled):
    application = FastAPI()
    application.add_middleware(AuthGateMiddleware)
    application.add_middleware(
        SessionMiddleware, secret_key="test-secret-key", session_cookie="map_rag_session"
    )
    application.include_router(pages.router)
    client = TestClient(application)
    r = client.get("/login")
    assert r.status_code == 200
    assert "Sign in" in r.text or "MAP-RAG" in r.text
