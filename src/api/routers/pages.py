"""HTML pages (engineer console + customer portal + login)."""

from __future__ import annotations

import os

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

router = APIRouter(tags=["pages"])

STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "static")


def _read_html(name: str) -> str:
    path = os.path.join(STATIC_DIR, name)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@router.get("/login", response_class=HTMLResponse)
async def read_login(request: Request):
    """Login page is always reachable (auth gate whitelist)."""
    from src.api.deps import get_session_user
    from src.core.config import get_settings

    if get_settings().auth_enabled and get_session_user(request):
        nxt = request.query_params.get("next") or "/"
        if not nxt.startswith("/") or nxt.startswith("//"):
            nxt = "/"
        return RedirectResponse(url=nxt, status_code=302)
    return _read_html("login.html")


@router.get("/", response_class=HTMLResponse)
async def read_root():
    return _read_html("index.html")


@router.get("/demo", response_class=HTMLResponse)
async def read_customer_portal():
    return _read_html("portal.html")


@router.get("/portal")
async def portal_alias():
    return RedirectResponse(url="/demo", status_code=302)
