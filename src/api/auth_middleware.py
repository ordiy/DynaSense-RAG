"""HTTP middleware: session cookie gate for demo console when AUTH_ENABLED."""

from __future__ import annotations

from urllib.parse import quote

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response


def _is_public_path(path: str) -> bool:
    if path == "/login":
        return True
    if path.startswith("/static/"):
        return True
    if path.startswith("/api/auth/login"):
        return True
    if path.startswith("/api/auth/logout"):
        return True
    if path.startswith("/api/auth/me"):
        return True
    if path in ("/docs", "/openapi.json", "/redoc"):
        return True
    return False


class AuthGateMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        from src.core.config import get_settings

        if not get_settings().auth_enabled:
            return await call_next(request)

        path = request.url.path
        if _is_public_path(path):
            return await call_next(request)

        user = None
        try:
            raw = request.session.get("user")
            if isinstance(raw, dict) and raw.get("username"):
                user = raw
        except AssertionError:
            user = None

        if user:
            return await call_next(request)

        if path.startswith("/api/"):
            return JSONResponse({"detail": "Not authenticated"}, status_code=401)

        nxt = quote(path + (("?" + request.url.query) if request.url.query else ""))
        return RedirectResponse(url=f"/login?next={nxt}", status_code=302)
