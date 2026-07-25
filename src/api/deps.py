"""FastAPI dependencies for demo authentication."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request


def get_session_user(request: Request) -> dict[str, Any] | None:
    """Return ``{username, display_name}`` from signed session cookie, if any."""
    try:
        user = request.session.get("user")
    except AssertionError:
        # SessionMiddleware not mounted
        return None
    if not isinstance(user, dict):
        return None
    username = user.get("username")
    if not username:
        return None
    return {
        "username": str(username),
        "display_name": str(user.get("display_name") or username),
    }


def require_user(request: Request) -> dict[str, Any]:
    """Require a logged-in user when AUTH_ENABLED; otherwise return anonymous demo user."""
    from src.core.config import get_settings

    user = get_session_user(request)
    if user:
        return user
    if not get_settings().auth_enabled:
        return {"username": "anonymous", "display_name": "Anonymous"}
    raise HTTPException(status_code=401, detail="Not authenticated")


def current_user_id(request: Request) -> str:
    return require_user(request)["username"]
