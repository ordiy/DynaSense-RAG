"""Demo login / logout / me endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.api.auth_users import authenticate
from src.api.deps import get_session_user
from src.core.config import get_settings

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=256)


@router.post("/login")
async def login(body: LoginRequest, request: Request):
    user = authenticate(body.username, body.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    request.session["user"] = {
        "username": user.username,
        "display_name": user.display_name,
    }
    return {"username": user.username, "display_name": user.display_name}


@router.post("/logout")
async def logout(request: Request):
    request.session.clear()
    return {"ok": True}


@router.get("/me")
async def me(request: Request):
    s = get_settings()
    user = get_session_user(request)
    if user:
        return {**user, "auth_enabled": s.auth_enabled}
    if not s.auth_enabled:
        return {
            "username": "anonymous",
            "display_name": "Anonymous",
            "auth_enabled": False,
        }
    raise HTTPException(status_code=401, detail="Not authenticated")
