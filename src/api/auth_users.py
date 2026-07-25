"""Parse and verify demo users from Settings.auth_users (bcrypt-hashed at load)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache

import bcrypt

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DemoUser:
    username: str
    display_name: str
    password_hash: bytes


def _hash_password(plain: str) -> bytes:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt(rounds=10))


def verify_password(plain: str, password_hash: bytes) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), password_hash)
    except Exception:
        return False


def parse_auth_users(raw: str) -> dict[str, DemoUser]:
    """Parse AUTH_USERS JSON into username → DemoUser (hashes passwords)."""
    try:
        data = json.loads(raw or "[]")
    except json.JSONDecodeError:
        logger.error("AUTH_USERS is not valid JSON; no demo users loaded.")
        return {}
    if not isinstance(data, list):
        logger.error("AUTH_USERS must be a JSON list.")
        return {}
    out: dict[str, DemoUser] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        username = str(item.get("username") or "").strip()
        password = str(item.get("password") or "")
        display = str(item.get("display_name") or username).strip() or username
        if not username or not password:
            continue
        out[username] = DemoUser(
            username=username,
            display_name=display,
            password_hash=_hash_password(password),
        )
    return out


@lru_cache
def get_demo_users() -> dict[str, DemoUser]:
    from src.core.config import get_settings

    return parse_auth_users(get_settings().auth_users)


def clear_demo_users_cache() -> None:
    get_demo_users.cache_clear()


def authenticate(username: str, password: str) -> DemoUser | None:
    users = get_demo_users()
    user = users.get((username or "").strip())
    if user is None:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user
