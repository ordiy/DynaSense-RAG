"""Mutable process state for upload tasks and chat sessions (not for multi-worker production)."""

from __future__ import annotations

import time
from typing import Any

from src.core.config import get_settings

tasks: dict[str, dict[str, Any]] = {}
chat_sessions: dict[str, dict[str, Any]] = {}
# Human feedback loop (MVP, in-memory; not for multi-worker production).
feedback_log: list[dict[str, Any]] = []
MAX_FEEDBACK_ENTRIES = 1000


def cleanup_tasks() -> None:
    now = time.time()
    ttl = float(get_settings().task_ttl_seconds)
    for task_id, info in list(tasks.items()):
        created_at = info.get("created_at")
        if created_at is None:
            continue
        if now - float(created_at) > ttl:
            tasks.pop(task_id, None)


def cleanup_chat_sessions() -> None:
    now = time.time()
    ttl = float(get_settings().chat_session_ttl_seconds)
    for conversation_id, info in list(chat_sessions.items()):
        updated_at = info.get("updated_at", info.get("created_at", now))
        if now - float(updated_at) > ttl:
            chat_sessions.pop(conversation_id, None)
