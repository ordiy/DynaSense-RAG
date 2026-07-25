"""PostgreSQL conversation / message store for demo chat history."""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def _title_from_message(message: str, max_len: int = 48) -> str:
    text = (message or "").strip().replace("\n", " ")
    if not text:
        return "New conversation"
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _ts(dt: Any) -> float | None:
    if dt is None:
        return None
    if hasattr(dt, "timestamp"):
        return float(dt.timestamp())
    return float(dt)


class ConversationStore:
    def __init__(self, pool):
        self._pool = pool

    def create_conversation(
        self,
        user_id: str,
        conversation_id: str | None = None,
        title: str = "",
    ) -> dict:
        cid = conversation_id or str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        with self._pool.connection() as conn:
            conn.execute(
                """
                INSERT INTO chat_conversation (id, user_id, title, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
                """,
                (cid, user_id, title or "New conversation", now, now),
            )
            conn.commit()
        return {
            "id": cid,
            "user_id": user_id,
            "title": title or "New conversation",
            "created_at": now.timestamp(),
            "updated_at": now.timestamp(),
        }

    def get_conversation(self, conversation_id: str, user_id: str) -> dict | None:
        with self._pool.connection() as conn:
            cur = conn.execute(
                """
                SELECT id, user_id, title, created_at, updated_at
                FROM chat_conversation
                WHERE id = %s AND user_id = %s
                """,
                (conversation_id, user_id),
            )
            row = cur.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "user_id": row[1],
            "title": row[2],
            "created_at": _ts(row[3]),
            "updated_at": _ts(row[4]),
        }

    def list_conversations(self, user_id: str, limit: int = 50) -> list[dict]:
        limit = max(1, min(int(limit), 200))
        with self._pool.connection() as conn:
            cur = conn.execute(
                """
                SELECT id, user_id, title, created_at, updated_at
                FROM chat_conversation
                WHERE user_id = %s
                ORDER BY updated_at DESC
                LIMIT %s
                """,
                (user_id, limit),
            )
            rows = cur.fetchall()
        return [
            {
                "id": r[0],
                "user_id": r[1],
                "title": r[2],
                "created_at": _ts(r[3]),
                "updated_at": _ts(r[4]),
            }
            for r in rows
        ]

    def delete_conversation(self, conversation_id: str, user_id: str) -> bool:
        with self._pool.connection() as conn:
            cur = conn.execute(
                """
                DELETE FROM chat_conversation
                WHERE id = %s AND user_id = %s
                """,
                (conversation_id, user_id),
            )
            conn.commit()
            return cur.rowcount > 0

    def list_messages(self, conversation_id: str, user_id: str) -> list[dict] | None:
        if self.get_conversation(conversation_id, user_id) is None:
            return None
        with self._pool.connection() as conn:
            cur = conn.execute(
                """
                SELECT id, role, content, meta, created_at
                FROM chat_message
                WHERE conversation_id = %s
                ORDER BY created_at ASC
                """,
                (conversation_id,),
            )
            rows = cur.fetchall()
        out: list[dict] = []
        for row in rows:
            meta = row[3]
            if isinstance(meta, str):
                meta = json.loads(meta)
            out.append(
                {
                    "id": row[0],
                    "role": row[1],
                    "content": row[2],
                    "meta": meta or {},
                    "created_at": _ts(row[4]),
                }
            )
        return out

    def append_message(
        self,
        conversation_id: str,
        user_id: str,
        role: str,
        content: str,
        meta: dict | None = None,
        message_id: str | None = None,
        set_title_if_empty: bool = False,
    ) -> dict | None:
        conv = self.get_conversation(conversation_id, user_id)
        if conv is None:
            return None
        mid = message_id or str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        meta_json = json.dumps(meta or {})
        with self._pool.connection() as conn:
            conn.execute(
                """
                INSERT INTO chat_message (id, conversation_id, role, content, meta, created_at)
                VALUES (%s, %s, %s, %s, %s::jsonb, %s)
                """,
                (mid, conversation_id, role, content, meta_json, now),
            )
            if set_title_if_empty and role == "user" and (
                not conv.get("title") or conv.get("title") == "New conversation"
            ):
                conn.execute(
                    """
                    UPDATE chat_conversation
                    SET title = %s, updated_at = %s
                    WHERE id = %s AND user_id = %s
                    """,
                    (_title_from_message(content), now, conversation_id, user_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE chat_conversation
                    SET updated_at = %s
                    WHERE id = %s AND user_id = %s
                    """,
                    (now, conversation_id, user_id),
                )
            conn.commit()
        return {
            "id": mid,
            "role": role,
            "content": content,
            "meta": meta or {},
            "created_at": now.timestamp(),
        }

    def ensure_conversation(
        self,
        user_id: str,
        conversation_id: str | None,
        first_user_message: str | None = None,
    ) -> str:
        """Return conversation id, creating the row when missing."""
        if conversation_id:
            existing = self.get_conversation(conversation_id, user_id)
            if existing:
                return conversation_id
        title = _title_from_message(first_user_message or "")
        created = self.create_conversation(
            user_id, conversation_id=conversation_id, title=title
        )
        return created["id"]


class MemoryConversationStore:
    """In-process fallback when PostgreSQL is unavailable (tests / degraded mode)."""

    def __init__(self) -> None:
        self._conversations: dict[str, dict] = {}
        self._messages: dict[str, list[dict]] = {}

    def create_conversation(
        self,
        user_id: str,
        conversation_id: str | None = None,
        title: str = "",
    ) -> dict:
        cid = conversation_id or str(uuid.uuid4())
        now = time.time()
        row = {
            "id": cid,
            "user_id": user_id,
            "title": title or "New conversation",
            "created_at": now,
            "updated_at": now,
        }
        self._conversations[cid] = row
        self._messages.setdefault(cid, [])
        return dict(row)

    def get_conversation(self, conversation_id: str, user_id: str) -> dict | None:
        row = self._conversations.get(conversation_id)
        if not row or row["user_id"] != user_id:
            return None
        return dict(row)

    def list_conversations(self, user_id: str, limit: int = 50) -> list[dict]:
        rows = [c for c in self._conversations.values() if c["user_id"] == user_id]
        rows.sort(key=lambda c: c["updated_at"], reverse=True)
        return [dict(c) for c in rows[: max(1, min(limit, 200))]]

    def delete_conversation(self, conversation_id: str, user_id: str) -> bool:
        row = self._conversations.get(conversation_id)
        if not row or row["user_id"] != user_id:
            return False
        self._conversations.pop(conversation_id, None)
        self._messages.pop(conversation_id, None)
        return True

    def list_messages(self, conversation_id: str, user_id: str) -> list[dict] | None:
        if self.get_conversation(conversation_id, user_id) is None:
            return None
        return [dict(m) for m in self._messages.get(conversation_id, [])]

    def append_message(
        self,
        conversation_id: str,
        user_id: str,
        role: str,
        content: str,
        meta: dict | None = None,
        message_id: str | None = None,
        set_title_if_empty: bool = False,
    ) -> dict | None:
        conv = self.get_conversation(conversation_id, user_id)
        if conv is None:
            return None
        mid = message_id or str(uuid.uuid4())
        now = time.time()
        msg = {
            "id": mid,
            "role": role,
            "content": content,
            "meta": meta or {},
            "created_at": now,
        }
        self._messages.setdefault(conversation_id, []).append(msg)
        if set_title_if_empty and role == "user" and (
            not conv.get("title") or conv.get("title") == "New conversation"
        ):
            conv["title"] = _title_from_message(content)
        conv["updated_at"] = now
        self._conversations[conversation_id] = conv
        return dict(msg)

    def ensure_conversation(
        self,
        user_id: str,
        conversation_id: str | None,
        first_user_message: str | None = None,
    ) -> str:
        if conversation_id:
            existing = self.get_conversation(conversation_id, user_id)
            if existing:
                return conversation_id
        title = _title_from_message(first_user_message or "")
        created = self.create_conversation(
            user_id, conversation_id=conversation_id, title=title
        )
        return created["id"]


_memory_store: MemoryConversationStore | None = None


def get_conversation_store():
    """Return PG store when pool is ready; otherwise a process-local memory store."""
    global _memory_store
    try:
        from src.infrastructure.persistence.postgres_connection import get_pool

        pool = get_pool()
        return ConversationStore(pool)
    except Exception as exc:
        logger.debug("ConversationStore falling back to memory: %s", exc)
        if _memory_store is None:
            _memory_store = MemoryConversationStore()
        return _memory_store


def reset_memory_conversation_store() -> None:
    """Test helper: clear in-memory fallback."""
    global _memory_store
    _memory_store = MemoryConversationStore()
