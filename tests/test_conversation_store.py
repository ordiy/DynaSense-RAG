"""Unit tests for ConversationStore (mocked PostgreSQL pool)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

from src.infrastructure.persistence.postgres_conversations import (
    ConversationStore,
    MemoryConversationStore,
    _title_from_message,
)


def test_title_from_message_truncates():
    assert _title_from_message("short") == "short"
    long = "x" * 80
    t = _title_from_message(long, max_len=48)
    assert len(t) == 48
    assert t.endswith("…")


def test_memory_store_crud_and_ownership():
    store = MemoryConversationStore()
    c = store.create_conversation("alice", title="Hello")
    cid = c["id"]
    assert store.get_conversation(cid, "bob") is None
    assert store.get_conversation(cid, "alice")["title"] == "Hello"

    store.append_message(cid, "alice", "user", "hi", set_title_if_empty=True)
    store.append_message(cid, "alice", "assistant", "hello")
    msgs = store.list_messages(cid, "alice")
    assert msgs is not None and len(msgs) == 2
    assert store.list_messages(cid, "bob") is None

    listed = store.list_conversations("alice")
    assert len(listed) == 1
    assert store.delete_conversation(cid, "bob") is False
    assert store.delete_conversation(cid, "alice") is True
    assert store.get_conversation(cid, "alice") is None


def test_memory_ensure_conversation():
    store = MemoryConversationStore()
    cid = store.ensure_conversation("u1", None, first_user_message="关联方有哪些？")
    assert store.get_conversation(cid, "u1")["title"].startswith("关联方")


def test_pg_store_create_and_list():
    mock_pool = MagicMock()
    mock_conn = MagicMock()
    mock_pool.connection.return_value.__enter__.return_value = mock_conn
    mock_cur = MagicMock()
    mock_conn.execute.return_value = mock_cur
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    mock_cur.fetchall.return_value = [
        ("c1", "demo", "Title", ts, ts),
    ]
    mock_cur.fetchone.return_value = ("c1", "demo", "Title", ts, ts)

    store = ConversationStore(mock_pool)
    created = store.create_conversation("demo", conversation_id="c1", title="Title")
    assert created["id"] == "c1"
    mock_conn.execute.assert_called()
    mock_conn.commit.assert_called()

    listed = store.list_conversations("demo", limit=10)
    assert listed[0]["id"] == "c1"
    assert listed[0]["title"] == "Title"

    got = store.get_conversation("c1", "demo")
    assert got is not None and got["user_id"] == "demo"


def test_pg_store_append_message_sets_title():
    mock_pool = MagicMock()
    mock_conn = MagicMock()
    mock_pool.connection.return_value.__enter__.return_value = mock_conn
    mock_cur = MagicMock()
    mock_conn.execute.return_value = mock_cur
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    mock_cur.fetchone.return_value = ("c1", "demo", "New conversation", ts, ts)

    store = ConversationStore(mock_pool)
    msg = store.append_message(
        "c1", "demo", "user", "what is X?", set_title_if_empty=True
    )
    assert msg is not None
    assert msg["role"] == "user"
    # INSERT message + UPDATE title
    assert mock_conn.execute.call_count >= 2
    sqls = [c.args[0] for c in mock_conn.execute.call_args_list]
    assert any("INSERT INTO chat_message" in s for s in sqls)
    assert any("UPDATE chat_conversation" in s and "title" in s for s in sqls)


def test_pg_store_list_messages_parses_meta():
    mock_pool = MagicMock()
    mock_conn = MagicMock()
    mock_pool.connection.return_value.__enter__.return_value = mock_conn
    mock_cur = MagicMock()
    mock_conn.execute.return_value = mock_cur
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    # get_conversation
    mock_cur.fetchone.return_value = ("c1", "demo", "T", ts, ts)
    mock_cur.fetchall.return_value = [
        ("m1", "user", "q", '{"route":"VECTOR"}', ts),
        ("m2", "assistant", "a", {"route": "VECTOR"}, ts),
    ]

    store = ConversationStore(mock_pool)
    msgs = store.list_messages("c1", "demo")
    assert msgs is not None and len(msgs) == 2
    assert msgs[0]["meta"]["route"] == "VECTOR"
    assert msgs[1]["meta"]["route"] == "VECTOR"


def test_pg_store_delete_returns_rowcount():
    mock_pool = MagicMock()
    mock_conn = MagicMock()
    mock_pool.connection.return_value.__enter__.return_value = mock_conn
    mock_cur = MagicMock()
    mock_cur.rowcount = 1
    mock_conn.execute.return_value = mock_cur

    store = ConversationStore(mock_pool)
    assert store.delete_conversation("c1", "demo") is True
