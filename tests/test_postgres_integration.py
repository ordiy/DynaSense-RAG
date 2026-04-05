"""
Integration tests for PostgreSQL + JSONB + Apache AGE (requires running DB).

Example:
  docker compose -f docker-compose.postgres.yml up -d
  DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:5433/map_rag \\
    pytest tests/test_postgres_integration.py -q
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="Set DATABASE_URL to run PostgreSQL integration tests.",
)


def test_schema_jsonb_and_graph_roundtrip(monkeypatch):
    """DDL + JSONB doc insert + AGE or relational triple + graph_store merge."""
    monkeypatch.setenv("DATABASE_URL", os.environ["DATABASE_URL"])
    from src.core.config import get_settings

    get_settings.cache_clear()

    from src.infrastructure.persistence.postgres_connection import close_pool, init_pool
    from src.infrastructure.persistence.postgres_schema import ensure_schema
    from src.infrastructure.persistence.postgres_jsonb_collection import PostgresJsonbDocCollection
    from src.infrastructure.persistence.postgres_age_setup import age_is_ready
    from src.graph_store import merge_triple, query_relationships_by_keywords

    close_pool()
    init_pool(os.environ["DATABASE_URL"])
    from src.infrastructure.persistence.postgres_connection import get_pool

    pool = get_pool()
    ensure_schema(pool)

    col = PostgresJsonbDocCollection(pool)
    col.insert_one(
        {
            "id": "parent_test_1",
            "type": "parent",
            "source": "t.txt",
            "full_content": "hello",
        }
    )
    children = list(col.find({"type": "child"}))
    assert isinstance(children, list)

    merge_triple(
        "中信银行",
        "关联方",
        "某科技公司",
        "chunk_test_1",
        "demo.txt",
    )
    rows = query_relationships_by_keywords(["中信"], limit=10)
    assert len(rows) >= 1
    assert any("中信" in str(r.get("subject", "")) for r in rows)

    if age_is_ready():
        from src.infrastructure.persistence.postgres_age_graph import global_graph_summary_age
        from src.core.config import get_settings

        s = get_settings()
        txt = global_graph_summary_age(pool, s.age_graph_name)
        assert "Graph summary" in txt

    close_pool()
