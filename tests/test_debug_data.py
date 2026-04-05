"""Unit tests for debug_data helpers (no FastAPI / Vertex import)."""
import os

import pytest

from src import debug_data


def test_graph_keyword_search_empty_keywords():
    rows, err = debug_data.graph_keyword_search([], limit=10)
    assert rows == []
    assert err
    assert "keyword" in err.lower()


@pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="Requires DATABASE_URL")
def test_kb_embedding_summary_with_pool(monkeypatch):
    from src.core.config import get_settings
    from src.infrastructure.persistence.postgres_connection import close_pool, init_pool

    monkeypatch.setenv("DATABASE_URL", os.environ["DATABASE_URL"])
    get_settings.cache_clear()
    close_pool()
    init_pool(os.environ["DATABASE_URL"])
    out = debug_data.kb_embedding_summary()
    assert "row_count" in out or "error" in out
    close_pool()
