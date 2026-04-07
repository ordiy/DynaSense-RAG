"""
Tests for PostgreSQL full-text search (postgres_fts.py) and fts_parent_documents (hybrid_rag.py).

No real database needed — the pool and collection are fully mocked.
The old BM25 cache tests are replaced here; postgres_fts has no in-process state to invalidate.
"""
from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pool(rows: list[tuple]) -> MagicMock:
    """Return a mock pool whose cursor returns *rows* on fetchall()."""
    mock_cur = MagicMock()
    mock_cur.fetchall.return_value = rows
    mock_cur.__enter__ = lambda s: s
    mock_cur.__exit__ = MagicMock(return_value=False)

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cur
    mock_conn.__enter__ = lambda s: s
    mock_conn.__exit__ = MagicMock(return_value=False)

    mock_pool = MagicMock()
    mock_pool.connection.return_value = mock_conn
    return mock_pool


# ---------------------------------------------------------------------------
# fulltext_search_children — postgres_fts
# ---------------------------------------------------------------------------

class TestFulltextSearchChildren:
    def test_returns_empty_on_blank_query(self):
        from src.infrastructure.persistence.postgres_fts import fulltext_search_children

        pool = MagicMock()
        assert fulltext_search_children("", pool) == []
        assert fulltext_search_children("   ", pool) == []

    def test_returns_rows_on_match(self):
        from src.infrastructure.persistence.postgres_fts import fulltext_search_children

        rows = [
            ("c1", "apple market content", {"parent_id": "p1"}, 0.75),
            ("c2", "apple supply chain", {"parent_id": "p2"}, 0.50),
        ]
        pool = _make_pool(rows)
        result = fulltext_search_children("apple", pool, top_k=5)

        assert len(result) == 2
        assert result[0][0] == "c1"
        assert result[0][3] == pytest.approx(0.75)

    def test_returns_empty_on_db_error(self, caplog):
        from src.infrastructure.persistence.postgres_fts import fulltext_search_children
        import logging

        pool = MagicMock()
        pool.connection.side_effect = Exception("db unavailable")

        with caplog.at_level(logging.WARNING, logger="src.infrastructure.persistence.postgres_fts"):
            result = fulltext_search_children("anything", pool)

        assert result == []
        assert any("FTS child search failed" in r.message for r in caplog.records)

    def test_top_k_passed_to_sql(self):
        from src.infrastructure.persistence.postgres_fts import fulltext_search_children

        pool = _make_pool([])
        fulltext_search_children("query", pool, top_k=7)

        # Verify the execute call included k=7
        conn = pool.connection().__enter__()
        cur = conn.cursor().__enter__()
        call_args = cur.execute.call_args
        assert call_args[0][1]["k"] == 7


# ---------------------------------------------------------------------------
# fts_parent_documents — hybrid_rag
# ---------------------------------------------------------------------------

def _make_child_row(child_id: str, content: str, parent_id: str, rank: float = 0.5):
    return (child_id, content, {"parent_id": parent_id, "source": "test.txt"}, rank)


def _make_parent_record(parent_id: str, content: str = "Full parent text."):
    return {"id": parent_id, "type": "parent", "source": "test.txt", "full_content": content}


@contextmanager
def _mock_fts_env(child_rows, parents):
    """Patch FTS + pool + collection for fts_parent_documents.

    ``get_pool`` and ``fulltext_search_children`` are imported locally inside
    ``fts_parent_documents``, so they must be patched at their source paths.
    """
    mock_col = MagicMock()
    mock_col.find.side_effect = lambda filt: iter(
        [p for p in parents if p["id"] in filt.get("id", {}).get("$in", [])]
    )
    mock_pool = MagicMock()

    with patch("src.hybrid_rag.collection", mock_col), \
         patch("src.infrastructure.persistence.postgres_fts.fulltext_search_children",
               return_value=child_rows) as mock_fts, \
         patch("src.infrastructure.persistence.postgres_connection.get_pool",
               return_value=mock_pool):
        yield mock_fts, mock_col


class TestFtsParentDocuments:
    def test_returns_parent_docs_for_matching_children(self):
        from src.hybrid_rag import fts_parent_documents

        rows = [_make_child_row("c1", "apple text", "p1")]
        parents = [_make_parent_record("p1", "Apple full content")]

        with _mock_fts_env(rows, parents):
            docs, logs = fts_parent_documents("apple", top_child=12)

        assert len(docs) == 1
        assert "Apple full content" in docs[0].page_content
        assert any("fts" in l.lower() or "parent" in l.lower() for l in logs)

    def test_deduplicates_same_parent(self):
        from src.hybrid_rag import fts_parent_documents

        rows = [
            _make_child_row("c1", "text A", "p1", 0.9),
            _make_child_row("c2", "text B", "p1", 0.6),  # same parent
        ]
        parents = [_make_parent_record("p1", "Parent content")]

        with _mock_fts_env(rows, parents):
            docs, _ = fts_parent_documents("query")

        assert len(docs) == 1  # deduplicated

    def test_returns_empty_on_no_fts_matches(self):
        from src.hybrid_rag import fts_parent_documents

        with _mock_fts_env([], []):
            docs, logs = fts_parent_documents("nonexistent term")

        assert docs == []

    def test_source_tag_is_fts(self):
        from src.hybrid_rag import fts_parent_documents

        rows = [_make_child_row("c1", "content", "p1")]
        parents = [_make_parent_record("p1")]

        with _mock_fts_env(rows, parents):
            docs, _ = fts_parent_documents("content")

        assert docs[0].metadata.get("source") == "fts"
