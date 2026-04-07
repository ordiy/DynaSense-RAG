"""
Tests for the BM25 index cache in hybrid_rag.

No database or external services needed — src.hybrid_rag.collection is patched
at the module level (it is None without a live DB connection).
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_child(content: str, parent_id: str = "p1") -> dict:
    return {"id": f"c_{content[:4]}", "type": "child", "content": content, "parent_id": parent_id}


@contextmanager
def _mock_collection(children=None, parents=None):
    """Patch src.hybrid_rag.collection with a mock that returns given data."""
    children = children or []
    parents = parents or []

    mock_col = MagicMock()

    def find_side_effect(filt):
        if filt.get("type") == "child":
            return iter(children)
        if filt.get("type") == "parent":
            ids = filt.get("id", {}).get("$in", [])
            return iter([p for p in parents if p["id"] in ids])
        return iter([])

    mock_col.find.side_effect = find_side_effect
    with patch("src.hybrid_rag.collection", mock_col):
        yield mock_col


# ---------------------------------------------------------------------------
# Cache build
# ---------------------------------------------------------------------------

def test_cache_is_none_initially(clean_bm25_cache):
    import src.hybrid_rag as hr

    assert hr._bm25_cache is None


def test_get_or_build_populates_cache(clean_bm25_cache):
    import src.hybrid_rag as hr

    children = [_make_child("hello world"), _make_child("foo bar baz")]
    with _mock_collection(children=children):
        result = hr._get_or_build_bm25()

    assert result is not None
    cached_children, bm25_obj = result
    assert len(cached_children) == 2
    assert hr._bm25_cache is not None


def test_get_or_build_returns_same_object_on_second_call(clean_bm25_cache):
    import src.hybrid_rag as hr

    children = [_make_child("hello world")]
    with _mock_collection(children=children):
        first = hr._get_or_build_bm25()

    # Second call must NOT re-query the DB (cache hit outside mock context)
    second = hr._get_or_build_bm25()
    assert first is second


def test_invalidate_clears_cache(clean_bm25_cache):
    import src.hybrid_rag as hr

    children = [_make_child("test data")]
    with _mock_collection(children=children):
        hr._get_or_build_bm25()

    assert hr._bm25_cache is not None
    hr.invalidate_bm25_cache()
    assert hr._bm25_cache is None


def test_cache_rebuilds_after_invalidation(clean_bm25_cache):
    import src.hybrid_rag as hr

    children_v1 = [_make_child("original content")]
    children_v2 = [_make_child("original content"), _make_child("new content added")]

    with _mock_collection(children=children_v1):
        first = hr._get_or_build_bm25()
    assert first is not None and len(first[0]) == 1

    hr.invalidate_bm25_cache()

    with _mock_collection(children=children_v2):
        second = hr._get_or_build_bm25()
    assert second is not None and len(second[0]) == 2


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

def test_cache_thread_safe_build(clean_bm25_cache):
    """Multiple threads competing on cache build must only build the index once."""
    import src.hybrid_rag as hr

    build_count = {"n": 0}
    children = [_make_child("shared content")]

    original_BM25 = None
    try:
        from rank_bm25 import BM25Okapi as _BM25
        original_BM25 = _BM25
    except ImportError:
        pytest.skip("rank_bm25 not installed")

    find_call_count = {"n": 0}

    mock_col = MagicMock()

    def counting_find(filt):
        if filt.get("type") == "child":
            find_call_count["n"] += 1
        return iter(children)

    mock_col.find.side_effect = counting_find

    results = []

    def worker():
        results.append(hr._get_or_build_bm25())

    with patch("src.hybrid_rag.collection", mock_col):
        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert all(r is not None for r in results)
    # DB should have been queried for children exactly once
    assert find_call_count["n"] == 1, f"Expected 1 child scan, got {find_call_count['n']}"


# ---------------------------------------------------------------------------
# Empty / no-data edge cases
# ---------------------------------------------------------------------------

def test_returns_none_when_no_children(clean_bm25_cache):
    import src.hybrid_rag as hr

    with _mock_collection(children=[]):
        result = hr._get_or_build_bm25()

    assert result is None


# ---------------------------------------------------------------------------
# bm25_parent_documents integration
# ---------------------------------------------------------------------------

def test_bm25_parent_documents_returns_empty_on_no_children(clean_bm25_cache):
    import src.hybrid_rag as hr

    with _mock_collection(children=[]):
        docs, logs = hr.bm25_parent_documents("test query")

    assert docs == []
    assert any("no child" in l.lower() or "unavailable" in l.lower() for l in logs)


def test_bm25_parent_documents_uses_cache(clean_bm25_cache):
    """Second call must not re-scan children — cache should absorb it."""
    import src.hybrid_rag as hr

    children = [
        _make_child("apple fruit market", "p1"),
        _make_child("banana yellow tropical", "p2"),
    ]
    parents = [
        {"id": "p1", "type": "parent", "source": "doc.txt", "full_content": "Apple content"},
        {"id": "p2", "type": "parent", "source": "doc.txt", "full_content": "Banana content"},
    ]

    child_scan_count = {"n": 0}
    mock_col = MagicMock()

    def mock_find(filt):
        if filt.get("type") == "child":
            child_scan_count["n"] += 1
            return iter(children)
        ids = filt.get("id", {}).get("$in", [])
        return iter([p for p in parents if p["id"] in ids])

    mock_col.find.side_effect = mock_find

    with patch("src.hybrid_rag.collection", mock_col):
        hr.bm25_parent_documents("apple")
        count_after_first = child_scan_count["n"]
        hr.bm25_parent_documents("banana")
        count_after_second = child_scan_count["n"]

    assert count_after_first == 1, "First call should scan children once"
    assert count_after_second == 1, "Second call must use cache, not re-scan"
