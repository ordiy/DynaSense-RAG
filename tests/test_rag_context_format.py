"""Unit tests for numbered passage formatting (grader/generator context)."""
import pytest

from src.core.rag_context_format import format_numbered_passages


def test_format_numbered_passages_empty():
    assert format_numbered_passages([]) == ""


def test_format_numbered_passages_single():
    out = format_numbered_passages(["hello world"])
    assert "[Passage 1]" in out
    assert "hello world" in out


def test_format_numbered_passages_multiple_order():
    out = format_numbered_passages(["a", "b", "c"])
    assert out.index("[Passage 1]") < out.index("[Passage 2]")
    assert out.index("[Passage 2]") < out.index("[Passage 3]")
    assert "a" in out and "b" in out and "c" in out


def test_format_numbered_passages_strips_whitespace():
    out = format_numbered_passages(["  x  ", ""])
    assert "x" in out
    assert "[Passage 2]" in out


def test_settings_rag_vector_rerank_top_n(monkeypatch):
    from src.core.config import get_settings

    monkeypatch.setenv("RAG_VECTOR_RERANK_TOP_N", "8")
    get_settings.cache_clear()
    try:
        assert get_settings().rag_vector_rerank_top_n == 8
    finally:
        get_settings.cache_clear()
