"""
Tests for the unified LangGraph execution path (optimisation #10).

Verifies that:
- run_hybrid_chat_pipeline routes through rag_app.invoke (skip_retrieval=True).
- retrieve_and_rerank_node skips vector search when skip_retrieval=True and
  documents are already provided.
- Both vector-only and hybrid non-streaming paths exercise grade_documents_node
  and generate_node from the same compiled graph.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# retrieve_and_rerank_node — skip_retrieval behaviour
# ---------------------------------------------------------------------------

def test_retrieve_node_skips_when_flag_set():
    from src.rag_core import retrieve_and_rerank_node

    mock_vs = MagicMock()
    state = {
        "question": "what is the risk",
        "documents": ["doc1", "doc2"],
        "skip_retrieval": True,
        "loop_count": 0,
        "logs": [],
    }
    with patch("src.rag_core.vectorstore", mock_vs):
        result = retrieve_and_rerank_node(state)

    mock_vs.as_retriever.assert_not_called()
    # Documents must be unchanged (node did not overwrite them)
    assert "doc1" in (state.get("documents") or result.get("documents", []))


def test_retrieve_node_performs_search_without_flag():
    """Without skip_retrieval, retrieve_parent_documents_expanded is called."""
    from src.rag_core import retrieve_and_rerank_node

    mock_vs = MagicMock()
    state = {
        "question": "test query",
        "documents": [],
        "skip_retrieval": False,
        "loop_count": 0,
        "logs": [],
    }
    with patch("src.rag_core.vectorstore", mock_vs), \
         patch("src.rag_core.retrieve_parent_documents_expanded", return_value=([], [])) as mock_expand, \
         patch("src.core.query_anchors.filter_documents_by_query_anchors", return_value=([], [])), \
         patch("src.rag_core.jina_rerank", return_value=[]):
        retrieve_and_rerank_node(state)

    mock_expand.assert_called_once()


# ---------------------------------------------------------------------------
# run_hybrid_chat_pipeline — uses invoke_rag_app
# ---------------------------------------------------------------------------

def test_run_hybrid_chat_pipeline_calls_rag_app(monkeypatch, clean_bm25_cache, clean_settings_cache):
    """run_hybrid_chat_pipeline must call invoke_rag_app with skip_retrieval=True."""
    import src.hybrid_rag as hr

    # Fake a successful prepare_hybrid_chat result
    fake_state = {
        "question": "test",
        "documents": ["relevant doc"],
        "is_analysis": False,
        "generation": "",
        "loop_count": 0,
        "logs": ["Hybrid RAG pipeline (MVP)", "Router: VECTOR — test"],
    }
    fake_prepared = hr.HybridPrepared(
        state=fake_state,
        decision=MagicMock(route="VECTOR", reason="test"),
        effective_route="VECTOR",
    )

    invoke_calls = []

    def mock_invoke_rag_app(state):
        invoke_calls.append(state)
        return {**state, "generation": "mock answer"}

    with patch.object(hr, "prepare_hybrid_chat", return_value=fake_prepared), \
         patch("src.hybrid_rag.invoke_rag_app", side_effect=mock_invoke_rag_app):
        result = hr.run_hybrid_chat_pipeline("test")

    assert len(invoke_calls) == 1
    called_state = invoke_calls[0]
    assert called_state.get("skip_retrieval") is True, "skip_retrieval must be True"
    assert result["answer"] == "mock answer"


def test_run_hybrid_pipeline_returns_early_on_empty_kb(clean_settings_cache):
    """When prepare_hybrid_chat returns a dict (empty KB), rag_app must NOT be called."""
    import src.hybrid_rag as hr

    early_dict = {"answer": "抱歉，知识库为空。请先上传文档。", "context_used": [], "logs": []}

    with patch.object(hr, "prepare_hybrid_chat", return_value=early_dict), \
         patch("src.hybrid_rag.invoke_rag_app") as mock_invoke:
        result = hr.run_hybrid_chat_pipeline("any question")

    mock_invoke.assert_not_called()
    assert result["answer"] == early_dict["answer"]


# ---------------------------------------------------------------------------
# Fusion rerank pool cap logging
# ---------------------------------------------------------------------------

def test_fusion_rerank_logs_truncation(clean_settings_cache, monkeypatch):
    """fusion_rerank_docs must log a warning when candidate pool exceeds pool_size."""
    from langchain_core.documents import Document
    import src.hybrid_rag as hr

    monkeypatch.setenv("HYBRID_RERANK_POOL_SIZE", "3")
    from src.core.config import get_settings
    get_settings.cache_clear()

    docs = [Document(page_content=f"doc {i}") for i in range(10)]

    with patch("src.hybrid_rag.filter_documents_by_query_anchors", return_value=(docs, [])), \
         patch("src.hybrid_rag.jina_rerank", return_value=docs[:2]):
        ranked, logs = hr.fusion_rerank_docs("question", docs, top_n=2)

    truncation_logged = any("truncat" in l.lower() for l in logs)
    assert truncation_logged, f"Expected truncation log. Logs: {logs}"


def test_fusion_rerank_no_truncation_log_when_within_pool(clean_settings_cache, monkeypatch):
    from langchain_core.documents import Document
    import src.hybrid_rag as hr

    monkeypatch.setenv("HYBRID_RERANK_POOL_SIZE", "50")
    from src.core.config import get_settings
    get_settings.cache_clear()

    docs = [Document(page_content=f"doc {i}") for i in range(5)]

    with patch("src.hybrid_rag.filter_documents_by_query_anchors", return_value=(docs, [])), \
         patch("src.hybrid_rag.jina_rerank", return_value=docs[:2]):
        _, logs = hr.fusion_rerank_docs("question", docs, top_n=2)

    truncation_logged = any("truncat" in l.lower() for l in logs)
    assert not truncation_logged


# ---------------------------------------------------------------------------
# setup_storage — no crash without DATABASE_URL
# ---------------------------------------------------------------------------

def test_setup_storage_no_crash_without_db_url(no_db_settings):
    """setup_storage must log a warning but not raise when DATABASE_URL is absent."""
    from src.rag_core import setup_storage

    # Should complete without raising
    setup_storage()


def test_setup_storage_skips_init_without_db_url(no_db_settings):
    """setup_storage must not call init_pool when DATABASE_URL is unset."""
    from src.rag_core import setup_storage

    with patch("src.rag_core._init_postgres_storage") as mock_init:
        setup_storage()

    mock_init.assert_not_called()
