"""
Tests for src/retrieval_tools.py — LangChain @tool wrappers.

No GCP, no real DB: vector_search and fulltext_search internals are fully mocked.
Thread-local registry deduplication and logging are tested explicitly.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.retrieval_tools import (
    get_collected_docs,
    get_tool_logs,
    reset_doc_registry,
)


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

class TestDocRegistry:
    def setup_method(self):
        reset_doc_registry()

    def test_empty_after_reset(self):
        assert get_collected_docs() == []
        assert get_tool_logs() == []

    def test_reset_clears_between_calls(self):
        # Manually populate registry via private import
        from src.retrieval_tools import _register

        _register([Document(page_content="some text")], "test")
        assert len(get_collected_docs()) == 1

        reset_doc_registry()
        assert get_collected_docs() == []

    def test_deduplicates_identical_content(self):
        from src.retrieval_tools import _register

        doc = Document(page_content="identical content here")
        _register([doc, doc], "label1")
        _register([doc], "label2")

        assert len(get_collected_docs()) == 1

    def test_accumulates_distinct_docs(self):
        from src.retrieval_tools import _register

        docs = [
            Document(page_content=f"unique content {i}" * 10)
            for i in range(3)
        ]
        _register(docs, "batch")
        assert len(get_collected_docs()) == 3

    def test_logs_include_added_count(self):
        from src.retrieval_tools import _register

        docs = [Document(page_content=f"doc {i}" * 10) for i in range(2)]
        _register(docs, "my_tool")

        logs = get_tool_logs()
        assert len(logs) == 1
        assert "+2" in logs[0]


# ---------------------------------------------------------------------------
# vector_search tool
# ---------------------------------------------------------------------------

class TestVectorSearchTool:
    def setup_method(self):
        reset_doc_registry()

    def _run(self, query: str, top_k: int = 3):
        """Invoke the tool function directly (bypassing LangChain decorator)."""
        from src.retrieval_tools import vector_search

        # Access the underlying function via .func (LangChain StructuredTool)
        fn = getattr(vector_search, "func", vector_search)
        return fn(query=query, top_k=top_k)

    def test_returns_preview_on_success(self):
        mock_doc = Document(page_content="relevant passage about topic X" * 5)

        with patch("src.rag_core.retrieve_parent_documents_expanded",
                   return_value=([mock_doc], [])), \
             patch("src.rag_core.jina_rerank", return_value=[mock_doc]):
            result = self._run("topic X")

        assert "Found 1 passage" in result
        assert "relevant passage" in result

    def test_registers_docs_in_registry(self):
        mock_doc = Document(page_content="passage about finance" * 5)

        with patch("src.rag_core.retrieve_parent_documents_expanded",
                   return_value=([mock_doc], [])), \
             patch("src.rag_core.jina_rerank", return_value=[mock_doc]):
            self._run("finance")

        assert len(get_collected_docs()) == 1

    def test_returns_no_results_message_on_empty(self):
        with patch("src.rag_core.retrieve_parent_documents_expanded",
                   return_value=([], [])), \
             patch("src.rag_core.jina_rerank", return_value=[]):
            result = self._run("obscure query")

        assert "No relevant" in result

    def test_returns_error_string_on_exception(self):
        with patch("src.rag_core.retrieve_parent_documents_expanded",
                   side_effect=RuntimeError("vector store down")):
            result = self._run("crash query")

        assert "error" in result.lower()
        assert "vector store down" in result


# ---------------------------------------------------------------------------
# fulltext_search tool
# ---------------------------------------------------------------------------

class TestFulltextSearchTool:
    def setup_method(self):
        reset_doc_registry()

    def _run(self, query: str, top_k: int = 3):
        from src.retrieval_tools import fulltext_search

        fn = getattr(fulltext_search, "func", fulltext_search)
        return fn(query=query, top_k=top_k)

    def test_returns_preview_on_success(self):
        mock_doc = Document(page_content="keyword match passage" * 5)

        with patch("src.hybrid_rag.fts_parent_documents",
                   return_value=([mock_doc], ["FTS: 1 hit"])):
            result = self._run("keyword match")

        assert "Found 1 passage" in result
        assert "keyword" in result

    def test_registers_docs_in_registry(self):
        mock_doc = Document(page_content="fts passage" * 10)

        with patch("src.hybrid_rag.fts_parent_documents",
                   return_value=([mock_doc], [])):
            self._run("query")

        assert len(get_collected_docs()) == 1

    def test_returns_no_matches_on_empty(self):
        with patch("src.hybrid_rag.fts_parent_documents",
                   return_value=([], [])):
            result = self._run("unusual term")

        assert "No keyword" in result

    def test_returns_error_string_on_exception(self):
        with patch("src.hybrid_rag.fts_parent_documents",
                   side_effect=RuntimeError("pool closed")):
            result = self._run("crash")

        assert "error" in result.lower()
