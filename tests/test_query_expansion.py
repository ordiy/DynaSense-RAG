"""Tests for LangGraph Query Expansion node (no GCP creds required)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.rag_core import AgentState, ExpandQuery, expand_query_node, retrieve_and_rerank_node


@pytest.fixture
def mock_settings():
    with patch("src.rag_core.get_settings") as mock:
        settings = MagicMock()
        settings.query_expansion_enabled = False
        settings.rag_vector_rerank_top_n = 5
        mock.return_value = settings
        yield settings


@pytest.fixture
def mock_expand_llm():
    with patch("src.rag_core.expand_llm") as mock:
        result = ExpandQuery(queries=["rephrase1", "rephrase2"])
        mock.invoke.return_value = result
        yield mock


@pytest.fixture
def mock_retrieve():
    with patch("src.rag_core.retrieve_parent_documents_expanded") as mock:
        yield mock


@pytest.fixture
def mock_vectorstore():
    with patch("src.rag_core.vectorstore", new=MagicMock()):
        yield


@pytest.fixture
def mock_query_anchors():
    with patch("src.core.query_anchors.filter_documents_by_query_anchors") as mock:
        mock.side_effect = lambda q, docs: (docs, [])
        yield mock


@pytest.fixture
def mock_jina():
    with patch("src.rag_core.jina_rerank") as mock:
        mock.side_effect = lambda q, docs, top_n: docs[:top_n]
        yield mock


def test_expand_query_disabled(mock_settings, mock_expand_llm):
    """When expansion is off, expanded_questions contains only the original question."""
    mock_settings.query_expansion_enabled = False
    state: AgentState = {"question": "original_question", "logs": []}

    result = expand_query_node(state)

    assert result["expanded_questions"] == ["original_question"]
    mock_expand_llm.invoke.assert_not_called()


def test_expand_query_enabled(mock_settings, mock_expand_llm):
    """When expansion is on, LLM is called and expanded_questions has 3 items."""
    mock_settings.query_expansion_enabled = True
    state: AgentState = {"question": "original_question", "logs": []}

    result = expand_query_node(state)

    assert result["expanded_questions"] == ["original_question", "rephrase1", "rephrase2"]
    mock_expand_llm.invoke.assert_called_once()


def test_expand_query_llm_failure_falls_back(mock_settings, mock_expand_llm):
    """If LLM expansion fails, falls back to original question without raising."""
    mock_settings.query_expansion_enabled = True
    mock_expand_llm.invoke.side_effect = RuntimeError("LLM unavailable")
    state: AgentState = {"question": "original_question", "logs": []}

    result = expand_query_node(state)

    assert result["expanded_questions"] == ["original_question"]
    assert any("failed" in log.lower() or "\u26a0\ufe0f" in log for log in result["logs"])


def test_retrieve_uses_expanded_questions(
    mock_settings, mock_retrieve, mock_vectorstore, mock_query_anchors, mock_jina
):
    """retrieve_and_rerank_node runs retrieval for each expanded query and deduplicates."""
    state: AgentState = {
        "question": "original_question",
        "expanded_questions": ["original_question", "rephrase1"],
        "logs": [],
    }

    def side_effect(q, dense_k):
        if q == "original_question":
            return [
                Document(page_content="doc1", metadata={"chunk_id": "1"}),
                Document(page_content="doc2", metadata={"chunk_id": "2"}),
            ], []
        return [
            Document(page_content="doc2", metadata={"chunk_id": "2"}),
            Document(page_content="doc3", metadata={"chunk_id": "3"}),
        ], []

    mock_retrieve.side_effect = side_effect

    result = retrieve_and_rerank_node(state)

    assert mock_retrieve.call_count == 2
    mock_retrieve.assert_any_call("original_question", dense_k=10)
    mock_retrieve.assert_any_call("rephrase1", dense_k=10)
    assert result["documents"].count("doc2") == 1
    assert len(result["documents"]) == 3
