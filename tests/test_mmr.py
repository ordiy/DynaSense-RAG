"""Unit tests for MMR diversification filter (pure Python, no GCP)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.core.mmr import _jaccard, _tokenize, mmr_filter
from src.rag_core import AgentState, retrieve_and_rerank_node


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _doc(text: str) -> Document:
    return Document(page_content=text, metadata={"chunk_id": text[:20]})


# ---------------------------------------------------------------------------
# Unit tests: mmr_filter
# ---------------------------------------------------------------------------

def test_mmr_returns_k_docs():
    docs = [_doc(f"document about topic {i} with unique words sentence paragraph") for i in range(5)]
    result = mmr_filter(docs, k=3)
    assert len(result) == 3


def test_mmr_empty_input():
    assert mmr_filter([], k=3) == []


def test_mmr_k_zero():
    docs = [_doc("some text")]
    assert mmr_filter(docs, k=0) == []


def test_mmr_k_larger_than_docs():
    docs = [_doc(f"doc {i}") for i in range(3)]
    result = mmr_filter(docs, k=10)
    assert len(result) == 3


def test_mmr_lambda_one_preserves_rank():
    """lambda=1.0 → pure relevance → first k docs returned in original order."""
    docs = [_doc(f"relevant topic number {i} alpha beta gamma") for i in range(5)]
    result = mmr_filter(docs, k=3, lambda_param=1.0)
    assert result == docs[:3]


def test_mmr_selects_diverse_docs():
    """Three near-identical docs + one unique doc, k=2, lambda=0.5 → unique doc selected."""
    repeated = "the quick brown fox jumps over the lazy dog " * 5
    unique = "quantum entanglement photon spin measurement collapse wavefunction"
    docs = [
        _doc(repeated + "a"),
        _doc(repeated + "b"),
        _doc(repeated + "c"),
        _doc(unique),
    ]
    # rank 0 = repeated+a (highest Jina relevance); unique doc is rank 3
    result = mmr_filter(docs, k=2, lambda_param=0.5)
    contents = [d.page_content for d in result]
    assert any(unique in c for c in contents), "Unique doc must be selected for diversity"


def test_jaccard_identical():
    t = _tokenize("hello world hello")
    assert _jaccard(t, t) == 1.0


def test_jaccard_disjoint():
    a = _tokenize("hello world")
    b = _tokenize("foo bar baz")
    assert _jaccard(a, b) == 0.0


def test_jaccard_empty():
    assert _jaccard(frozenset(), frozenset()) == 1.0


# ---------------------------------------------------------------------------
# Integration: retrieve_and_rerank_node calls mmr_filter when enabled
# ---------------------------------------------------------------------------

def test_retrieve_node_applies_mmr_when_enabled():
    docs = [_doc(f"passage {i}") for i in range(3)]

    mock_settings = MagicMock()
    mock_settings.rag_vector_rerank_top_n = 3
    mock_settings.mmr_enabled = True
    mock_settings.mmr_lambda = 0.7
    mock_settings.query_expansion_enabled = False

    with patch("src.rag_core.get_settings", return_value=mock_settings), \
         patch("src.rag_core.vectorstore", new=MagicMock()), \
         patch("src.rag_core.retrieve_parent_documents_expanded", return_value=(docs, [])), \
         patch("src.core.query_anchors.filter_documents_by_query_anchors",
               side_effect=lambda q, d: (d, [])), \
         patch("src.rag_core.jina_rerank", return_value=docs), \
         patch("src.core.mmr.mmr_filter", return_value=docs[:2]) as mock_mmr:

        state: AgentState = {
            "question": "test query",
            "expanded_questions": ["test query"],
            "logs": [],
        }
        result = retrieve_and_rerank_node(state)

    mock_mmr.assert_called_once_with(docs, k=3, lambda_param=0.7)
    assert len(result["documents"]) == 2


def test_retrieve_node_skips_mmr_when_disabled():
    docs = [_doc(f"passage {i}") for i in range(3)]

    mock_settings = MagicMock()
    mock_settings.rag_vector_rerank_top_n = 3
    mock_settings.mmr_enabled = False
    mock_settings.query_expansion_enabled = False

    with patch("src.rag_core.get_settings", return_value=mock_settings), \
         patch("src.rag_core.vectorstore", new=MagicMock()), \
         patch("src.rag_core.retrieve_parent_documents_expanded", return_value=(docs, [])), \
         patch("src.core.query_anchors.filter_documents_by_query_anchors",
               side_effect=lambda q, d: (d, [])), \
         patch("src.rag_core.jina_rerank", return_value=docs), \
         patch("src.core.mmr.mmr_filter") as mock_mmr:

        state: AgentState = {
            "question": "test query",
            "expanded_questions": ["test query"],
            "logs": [],
        }
        retrieve_and_rerank_node(state)

    mock_mmr.assert_not_called()
