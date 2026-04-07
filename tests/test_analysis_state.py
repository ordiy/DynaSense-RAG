"""
Tests for the is_analysis / skip_retrieval fields in AgentState and the
single-compute guarantee in retrieve_and_rerank_node / grade / generate nodes.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# _is_analysis_query
# ---------------------------------------------------------------------------

def test_analysis_keywords_chinese():
    from src.rag_core import _is_analysis_query

    assert _is_analysis_query("请分析该公司的风险") is True
    assert _is_analysis_query("预测未来三年趋势") is True
    assert _is_analysis_query("如何改进现有方案") is True


def test_analysis_keywords_english():
    from src.rag_core import _is_analysis_query

    assert _is_analysis_query("analyze the market risk") is True
    assert _is_analysis_query("predict quarterly revenue") is True
    assert _is_analysis_query("what is the feasibility of this plan") is True


def test_factual_query_not_analysis():
    from src.rag_core import _is_analysis_query

    assert _is_analysis_query("中信银行2024年净利润是多少") is False
    assert _is_analysis_query("who is the CEO") is False
    assert _is_analysis_query("what date was the contract signed") is False


# ---------------------------------------------------------------------------
# is_analysis stored in AgentState by retrieve_and_rerank_node
# ---------------------------------------------------------------------------

def test_retrieve_node_sets_is_analysis_for_analysis_question():
    """retrieve_and_rerank_node must set is_analysis=True for analysis queries."""
    from src.rag_core import retrieve_and_rerank_node

    state = {
        "question": "分析该公司的市场风险",
        "documents": [],
        "loop_count": 0,
        "logs": [],
    }
    with patch("src.rag_core.vectorstore", None):
        result = retrieve_and_rerank_node(state)

    assert result.get("is_analysis") is True


def test_retrieve_node_sets_is_analysis_false_for_factual_question():
    from src.rag_core import retrieve_and_rerank_node

    state = {
        "question": "中信银行2024年净利润",
        "documents": [],
        "loop_count": 0,
        "logs": [],
    }
    with patch("src.rag_core.vectorstore", None):
        result = retrieve_and_rerank_node(state)

    assert result.get("is_analysis") is False


# ---------------------------------------------------------------------------
# skip_retrieval bypasses vector search
# ---------------------------------------------------------------------------

def test_skip_retrieval_prevents_vector_search():
    """When skip_retrieval=True, retrieve_and_rerank_node must not touch vectorstore."""
    from src.rag_core import retrieve_and_rerank_node

    mock_vs = MagicMock()
    state = {
        "question": "test question",
        "documents": ["pre-populated doc"],
        "skip_retrieval": True,
        "loop_count": 0,
        "logs": [],
    }
    with patch("src.rag_core.vectorstore", mock_vs):
        result = retrieve_and_rerank_node(state)

    mock_vs.as_retriever.assert_not_called()
    assert "skip" in " ".join(result.get("logs", [])).lower()


def test_skip_retrieval_still_computes_is_analysis():
    from src.rag_core import retrieve_and_rerank_node

    state = {
        "question": "如何优化分析流程",
        "documents": ["some pre-fetched content"],
        "skip_retrieval": True,
        "loop_count": 0,
        "logs": [],
    }
    with patch("src.rag_core.vectorstore", None):
        result = retrieve_and_rerank_node(state)

    assert result.get("is_analysis") is True


# ---------------------------------------------------------------------------
# grade_documents_node uses is_analysis from state (no re-computation)
# ---------------------------------------------------------------------------

def test_grade_node_uses_state_is_analysis(monkeypatch):
    """grade_documents_node must not call _is_analysis_query when state already has is_analysis."""
    from src.rag_core import grade_documents_node

    call_count = {"n": 0}

    def counting_analysis_query(q):
        call_count["n"] += 1
        return False

    monkeypatch.setattr("src.rag_core._is_analysis_query", counting_analysis_query)

    # Pre-supply is_analysis in state — grade node must NOT recompute
    mock_grader = MagicMock()
    mock_grader.invoke.return_value = MagicMock(binary_score="yes")

    state = {
        "question": "some question",
        "documents": ["[Passage 1]\nsome content"],
        "is_analysis": False,
        "loop_count": 0,
        "logs": [],
    }
    with patch("src.rag_core.grader_llm", mock_grader), \
         patch("src.rag_core.format_numbered_passages", return_value="[Passage 1]\nsome content"):
        grade_documents_node(state)

    assert call_count["n"] == 0, "_is_analysis_query should not be called when is_analysis is in state"


# ---------------------------------------------------------------------------
# generate_node uses is_analysis from state (no re-computation)
# ---------------------------------------------------------------------------

def test_generate_node_uses_state_is_analysis(monkeypatch):
    """generate_node must not call _is_analysis_query when state already has is_analysis."""
    from src.rag_core import generate_node

    call_count = {"n": 0}

    def counting_analysis_query(q):
        call_count["n"] += 1
        return True

    monkeypatch.setattr("src.rag_core._is_analysis_query", counting_analysis_query)

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="generated answer")

    state = {
        "question": "some question",
        "documents": ["some content"],
        "is_analysis": True,
        "loop_count": 0,
        "logs": [],
    }
    with patch("src.rag_core.llm", mock_llm), \
         patch("src.rag_core.format_numbered_passages", return_value="[Passage 1]\nsome content"):
        generate_node(state)

    assert call_count["n"] == 0, "_is_analysis_query should not be called when is_analysis is in state"


def test_generate_node_picks_analysis_prompt_when_flag_true(monkeypatch):
    from src.rag_core import generate_node, GEN_ANALYSIS_PROMPT, GEN_PROMPT

    prompts_used = []

    def capture_invoke(msgs):
        prompts_used.append(msgs)
        return MagicMock(content="ok")

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = capture_invoke

    state = {
        "question": "请分析风险",
        "documents": ["doc text"],
        "is_analysis": True,
        "loop_count": 0,
        "logs": [],
    }
    with patch("src.rag_core.llm", mock_llm), \
         patch("src.rag_core.format_numbered_passages", return_value="[Passage 1]\ndoc text"):
        generate_node(state)

    assert mock_llm.invoke.called
    # Log must mention analysis
    assert any("analysis" in l.lower() or "Analysis" in l for l in state["logs"])


# ---------------------------------------------------------------------------
# stream_generation_chunks accepts is_analysis parameter
# ---------------------------------------------------------------------------

def test_stream_generation_chunks_accepts_is_analysis(monkeypatch):
    from src.rag_core import stream_generation_chunks

    call_count = {"n": 0}

    def counting_analysis_query(q):
        call_count["n"] += 1
        return False

    monkeypatch.setattr("src.rag_core._is_analysis_query", counting_analysis_query)

    mock_llm = MagicMock()
    mock_llm.stream.return_value = [MagicMock(content="hello")]

    with patch("src.rag_core.llm", mock_llm), \
         patch("src.rag_core.format_numbered_passages", return_value="[Passage 1]\nsome content"):
        chunks = list(stream_generation_chunks("question", ["some content"], is_analysis=False))

    # is_analysis was provided — should not call _is_analysis_query
    assert call_count["n"] == 0


def test_stream_generation_chunks_falls_back_to_compute_when_none(monkeypatch):
    from src.rag_core import stream_generation_chunks

    call_count = {"n": 0}

    def counting_analysis_query(q):
        call_count["n"] += 1
        return False

    monkeypatch.setattr("src.rag_core._is_analysis_query", counting_analysis_query)

    mock_llm = MagicMock()
    mock_llm.stream.return_value = [MagicMock(content="hello")]

    with patch("src.rag_core.llm", mock_llm), \
         patch("src.rag_core.format_numbered_passages", return_value="[Passage 1]\nsome content"):
        list(stream_generation_chunks("question", ["some content"], is_analysis=None))

    assert call_count["n"] == 1
