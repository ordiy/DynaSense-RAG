"""
Tests for src/agentic_rag.py — ReAct agentic retrieval pipeline.

All external dependencies (LLM, tools, DB) are mocked.
Tests verify routing, document collection, grade/generate integration,
and config flag dispatch in run_chat_pipeline.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# run_agentic_retrieval
# ---------------------------------------------------------------------------

class TestRunAgenticRetrieval:
    def _invoke(self, question: str = "what is inflation?", max_steps: int = 3):
        from src.agentic_rag import run_agentic_retrieval

        return run_agentic_retrieval(question, max_steps=max_steps)

    def test_returns_docs_and_logs(self):
        mock_doc = Document(page_content="inflation definition" * 5)

        fake_agent = MagicMock()
        fake_agent.invoke.return_value = {"messages": []}

        with patch("src.retrieval_tools.reset_doc_registry"), \
             patch("src.retrieval_tools.get_collected_docs", return_value=[mock_doc]), \
             patch("src.retrieval_tools.get_tool_logs", return_value=["vector_search: +1"]), \
             patch("langgraph.prebuilt.create_react_agent", return_value=fake_agent), \
             patch("src.rag_core.jina_rerank", return_value=[mock_doc]):
            docs, logs = self._invoke()

        assert len(docs) == 1
        assert any("Jina rerank" in l for l in logs)

    def test_returns_empty_when_no_docs_collected(self):
        fake_agent = MagicMock()
        fake_agent.invoke.return_value = {"messages": []}

        with patch("src.retrieval_tools.reset_doc_registry"), \
             patch("src.retrieval_tools.get_collected_docs", return_value=[]), \
             patch("src.retrieval_tools.get_tool_logs", return_value=[]), \
             patch("langgraph.prebuilt.create_react_agent", return_value=fake_agent):
            docs, logs = self._invoke()

        assert docs == []

    def test_continues_on_agent_exception(self):
        mock_doc = Document(page_content="fallback doc" * 5)
        fake_agent = MagicMock()
        fake_agent.invoke.side_effect = RuntimeError("LLM timeout")

        with patch("src.retrieval_tools.reset_doc_registry"), \
             patch("src.retrieval_tools.get_collected_docs", return_value=[mock_doc]), \
             patch("src.retrieval_tools.get_tool_logs", return_value=[]), \
             patch("langgraph.prebuilt.create_react_agent", return_value=fake_agent), \
             patch("src.rag_core.jina_rerank", return_value=[mock_doc]):
            docs, logs = self._invoke()

        # Should not raise; should still return whatever was collected
        assert any("warning" in l.lower() or "error" in l.lower() for l in logs)


# ---------------------------------------------------------------------------
# run_agentic_chat_pipeline
# ---------------------------------------------------------------------------

class TestRunAgenticChatPipeline:
    def _run(self, query: str = "what is GDP?"):
        from src.agentic_rag import run_agentic_chat_pipeline

        return run_agentic_chat_pipeline(query)

    def test_returns_answer_dict_structure(self):
        mock_doc = Document(page_content="GDP is the total output" * 5)

        with patch("src.agentic_rag.run_agentic_retrieval",
                   return_value=([mock_doc], ["retrieval done"])), \
             patch("src.rag_core.grade_documents_node",
                   return_value={"documents": ["GDP is the total output" * 5], "logs": []}), \
             patch("src.rag_core.generate_node",
                   return_value={"generation": "GDP measures total output.", "logs": []}), \
             patch("src.core.citations.build_citations_from_context", return_value=[]):
            result = self._run()

        assert "answer" in result
        assert "context_used" in result
        assert "logs" in result
        assert "citations" in result
        assert result["route"] == "agentic"

    def test_returns_blocked_message_when_no_docs(self):
        with patch("src.agentic_rag.run_agentic_retrieval", return_value=([], [])), \
             patch("src.rag_core.grade_documents_node",
                   return_value={"documents": [], "logs": []}), \
             patch("src.rag_core.generate_node",
                   return_value={"generation": "抱歉，知识库中未能找到", "logs": []}), \
             patch("src.core.citations.build_citations_from_context", return_value=[]):
            result = self._run("unanswerable question")

        assert "抱歉" in result["answer"] or result["answer"] != ""


# ---------------------------------------------------------------------------
# run_chat_pipeline dispatch (agentic_retrieval_enabled flag)
# ---------------------------------------------------------------------------

class TestRunChatPipelineDispatch:
    def test_agentic_branch_called_when_enabled(self, clean_settings_cache, monkeypatch):
        monkeypatch.setenv("AGENTIC_RETRIEVAL_ENABLED", "true")
        from src.core.config import get_settings
        get_settings.cache_clear()

        fake_result = {
            "answer": "agentic answer",
            "context_used": [],
            "logs": [],
            "citations": [],
            "route": "agentic",
        }

        with patch("src.agentic_rag.run_agentic_chat_pipeline",
                   return_value=fake_result) as mock_agentic:
            from src.rag_core import run_chat_pipeline
            result = run_chat_pipeline("test question")

        mock_agentic.assert_called_once_with("test question")
        assert result["route"] == "agentic"

    def test_agentic_branch_skipped_when_disabled(self, clean_settings_cache, monkeypatch):
        monkeypatch.setenv("AGENTIC_RETRIEVAL_ENABLED", "false")
        from src.core.config import get_settings
        get_settings.cache_clear()

        hybrid_result = {
            "answer": "hybrid answer",
            "context_used": [],
            "logs": [],
            "citations": [],
        }

        with patch("src.agentic_rag.run_agentic_chat_pipeline") as mock_agentic, \
             patch("src.hybrid_rag.run_hybrid_chat_pipeline", return_value=hybrid_result):
            from src.rag_core import run_chat_pipeline
            result = run_chat_pipeline("test question")

        mock_agentic.assert_not_called()
        assert result["answer"] == "hybrid answer"

    def test_agentic_falls_back_on_exception(self, clean_settings_cache, monkeypatch):
        monkeypatch.setenv("AGENTIC_RETRIEVAL_ENABLED", "true")
        from src.core.config import get_settings
        get_settings.cache_clear()

        hybrid_result = {
            "answer": "hybrid fallback",
            "context_used": [],
            "logs": [],
            "citations": [],
        }

        with patch("src.agentic_rag.run_agentic_chat_pipeline",
                   side_effect=RuntimeError("agent crash")), \
             patch("src.hybrid_rag.run_hybrid_chat_pipeline", return_value=hybrid_result):
            from src.rag_core import run_chat_pipeline
            result = run_chat_pipeline("test question")

        assert result["answer"] == "hybrid fallback"
