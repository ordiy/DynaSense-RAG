"""Observability init (no LangChain import)."""
import os

from src.observability import init_langsmith_tracing


def test_init_without_key_does_not_enable(monkeypatch):
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    assert init_langsmith_tracing() is False


def test_init_with_key_sets_env(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_API_KEY", "test-key-not-real")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGCHAIN_PROJECT", "unit-test-project")
    assert init_langsmith_tracing() is True
    assert os.environ.get("LANGCHAIN_TRACING_V2") == "true"
    assert os.environ.get("LANGCHAIN_API_KEY") == "test-key-not-real"
    assert os.environ.get("LANGCHAIN_PROJECT") == "unit-test-project"


def test_init_respects_tracing_false(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_API_KEY", "x")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    assert init_langsmith_tracing() is False
