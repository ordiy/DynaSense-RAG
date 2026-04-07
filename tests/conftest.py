"""
Shared pytest fixtures for the MAP-RAG test suite.

Module-level: mock the GCP/Vertex AI heavy imports before rag_core is first
imported so unit tests can run without GCP credentials or a live database.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Intercept GCP / Vertex AI before any test file triggers rag_core import.
# These stubs must be in place at conftest load time (before collection).
# ---------------------------------------------------------------------------

def _stub_vertexai():
    """Return a module stub for langchain_google_vertexai if not already loaded."""
    if "langchain_google_vertexai" in sys.modules:
        return  # Real module already loaded (e.g., full-stack test run with creds)
    stub = MagicMock()
    # VertexAIEmbeddings / ChatVertexAI instantiated at rag_core module level
    stub.VertexAIEmbeddings.return_value = MagicMock()
    stub.ChatVertexAI.return_value = MagicMock()
    sys.modules["langchain_google_vertexai"] = stub


_stub_vertexai()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

import pytest  # noqa: E402 (must come after sys.modules manipulation above)


@pytest.fixture(autouse=False)
def clean_settings_cache():
    """Reset the lru_cache on get_settings() before and after each test."""
    from src.core.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=False)
def clean_bm25_cache():
    """Legacy fixture — BM25 cache was replaced by PostgreSQL FTS; kept as no-op for compatibility."""
    yield


@pytest.fixture()
def no_db_settings(monkeypatch, clean_settings_cache):
    """Ensure DATABASE_URL is unset so setup_storage() is a safe no-op.

    Uses setenv("") rather than delenv so that the env-var layer (higher priority
    than .env file in Pydantic Settings) wins and database_url resolves to a
    falsy empty string even when .env contains a real DATABASE_URL.
    """
    monkeypatch.setenv("DATABASE_URL", "")
    from src.core.config import get_settings

    get_settings.cache_clear()
    return get_settings()
