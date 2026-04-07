"""Tests for the new Settings fields added in the optimisation pass."""
from __future__ import annotations

import pytest


def test_default_hybrid_rag_enabled(clean_settings_cache):
    from src.core.config import get_settings

    assert get_settings().hybrid_rag_enabled is True


def test_hybrid_rag_enabled_env_override(monkeypatch, clean_settings_cache):
    monkeypatch.setenv("HYBRID_RAG_ENABLED", "false")
    from src.core.config import get_settings

    get_settings.cache_clear()
    assert get_settings().hybrid_rag_enabled is False


def test_default_pg_pool_max_size(clean_settings_cache):
    from src.core.config import get_settings

    assert get_settings().pg_pool_max_size == 10


def test_pg_pool_max_size_env_override(monkeypatch, clean_settings_cache):
    monkeypatch.setenv("PG_POOL_MAX_SIZE", "20")
    from src.core.config import get_settings

    get_settings.cache_clear()
    assert get_settings().pg_pool_max_size == 20


def test_default_hybrid_fusion_top_n(clean_settings_cache):
    from src.core.config import get_settings

    assert get_settings().hybrid_fusion_top_n == 5


def test_default_hybrid_bm25_top_child(clean_settings_cache):
    from src.core.config import get_settings

    assert get_settings().hybrid_bm25_top_child == 12


def test_default_hybrid_dense_k(clean_settings_cache):
    from src.core.config import get_settings

    assert get_settings().hybrid_dense_k == 10


def test_default_hybrid_rerank_pool_size(clean_settings_cache):
    from src.core.config import get_settings

    assert get_settings().hybrid_rerank_pool_size == 40


def test_default_skip_graph_ingest(clean_settings_cache):
    from src.core.config import get_settings

    assert get_settings().skip_graph_ingest is False


def test_skip_graph_ingest_env_override(monkeypatch, clean_settings_cache):
    monkeypatch.setenv("SKIP_GRAPH_INGEST", "true")
    from src.core.config import get_settings

    get_settings.cache_clear()
    assert get_settings().skip_graph_ingest is True
