"""
DDL for PostgreSQL unified storage: **pgvector**, **JSONB** documents, optional **Apache AGE** graph.

- ``kb_doc``: replaces MongoMock / ``doc_parent``+``doc_child`` (one JSONB document per id).
- ``kb_embedding``: dense vectors for retrieval.
- Graph: prefer **AGE** + Cypher; if extension missing, create ``kg_triple`` (relational fallback).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 768


def _kb_tables_ddl() -> list[str]:
    return [
        """
        CREATE TABLE IF NOT EXISTS kb_doc (
            id   TEXT PRIMARY KEY,
            doc  JSONB NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_kb_doc_type ON kb_doc ((doc->>'type'))",
        f"""
        CREATE TABLE IF NOT EXISTS kb_embedding (
            id          TEXT PRIMARY KEY,
            content     TEXT NOT NULL,
            meta        JSONB NOT NULL,
            embedding   vector({EMBEDDING_DIM}) NOT NULL
        )
        """,
        # GIN index for full-text search (replaces in-memory BM25).
        # 'simple' config: language-agnostic, no stemming, CJK-friendly.
        "CREATE INDEX IF NOT EXISTS kb_embedding_fts_idx "
        "ON kb_embedding USING GIN(to_tsvector('simple', content))",
    ]


def _kg_triple_ddl() -> list[str]:
    """Relational triple store — used only when Apache AGE is not available."""
    return [
        """
        CREATE TABLE IF NOT EXISTS kg_triple (
            id             BIGSERIAL PRIMARY KEY,
            subject_norm   TEXT NOT NULL,
            subject_name   TEXT NOT NULL,
            predicate      TEXT NOT NULL,
            object_norm    TEXT NOT NULL,
            object_name    TEXT NOT NULL,
            chunk_id       TEXT NOT NULL,
            source         TEXT,
            UNIQUE (subject_norm, object_norm, predicate, chunk_id)
        )
        """,
        "CREATE INDEX IF NOT EXISTS kg_triple_subj ON kg_triple (subject_norm)",
        "CREATE INDEX IF NOT EXISTS kg_triple_obj ON kg_triple (object_norm)",
        "CREATE INDEX IF NOT EXISTS kg_triple_chunk ON kg_triple (chunk_id)",
    ]


def ensure_schema(pool) -> None:
    """Create extensions and tables; attempt AGE + graph, else relational ``kg_triple``."""
    from src.core.config import get_settings
    from src.infrastructure.persistence.postgres_age_setup import (
        ensure_age_extension_and_graph,
        set_age_ready,
    )

    s = get_settings()
    with pool.connection() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        for stmt in _kb_tables_ddl():
            conn.execute(stmt)
        conn.commit()

    age_ok = False
    if s.graph_backend == "age":
        age_ok = ensure_age_extension_and_graph(pool, s.age_graph_name)
    if not age_ok:
        set_age_ready(False)
        logger.info("Using relational kg_triple (AGE unavailable or graph_backend=relational).")
        with pool.connection() as conn:
            for stmt in _kg_triple_ddl():
                conn.execute(stmt)
            conn.commit()
    else:
        logger.info("Apache AGE graph ready: %s", s.age_graph_name)

    logger.info("PostgreSQL schema ensured (vector + JSONB kb_doc + graph).")


def truncate_kb_storage(pool) -> None:
    """
    Benchmark / reset: drop-recreate AGE graph when active, then TRUNCATE ``kb_*`` (+ ``kg_triple`` if present).
    """
    from src.core.config import get_settings
    from src.infrastructure.persistence.postgres_age_setup import age_is_ready

    s = get_settings()
    if age_is_ready() and s.graph_backend == "age":
        from src.infrastructure.persistence.postgres_age_graph import reset_age_graph_if_configured

        try:
            reset_age_graph_if_configured(pool, s.age_graph_name)
        except Exception as e:
            logger.warning("AGE graph truncate skipped: %s", e)
    with pool.connection() as conn:
        try:
            conn.execute("TRUNCATE TABLE kg_triple RESTART IDENTITY CASCADE")
        except Exception:
            pass
        conn.execute("TRUNCATE TABLE kb_embedding, kb_doc RESTART IDENTITY CASCADE")
        conn.commit()
