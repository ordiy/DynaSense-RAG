"""
Graph layer for Hybrid RAG (MVP): PostgreSQL (Apache AGE or relational ``kg_triple``).
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Truthy sentinel when graph persistence is PostgreSQL (``get_driver()`` is only used as boolean).
_PG_GRAPH = object()


def get_driver():
    """
    Legacy name: callers only check truthiness.

    Returns a sentinel when PostgreSQL graph storage is reachable.
    """
    from src.core.config import get_settings
    from src.infrastructure.persistence.postgres_connection import get_pool
    from src.infrastructure.persistence.postgres_graph import ping

    s = get_settings()
    if not s.database_url:
        return None
    try:
        if ping(get_pool()):
            return _PG_GRAPH
    except Exception as e:
        logger.debug("PostgreSQL graph backend unavailable: %s", e)
    return None


def ensure_schema() -> None:
    """DDL is owned by ``postgres_schema.ensure_schema``; no-op here."""
    return


def merge_triple(
    subject: str,
    predicate: str,
    obj: str,
    chunk_id: str,
    source: str,
) -> None:
    """MERGE two entities and a typed relationship with provenance."""
    from src.core.config import get_settings
    from src.infrastructure.persistence.postgres_connection import get_pool
    from src.infrastructure.persistence.postgres_age_setup import age_is_ready
    from src.infrastructure.persistence.postgres_graph import merge_triple as _pg_merge

    s = get_settings()
    if not s.database_url:
        return
    pool = get_pool()
    if age_is_ready() and s.graph_backend == "age":
        from src.infrastructure.persistence.postgres_age_graph import merge_triple_age

        merge_triple_age(subject, predicate, obj, chunk_id, source, pool, s.age_graph_name)
        return
    _pg_merge(subject, predicate, obj, chunk_id, source, pool)


def query_relationships_by_keywords(keywords: list[str], limit: int = 40) -> list[dict[str, Any]]:
    """Return relationship rows matching any keyword in entity names or relation type."""
    from src.core.config import get_settings
    from src.infrastructure.persistence.postgres_connection import get_pool
    from src.infrastructure.persistence.postgres_age_setup import age_is_ready
    from src.infrastructure.persistence.postgres_graph import query_relationships_by_keywords as _pg_q

    s = get_settings()
    if not s.database_url:
        return []
    pool = get_pool()
    if age_is_ready() and s.graph_backend == "age":
        from src.infrastructure.persistence.postgres_age_graph import query_relationships_by_keywords_age

        return query_relationships_by_keywords_age(keywords, limit, pool, s.age_graph_name)
    return _pg_q(keywords, limit, pool)


def global_graph_summary() -> str:
    """MVP 'community/global' signal: entity and relationship counts + sample entities."""
    from src.core.config import get_settings
    from src.infrastructure.persistence.postgres_connection import get_pool
    from src.infrastructure.persistence.postgres_age_setup import age_is_ready
    from src.infrastructure.persistence.postgres_graph import global_graph_summary as _pg_g

    s = get_settings()
    if not s.database_url:
        return ""
    pool = get_pool()
    if age_is_ready() and s.graph_backend == "age":
        from src.infrastructure.persistence.postgres_age_graph import global_graph_summary_age

        return global_graph_summary_age(pool, s.age_graph_name)
    return _pg_g(pool)


def linearize_rows(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    parts = []
    for r in rows:
        parts.append(
            f"{r.get('subject','')} —[{r.get('predicate','REL')}]→ {r.get('object','')} "
            f"(chunk_id={r.get('chunk_id','')}, source={r.get('source','')})"
        )
    return "\n".join(parts)
