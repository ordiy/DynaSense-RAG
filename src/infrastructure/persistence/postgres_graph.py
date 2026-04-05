"""
Knowledge-graph triples stored relationally (replaces Neo4j for the PostgreSQL backend).

Semantic parity with ``graph_store.merge_triple`` / ``query_relationships_by_keywords`` / ``global_graph_summary``.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def _norm_key(name: str) -> str:
    """Same normalization as ``graph_store._norm_key`` for consistent entity keys."""
    s = name.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s[:512]


def merge_triple(
    subject: str,
    predicate: str,
    obj: str,
    chunk_id: str,
    source: str,
    pool,
) -> None:
    """Insert a triple row; duplicates resolved by UNIQUE(subject_norm, object_norm, predicate, chunk_id)."""
    sn = _norm_key(subject)
    on = _norm_key(obj)
    if not sn or not on:
        return
    pred = (predicate or "RELATED_TO").strip()[:128] or "RELATED_TO"
    with pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO kg_triple
                (subject_norm, subject_name, predicate, object_norm, object_name, chunk_id, source)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (subject_norm, object_norm, predicate, chunk_id) DO UPDATE SET
                subject_name = EXCLUDED.subject_name,
                object_name = EXCLUDED.object_name,
                source = EXCLUDED.source
            """,
            (
                sn,
                subject.strip()[:512],
                pred,
                on,
                obj.strip()[:512],
                chunk_id,
                source[:512],
            ),
        )
        conn.commit()


def query_relationships_by_keywords(
    keywords: list[str],
    limit: int,
    pool,
) -> list[dict[str, Any]]:
    """Match any keyword against subject/object names or predicate (case-insensitive substring)."""
    kws = [k.strip() for k in keywords if k and len(k.strip()) > 1][:20]
    if not kws:
        return []
    # Build OR conditions — same spirit as Neo4j CONTAINS (substring match).
    clauses = []
    params: list[Any] = []
    for kw in kws:
        p = f"%{kw.lower()}%"
        clauses.append(
            "(LOWER(subject_name) LIKE %s OR LOWER(object_name) LIKE %s OR LOWER(predicate) LIKE %s)"
        )
        params.extend([p, p, p])
    where_sql = " OR ".join(clauses)
    sql = f"""
        SELECT DISTINCT subject_name, predicate, object_name, chunk_id, source
        FROM kg_triple
        WHERE {where_sql}
        LIMIT %s
    """
    params.append(int(limit))
    with pool.connection() as conn:
        rows = conn.execute(sql, params).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "subject": row[0],
                "predicate": row[1],
                "object": row[2],
                "chunk_id": row[3],
                "source": row[4],
            }
        )
    return out


def global_graph_summary(pool) -> str:
    """Entity/edge counts + sample names (MVP parity with Neo4j ``global_graph_summary``)."""
    with pool.connection() as conn:
        n_ent = conn.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT subject_norm AS n FROM kg_triple
                UNION
                SELECT object_norm FROM kg_triple
            ) u
            """
        ).fetchone()[0]
        n_rel = conn.execute("SELECT COUNT(*) FROM kg_triple").fetchone()[0]
        names = conn.execute(
            """
            SELECT DISTINCT subject_name AS n FROM kg_triple
            UNION
            SELECT DISTINCT object_name FROM kg_triple
            ORDER BY 1
            LIMIT 30
            """
        ).fetchall()
    sample = [r[0] for r in names if r[0]]
    lines = [
        f"[Graph summary] Distinct entity keys (approx): {n_ent}, Triple rows: {n_rel}.",
        "Sample names: " + ", ".join(sample[:15]) + ("..." if len(sample) > 15 else ""),
    ]
    return "\n".join(lines)


def ping(pool) -> bool:
    """
    Return True if graph storage is reachable: AGE (when enabled) or ``kg_triple`` exists.

    Used by ``graph_store.get_driver()`` for PostgreSQL.
    """
    from src.core.config import get_settings
    from src.infrastructure.persistence.postgres_age_setup import age_is_ready
    from src.infrastructure.persistence.postgres_age_graph import ping_age

    s = get_settings()
    try:
        with pool.connection() as conn:
            conn.execute("SELECT 1")
    except Exception as e:
        logger.debug("postgres ping failed: %s", e)
        return False
    if s.graph_backend == "age" and age_is_ready():
        return ping_age(pool, s.age_graph_name)
    try:
        with pool.connection() as conn:
            conn.execute("SELECT 1 FROM kg_triple LIMIT 1")
        return True
    except Exception:
        return False
