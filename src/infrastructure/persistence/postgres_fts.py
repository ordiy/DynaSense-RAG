"""
PostgreSQL full-text search over ``kb_embedding`` child chunks.

Replaces the in-memory BM25 cache (``rank_bm25 + threading.Lock``) used by
``hybrid_rag.bm25_parent_documents``.  No in-process state; no cache
invalidation needed on ingest.

Text-search config: ``'simple'``
  - Language-agnostic; no stemming; CJK characters become individual tokens.
  - Always available (built-in PostgreSQL config — no extensions required).
  - Adequate for mixed Chinese/English recall; Jina reranker handles precision.

GIN index ``kb_embedding_fts_idx`` is created by ``postgres_schema.ensure_schema``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_FTS_CFG = "simple"  # Language-agnostic, CJK-compatible

# SQL: FTS child chunk search, returns id/content/meta/rank
_FTS_SQL = """
SELECT  id,
        content,
        meta,
        ts_rank(to_tsvector(%(cfg)s, content),
                plainto_tsquery(%(cfg)s, %(q)s))  AS rank
FROM    kb_embedding
WHERE   to_tsvector(%(cfg)s, content)
        @@ plainto_tsquery(%(cfg)s, %(q)s)
ORDER   BY rank DESC
LIMIT   %(k)s
"""


def fulltext_search_children(
    query: str,
    pool,
    *,
    top_k: int = 12,
) -> list[tuple[str, str, dict, float]]:
    """
    Return at most *top_k* matching child chunks as ``(id, content, meta, rank)`` tuples.

    Returns an empty list when:
    - *query* is blank (``plainto_tsquery`` would error on empty string)
    - the ``kb_embedding`` table is empty or has no FTS match
    - any database error occurs (logged as WARNING)
    """
    if not query or not query.strip():
        return []
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(_FTS_SQL, {"cfg": _FTS_CFG, "q": query.strip(), "k": top_k})
                return cur.fetchall()
    except Exception as exc:
        logger.warning("FTS child search failed: %s", exc)
        return []
