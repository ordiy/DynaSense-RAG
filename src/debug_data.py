"""
Read-only helpers for PostgreSQL inspection (debug UI).

Design: no arbitrary Cypher; bounded limits; separate from rag_core to avoid
pulling LLM/vector logic into data browsing.
"""
from __future__ import annotations

from typing import Any

MAX_ROWS_PAGE = 80
MAX_SCAN_ROWS = 2000  # cap SQL result size for safety


def postgres_storage_summary() -> dict[str, Any]:
    """
    Row counts for unified PostgreSQL storage (parents, children, vectors, triples).

    Returns a small dict suitable for ``GET /api/debug/pg/summary`` when the pool
    is initialized (typically after ``rag_core`` import).
    """
    from src.core.config import get_settings

    s = get_settings()
    if not s.database_url:
        return {"backend": "no_database_url", "hint": "Set DATABASE_URL."}
    try:
        from src.infrastructure.persistence.postgres_connection import get_pool
    except Exception as e:
        return {"backend": "postgresql", "error": f"pool: {e}"}
    try:
        pool = get_pool()
    except Exception as e:
        return {"backend": "postgresql", "error": str(e)}
    with pool.connection() as conn:
        np = conn.execute(
            "SELECT COUNT(*) FROM kb_doc WHERE doc->>'type' = 'parent'"
        ).fetchone()[0]
        nc = conn.execute(
            "SELECT COUNT(*) FROM kb_doc WHERE doc->>'type' = 'child'"
        ).fetchone()[0]
        ne = conn.execute("SELECT COUNT(*) FROM kb_embedding").fetchone()[0]
        try:
            nt = conn.execute("SELECT COUNT(*) FROM kg_triple").fetchone()[0]
        except Exception:
            nt = None
    out = {
        "backend": "postgresql",
        "kb_doc_parent_rows": int(np),
        "kb_doc_child_rows": int(nc),
        "kb_embedding_rows": int(ne),
        "kg_triple_rows": nt,
    }
    try:
        from src.infrastructure.persistence.postgres_age_setup import age_is_ready
        from src.core.config import get_settings as _gs

        gs = _gs()
        out["apache_age_ready"] = bool(age_is_ready())
        out["age_graph_name"] = gs.age_graph_name if age_is_ready() else None
    except Exception:
        pass
    return out


def kb_embedding_summary() -> dict[str, Any]:
    """Row count for ``kb_embedding`` (pgvector)."""
    from src.core.config import get_settings

    s = get_settings()
    if not s.database_url:
        return {"error": "DATABASE_URL not set", "row_count": None}
    try:
        from src.infrastructure.persistence.postgres_connection import get_pool

        pool = get_pool()
        with pool.connection() as conn:
            n = conn.execute("SELECT COUNT(*) FROM kb_embedding").fetchone()[0]
        return {"table": "kb_embedding", "row_count": int(n)}
    except Exception as e:
        return {"error": str(e), "row_count": None}


def kb_embedding_rows(
    limit: int,
    offset: int,
    source_substring: str | None,
    parent_id_substring: str | None,
) -> dict[str, Any]:
    """Paginated rows from ``kb_embedding`` with optional JSONB ``meta`` filters."""
    from src.core.config import get_settings

    s = get_settings()
    if not s.database_url:
        return {"error": "DATABASE_URL not set", "rows": [], "total_after_filter": 0}

    limit = max(1, min(int(limit), MAX_ROWS_PAGE))
    offset = max(0, int(offset))
    src_q = (source_substring or "").strip().lower()
    pid_q = (parent_id_substring or "").strip().lower()

    from src.infrastructure.persistence.postgres_connection import get_pool

    pool = get_pool()
    where_parts: list[str] = []
    params: list[Any] = []
    if src_q:
        where_parts.append("LOWER(meta->>'source') LIKE %s")
        params.append(f"%{src_q}%")
    if pid_q:
        where_parts.append("LOWER(meta->>'parent_id') LIKE %s")
        params.append(f"%{pid_q}%")
    where_sql = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""
    count_sql = f"SELECT COUNT(*) FROM kb_embedding{where_sql}"
    page_sql = f"""
            SELECT id, content, meta
            FROM kb_embedding
            {where_sql}
            ORDER BY id
            LIMIT %s OFFSET %s
            """

    with pool.connection() as conn:
        if params:
            total = int(conn.execute(count_sql, params).fetchone()[0])
            rows = conn.execute(page_sql, (*params, limit, offset)).fetchall()
        else:
            total = int(conn.execute(count_sql).fetchone()[0])
            rows = conn.execute(page_sql, (limit, offset)).fetchall()
        capped = min(total, MAX_SCAN_ROWS)

    rows_out: list[dict[str, Any]] = []
    for rid, content, meta in rows:
        text = str(content or "")
        preview = text[:500] + ("…" if len(text) > 500 else "")
        md = dict(meta) if isinstance(meta, dict) else {}
        rows_out.append(
            {
                "id": rid,
                "text_preview": preview,
                "text_len": len(text),
                "metadata": md,
                "has_vector": True,
            }
        )

    return {
        "table": "kb_embedding",
        "total_after_filter": total,
        "limit": limit,
        "offset": offset,
        "rows": rows_out,
        "truncated_scan": total > MAX_SCAN_ROWS,
        "capped_rows": capped,
    }


def graph_summary_text() -> tuple[str | None, str | None]:
    """Returns (summary_text, error_message)."""
    try:
        from src.graph_store import global_graph_summary, get_driver
    except Exception as e:
        return None, f"Import error: {e}"

    if not get_driver():
        return None, "Graph backend not available (set DATABASE_URL and ensure PostgreSQL is up)."
    text = global_graph_summary()
    if not text.strip():
        return None, "Empty summary (no data or connection issue)."
    return text, None


def graph_keyword_search(keywords: list[str], limit: int) -> tuple[list[dict[str, Any]], str | None]:
    try:
        from src.graph_store import get_driver, query_relationships_by_keywords
    except Exception as e:
        return [], f"Import error: {e}"

    lim = max(1, min(int(limit), 100))
    kws = [k.strip() for k in keywords if k and k.strip()][:20]
    if not kws:
        return [], "Provide at least one keyword (length > 1)."

    if not get_driver():
        return [], "Graph backend not available."

    rows = query_relationships_by_keywords(kws, limit=lim)
    return rows, None
