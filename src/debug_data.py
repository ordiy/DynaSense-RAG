"""
Read-only helpers for LanceDB / Neo4j inspection (debug UI).

Design: no arbitrary Cypher; bounded limits; separate from rag_core to avoid
pulling LLM/vector logic into data browsing.
"""
from __future__ import annotations

import os
from typing import Any

import lancedb

# Keep in sync with rag_core defaults
LANCEDB_URI = os.environ.get("LANCEDB_URI", "./data/lancedb_store")
TABLE_NAME = os.environ.get("LANCEDB_TABLE", "knowledge_base")

MAX_ROWS_PAGE = 80
MAX_SCAN_ROWS = 2000  # cap pandas materialization for safety


def lancedb_summary() -> dict[str, Any]:
    """Table names and row counts for the configured LanceDB path."""
    os.makedirs(os.path.dirname(LANCEDB_URI) or ".", exist_ok=True)
    db = lancedb.connect(LANCEDB_URI)
    tables: list[dict[str, Any]] = []
    for name in sorted(db.table_names()):
        try:
            t = db.open_table(name)
            n = int(t.count_rows())
        except Exception as e:
            n = None
            err = str(e)
        else:
            err = None
        tables.append({"name": name, "row_count": n, "error": err})
    return {"lancedb_uri": LANCEDB_URI, "tables": tables}


def _metadata_dict(m: Any) -> dict[str, Any]:
    if m is None:
        return {}
    if isinstance(m, dict):
        return dict(m)
    if hasattr(m, "as_py"):
        try:
            v = m.as_py()
            return dict(v) if isinstance(v, dict) else {"value": str(v)}
        except Exception:
            return {"value": str(m)}
    return {"value": str(m)}


def lancedb_rows(
    table_name: str | None,
    limit: int,
    offset: int,
    source_substring: str | None,
    parent_id_substring: str | None,
) -> dict[str, Any]:
    """
    Return a page of rows with text preview and metadata (JSON-serializable).
    Filters are simple substring matches on metadata fields.
    """
    name = table_name or TABLE_NAME
    limit = max(1, min(int(limit), MAX_ROWS_PAGE))
    offset = max(0, int(offset))

    db = lancedb.connect(LANCEDB_URI)
    if name not in db.table_names():
        return {"error": f"Table not found: {name}", "rows": [], "total_after_filter": 0}

    tbl = db.open_table(name)
    df = tbl.to_pandas()
    if len(df) > MAX_SCAN_ROWS:
        df = df.head(MAX_SCAN_ROWS)

    src_q = (source_substring or "").strip().lower()
    pid_q = (parent_id_substring or "").strip().lower()

    if src_q and "metadata" in df.columns:
        def _src_match(m):
            md = _metadata_dict(m)
            s = str(md.get("source", "")).lower()
            return src_q in s

        df = df[df["metadata"].apply(_src_match)]

    if pid_q and "metadata" in df.columns:
        def _pid_match(m):
            md = _metadata_dict(m)
            p = str(md.get("parent_id", "")).lower()
            return pid_q in p

        df = df[df["metadata"].apply(_pid_match)]

    total = len(df)
    page = df.iloc[offset : offset + limit]

    rows_out: list[dict[str, Any]] = []
    for _, row in page.iterrows():
        text = str(row.get("text", "") or "")
        preview = text[:500] + ("…" if len(text) > 500 else "")
        md = _metadata_dict(row.get("metadata"))
        vec = row.get("vector")
        rows_out.append(
            {
                "text_preview": preview,
                "text_len": len(text),
                "metadata": md,
                "has_vector": vec is not None,
                "vector_dim": len(vec) if hasattr(vec, "__len__") and vec is not None else None,
            }
        )

    return {
        "table": name,
        "total_after_filter": total,
        "limit": limit,
        "offset": offset,
        "rows": rows_out,
        "truncated_scan": tbl.count_rows() > MAX_SCAN_ROWS,
    }


def neo4j_summary_text() -> tuple[str | None, str | None]:
    """Returns (summary_text, error_message)."""
    try:
        from src.graph_store import global_graph_summary, get_driver
    except Exception as e:
        return None, f"Import error: {e}"

    if not get_driver():
        return None, "Neo4j driver not available (check NEO4J_URI / password or service)."
    text = global_graph_summary()
    if not text.strip():
        return None, "Empty summary (no data or connection issue)."
    return text, None


def neo4j_keyword_search(keywords: list[str], limit: int) -> tuple[list[dict[str, Any]], str | None]:
    try:
        from src.graph_store import get_driver, query_relationships_by_keywords
    except Exception as e:
        return [], f"Import error: {e}"

    if not get_driver():
        return [], "Neo4j driver not available."

    lim = max(1, min(int(limit), 100))
    kws = [k.strip() for k in keywords if k and k.strip()][:20]
    if not kws:
        return [], "Provide at least one keyword (length > 1)."

    rows = query_relationships_by_keywords(kws, limit=lim)
    return rows, None
