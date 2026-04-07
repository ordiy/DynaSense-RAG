"""
Apache AGE bootstrap: extension, ``LOAD``, ``search_path``, and graph creation.

AGE runs Cypher inside PostgreSQL via ``cypher(graph_name, query)``. Each new
connection should set ``search_path`` and load the library (see AGE manual).
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_age_ready: bool = False


def age_is_ready() -> bool:
    return _age_ready


def set_age_ready(value: bool) -> None:
    global _age_ready
    _age_ready = value


def prepare_connection(conn) -> None:
    """Per-connection AGE setup (safe on every checkout)."""
    try:
        conn.execute("LOAD 'age'")
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        return  # AGE not installed; skip search_path setup
    try:
        conn.execute("SET search_path = ag_catalog, public")
    except Exception as e:
        logger.debug("AGE search_path: %s", e)
        try:
            conn.rollback()
        except Exception:
            pass


def ensure_age_extension_and_graph(pool, graph_name: str) -> bool:
    """
    Create extension + graph if missing. Returns True if Cypher graph is usable.

    If the image has no AGE build, returns False and callers create ``kg_triple`` instead.
    """
    global _age_ready
    try:
        with pool.connection() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS age")
            prepare_connection(conn)
            ext = conn.execute(
                "SELECT 1 FROM pg_extension WHERE extname = 'age'"
            ).fetchone()
            if not ext:
                logger.warning("Extension 'age' not present after CREATE.")
                conn.rollback()
                _age_ready = False
                return False
            exists = conn.execute(
                "SELECT 1 FROM ag_catalog.ag_graph WHERE name = %s",
                (graph_name,),
            ).fetchone()
            if not exists:
                if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", graph_name):
                    raise ValueError(f"Invalid graph name: {graph_name!r}")
                conn.execute(f"SELECT create_graph('{graph_name}')")
                logger.info("AGE graph created: %s", graph_name)
            conn.commit()
        _age_ready = True
        return True
    except Exception as e:
        logger.warning("Apache AGE unavailable (%s); using relational kg_triple.", e)
        _age_ready = False
        return False


def run_cypher(
    pool,
    graph_name: str,
    cypher_body: str,
    result_columns: list[tuple[str, str]],
) -> list[tuple[Any, ...]]:
    """
    ``SELECT * FROM cypher('graph_name', $$...$$) AS (...)``.

    The graph name must be a SQL literal (AGE does not accept it as a bound
    parameter in all versions), so we validate ``[a-zA-Z_][a-zA-Z0-9_]*``.
    """
    if not result_columns:
        raise ValueError("result_columns required")
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", graph_name):
        raise ValueError(f"Invalid AGE graph name: {graph_name!r}")
    as_clause = ", ".join(f"{name} {typ}" for name, typ in result_columns)
    # AGE expects the second argument as a dollar-quoted *SQL literal*, not a bound param.
    if "$$" in cypher_body:
        dq = f"$cypher${cypher_body}$cypher$"
    else:
        dq = f"$${cypher_body}$$"
    sql = f"SELECT * FROM cypher('{graph_name}', {dq}) AS ({as_clause})"
    with pool.connection() as conn:
        prepare_connection(conn)
        cur = conn.execute(sql)
        rows = cur.fetchall()
        conn.commit()
    return list(rows)
