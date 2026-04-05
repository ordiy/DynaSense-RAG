"""
PostgreSQL connection pool for the unified storage backend.

Purpose: one process-wide pool so ingest + retrieval + graph share connections safely.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)

_pool: ConnectionPool | None = None


def get_pool() -> "ConnectionPool":
    """Return the singleton pool; raises if ``init_pool`` was not called."""
    if _pool is None:
        raise RuntimeError("PostgreSQL pool not initialized. Call init_pool(database_url) first.")
    return _pool


def _configure_connection(conn) -> None:
    """Prepare each pooled connection (Apache AGE ``LOAD`` / ``search_path`` when present)."""
    try:
        from src.infrastructure.persistence.postgres_age_setup import prepare_connection

        prepare_connection(conn)
        # Pool requires connections not left mid-transaction after configure.
        conn.commit()
    except Exception as e:
        logger.debug("pool configure: %s", e)
        try:
            conn.rollback()
        except Exception:
            pass


def init_pool(database_url: str) -> None:
    """
    Create the connection pool (idempotent: replaces existing pool if URL changes in tests).

    Uses psycopg3 ConnectionPool with conservative defaults suitable for API workers.
    """
    global _pool
    from psycopg_pool import ConnectionPool

    if _pool is not None:
        try:
            _pool.close()
        except Exception as e:
            logger.debug("pool close: %s", e)
        _pool = None

    # min_size=1 keeps a warm connection; max_size works for typical FastAPI concurrency
    _pool = ConnectionPool(
        conninfo=database_url,
        min_size=1,
        max_size=10,
        kwargs={"autocommit": False},
        configure=_configure_connection,
        open=True,
    )
    logger.info("PostgreSQL connection pool ready.")


def close_pool() -> None:
    """Release pool resources (tests / shutdown)."""
    global _pool
    if _pool is not None:
        try:
            _pool.close()
        except Exception as e:
            logger.debug("pool close: %s", e)
        _pool = None
