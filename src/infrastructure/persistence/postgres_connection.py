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
    """
    Prepare each pooled connection once at creation time.

    Runs Apache AGE LOAD / search_path setup and registers pgvector type adapters
    so callers never need to call register_vector(conn) per-method.
    """
    try:
        from src.infrastructure.persistence.postgres_age_setup import prepare_connection

        prepare_connection(conn)
    except Exception as e:
        logger.debug("pool configure (AGE): %s", e)
    finally:
        # AGE LOAD may fail and leave the connection in INERROR; always clean up.
        try:
            conn.rollback()
        except Exception:
            pass

    try:
        from pgvector.psycopg import register_vector

        register_vector(conn)
    except Exception as e:
        logger.debug("pool configure (pgvector): %s", e)

    # Pool requires connections not left mid-transaction after configure.
    try:
        conn.commit()
    except Exception as e:
        logger.debug("pool configure commit: %s", e)
        try:
            conn.rollback()
        except Exception:
            pass


def init_pool(database_url: str) -> None:
    """
    Create the connection pool (idempotent: replaces existing pool if URL changes in tests).

    Uses psycopg3 ConnectionPool; pool size is read from Settings so it can be
    overridden via the ``PG_POOL_MAX_SIZE`` environment variable.
    """
    global _pool
    from psycopg_pool import ConnectionPool
    from src.core.config import get_settings

    if _pool is not None:
        try:
            _pool.close()
        except Exception as e:
            logger.debug("pool close: %s", e)
        _pool = None

    max_size = get_settings().pg_pool_max_size
    # min_size=1 keeps a warm connection; max_size from settings
    _pool = ConnectionPool(
        conninfo=database_url,
        min_size=1,
        max_size=max_size,
        kwargs={"autocommit": False},
        configure=_configure_connection,
        open=True,
    )
    logger.info("PostgreSQL connection pool ready (max_size=%s).", max_size)


def close_pool() -> None:
    """Release pool resources (tests / shutdown)."""
    global _pool
    if _pool is not None:
        try:
            _pool.close()
        except Exception as e:
            logger.debug("pool close: %s", e)
        _pool = None
