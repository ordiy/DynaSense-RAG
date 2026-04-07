"""
JSONB-backed document store — replaces MongoMock / ``doc_parent``+``doc_child`` for PostgreSQL.

Each row is ``(id, doc)`` where ``doc`` is the full legacy Mongo document as JSONB:
``{ "id", "type": "parent"|"child", "source", "parent_id"?, "content"?, "full_content"? }``.

Purpose: one table, flexible fields, GIN-friendly filters on ``doc->>'type'``.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

from psycopg.types.json import Json

logger = logging.getLogger(__name__)


class PostgresJsonbDocCollection:
    """
    Subset of pymongo API used by ``hybrid_rag`` / ``rag_core`` — backed by ``kb_doc``.
    """

    def __init__(self, pool) -> None:
        self._pool = pool

    def insert_one(self, doc: dict[str, Any]) -> None:
        did = doc.get("id")
        if not did:
            raise ValueError("document must have id")
        with self._pool.connection() as conn:
            conn.execute(
                """
                INSERT INTO kb_doc (id, doc)
                VALUES (%s, %s)
                ON CONFLICT (id) DO UPDATE SET doc = EXCLUDED.doc
                """,
                (did, Json(doc)),
            )
            conn.commit()

    def insert_many(self, docs: list[dict[str, Any]]) -> None:
        params = [(d["id"], Json(d)) for d in docs if d.get("id")]
        if not params:
            return
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO kb_doc (id, doc)
                    VALUES (%s, %s)
                    ON CONFLICT (id) DO UPDATE SET doc = EXCLUDED.doc
                    """,
                    params,
                )
            conn.commit()

    def find(self, filt: dict[str, Any]) -> Iterator[dict[str, Any]]:
        with self._pool.connection() as conn:
            if filt.get("type") == "child":
                res = conn.execute(
                    """
                    SELECT doc FROM kb_doc
                    WHERE doc->>'type' = 'child'
                    ORDER BY id
                    """
                )
                for (doc,) in res.fetchall():
                    yield dict(doc) if not isinstance(doc, dict) else doc
            elif filt.get("type") == "parent" and "$in" in filt.get("id", {}):
                ids = filt["id"]["$in"]
                if not ids:
                    return
                res = conn.execute(
                    """
                    SELECT doc FROM kb_doc
                    WHERE doc->>'type' = 'parent' AND doc->>'id' = ANY(%s)
                    """,
                    (list(ids),),
                )
                for (doc,) in res.fetchall():
                    yield dict(doc) if not isinstance(doc, dict) else doc
            else:
                logger.warning("PostgresJsonbDocCollection.find: unsupported filter %s", filt)

    def find_one(self, filt: dict[str, Any]) -> dict[str, Any] | None:
        if filt.get("type") != "parent" or "id" not in filt:
            return None
        pid = filt["id"]
        with self._pool.connection() as conn:
            row = conn.execute(
                """
                SELECT doc FROM kb_doc
                WHERE doc->>'type' = 'parent' AND doc->>'id' = %s
                LIMIT 1
                """,
                (pid,),
            ).fetchone()
            if not row:
                return None
            doc = row[0]
            return dict(doc) if not isinstance(doc, dict) else doc

    def delete_many(self, filt: dict[str, Any]) -> None:
        """Full KB reset — delegates to :func:`postgres_schema.truncate_kb_storage`."""
        from src.infrastructure.persistence.postgres_schema import truncate_kb_storage

        truncate_kb_storage(self._pool)


# Backward-compatible name for ``rag_core`` imports
PostgresChunkCollection = PostgresJsonbDocCollection
