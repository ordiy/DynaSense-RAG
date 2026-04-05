"""
Vector retrieval over ``kb_embedding`` using pgvector cosine distance.

Purpose: LangChain-compatible ``similarity_search`` + ``as_retriever().invoke()`` used by ``rag_core``.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document
from pgvector.psycopg import register_vector
from psycopg.types.json import Json

logger = logging.getLogger(__name__)


class PostgresVectorStore:
    """
    Minimal vector store backed by PostgreSQL + pgvector.

    Uses cosine distance (``<=>``) to match Vertex embedding space behavior.
    """

    def __init__(self, pool, embedding_model) -> None:
        self._pool = pool
        self._embed = embedding_model

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        """Embed the question and fetch the *k* nearest child chunks by cosine distance."""
        qvec = self._embed.embed_query(query)
        with self._pool.connection() as conn:
            register_vector(conn)
            rows = conn.execute(
                """
                SELECT id, content, meta, embedding <=> %s::vector AS dist
                FROM kb_embedding
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (qvec, qvec, k),
            ).fetchall()
        out: list[Document] = []
        for row in rows:
            _cid, content, meta, _dist = row
            md = dict(meta) if meta is not None else {}
            out.append(Document(page_content=content, metadata=md))
        return out

    def as_retriever(self, *, search_kwargs: dict[str, Any] | None = None):
        """Return an object with ``invoke(query) -> list[Document]`` (LangChain retriever protocol)."""
        sk = search_kwargs or {}
        k = int(sk.get("k", 4))

        class _Retriever:
            def __init__(self, store: PostgresVectorStore, kk: int) -> None:
                self._store = store
                self._k = kk

            def invoke(self, query: str) -> list[Document]:
                return self._store.similarity_search(query, k=self._k)

        return _Retriever(self, k)

    def add_embedding_rows(
        self,
        rows: list[tuple[str, str, dict, list[float]]],
    ) -> None:
        """
        Bulk insert chunk embeddings.

        Each tuple: (chunk_id, plain_text, metadata_dict, embedding_vector).
        """
        if not rows:
            return
        with self._pool.connection() as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                for rid, text, meta, emb in rows:
                    cur.execute(
                        """
                        INSERT INTO kb_embedding (id, content, meta, embedding)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET
                            content = EXCLUDED.content,
                            meta = EXCLUDED.meta,
                            embedding = EXCLUDED.embedding
                        """,
                        (rid, text, Json(meta), emb),
                    )
            conn.commit()
