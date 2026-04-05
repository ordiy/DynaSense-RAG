"""Backward-compatible re-export — storage is JSONB (``kb_doc``)."""

from __future__ import annotations

from src.infrastructure.persistence.postgres_jsonb_collection import (
    PostgresChunkCollection,
    PostgresJsonbDocCollection,
)

__all__ = ["PostgresChunkCollection", "PostgresJsonbDocCollection"]
