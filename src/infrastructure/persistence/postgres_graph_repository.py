"""
PostgreSQL graph adapter implementing :class:`IGraphRepository`.

Delegates to ``src.graph_store`` (AGE or relational ``kg_triple``).
"""
from __future__ import annotations

from typing import Any

from src.domain.interfaces.graph_repository import IGraphRepository
from src import graph_store


class PostgresGraphRepository(IGraphRepository):
    def query_relationships_by_keywords(self, keywords: list[str], limit: int = 40) -> list[dict[str, Any]]:
        return graph_store.query_relationships_by_keywords(keywords, limit=limit)

    def global_summary_text(self) -> str:
        return graph_store.global_graph_summary()
