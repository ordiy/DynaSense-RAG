"""
Neo4j adapter implementing :class:`IGraphRepository`.

Delegates to existing ``src.graph_store`` (incremental migration — logic stays there for now).
"""
from __future__ import annotations

from typing import Any

from src.domain.interfaces.graph_repository import IGraphRepository
from src import graph_store


class Neo4jGraphRepository(IGraphRepository):
    def query_relationships_by_keywords(self, keywords: list[str], limit: int = 40) -> list[dict[str, Any]]:
        return graph_store.query_relationships_by_keywords(keywords, limit=limit)

    def global_summary_text(self) -> str:
        return graph_store.global_graph_summary()
