"""
Graph persistence port (PostgreSQL AGE / relational triples).

Agent / use-case code should depend on this protocol, not on concrete driver types.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypedDict


class GraphRelationship(TypedDict, total=False):
    subject: str
    predicate: str
    object: str
    chunk_id: str
    source: str


class IGraphRepository(ABC):
    """Read-side operations used by debug UI and (optionally) hybrid retrieval."""

    @abstractmethod
    def query_relationships_by_keywords(self, keywords: list[str], limit: int = 40) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    def global_summary_text(self) -> str:
        ...
