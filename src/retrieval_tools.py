"""
LangChain ``@tool`` wrappers for retrieval backends.

Used by ``agentic_rag.run_agentic_retrieval`` (ReAct loop).

Document accumulation
---------------------
Each tool call stores *full* parent Documents in a **thread-local registry**
keyed by a content hash (deduplication).  After the agent loop, callers
retrieve them via ``get_collected_docs()``.

This design keeps tool response strings short (LLM context budget) while
preserving full document text for the downstream Jina rerank + grade + generate
nodes.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from typing import TYPE_CHECKING

from langchain_core.tools import tool

if TYPE_CHECKING:
    from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ── Thread-local document registry ───────────────────────────────────────────
_ctx = threading.local()


def reset_doc_registry() -> None:
    """Clear the registry before each agentic retrieval run."""
    _ctx.docs = {}
    _ctx.logs = []


def get_collected_docs() -> list[Document]:
    """Return all unique Documents accumulated across tool calls in this thread."""
    return list(getattr(_ctx, "docs", {}).values())


def get_tool_logs() -> list[str]:
    return list(getattr(_ctx, "logs", []))


def _doc_key(doc: Document) -> str:
    return hashlib.sha256(doc.page_content[:400].encode(errors="ignore")).hexdigest()[:20]


def _register(docs: list[Document], label: str) -> None:
    if not hasattr(_ctx, "docs"):
        _ctx.docs = {}
    if not hasattr(_ctx, "logs"):
        _ctx.logs = []
    added = 0
    for doc in docs:
        k = _doc_key(doc)
        if k not in _ctx.docs:
            _ctx.docs[k] = doc
            added += 1
    _ctx.logs.append(f"{label}: +{added} new docs (registry size={len(_ctx.docs)}).")


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def vector_search(query: str, top_k: int = 5) -> str:
    """
    Semantic similarity search over the knowledge base.

    Best for: conceptual questions, explanations, definitions, narrative facts.
    Returns a short preview of the top result so the agent can decide next steps.
    """
    from src.rag_core import retrieve_parent_documents_expanded, jina_rerank

    try:
        docs, _ = retrieve_parent_documents_expanded(query, dense_k=min(top_k * 2, 20))
        reranked = jina_rerank(query, docs, top_n=top_k)
    except Exception as exc:
        logger.warning("vector_search failed: %s", exc)
        return f"vector_search error: {exc}"

    _register(reranked, f"vector_search('{query[:60]}')")
    if not reranked:
        return "No relevant passages found for this query."
    preview = reranked[0].page_content[:500].replace("\n", " ")
    return (
        f"Found {len(reranked)} passage(s). "
        f"Top passage preview (first 500 chars):\n{preview}"
    )


@tool
def fulltext_search(query: str, top_k: int = 5) -> str:
    """
    Keyword-based full-text search over the knowledge base (PostgreSQL tsvector).

    Best for: specific entity names, product codes, dates, or when semantic search
    misses exact terminology.  Complements vector_search with lexical recall.
    """
    from src.hybrid_rag import fts_parent_documents

    try:
        docs, logs = fts_parent_documents(query, top_child=min(top_k * 2, 24))
    except Exception as exc:
        logger.warning("fulltext_search failed: %s", exc)
        return f"fulltext_search error: {exc}"

    _register(docs[:top_k], f"fulltext_search('{query[:60]}')")
    if not docs:
        return "No keyword matches found for this query."
    preview = docs[0].page_content[:500].replace("\n", " ")
    return (
        f"Found {len(docs[:top_k])} passage(s) via keyword search. "
        f"Top passage preview:\n{preview}"
    )
