"""
Hybrid RAG MVP: Query Router + Dense/BM25 vector path + PostgreSQL graph path + unified rerank.

Implements readme-v2-1.md topology at MVP scope:
  - Intent router: VECTOR | GRAPH | GLOBAL | HYBRID (LLM structured output)
  - VECTOR: dense retrieval + BM25 over child chunks (parent expansion) -> fusion rerank Top-5
  - GRAPH: keyword-based subgraph -> linearized text as context
  - GLOBAL: graph summary statistics + optional dense hint
  - HYBRID: concurrent vector + graph candidates -> single rerank Top-5

Downstream grading/generation reuses rag_core.grade_documents_node / generate_node logic
by invoking those functions on a synthetic AgentState.
"""
from __future__ import annotations

import hashlib
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Iterator, Literal

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.graph_store import (
    get_driver,
    global_graph_summary,
    linearize_rows,
    merge_triple,
    query_relationships_by_keywords,
)
from src.core.query_anchors import filter_documents_by_query_anchors
from src.rag_core import (
    AgentState,
    _is_analysis_query,
    collection,
    grade_documents_node,
    invoke_rag_app,
    jina_rerank,
    langgraph_stream_log_enabled,
    llm,
    retrieve_parent_documents_expanded,
    vectorstore,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BM25 index cache — built once from all child chunks, invalidated on ingest.
# ---------------------------------------------------------------------------
_bm25_cache: tuple[list[dict], Any] | None = None  # (children_list, BM25Okapi)
_bm25_lock = threading.Lock()


def invalidate_bm25_cache() -> None:
    """Drop the cached BM25 index so the next query rebuilds it from fresh data."""
    global _bm25_cache
    with _bm25_lock:
        _bm25_cache = None
    logger.debug("BM25 cache invalidated.")


def _get_or_build_bm25() -> tuple[list[dict], Any] | None:
    """
    Return the cached (children, BM25Okapi) pair, building it on first call.
    Thread-safe double-checked locking.
    """
    global _bm25_cache
    if _bm25_cache is not None:
        return _bm25_cache
    with _bm25_lock:
        if _bm25_cache is not None:
            return _bm25_cache
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("BM25: rank_bm25 not installed; skipping.")
            return None
        children = list(collection.find({"type": "child"}))
        if not children:
            return None
        tokenized = [_tokenize(str(c.get("content", ""))) for c in children]
        if not any(tokenized):
            return None
        _bm25_cache = (children, BM25Okapi(tokenized))
        logger.debug("BM25 index built: %d child chunks.", len(children))
        return _bm25_cache


class RouteDecision(BaseModel):
    route: Literal["VECTOR", "GRAPH", "GLOBAL", "HYBRID"] = Field(
        description="VECTOR=facts/semantic in text; GRAPH=multi-hop relations; "
        "GLOBAL=whole-corpus summary; HYBRID=both text and relations needed."
    )
    reason: str = Field(default="", description="One short line (English or Chinese).")


class TripleItem(BaseModel):
    subject: str = Field(description="Entity name, e.g. company or person")
    predicate: str = Field(description="Relationship type in few words")
    object: str = Field(description="Target entity or value")


class TripleExtraction(BaseModel):
    triples: list[TripleItem] = Field(default_factory=list, description="3-12 triples when possible")


class KeywordList(BaseModel):
    keywords: list[str] = Field(default_factory=list, description="2-8 short keywords for graph lookup")


ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a query router for a Hybrid RAG system.\n"
            "Choose exactly one label:\n"
            "- VECTOR: user wants passage-level facts, definitions, troubleshooting, or lexical detail in documents.\n"
            "- GRAPH: user asks about relationships (who owns whom, subsidiaries, investment structure, dependencies, '关联方').\n"
            "- GLOBAL: user asks for overview of the whole knowledge base (总结/概况/整体).\n"
            "- HYBRID: needs both narrative evidence and structured relations (e.g. '介绍X并说明其与Y的关系').\n"
            "Output JSON only matching the schema.",
        ),
        ("human", "User question:\n{question}"),
    ]
)

KEYWORD_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Extract 2-8 short keywords for searching a knowledge graph (entity names, relation cues). "
            "Prefer Chinese entity tokens if present. JSON only.",
        ),
        ("human", "Question:\n{question}"),
    ]
)

EXTRACT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "From the passage, extract factual triples (subject, predicate, object). "
            "Focus on organizations, persons, percentages, locations, contractual relations. "
            "Skip generic boilerplate. JSON only.",
        ),
        ("human", "Passage:\n{text}"),
    ]
)

_router = llm.with_structured_output(RouteDecision)
_kw = llm.with_structured_output(KeywordList)
_extractor = llm.with_structured_output(TripleExtraction)


def _tokenize(s: str) -> list[str]:
    return re.findall(r"[\w\u4e00-\u9fff]+", (s or "").lower())


def _dedupe_docs(docs: list[Document]) -> list[Document]:
    seen: set[str] = set()
    out: list[Document] = []
    for d in docs:
        key = hashlib.sha256(d.page_content[:400].encode("utf-8", errors="ignore")).hexdigest()[:24]
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def route_query(question: str) -> RouteDecision:
    try:
        return _router.invoke(ROUTER_PROMPT.format_messages(question=question))
    except Exception as e:
        logger.warning("Router LLM failed (%s); defaulting to VECTOR", e)
        return RouteDecision(route="VECTOR", reason="router fallback")


def extract_graph_keywords(question: str) -> list[str]:
    try:
        kl = _kw.invoke(KEYWORD_PROMPT.format_messages(question=question))
        kws = [k.strip() for k in (kl.keywords or []) if k and len(k.strip()) > 1]
        return kws[:12]
    except Exception as e:
        logger.warning("Keyword extraction failed: %s", e)
        return _tokenize(question)[:8]


def extract_triples_from_text(text: str) -> list[TripleItem]:
    if not text.strip():
        return []
    try:
        out = _extractor.invoke(EXTRACT_PROMPT.format_messages(text=text[:12000]))
        return list(out.triples or [])[:15]
    except Exception as e:
        logger.warning("Triple extraction failed: %s", e)
        return []


def ingest_chunks_to_graph(chunks: list[str], chunk_ids: list[str], source: str) -> int:
    """
    Offline: LLM triple extraction + merge into PostgreSQL graph.

    Triple extraction is I/O-bound (LLM calls), so chunks are processed in parallel
    using a bounded thread pool (max 5 workers) to respect Vertex AI rate limits
    while cutting ingestion time by ~5×. Returns total triples written.
    """
    if not get_driver():
        return 0

    def _extract_for_chunk(ch: str, cid: str) -> list[tuple[str, str, str, str]]:
        """Extract triples for one chunk; returns list of (subj, pred, obj, cid)."""
        results = []
        for t in extract_triples_from_text(ch):
            results.append((t.subject, t.predicate, t.object, cid))
        return results

    all_triples: list[tuple[str, str, str, str]] = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {
            pool.submit(_extract_for_chunk, ch, cid): cid
            for ch, cid in zip(chunks, chunk_ids)
        }
        for fut in as_completed(futures):
            try:
                all_triples.extend(fut.result())
            except Exception as e:
                logger.warning("Triple extraction failed for chunk %s: %s", futures[fut], e)

    n = 0
    for subj, pred, obj, cid in all_triples:
        try:
            merge_triple(subj, pred, obj, cid, source)
            n += 1
        except Exception as e:
            logger.debug("merge_triple skip: %s", e)

    # New child docs were added — invalidate the BM25 cache so the next query sees them.
    invalidate_bm25_cache()
    return n


def bm25_parent_documents(question: str, top_child: int | None = None) -> tuple[list[Document], list[str]]:
    from src.core.config import get_settings

    if top_child is None:
        top_child = get_settings().hybrid_bm25_top_child
    logs: list[str] = []
    qtok = _tokenize(question)
    if not qtok:
        return [], ["BM25: empty query tokens; skipping."]

    cached = _get_or_build_bm25()
    if cached is None:
        return [], ["BM25: no child chunks in store or rank_bm25 unavailable."]

    children, bm25 = cached
    scores = bm25.get_scores(qtok)
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_child]

    parent_ids_ordered: list[str] = []
    seen: set[str] = set()
    for i in ranked_idx:
        pid = children[i].get("parent_id")
        if pid and pid not in seen:
            seen.add(pid)
            parent_ids_ordered.append(pid)

    parent_records: dict[str, dict] = {}
    if parent_ids_ordered:
        for rec in collection.find({"type": "parent", "id": {"$in": parent_ids_ordered}}):
            pid = rec.get("id")
            if pid:
                parent_records[pid] = rec

    docs: list[Document] = []
    for pid in parent_ids_ordered:
        pr = parent_records.get(pid)
        if not pr:
            continue
        full_text = f"[Source: {pr.get('source', 'Unknown')}]\n[bm25]\n{pr.get('full_content', '')}"
        docs.append(Document(page_content=full_text, metadata={"parent_id": pid, "source": "bm25"}))

    logs.append(f"BM25: top {len(ranked_idx)} child hits -> {len(docs)} parent docs.")
    return docs, logs


def graph_context_documents(question: str) -> tuple[list[Document], list[str]]:
    logs: list[str] = []
    if not get_driver():
        return [], ["Graph: backend not connected."]
    kws = extract_graph_keywords(question)
    if not kws:
        kws = _tokenize(question)[:6]
    rows = query_relationships_by_keywords(kws, limit=40)
    text = linearize_rows(rows)
    if not text.strip():
        return [], [f"Graph: no edges for keywords {kws!r}."]
    doc = Document(
        page_content="[Graph retrieval — linearized triples]\n" + text,
        metadata={"source": "graph"},
    )
    logs.append(f"Graph: {len(rows)} triples linearized (keywords={kws[:6]}).")
    return [doc], logs


def global_context_documents(question: str) -> tuple[list[Document], list[str]]:
    logs: list[str] = []
    if not get_driver():
        return [], ["Global: graph backend not connected."]
    summary = global_graph_summary()
    if not summary:
        return [], ["Global: empty summary."]
    logs.append("Global: graph summary context.")
    # Add light dense anchor so answers can cite text when needed
    anchor_docs: list[Document] = []
    if vectorstore:
        ddocs, _ = retrieve_parent_documents_expanded(question, dense_k=3)
        anchor_docs.extend(ddocs[:2])
    parts = [summary]
    for d in anchor_docs:
        parts.append(d.page_content[:6000])
    merged = "\n\n---\n\n".join(parts)
    return [Document(page_content=merged, metadata={"source": "global"})], logs


def fusion_rerank_docs(
    question: str, candidates: list[Document], top_n: int | None = None
) -> tuple[list[Document], list[str]]:
    """Deduplicate, optional anchor filter, cap pool, Jina cross-encoder rerank; returns ranked Documents."""
    from src.core.config import get_settings

    s = get_settings()
    if top_n is None:
        top_n = s.hybrid_fusion_top_n
    pool_size = s.hybrid_rerank_pool_size

    logs: list[str] = []
    cand = _dedupe_docs(candidates)
    if not cand:
        return [], ["Fusion: no candidates."]
    cand, alogs = filter_documents_by_query_anchors(question, cand)
    logs.extend(alogs)
    if len(cand) > pool_size:
        logs.append(
            f"Fusion: truncating candidate pool from {len(cand)} to {pool_size} "
            f"(hybrid_rerank_pool_size). Increase setting to pass more to reranker."
        )
    pool = cand[:pool_size]
    logs.append(f"Fusion+rerank: pool_size={len(pool)} -> top_n={top_n}")
    ranked = jina_rerank(question, pool, top_n=top_n)
    return ranked, logs


def fusion_rerank_all(question: str, candidates: list[Document], top_n: int | None = None) -> tuple[list[str], list[str]]:
    ranked, logs = fusion_rerank_docs(question, candidates, top_n=top_n)
    return [d.page_content for d in ranked], logs


def collect_vector_path(question: str) -> tuple[list[Document], list[str]]:
    from src.core.config import get_settings

    logs: list[str] = []
    dense_docs, dl = retrieve_parent_documents_expanded(question, dense_k=get_settings().hybrid_dense_k)
    logs.extend(dl)
    bm_docs, bl = bm25_parent_documents(question)
    logs.extend(bl)
    return dense_docs + bm_docs, logs


def gather_route_candidates(question: str, effective_route: str) -> tuple[list[Document], list[str]]:
    """Collect raw candidate Documents for a fixed route (VECTOR / GRAPH / GLOBAL / HYBRID)."""
    logs: list[str] = []
    candidates: list[Document] = []

    if effective_route == "VECTOR":
        cdocs, sub = collect_vector_path(question)
        logs.extend(sub)
        candidates.extend(cdocs)

    elif effective_route == "GRAPH":
        gdocs, sub = graph_context_documents(question)
        logs.extend(sub)
        candidates.extend(gdocs)
        if not gdocs:
            cdocs, sub2 = collect_vector_path(question)
            logs.extend(sub2)
            candidates.extend(cdocs)

    elif effective_route == "GLOBAL":
        gdocs, sub = global_context_documents(question)
        logs.extend(sub)
        candidates.extend(gdocs)

    else:  # HYBRID
        cdocs, sub = collect_vector_path(question)
        logs.extend(sub)
        gdocs, sub2 = graph_context_documents(question)
        logs.extend(sub2)
        candidates.extend(cdocs + gdocs)

    return candidates, logs


def retrieve_hybrid_ranked_documents(question: str, top_n: int = 10) -> tuple[list[Document], dict[str, Any]]:
    """
    Same routing + fusion as chat pipeline, but returns ranked Documents (for Recall/NDCG eval).
    Does not run grader/generator.
    """
    meta: dict[str, Any] = {"logs": ["Hybrid retrieval (eval)"]}
    if not vectorstore:
        meta["error"] = "Vector store empty."
        return [], meta

    decision = route_query(question)
    meta["route"] = decision.route
    meta["router_reason"] = decision.reason

    graph_ok = bool(get_driver())
    effective_route = decision.route
    if not graph_ok and effective_route in ("GRAPH", "GLOBAL", "HYBRID"):
        meta["logs"].append("Graph backend unavailable; falling back to VECTOR-style retrieval.")
        effective_route = "VECTOR"
    meta["effective_route"] = effective_route

    candidates, sublogs = gather_route_candidates(question, effective_route)
    meta["logs"].extend(sublogs)

    ranked, flogs = fusion_rerank_docs(question, candidates, top_n=top_n)
    meta["logs"].extend(flogs)
    return ranked, meta


@dataclass(frozen=True)
class HybridPrepared:
    """State after hybrid retrieval + grader; ready for generate or streaming."""

    state: AgentState
    decision: RouteDecision
    effective_route: str


def prepare_hybrid_chat(question: str) -> HybridPrepared | dict[str, Any]:
    """
    Run router → candidates → fusion rerank → grade. Returns either an early
    response dict (empty KB / no hits) or ``HybridPrepared`` for generate/stream.
    """
    logs: list[str] = ["Hybrid RAG pipeline (MVP)"]

    if not vectorstore:
        logs.append("Vector store empty; cannot answer.")
        return {"answer": "抱歉，知识库为空。请先上传文档。", "context_used": [], "logs": logs}

    decision = route_query(question)
    logs.append(f"Router: {decision.route} — {decision.reason}")

    graph_ok = bool(get_driver())
    effective_route = decision.route
    if not graph_ok and effective_route in ("GRAPH", "GLOBAL", "HYBRID"):
        logs.append("Graph backend unavailable; falling back to VECTOR-style retrieval.")
        effective_route = "VECTOR"

    candidates, sub = gather_route_candidates(question, effective_route)
    logs.extend(sub)

    final_texts, flogs = fusion_rerank_all(question, candidates)
    logs.extend(flogs)

    if not final_texts:
        return {
            "answer": "抱歉，知识库中未能找到与您问题高度相关的信息，为避免产生误导（幻觉），系统已阻断回答。",
            "context_used": [],
            "logs": logs,
            "route": decision.route,
            "effective_route": effective_route,
            "router_reason": decision.reason,
        }

    is_analysis = _is_analysis_query(question)
    state: AgentState = {
        "question": question,
        "documents": final_texts,
        "is_analysis": is_analysis,
        "generation": "",
        "loop_count": 0,
        "logs": logs,
    }
    if langgraph_stream_log_enabled():
        logger.info(
            "[Hybrid pipeline] grading (manual node) | candidates=%s",
            len(final_texts),
        )
    state.update(grade_documents_node(state))
    if langgraph_stream_log_enabled():
        logger.info(
            "[Hybrid pipeline] after grade | documents=%s",
            len(state.get("documents") or []),
        )
    return HybridPrepared(state=state, decision=decision, effective_route=effective_route)


def run_hybrid_chat_pipeline(question: str) -> dict[str, Any]:
    """
    End-to-end hybrid retrieval → grade → generate.

    Uses the same compiled ``rag_app`` LangGraph as the vector-only path so that
    grade and generate nodes are shared.  The hybrid retrieval result is injected
    into ``AgentState`` with ``skip_retrieval=True``; the retrieve node detects this
    and skips vector search, then the graph continues normally through grade → generate.

    Returns the same shape as ``rag_core.run_chat_pipeline`` (legacy key set).
    """
    prepared = prepare_hybrid_chat(question)
    if isinstance(prepared, dict):
        return prepared

    # prepared.state already has graded documents from prepare_hybrid_chat.
    # Feed it back into the full graph with skip_retrieval=True so that the
    # retrieve node is a no-op and grade/generate run through the unified path.
    pre_state = prepared.state
    unified_state: AgentState = {
        **pre_state,  # type: ignore[arg-type]
        "skip_retrieval": True,
    }
    result = invoke_rag_app(unified_state)

    if langgraph_stream_log_enabled():
        logger.info(
            "[Hybrid pipeline] after generate (unified rag_app) | answer_chars=%s",
            len((result.get("generation") or "")),
        )
    decision = prepared.decision
    effective_route = prepared.effective_route
    return {
        "answer": result.get("generation", ""),
        "context_used": result.get("documents", []),
        "logs": result.get("logs", []),
        "route": decision.route,
        "effective_route": effective_route,
        "router_reason": decision.reason,
    }


def iter_hybrid_chat_stream_events(question: str) -> Iterator[dict[str, Any]]:
    """
    Yield SSE-friendly event dicts: ``meta`` (citations, route), ``token`` chunks, ``done`` (full answer).

    Reuses ``prepare_hybrid_chat`` then ``stream_generation_chunks`` instead of ``generate_node``.
    """
    from src.core.citations import build_citations_from_context
    from src.rag_core import stream_generation_chunks

    prepared = prepare_hybrid_chat(question)
    if isinstance(prepared, dict):
        yield {
            "type": "meta",
            "citations": [],
            "route": prepared.get("route"),
            "effective_route": prepared.get("effective_route"),
            "router_reason": prepared.get("router_reason"),
            "logs": prepared.get("logs") or [],
        }
        yield {"type": "done", "answer": prepared.get("answer", "")}
        return

    state = prepared.state
    decision = prepared.decision
    effective_route = prepared.effective_route
    docs = state.get("documents") or []
    citations = build_citations_from_context(docs)
    yield {
        "type": "meta",
        "citations": citations,
        "route": decision.route,
        "effective_route": effective_route,
        "router_reason": decision.reason,
        "logs": state.get("logs") or [],
    }
    blocked = "抱歉，知识库中未能找到与您问题高度相关的信息，为避免产生误导（幻觉），系统已阻断回答。"
    if not docs:
        yield {"type": "done", "answer": blocked}
        return
    parts: list[str] = []
    for piece in stream_generation_chunks(question, docs, is_analysis=state.get("is_analysis")):
        parts.append(piece)
        yield {"type": "token", "text": piece}
    yield {"type": "done", "answer": "".join(parts)}


def run_legacy_vector_pipeline(question: str) -> dict[str, Any]:
    """Original LangGraph vector-only pipeline."""
    inputs: AgentState = {"question": question, "loop_count": 0, "logs": []}
    result = invoke_rag_app(inputs)
    return {
        "answer": result["generation"],
        "context_used": result["documents"],
        "logs": result["logs"],
    }
