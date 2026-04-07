"""
ReAct agentic retrieval pipeline.

Uses ``langgraph.prebuilt.create_react_agent`` to drive an iterative
tool-calling loop over ``vector_search`` and ``fulltext_search`` tools
(defined in ``src.retrieval_tools``).  The agent decides which tools to call
and in what order; collected Documents are Jina-reranked, then passed through
the same ``grade_documents_node`` → ``generate_node`` LangGraph nodes used by
the vector-only and hybrid paths.

Feature flag
------------
Enabled by ``Settings.agentic_retrieval_enabled = True`` (default: False).
``run_chat_pipeline`` in ``rag_core`` dispatches here when the flag is on.

Why ``threading.local`` instead of ``contextvars``
---------------------------------------------------
The existing pipeline (including retrieval_tools) is fully synchronous.
``threading.local`` is safe and has zero overhead in sync code.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from langchain_core.messages import SystemMessage

logger = logging.getLogger(__name__)

# ── System prompt ──────────────────────────────────────────────────────────────

RETRIEVAL_AGENT_SYSTEM_PROMPT = """You are a precise retrieval agent with access to a knowledge base.
Your job is to gather the most relevant evidence for the user's question before answering.

Available tools:
- vector_search(query, top_k): semantic/conceptual search — use for explanations, definitions, narratives.
- fulltext_search(query, top_k): keyword/exact-term search — use for product codes, names, dates.

Strategy:
1. Issue a broad vector_search with the user's question first.
2. If the top preview is only partially relevant, issue a targeted fulltext_search with specific terms.
3. You may reformulate the query and search again (up to the step limit).
4. Stop when you have gathered enough evidence or have exhausted useful queries.

Do NOT attempt to answer the question yourself — only collect evidence via the tools.
Return FINAL ANSWER once you believe retrieval is sufficient or no more useful queries remain."""


# ── Core: run the ReAct tool loop ─────────────────────────────────────────────

def run_agentic_retrieval(
    question: str,
    *,
    max_steps: int = 5,
) -> tuple[list, list[str]]:
    """
    Drive a ReAct loop that calls ``vector_search`` / ``fulltext_search`` tools,
    then return ``(reranked_docs, log_lines)``.

    The agent runs for at most *max_steps* tool iterations.  After the loop,
    all accumulated Documents are Jina-reranked against the original question.
    """
    from langgraph.prebuilt import create_react_agent

    from src.retrieval_tools import (
        reset_doc_registry,
        get_collected_docs,
        get_tool_logs,
        vector_search,
        fulltext_search,
    )
    from src.rag_core import llm, jina_rerank
    from src.core.config import get_settings

    s = get_settings()
    top_n = s.rag_vector_rerank_top_n

    reset_doc_registry()

    agent = create_react_agent(
        model=llm,
        tools=[vector_search, fulltext_search],
        prompt=SystemMessage(content=RETRIEVAL_AGENT_SYSTEM_PROMPT),
    )

    logs: list[str] = [f"Agentic retrieval started (max_steps={max_steps})."]
    try:
        agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config={"recursion_limit": max_steps * 3 + 2},
        )
    except Exception as exc:
        logger.warning("Agentic ReAct loop error (continuing with collected docs): %s", exc)
        logs.append(f"ReAct loop warning: {exc}")

    tool_logs = get_tool_logs()
    logs.extend(tool_logs)

    raw_docs = get_collected_docs()
    logs.append(f"Tool loop done: {len(raw_docs)} unique docs collected.")

    if not raw_docs:
        return [], logs

    reranked = jina_rerank(question, raw_docs, top_n=top_n)
    logs.append(f"Jina rerank: {len(reranked)} docs after top_n={top_n}.")
    return reranked, logs


# ── Chat pipeline: reuse grade + generate from rag_core ───────────────────────

def run_agentic_chat_pipeline(query: str) -> Dict[str, Any]:
    """
    Full agentic RAG pipeline:
      1. ReAct retrieval loop → Jina reranked docs
      2. grade_documents_node  (hallucination guard)
      3. generate_node          (LLM answer synthesis)

    Returns a dict with keys: answer, context_used, logs, citations, route.
    """
    from src.core.config import get_settings
    from src.core.citations import build_citations_from_context
    from src.rag_core import (
        grade_documents_node,
        generate_node,
        _is_analysis_query,
        AgentState,
    )

    s = get_settings()
    reranked_docs, retrieval_logs = run_agentic_retrieval(
        query, max_steps=s.agentic_max_steps
    )

    doc_strings: List[str] = [d.page_content for d in reranked_docs]
    is_analysis = _is_analysis_query(query)

    state: AgentState = {
        "question": query,
        "documents": doc_strings,
        "is_analysis": is_analysis,
        "loop_count": 0,
        "logs": retrieval_logs,
        "skip_retrieval": True,  # docs already populated
    }

    state.update(grade_documents_node(state))
    state.update(generate_node(state))

    answer: str = state.get("generation") or (
        "抱歉，知识库中未能找到与您问题高度相关的信息，为避免产生误导（幻觉），系统已阻断回答。"
    )
    ctx: List[str] = state.get("documents") or []

    return {
        "answer": answer,
        "context_used": ctx,
        "logs": state.get("logs") or [],
        "citations": build_citations_from_context(ctx),
        "route": "agentic",
    }
