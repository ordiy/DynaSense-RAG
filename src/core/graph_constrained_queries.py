"""
Constrained Neo4j access (alternative to GraphCypherQAChain).

Design:
- Only **whitelisted template ids** map to **fixed, parameterized** Cypher or to
  existing helpers in ``graph_store`` (already reviewed).
- User-supplied strings are passed **only** as query parameters (no string
  interpolation into Cypher), with length / shape validation.
- Optional ``suggest_template_from_question`` uses **heuristics** for tests; a
  future LLM can fill the same ``template`` + ``params`` JSON without generating
  arbitrary Cypher.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from src.graph_store import (
    get_driver,
    global_graph_summary,
    query_relationships_by_keywords,
)


@dataclass(frozen=True)
class ConstrainedGraphResult:
    """Unified result for API / tests."""

    template_id: str
    rows: list[dict[str, Any]]
    logs: list[str]


def _validate_text(s: str, *, max_len: int = 256) -> str:
    t = (s or "").strip()
    if not t:
        raise ValueError("parameter must be non-empty")
    if len(t) > max_len:
        raise ValueError(f"parameter exceeds max length {max_len}")
    if "\x00" in t:
        raise ValueError("invalid character")
    return t


def _validate_limit(n: Any, *, default: int = 40, cap: int = 100) -> int:
    if n is None:
        return default
    try:
        v = int(n)
    except (TypeError, ValueError):
        raise ValueError("limit must be an integer")
    if v < 1 or v > cap:
        raise ValueError(f"limit must be in [1, {cap}]")
    return v


def _tpl_edges_from_entity(params: dict[str, Any]) -> ConstrainedGraphResult:
    """Single focus keyword: reuses bounded keyword search (one term)."""
    sub = _validate_text(str(params.get("name_substring", "")), max_len=128)
    lim = _validate_limit(params.get("limit"), default=40, cap=100)
    if not get_driver():
        return ConstrainedGraphResult(
            "edges_from_entity",
            [],
            ["Neo4j unavailable; no rows."],
        )
    rows = query_relationships_by_keywords([sub], limit=lim)
    logs = [f"edges_from_entity: keywords=[{sub!r}] limit={lim} rows={len(rows)}"]
    return ConstrainedGraphResult("edges_from_entity", rows, logs)


def _tpl_multi_keyword_edges(params: dict[str, Any]) -> ConstrainedGraphResult:
    raw = params.get("keywords")
    if not isinstance(raw, list):
        raise ValueError("keywords must be a list of strings")
    kws: list[str] = []
    for x in raw[:20]:
        kws.append(_validate_text(str(x), max_len=64))
    if not kws:
        raise ValueError("at least one keyword required")
    lim = _validate_limit(params.get("limit"), default=40, cap=100)
    if not get_driver():
        return ConstrainedGraphResult("multi_keyword_edges", [], ["Neo4j unavailable; no rows."])
    rows = query_relationships_by_keywords(kws, limit=lim)
    logs = [f"multi_keyword_edges: n_kw={len(kws)} limit={lim} rows={len(rows)}"]
    return ConstrainedGraphResult("multi_keyword_edges", rows, logs)


def _tpl_graph_global_summary(_params: dict[str, Any]) -> ConstrainedGraphResult:
    if not get_driver():
        return ConstrainedGraphResult(
            "graph_global_summary",
            [],
            ["Neo4j unavailable; empty summary."],
        )
    text = global_graph_summary()
    rows = [{"summary": text}] if text else []
    logs = [f"graph_global_summary: chars={len(text or '')}"]
    return ConstrainedGraphResult("graph_global_summary", rows, logs)


_TEMPLATES: dict[str, Callable[[dict[str, Any]], ConstrainedGraphResult]] = {
    "edges_from_entity": _tpl_edges_from_entity,
    "multi_keyword_edges": _tpl_multi_keyword_edges,
    "graph_global_summary": _tpl_graph_global_summary,
}


def execute_constrained_template(template_id: str, params: dict[str, Any] | None) -> ConstrainedGraphResult:
    """
    Run a whitelisted template. Raises ``ValueError`` on unknown template or bad params.

    This is the integration point for HTTP handlers and for a future LLM that
    returns **structured** ``{template_id, params}`` instead of raw Cypher.
    """
    if template_id not in _TEMPLATES:
        raise ValueError(f"unknown template: {template_id!r}")
    return _TEMPLATES[template_id](dict(params or {}))


def suggest_template_from_question(question: str) -> tuple[str | None, dict[str, Any]]:
    """
    Lightweight heuristic slot-filler (no LLM). For demos and unit expectations.

    Returns (template_id, params) or (None, {}) if no match.
    """
    q = (question or "").strip()
    if not q:
        return None, {}
    ql = q.lower()
    if any(
        x in q
        for x in ("图谱摘要", "图数据库摘要", "全局图", "整体图谱", "graph summary")
    ) or "global graph" in ql:
        return "graph_global_summary", {}
    m = re.search(r"(?:与|关于|围绕|涉及)\s*(.+?)\s*(?:相关|的关系|的边|三元组)", q)
    if m:
        return "edges_from_entity", {"name_substring": m.group(1).strip().strip("'\""), "limit": 40}
    m2 = re.search(r"(?:keywords?|关键词)\s*[:：]\s*([^\n]+)", q, re.I)
    if m2:
        parts = [p.strip() for p in re.split(r"[,，;；\s]+", m2.group(1)) if p.strip()]
        if parts:
            return "multi_keyword_edges", {"keywords": parts[:12], "limit": 40}
    return None, {}
