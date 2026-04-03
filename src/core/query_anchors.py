"""
Query anchor extraction and optional document filtering.

Problem addressed: dense retrieval in a heterogeneous KB can return parents from
unrelated domains (e.g. bank disclosure + internal project docs). When the query
contains strong anchors such as an issuer name or stock code, we keep only
candidates whose text mentions at least one anchor before cross-encoder rerank.

Design: fail-open — if no anchors are extracted, or filtering would drop
everything, the original list is restored and a log line explains why.
"""

from __future__ import annotations

import os
import re

from langchain_core.documents import Document

# 机构 / 证券实体：中信 + 银行、XX股份有限公司 等
_ORG_TAIL = re.compile(
    r"[\u4e00-\u9fff]{1,14}(?:银行|证券|保险|信托|基金|期货)"
    r"|[\u4e00-\u9fff]{2,16}(?:股份有限公司|有限责任公司|有限公司|集团)"
)
# A 股代码（6 位数字，与常见年份区分：要求前两位常见板块或全数字上下文）
_STOCK_CODE = re.compile(r"(?<![0-9])(?:60[0-9]{4}|00[0-9]{4}|30[0-9]{4}|68[0-9]{4})(?![0-9])")

def anchor_filter_enabled() -> bool:
    return os.environ.get("QUERY_ANCHOR_FILTER", "true").lower() in ("1", "true", "yes")


def extract_anchor_keywords(question: str) -> list[str]:
    """
    Pull high-precision substrings from the user question.

    - Organization / legal-entity style phrases (中文 + 银行/证券/有限公司…)
    - Shanghai / Shenzhen / ChiNext style 6-digit listing codes

    If nothing matches, returns empty list and the caller skips filtering (no weak
    token heuristics — those caused false anchors like generic report phrases).
    """
    q = (question or "").strip()
    if not q:
        return []

    seen: set[str] = set()
    out: list[str] = []

    for m in _ORG_TAIL.finditer(q):
        s = m.group(0).strip()
        if len(s) >= 4 and s not in seen:
            seen.add(s)
            out.append(s)

    for m in _STOCK_CODE.finditer(q):
        s = m.group(0)
        if s not in seen:
            seen.add(s)
            out.append(s)

    return out[:8]


def _doc_matches_any(doc: Document, anchors: list[str]) -> bool:
    text = doc.page_content or ""
    return any(a in text for a in anchors)


def filter_documents_by_query_anchors(
    question: str,
    documents: list[Document],
    *,
    fail_open: bool = True,
) -> tuple[list[Document], list[str]]:
    """
    Keep documents that mention at least one extracted anchor.

    If ``fail_open`` and no document matches, returns the original ``documents``.
    """
    logs: list[str] = []
    if not anchor_filter_enabled():
        logs.append("Anchor filter: disabled (QUERY_ANCHOR_FILTER).")
        return documents, logs
    if not documents:
        return [], logs

    anchors = extract_anchor_keywords(question)
    if not anchors:
        logs.append("Anchor filter: no anchors extracted; skip.")
        return documents, logs

    logs.append(f"Anchor filter: anchors={anchors!r}")
    kept = [d for d in documents if _doc_matches_any(d, anchors)]
    if not kept and fail_open:
        logs.append("Anchor filter: zero matches; fail-open (keep unfiltered pool).")
        return documents, logs
    if not kept:
        logs.append("Anchor filter: zero matches; returning empty.")
        return [], logs

    logs.append(f"Anchor filter: {len(documents)} -> {len(kept)} candidates.")
    return kept, logs
