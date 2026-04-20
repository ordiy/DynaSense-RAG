"""
Maximal Marginal Relevance (MMR) diversification for reranked document lists.

Inserts after Jina cross-encoder reranking to reduce redundancy while keeping
relevance. Pure stdlib + langchain_core only — no external ML dependencies.

Reference: Carbonell & Goldstein (1998), "The use of MMR, diversity-based
reranking for reordering documents and producing summaries."
"""
from __future__ import annotations

import re

from langchain_core.documents import Document


def _tokenize(text: str) -> frozenset[str]:
    """Case-fold and extract word tokens (ASCII + CJK)."""
    return frozenset(re.findall(r"[a-zA-Z0-9\u4e00-\u9fff]+", text.lower()))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard similarity on token sets. Empty∩empty = 1.0 (identical empty docs)."""
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def mmr_filter(
    docs: list[Document],
    k: int,
    lambda_param: float = 0.7,
) -> list[Document]:
    """
    Select k documents balancing relevance and diversity via MMR.

    Args:
        docs: Documents pre-ordered by Jina rerank score (index 0 = most relevant).
        k: Number of documents to return.
        lambda_param: Trade-off weight. 1.0 = pure relevance (preserves rank order).
                      0.0 = pure diversity. Default 0.7 is relevance-biased.

    Returns:
        List of at most min(k, len(docs)) selected documents.
    """
    if not docs or k <= 0:
        return []
    if k >= len(docs):
        return docs[:]

    # Relevance score derived from Jina rank position (rank 0 = highest relevance).
    relevance = [1.0 / (1.0 + i) for i in range(len(docs))]
    token_sets = [_tokenize(d.page_content) for d in docs]

    selected_idx: list[int] = []
    remaining_idx = list(range(len(docs)))

    for _ in range(k):
        best_idx, best_score = -1, float("-inf")
        for idx in remaining_idx:
            if selected_idx:
                max_sim = max(
                    _jaccard(token_sets[idx], token_sets[s]) for s in selected_idx
                )
            else:
                max_sim = 0.0
            score = lambda_param * relevance[idx] - (1.0 - lambda_param) * max_sim
            if score > best_score:
                best_score, best_idx = score, idx
        selected_idx.append(best_idx)
        remaining_idx.remove(best_idx)

    return [docs[i] for i in selected_idx]
