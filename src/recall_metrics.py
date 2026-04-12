"""
Retrieval metrics for binary relevance (single ground-truth document).

Definitions (hit_rank is 0-based index in the ranked list; -1 = not found):
- Recall@K: 1 if the relevant document appears at any position < K, else 0.
- NDCG@K: binary relevance, one relevant item; ideal DCG places it at rank 1.

DCG@K = rel / log2(p + 1) where p is 1-based position of the hit (only one non-zero term).
IDCG@K = 1 / log2(2) = 1.0 (best possible: relevant at rank 1).
NDCG@K = DCG@K / IDCG@K = 1 / log2(p + 1) when hit is within top K.
"""
from __future__ import annotations

import math
from typing import Iterable


def recall_at_k(hit_rank_0: int, k: int) -> int:
    """Binary recall: 1 iff first hit position (0-based) is in [0, k-1]."""
    if hit_rank_0 < 0:
        return 0
    return 1 if hit_rank_0 < k else 0


def dcg_at_k_binary(hit_rank_0: int, k: int) -> float:
    if hit_rank_0 < 0 or hit_rank_0 >= k:
        return 0.0
    p = hit_rank_0 + 1  # 1-based rank
    return 1.0 / math.log2(p + 1)


def idcg_at_k_binary_single_relevant(k: int) -> float:
    """IDCG@K for a single relevant document (ideal rank at position 1)."""
    return 1.0 / math.log2(2)


def ndcg_at_k_binary(hit_rank_0: int, k: int) -> float:
    if hit_rank_0 < 0 or hit_rank_0 >= k:
        return 0.0
    p = hit_rank_0 + 1
    dcg = 1.0 / math.log2(p + 1)
    idcg = idcg_at_k_binary_single_relevant(k)
    return dcg / idcg if idcg > 0 else 0.0


def find_hit_rank(
    ranked_texts: Iterable[str],
    expected_substring: str,
) -> int:
    """Return 0-based index of first doc containing expected_substring (case-insensitive), or -1."""
    needle = (expected_substring or "").strip().lower()
    if not needle:
        return -1
    for i, text in enumerate(ranked_texts):
        if needle in (text or "").lower():
            return i
    return -1


def metrics_for_hit(hit_rank_0: int) -> dict[str, float | int]:
    """Recall@1,3,5,10 and NDCG@K for K in 1,3,5,10 (primary report: ndcg@10)."""
    out: dict[str, float | int] = {}
    for k in (1, 3, 5, 10):
        out[f"recall@{k}"] = recall_at_k(hit_rank_0, k)
        out[f"ndcg@{k}"] = round(ndcg_at_k_binary(hit_rank_0, k), 6)
    return out


def aggregate_mean(metrics_list: list[dict[str, float | int]]) -> dict[str, float]:
    """
    Mean over queries for retrieval metrics (recall@K, ndcg@K) and, when present,
    the faithfulness score produced by ``src.core.faithfulness.judge_faithfulness``.
    """
    if not metrics_list:
        return {}
    keys = [
        k for k in metrics_list[0].keys()
        if k.startswith("recall@") or k.startswith("ndcg@") or k == "faithfulness_score"
    ]
    totals: dict[str, float] = {k: 0.0 for k in keys}
    counts: dict[str, int] = {k: 0 for k in keys}
    for m in metrics_list:
        for k in keys:
            v = m.get(k)
            if v is not None:
                totals[k] += float(v)
                counts[k] += 1
    return {k: totals[k] / counts[k] for k in keys if counts[k] > 0}
