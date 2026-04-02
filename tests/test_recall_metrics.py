"""Unit tests for recall / NDCG metrics (no external services)."""
import math

import pytest

from src.recall_metrics import (
    aggregate_mean,
    find_hit_rank,
    metrics_for_hit,
    ndcg_at_k_binary,
    recall_at_k,
)


def test_recall_at_k():
    assert recall_at_k(-1, 10) == 0
    assert recall_at_k(0, 1) == 1
    assert recall_at_k(1, 1) == 0
    assert recall_at_k(1, 3) == 1
    assert recall_at_k(9, 10) == 1
    assert recall_at_k(10, 10) == 0


def test_ndcg_at_k_binary():
    # Rank 1 (0-based 0): NDCG@10 = 1/log2(2) / (1/log2(2)) = 1
    assert abs(ndcg_at_k_binary(0, 10) - 1.0) < 1e-9
    # Rank 2 (0-based 1): DCG = 1/log2(3), IDCG = 1
    p = 2
    expected = (1.0 / math.log2(p + 1)) / (1.0 / math.log2(2))
    assert abs(ndcg_at_k_binary(1, 10) - expected) < 1e-9
    # Miss in top 10
    assert ndcg_at_k_binary(10, 10) == 0.0


def test_find_hit_rank():
    docs = ["hello", "world foo", "bar"]
    assert find_hit_rank(docs, "FOO") == 1
    assert find_hit_rank(docs, "baz") == -1


def test_metrics_for_hit():
    m = metrics_for_hit(0)
    assert m["recall@1"] == 1
    assert m["recall@10"] == 1
    assert m["ndcg@10"] == 1.0

    m2 = metrics_for_hit(-1)
    assert m2["recall@10"] == 0
    assert m2["ndcg@10"] == 0.0


def test_aggregate_mean():
    a = metrics_for_hit(0)
    b = metrics_for_hit(2)
    mean = aggregate_mean([a, b])
    assert "recall@1" in mean
    assert mean["recall@1"] == 0.5
