"""
Tests for parallel triple extraction in ingest_chunks_to_graph.

Verifies that:
- All chunks are processed (parallel results match sequential count).
- Failures in one chunk do not abort others.
- BM25 cache is invalidated after ingestion.
- ThreadPoolExecutor is used (not a sequential loop).
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _triple(s, p, o):
    t = MagicMock()
    t.subject = s
    t.predicate = p
    t.object = o
    return t


# ---------------------------------------------------------------------------
# Core correctness
# ---------------------------------------------------------------------------

def test_all_chunks_processed(clean_bm25_cache):
    """ingest_chunks_to_graph must extract triples from every chunk."""
    import src.hybrid_rag as hr

    chunks = ["chunk A text", "chunk B text", "chunk C text"]
    chunk_ids = ["c1", "c2", "c3"]
    triples_per_chunk = [
        [_triple("A", "rel", "X")],
        [_triple("B", "rel", "Y"), _triple("B2", "rel", "Y2")],
        [],
    ]

    extract_calls = []

    def mock_extract(text):
        extract_calls.append(text)
        idx = chunks.index(text)
        return triples_per_chunk[idx]

    with patch("src.hybrid_rag.get_driver", return_value=object()), \
         patch("src.hybrid_rag.extract_triples_from_text", side_effect=mock_extract), \
         patch("src.hybrid_rag.merge_triple") as mock_merge:
        n = hr.ingest_chunks_to_graph(chunks, chunk_ids, "test.txt")

    assert set(extract_calls) == set(chunks), "All chunks must be processed"
    assert n == 3  # 1 + 2 + 0
    assert mock_merge.call_count == 3


def test_partial_failure_does_not_abort(clean_bm25_cache):
    """A failing chunk should not prevent other chunks from being processed."""
    import src.hybrid_rag as hr

    chunks = ["good chunk", "bad chunk", "another good chunk"]
    chunk_ids = ["c1", "c2", "c3"]

    def mock_extract(text):
        if "bad" in text:
            raise RuntimeError("simulated LLM failure")
        return [_triple("E", "rel", "F")]

    with patch("src.hybrid_rag.get_driver", return_value=object()), \
         patch("src.hybrid_rag.extract_triples_from_text", side_effect=mock_extract), \
         patch("src.hybrid_rag.merge_triple"):
        n = hr.ingest_chunks_to_graph(chunks, chunk_ids, "test.txt")

    # 2 good chunks × 1 triple each
    assert n == 2


def test_returns_zero_when_driver_unavailable(clean_bm25_cache):
    import src.hybrid_rag as hr

    with patch("src.hybrid_rag.get_driver", return_value=None):
        n = hr.ingest_chunks_to_graph(["any"], ["c1"], "f.txt")

    assert n == 0


def test_bm25_cache_invalidated_after_ingest(clean_bm25_cache):
    """BM25 cache must be None after ingest_chunks_to_graph completes."""
    import src.hybrid_rag as hr
    from langchain_core.documents import Document

    # Pre-populate cache
    hr._bm25_cache = ([], MagicMock())
    assert hr._bm25_cache is not None

    with patch("src.hybrid_rag.get_driver", return_value=object()), \
         patch("src.hybrid_rag.extract_triples_from_text", return_value=[]), \
         patch("src.hybrid_rag.merge_triple"):
        hr.ingest_chunks_to_graph(["chunk"], ["c1"], "f.txt")

    assert hr._bm25_cache is None, "BM25 cache must be invalidated after ingestion"


# ---------------------------------------------------------------------------
# Parallelism
# ---------------------------------------------------------------------------

def test_parallel_extraction_is_faster_than_sequential():
    """
    Parallel extraction with I/O-bound (sleep) tasks must complete faster
    than the sum of individual delays.

    Uses 5 chunks each sleeping 0.05 s → sequential would take ~0.25 s;
    parallel (max_workers=5) should finish in ~0.05 s.
    """
    import src.hybrid_rag as hr

    DELAY = 0.05
    N = 5
    chunks = [f"chunk {i}" for i in range(N)]
    chunk_ids = [f"c{i}" for i in range(N)]

    def slow_extract(text):
        time.sleep(DELAY)
        return []

    with patch("src.hybrid_rag.get_driver", return_value=object()), \
         patch("src.hybrid_rag.extract_triples_from_text", side_effect=slow_extract), \
         patch("src.hybrid_rag.merge_triple"), \
         patch("src.hybrid_rag.invalidate_bm25_cache"):
        start = time.monotonic()
        hr.ingest_chunks_to_graph(chunks, chunk_ids, "f.txt")
        elapsed = time.monotonic() - start

    sequential_lower_bound = DELAY * N * 0.8  # 80 % of sequential time
    assert elapsed < sequential_lower_bound, (
        f"Expected parallel time < {sequential_lower_bound:.3f}s, got {elapsed:.3f}s"
    )


def test_merge_triple_called_with_correct_source(clean_bm25_cache):
    """Provenance (source) must be passed through to merge_triple."""
    import src.hybrid_rag as hr

    chunks = ["text about company X"]
    chunk_ids = ["cx1"]

    with patch("src.hybrid_rag.get_driver", return_value=object()), \
         patch("src.hybrid_rag.extract_triples_from_text",
               return_value=[_triple("CompanyX", "founded_by", "Alice")]), \
         patch("src.hybrid_rag.merge_triple") as mock_merge, \
         patch("src.hybrid_rag.invalidate_bm25_cache"):
        hr.ingest_chunks_to_graph(chunks, chunk_ids, "annual_report.txt")

    mock_merge.assert_called_once_with(
        "CompanyX", "founded_by", "Alice", "cx1", "annual_report.txt"
    )
