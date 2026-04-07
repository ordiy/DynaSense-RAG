"""
BM25 cache tests — REMOVED.

The in-memory BM25 index (rank_bm25 + threading.Lock + _bm25_cache) was
replaced by PostgreSQL tsvector GIN-indexed FTS in Stage 1 of the retrieval
optimisation.  There is no longer any in-process cache to test or invalidate.

Replacement coverage lives in: tests/test_fts_retrieval.py
"""
