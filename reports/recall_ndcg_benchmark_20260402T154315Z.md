# Recall / NDCG Benchmark Report (dry-run)

## 1. Test objective

Measure **mean Recall@1,3,5,10** and **mean NDCG@K** (K=1,3,5,10) on a public scientific-QA corpus, using the same retrieval + rerank stack as production evaluation (`run_evaluation`).

## 2. Dataset

- **Corpus**: sciq (HuggingFace)
- **Indexing**: each unique `support` paragraph is stored as one parent document; a deterministic marker `[benchmark_doc_id=N]` is appended so relevance is unambiguous (no accidental substring collisions).
- **Indexed documents**: 100
- **Evaluation queries**: 50
- **Retrieval mode**: `use_hybrid=false` (see `docs/recall_evaluation.md`).

## 3. Environment

- `LANCEDB_URI`: isolated benchmark directory (default `./data/lancedb_recall_benchmark`)
- `SKIP_NEO4J_INGEST=1`: skip graph extraction during ingest
- Vertex AI embeddings + Jina reranker (if `JINA_API_KEY` set)

## 4. Aggregated results

- Successful queries: **0**, errors: **0**

| Metric | Mean |
|--------|------|
| *(no successful runs)* | — |

## 5. Interpretation

- **Recall@K**: fraction of queries where the relevant parent (identified by marker) appears in the **top-K** after dense retrieval → small-to-big → Jina rerank (and hybrid routing when enabled).
- **NDCG@K**: binary relevance, single relevant document; normalized DCG vs ideal rank-1 (see `src/recall_metrics.py`).

## 6. Notes

Dry run: no indexing, no Vertex/Jina calls.
