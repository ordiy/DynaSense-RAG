# Recall / NDCG Benchmark Report run 20260402T154831Z

## 1. Test objective

Measure **mean Recall@1,3,5,10** and **mean NDCG@K** (K=1,3,5,10) on a public scientific-QA corpus, using the same retrieval + rerank stack as production evaluation (`run_evaluation`).

## 2. Dataset

- **Corpus**: sciq (HuggingFace `allenai/sciq`, train split)
- **Indexing**: each unique `support` paragraph is stored as one parent document; a deterministic marker `[benchmark_doc_id=N]` is appended so relevance is unambiguous (no accidental substring collisions).
- **Indexed documents**: 60
- **Evaluation queries**: 30
- **Retrieval mode**: `use_hybrid=false` (see `docs/recall_evaluation.md`).

## 3. Environment

- `LANCEDB_URI`: `./data/lancedb_recall_benchmark_jina` (isolated from production / prior dry run)
- `SKIP_NEO4J_INGEST=1`: skip graph extraction during ingest
- **Vertex AI** `text-embedding-004` (embeddings + query vectors)
- **Jina**: `segment.jina.ai` semantic chunking + `jina-reranker-v2-base-multilingual` cross-encoder rerank (API key supplied via `JINA_API_KEY` at runtime — never commit keys to git)

## 4. Aggregated results

- Successful queries: **30**, errors: **0**

| Metric | Mean |
|--------|------|
| `ndcg@1` | 1.000000 |
| `ndcg@10` | 1.000000 |
| `ndcg@3` | 1.000000 |
| `ndcg@5` | 1.000000 |
| `recall@1` | 1.000000 |
| `recall@10` | 1.000000 |
| `recall@3` | 1.000000 |
| `recall@5` | 1.000000 |

## 5. Interpretation

- **Recall@K**: fraction of queries where the relevant parent (identified by marker) appears in the **top-K** after dense retrieval → small-to-big → Jina rerank (and hybrid routing when enabled).
- **NDCG@K**: binary relevance, single relevant document; normalized DCG vs ideal rank-1 (see `src/recall_metrics.py`).

## 6. Notes

Ground truth: synthetic marker `[benchmark_doc_id=N]` appended to each support passage; Recall/NDCG use first hit in reranked top-10 contexts (see `docs/recall_evaluation.md`). Neo4j ingest skipped (`SKIP_NEO4J_INGEST=1`). Raw JSON: `recall_ndcg_benchmark_20260402T154831Z.json`.

**Security:** If you pasted a Jina API key in a chat or ticket, **rotate it** in the Jina dashboard and use only `export JINA_API_KEY=...` or a local `.env` (gitignored).
