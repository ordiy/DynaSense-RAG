# Recall@K & NDCG Evaluation

## Definitions (binary relevance)

- **Ground truth**: each test case provides `expected_substring`. The **first** ranked context whose text contains this substring (case-insensitive) is treated as the relevant hit.
- **Recall@K** ∈ {0,1}: 1 iff the hit appears at rank ≤ K (1-based), i.e. 0-based index `< K`.
- **NDCG@K**: one relevant document; ideal rank is 1.  
  \(\text{NDCG@K} = \dfrac{\text{DCG@K}}{\text{IDCG@K}}\), with \(\text{DCG@K} = 1/\log_2(p+1)\) if the hit is at 1-based position \(p \le K\), else 0; \(\text{IDCG@K} = 1/\log_2(2) = 1\).

Reported metrics: **Recall@1, @3, @5, @10** and **NDCG@1, @3, @5, @10** (primary aggregate: **NDCG@10**).

## Test cases

- JSON file: [`data/recall_eval_cases.json`](../data/recall_eval_cases.json) — tuned to [`data/demo_related_party.txt`](../data/demo_related_party.txt). **Upload that document** (or a corpus containing the same facts) before running eval.

## API

- `POST /api/evaluate` — body: `{ "query", "expected_substring", "use_hybrid": false }`  
- `POST /api/evaluate/batch` — body: `{ "cases": [ { "id", "query", "expected_substring" } ], "use_hybrid": false }`  
  Response includes `mean_metrics` over successful cases.

## CLI

```bash
cd /path/to/repo
.venv/bin/python scripts/run_recall_eval.py
# Hybrid RAG retrieval path:
.venv/bin/python scripts/run_recall_eval.py --hybrid
```

Writes `reports/recall_eval_last.json`.

## SciQ 批量基准（公开数据集）

见 **[recall_ndcg_benchmark_plan.md](./recall_ndcg_benchmark_plan.md)** 与 `scripts/benchmark_recall_ndcg.py`：对 Hugging Face **SciQ** 建库并输出 **mean Recall@K / NDCG@K** 与 `reports/recall_ndcg_benchmark_*.md`。

## Unit tests (metrics only)

```bash
.venv/bin/pytest tests/test_recall_metrics.py -v
```

Requires `pytest` (install: `pip install pytest`).

## Modes

| `use_hybrid` | Retrieval |
|----------------|-----------|
| `false` | Dense + Small-to-Big + Jina rerank **top 10** (vector-only). |
| `true` | Same as Hybrid RAG: router + route-specific candidates + fusion rerank **top 10**. |
