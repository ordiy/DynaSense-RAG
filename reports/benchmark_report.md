# MAP-RAG Retrieval Benchmark Report

## 1. Test Configuration
- **Dataset**: `sciq` (HuggingFace, Scientific QA)
- **Knowledge Base Size**: 1000 unique text chunks (raw paragraphs).
- **Test Queries**: 100 factual questions.
- **Base Vector Model**: Vertex AI `text-embedding-004`
- **Vector DB**: LanceDB
- **Reranker Model**: `jina-reranker-v2-base-multilingual` (Reranking top 20 candidates)

*(Note: The scale is restricted to 1k documents / 100 queries to strictly respect API rate limits and ensure execution stability during testing.)*

## 2. Benchmark Metrics

### Recall@K Comparison

| Metric | Base Vector Search (Vertex AI) | Pipeline (Vector + Jina Reranker) | Improvement |
|---|---|---|---|
| **Recall@1** | 86.0% | 96.0% | +10.0% |
| **Recall@3** | 96.0% | 100.0% | +4.0% |
| **Recall@5** | 99.0% | 100.0% | +1.0% |
| **Recall@10** | 100.0% | 100.0% | +0.0% |

## 3. Conclusion
- **High-Dimensional Precision**: The baseline Vector Search performs reasonably well, but can sometimes fail to float the precise document to the absolute top position.
- **Reranker Impact**: The Cross-Encoder (Jina Rerank) specifically excels at pushing correct candidates from the top 5-20 range into the top 1 or top 3 positions. This validates that the architectural pipeline design drastically reduces the context payload needed for the LLM generator, lowering hallucination risks.
