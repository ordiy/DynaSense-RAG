# MAP-RAG Architecture MVP

> **MAP-RAG**: Multi-resolution Agentic Perception Retrieval-Augmented Generation

An enterprise-grade RAG (Retrieval-Augmented Generation) architecture prototype focused on strict Anti-Hallucination mechanisms, intelligent semantic chunking, and Cross-Encoder reranking. 

## 🎯 Core Philosophy
**"No Answer is better than a Bad/Toxic Answer."**

In enterprise environments (legal, financial, internal HR policies), LLM hallucinations are unacceptable. This MVP explicitly **rejects** real-time generic Query Rewriting on the main pipeline to prevent "Intent Drift" (where specialized internal terms are rewritten into generic terms, losing their exact meaning) and to avoid unnecessary LLM latency.

Instead, this architecture achieves high precision through:
1. **Intelligent Chunking** (Jina Segmenter)
2. **High-Dimensional Vector Retrieval** (Google Vertex AI `text-embedding-004` + LanceDB)
3. **Cross-Encoder Semantic Reranking** (Jina Multilingual Reranker)
4. **Strict Grader Node** (LangGraph state machine for hallucination prevention)

## 🏗️ Architecture (LangGraph)

The system uses a directed graph (State Machine) to process user queries:
1. **Retrieve & Rerank Node**: Fetches Top-K (e.g., K=10) from LanceDB and reranks to Top-N (e.g., N=2) using Jina Reranker.
2. **Grade Node (Anti-Hallucination)**: The LLM strictly evaluates if the retrieved context *actually* answers the query. If `no`, the pipeline stops or falls back to a rewrite loop (disabled in strict mode).
3. **Generate Node**: Synthesizes the final answer *strictly* using the approved context.

## 📊 Benchmark Results (SciQ Dataset)
We benchmarked this pipeline against a subset of the HuggingFace `sciq` dataset (1000 documents, 100 questions).

| Metric | Base Vector Search (Vertex AI) | Pipeline (Vector + Jina Reranker) | Improvement |
|---|---|---|---|
| **Recall@1** | 86.0% | **96.0%** | 🚀 **+10.0%** |
| **Recall@3** | 96.0% | **100.0%** | 🚀 **+4.0%** |
| **Recall@5** | 99.0% | **100.0%** | +1.0% |
| **Recall@10** | 100.0% | 100.0% | Maxed |

*Conclusion*: The Reranker effectively acts as a precision "sniper," ensuring that the LLM only needs to process 1-3 chunks of text to get the correct context 100% of the time. This saves massive token costs, drastically reduces latency, and closes the window for hallucination.

## 🛠️ Tech Stack
* **Orchestration**: `LangGraph` & `LangChain`
* **Embedding Model**: Google Vertex AI `text-embedding-004`
* **LLM**: Google Vertex AI `gemini-2.5-pro`
* **Vector Database**: `LanceDB`
* **Semantic Chunking**: `Jina Segmenter API`
* **Reranker**: `jina-reranker-v2-base-multilingual`

## 🚀 Getting Started
```bash
# 1. Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install langchain langchain-google-vertexai langgraph lancedb==0.5.2 pydantic bs4 pandas numpy jina requests mongomock datasets polars

# 3. Set your API Keys and GCP config
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/gcp-sa.json"

# 4. Run scripts
python run_mvp_v3.py        # Run the MVP pipeline with Jina Chunking and Rerank
python run_benchmark.py     # Run the recall benchmark on the SciQ dataset
```

## 📄 Documentation
- [doc-feauture-v1.md](./doc-feauture-v1.md) - Initial Architecture RFC.
- [doc-future.md](./doc-future.md) - Enterprise principles and guidelines preventing bad answers.
- [benchmark_report.md](./benchmark_report.md) - Full benchmark logs.
