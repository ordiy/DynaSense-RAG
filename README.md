# 🧠 DynaSense-RAG (MAP-RAG Architecture)

> **MAP-RAG**: Multi-resolution Agentic Perception Retrieval-Augmented Generation

An enterprise-grade RAG (Retrieval-Augmented Generation) architecture prototype focused on strict Anti-Hallucination mechanisms, intelligent semantic chunking, and Cross-Encoder reranking.

## 🌐 Translations
[日本語-🇯🇵](README-jp.md) [Deutsch-🇩🇪](README-de.md) ｜ [简体汉字](README-cn.md) ｜ [繁體汉字](README-ch.md)

## 🖥️ Web Preview

| Chat Interface | RAG Admin Panel |
|:-:|:-:|
| ![Chat Interface](https://raw.githubusercontent.com/ordiy/study_notes/master/res/image/2026/20260412111147128.png) | ![RAG Admin Panel](https://raw.githubusercontent.com/ordiy/study_notes/master/res/image/2026/20260412111104002.png) |

## 🎯 Core Philosophy
**"No Answer is better than a Bad/Toxic Answer."**

In enterprise environments (legal, financial, internal HR policies), LLM hallucinations are unacceptable. This MVP explicitly **rejects** real-time generic Query Rewriting on the main pipeline to prevent "Intent Drift" (where specialized internal terms are rewritten into generic terms, losing their exact meaning) and to avoid unnecessary LLM latency.

Instead, this architecture achieves high precision through:
1. **Intelligent Chunking** (Jina Segmenter)
2. **High-Dimensional Vector Retrieval** (Google Vertex AI `text-embedding-004` + PostgreSQL pgvector)
3. **Cross-Encoder Semantic Reranking** (Jina Multilingual Reranker)
4. **Dual-Track Grader + Generator** (LangGraph state machine — strict for factual queries, analytically capable for reasoning queries)
5. **Server-Side Multi-Turn Memory** (conversation session with context-length control)
6. **Hybrid RAG (MVP)** — **Query Router** + **Dense + BM25** + **PostgreSQL graph recall** + unified **Top‑K rerank** before grading (see `docs/mvp_hybrid_rag.md`)



## 🏗️ Architecture Design (MAP-RAG)

![MAP-RAG Architecture Diagram](https://raw.githubusercontent.com/ordiy/study_notes/master/res/image/2026/20260420105838989.png)



The system uses a directed LangGraph state machine. Key design decisions:
- **No Query Rewrite on critical path** — prevents Intent Drift, reduces latency
- **Dual-Track Routing** — analysis queries are not blocked by a strict factual grader; the LLM is explicitly instructed to label reasoning vs. retrieved facts
- **Fail-Closed by Default** — if the grader returns an error, the pipeline blocks the answer rather than passing through unverified context



## 📊 Benchmark Results (SciQ Dataset)
We benchmarked this pipeline against a subset of the HuggingFace `sciq` dataset (1000 documents, 100 questions).

| Metric | Base Vector Search (Vertex AI) | Pipeline (Vector + Jina Reranker) | Improvement |
|---|---|---|---|
| **Recall@1** | 86.0% | **96.0%** | 🚀 **+10.0%** |
| **Recall@3** | 96.0% | **100.0%** | 🚀 **+4.0%** |
| **Recall@5** | 99.0% | **100.0%** | +1.0% |
| **Recall@10** | 100.0% | 100.0% | Maxed |

*Conclusion*: The Reranker effectively acts as a precision "sniper," ensuring that the LLM only needs to process 1-3 chunks of text to get the correct context 100% of the time. This saves massive token costs, drastically reduces latency, and closes the window for hallucination.

### Recall@K / NDCG@K (batch script, SciQ)
Automated run via `scripts/benchmark_recall_ndcg.py` — same retrieval stack as evaluation (`run_evaluation`), **vector path only** (`use_hybrid=false`). Latest report: [`reports/recall_ndcg_benchmark_latest.md`](reports/recall_ndcg_benchmark_latest.md).

| Setting | Value |
|--------|--------|
| Corpus | HuggingFace `allenai/sciq` (train), unique `support` paragraphs as parents |
| Indexed documents | 60 |
| Evaluation queries | 30 |
| Retrieval mode | Dense → Small-to-Big → Jina rerank (hybrid routing off) |

| Metric (mean) | Value |
|---------------|-------|
| Recall@1,3,5,10 | 1.000 |
| NDCG@1,3,5,10 | 1.000 |

Raw JSON and timestamped reports live under `reports/recall_ndcg_benchmark_*.json` / `*.md`. See [`docs/recall_evaluation.md`](docs/recall_evaluation.md).

## ✨ Feature Highlights

### Dual-Track Query Routing (Analysis vs. Factual)
The pipeline automatically detects whether a query requires a **factual lookup** or **analytical reasoning**, and routes it to the appropriate grader and generator policy:

| | Factual Track | Analysis Track |
|---|---|---|
| **Trigger** | Default | Keywords: 分析/影响/如何/规划/evaluate/impact… |
| **Grader** | Strict: requires direct answer in context | Relaxed: requires any topic-related fact |
| **Generator** | `GEN_PROMPT`: "strictly use context" | `GEN_ANALYSIS_PROMPT`: facts + domain reasoning |
| **Output Format** | Direct answer | `【文档事实】` + `【分析推理】` labelled sections |

**Demo — Analysis query on partial context:**
> **User**: 介绍"豌豆苗期货"，分析天气对该期货交易的影响
>
> **Context retrieved**: growth cycle 3 months, region: east coast farms, yield 10 tons/day
>
> **Response** *(abridged)*:
> **【文档事实】** 豌豆苗期货作物生长周期3个月，日产量10吨。
> **【分析推理】** 基于行业经验：① 极端天气（霜冻/高温）可直接导致减产，推高期货价格；② 高温高湿促进病虫害，降低可交割品质；③ 恶劣天气阻断运输，增加物流成本并传导至期货端。

See [docs/dual-track-query-routing.md](./docs/dual-track-query-routing.md) for full design, implementation details, and 4 demo cases.

### Server-Side Multi-Turn Memory
Conversation sessions managed by `conversation_id` on the backend, with context-length control and TTL cleanup. See [docs/chat_test_memory_design.md](./docs/chat_test_memory_design.md).

### A/B Memory Strategy Comparison
`POST /api/chat/session/ab` runs both `prioritized` and `legacy` memory modes in parallel for the same message, returning side-by-side query content, answers, and blocking status—enabling rapid diagnosis of memory strategy effects.

### Hybrid RAG — Routing + Dual Recall + graph (MVP)
Implements **`readme-v2-1.md`**: an LLM **intent router** (`VECTOR` / `GRAPH` / `GLOBAL` / `HYBRID`), **unified indexing** in PostgreSQL (pgvector + JSONB + graph triples with `chunk_id` provenance), **online** dense + BM25 recall with **graph linearization**, and a **single Jina rerank** cut to Top‑5 before the existing grader/generator.

```text
User Query
    │
    ▼
[ Query Router (LLM) ] ──► VECTOR | GRAPH | GLOBAL | HYBRID
    │
    ├─ VECTOR ──► Dense(Small-to-Big) + BM25(child→parent) ──┐
    ├─ GRAPH ───► Graph subgraph → linearized triples text ──┤──► [ Jina Rerank Top‑5 ]
    ├─ GLOBAL ──► Graph summary + small dense anchor ──────────┤
    └─ HYBRID ──► merge VECTOR + GRAPH candidates ───────────┘
                                        │
                                        ▼
                           Grader (anti-hallucination) → Generator
```

```mermaid
flowchart TB
  subgraph ingest["Offline: dual-engine indexing"]
    D[Raw documents] --> J[Jina chunking]
    J --> E[Vertex embeddings + pgvector]
    J --> T[LLM triple extraction]
    T --> N[(PostgreSQL graph + chunk_id provenance)]
  end
  subgraph online["Online: router + unified rerank"]
    Q[User query] --> R[Query router]
    R --> V[VECTOR: dense + BM25]
    R --> G[GRAPH: PG graph + linearize]
    R --> GL[GLOBAL: graph summary + dense anchor]
    R --> HY[HYBRID: merge candidates]
    V --> RR[Jina cross-encoder rerank Top-K]
    G --> RR
    GL --> RR
    HY --> RR
    RR --> P[Grader + generator]
  end
```

- **PostgreSQL**: `docker compose -f docker-compose.postgres.yml up -d` and set `DATABASE_URL` (see compose file comments).
- **Demo corpus**: upload `data/demo_related_party.txt`, then try *「中国中信银行的关联方有哪些？」* — expect `GRAPH` or `HYBRID` with graph-backed context in logs.
- **Disable hybrid** (legacy vector-only LangGraph): `export HYBRID_RAG_ENABLED=false`.

Full design, env vars, and Q&A: [docs/mvp_hybrid_rag.md](./docs/mvp_hybrid_rag.md).

### Query Expansion (S4)
When `QUERY_EXPANSION_ENABLED=true`, the LLM generates 2 semantic rephrasings of the user query before retrieval. All 3 questions are run through pgvector in parallel; results are merged and deduplicated by `chunk_id`. Improves recall on synonym-heavy or terminology-shifting queries at the cost of ~1 extra LLM call per request.

### Rich PDF Ingestion — OCR, Tables, Images (S1 + S5)
`extract_pdf_content()` is now the single entry point for all PDF content:
- **Text layer**: pypdf (existing behavior)
- **OCR fallback**: pytesseract + pdf2image for scanned/image-only PDFs (graceful degradation if not installed)
- **Tables → Markdown**: pdfplumber preserves row/column structure so tabular data is searchable and citable
- **Image captions**: When `IMAGE_CAPTION_ENABLED=true`, embedded images ≥ 10 KB are sent to Gemini Vision, captions prefixed `[图片描述]` are indexed alongside text

### Production Feedback Loop (S6)
`POST /api/feedback` now persists every thumbs-up/down to a PostgreSQL `feedback` table (append-only, with `trace_id` for LangSmith correlation). New endpoints:
- `GET /api/feedback/negative` — last 50 downvotes (DB-first, in-memory fallback)
- `python scripts/export_negative_feedback.py --out feedback_eval.jsonl` — JSONL export for RAGAS / annotation tools

### MMR Diversity Filter (S7)
When `MMR_ENABLED=true`, a Maximal Marginal Relevance filter (pure stdlib, Jaccard token-set similarity) runs after Jina reranking to reduce redundant passages. Controlled by `MMR_LAMBDA` (default 0.7; 1.0 = pure relevance, 0.0 = pure diversity). Most useful for overview/synthesis queries.

### Multi-Provider Inference (S3)
`src/core/inference.py` abstracts LLM and embedding providers behind a factory. Set `INFERENCE_PROVIDER` to switch at runtime:

| Provider | `INFERENCE_PROVIDER` | Notes |
|---|---|---|
| Google Vertex AI | `vertex` (default) | ADC or `GOOGLE_APPLICATION_CREDENTIALS` |
| OpenAI-compatible | `openai_compat` | NIM, Ollama, vLLM — set `INFERENCE_BASE_URL` |
| Anthropic Claude | `anthropic` | Set `INFERENCE_API_KEY`; no embeddings |

### RAGAS CI Regression Gate (S2)
`tests/eval/test_ragas_regression.py` enforces minimum quality thresholds on a golden Q&A set:

| Metric | Threshold |
|---|---|
| `faithfulness` | ≥ 0.70 |
| `answer_relevancy` | ≥ 0.65 |
| `context_precision` | ≥ 0.60 |

Run with `make eval` (marked `@pytest.mark.slow`, excluded from `make test`).

---

### Why Not ReAct Agentic Retrieval (for now)

We evaluated the [NVIDIA NeMo Agentic Retrieval](https://huggingface.co/blog/nvidia/nemo-retriever-agentic-retrieval) pattern (ReAct loop with iterative `retrieve → observe → rethink` tool calls) against the current Hybrid RAG pipeline. The approach shows real NDCG@10 gains on [ViDoRe v3](https://huggingface.co/spaces/vidore/vidore-leaderboard?tab=vidore-v3-pipeline) — particularly on multi-hop and open-ended queries — but the cost profile is not yet viable for a real-time chat interface:

| | Current Hybrid RAG | ReAct Agentic (Opus 4.5) |
|---|---|---|
| **Latency / query** | ~2–4 s | ~136 s |
| **Input tokens / query** | ~4k–12k | ~760k |
| **Retrieval passes** | 1 (multi-source, single shot) | 9.2 average |
| **NDCG@10 (ViDoRe v3)** | comparable to dense baseline ~65 | **69.22** (+3.6 pts) |
| **Cold-start / new domains** | requires routing heuristics | adapts via reasoning |
| **Operational complexity** | moderate (BM25 cache, graph) | simple (tools only) |

**Where the gap shows up most:** open-ended synthesis queries (0.438 vs higher with agent) and multi-hop queries (0.515 baseline). For extractive/boolean/numerical queries — the majority of enterprise FAQ use cases — the current pipeline is already near-ceiling.

**Planned migration path** (see `docs/architecture.md` for detail):
1. **Near-term**: Replace in-memory BM25 cache with PostgreSQL `tsvector` full-text search — removes stateful cache invalidation without changing the pipeline shape.
2. **Mid-term**: Wrap retrieval backends as LangChain tools; add a **dual-path router** — fast path (current pipeline, <4 s) for simple queries, ReAct loop (3–5 steps, Gemini Flash) for complex/multi-hop.
3. **Long-term**: Evaluate upgrading from `text-embedding-004` to `jina-embeddings-v4` or `nemotron-colembed` once the retrieval benchmark harness (`POST /api/evaluate/batch`) has enough domain-specific test cases to measure the delta.

---

## 🛠️ Tech Stack
* **Orchestration**: `LangGraph` & `LangChain`
* **Embedding Model**: Google Vertex AI `text-embedding-004` (default) — swappable via `INFERENCE_PROVIDER`
* **LLM**: Google Vertex AI `gemini-2.5-flash` (default) — or OpenAI-compat / Anthropic via `INFERENCE_PROVIDER`
* **Database**: PostgreSQL (`pgvector` + JSONB + `feedback` table + optional Apache AGE)
* **Semantic Chunking**: `Jina Segmenter API`
* **Reranker**: `jina-reranker-v2-base-multilingual`
* **Graph (Hybrid MVP)**: PostgreSQL (Apache AGE Cypher or relational `kg_triple` fallback)
* **Lexical retrieval**: PostgreSQL `tsvector` full-text search (FTS) replacing rank-bm25
* **PDF extraction**: `pypdf` (text) + `pdfplumber` (tables) + `pytesseract`/`pdf2image` (OCR fallback)
* **Quality evaluation**: `ragas>=0.2.0` (faithfulness / answer_relevancy / context_precision CI gate)
* **Session Store**: In-memory `dict` with TTL (upgradeable to Redis)

## 🚀 Getting Started
```bash
# 1. Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
#    Full app (Vertex, LangChain, etc.): use project requirements or the line below.
#    CI / unit tests only: `pip install -r requirements-ci.txt` (no Vertex stack).
pip install -r requirements.txt

# 3. Set your API Keys, GCP config, and PostgreSQL URL
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/gcp-sa.json"
export JINA_API_KEY="your-jina-api-key"

# 4. PostgreSQL + pgvector (required for storage)
docker compose up --build   # or: docker compose -f docker-compose.postgres.yml up -d
export DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:5433/map_rag

# 5. (Optional) Feature flags — all default to false / vertex
export QUERY_EXPANSION_ENABLED=true    # LLM query rephrasing (+1 LLM call/request)
export MMR_ENABLED=true                # MMR diversity filter after Jina rerank
export IMAGE_CAPTION_ENABLED=true      # Gemini Vision captions for PDF images
export SKIP_GRAPH_INGEST=false         # set true to skip triple extraction during upload
export INFERENCE_PROVIDER=vertex       # vertex | openai_compat | anthropic
# export INFERENCE_BASE_URL=http://localhost:11434/v1  # for Ollama/vLLM

# 6. Start the web server
.venv/bin/uvicorn src.app:app --host 0.0.0.0 --port 8000

# Open http://localhost:8000 in your browser
# Tab 1: Upload documents
# Tab 2: Single-turn chat
# Tab 3: Evaluation
# Tab 4: Multi-turn Chat Test (with Memory + A/B Compare)
```

## 📄 Documentation

| Document | Description |
|---|---|
| [docs/langsmith_observability.md](./docs/langsmith_observability.md) | **LangSmith** — tracing env vars, init order (`src/observability.py`), [official observability docs](https://docs.langchain.com/langsmith/observability) |
| [docs/langgraph_stream_log.md](./docs/langgraph_stream_log.md) | **LangGraph stream logs** — `LANGGRAPH_STREAM_LOG`, `stream_mode="values"` step logs in `invoke_rag_app` |
| [docs/architecture.md](./docs/architecture.md) | **Clean Architecture** — `api/` / `core/` / `domain/` / `infrastructure/` layout; **`POST /api/analytics/profile`** (controlled CSV/TSV/XLSX profiling, no arbitrary code) |
| [docs/bitter_lesson_roadmap.md](./docs/bitter_lesson_roadmap.md) | **The Bitter Lesson** — phased roadmap (metrics, learn vs hand-rules, compliance boundaries) |
| [docs/postgresql_storage_roadmap.md](./docs/postgresql_storage_roadmap.md) | **PostgreSQL + pgvector** — replace LanceDB / Neo4j / MongoMock for simpler GB-scale deployment |
| [docs/TODO.md](./docs/TODO.md) | **Backlog** — OpenClaw vs RAG boundaries and follow-ups |
| [docs/testing.md](./docs/testing.md) | **Testing** — pytest layers, `DATABASE_URL`, troubleshooting |
| [docs/graph_constrained_queries.md](./docs/graph_constrained_queries.md) | **Constrained graph queries** — whitelist templates, debug APIs |
| [docs/whatif_tools.md](./docs/whatif_tools.md) | **What-If tools** — `/api/whatif/loan/compare`, no-RAG DAG |
| [docs/mvp_hybrid_rag.md](./docs/mvp_hybrid_rag.md) | **Hybrid RAG MVP** — router, dense+BM25, Neo4j, fusion rerank (`readme-v2-1.md`) |
| [docs/recall_evaluation.md](./docs/recall_evaluation.md) | **Recall@K / NDCG@K** — test cases, batch API, `scripts/run_recall_eval.py` |
| [docs/recall_ndcg_benchmark_plan.md](./docs/recall_ndcg_benchmark_plan.md) | **SciQ benchmark plan** — `scripts/benchmark_recall_ndcg.py`, reports `reports/recall_ndcg_benchmark_*.md` |
| [docs/dual-track-query-routing.md](./docs/dual-track-query-routing.md) | **Dual-Track Query Routing** — analysis vs. factual, grader/generator policies, demo Q&A |
| [docs/chat_test_memory_design.md](./docs/chat_test_memory_design.md) | Server-side multi-turn memory, `conversation_id` session design |
| [docs/doc-small-to-big-retrieval.md](./docs/doc-small-to-big-retrieval.md) | Parent-child chunk expansion (Small-to-Big retrieval) |
| [docs/doc-feauture-v1.md](./docs/doc-feauture-v1.md) | Initial Architecture RFC |
| [docs/doc-future.md](./docs/doc-future.md) | Enterprise principles for preventing bad answers |
| [readme-v2-1.md](./readme-v2-1.md) | Dual-track Hybrid RAG product spec + **Q&A test data** (related-party demo link & sample questions) |
| [docs/specs/2026-04-19-nvidia-rag-blueprint-gap-analysis.md](./docs/specs/2026-04-19-nvidia-rag-blueprint-gap-analysis.md) | **NVIDIA RAG Blueprint gap analysis** — S1-S7 feature roadmap, priority matrix, design decisions |

## 🤖 Agent Skills

This repository includes a specialized agent skill to help AI assistants (like Gemini CLI) better understand and interact with the MAP-RAG system.

*   **[map-rag-api](docs/skills/map-rag-api/SKILL.md)**: Provides the AI agent with a complete reference for the MAP-RAG API endpoints, feature flags, and architectural constraints.

**How to prompt your agent to use it:**
> "Please activate the `map-rag-api` skill. Use it to upload `data/demo_related_party.txt` to the API and then query the knowledge base about the related parties."
