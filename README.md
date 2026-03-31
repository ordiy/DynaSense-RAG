# 🧠 DynaSense-RAG (MAP-RAG Architecture)

> **MAP-RAG**: Multi-resolution Agentic Perception Retrieval-Augmented Generation

An enterprise-grade RAG (Retrieval-Augmented Generation) architecture prototype focused on strict Anti-Hallucination mechanisms, intelligent semantic chunking, and Cross-Encoder reranking. 

## 🎯 Core Philosophy
**"No Answer is better than a Bad/Toxic Answer."**

In enterprise environments (legal, financial, internal HR policies), LLM hallucinations are unacceptable. This MVP explicitly **rejects** real-time generic Query Rewriting on the main pipeline to prevent "Intent Drift" (where specialized internal terms are rewritten into generic terms, losing their exact meaning) and to avoid unnecessary LLM latency.

Instead, this architecture achieves high precision through:
1. **Intelligent Chunking** (Jina Segmenter)
2. **High-Dimensional Vector Retrieval** (Google Vertex AI `text-embedding-004` + LanceDB)
3. **Cross-Encoder Semantic Reranking** (Jina Multilingual Reranker)
4. **Dual-Track Grader + Generator** (LangGraph state machine — strict for factual queries, analytically capable for reasoning queries)
5. **Server-Side Multi-Turn Memory** (conversation session with context-length control)



## 🏗️ Architecture Design (MAP-RAG)

```text
╔══════════════════════════════════════════════════════════════════════╗
║                     DATA INGESTION PIPELINE                         ║
╚══════════════════════════════════════════════════════════════════════╝

Raw Documents (TXT/MD)
      │
      ▼
[ Jina Semantic Segmenter ] ──(Chunking)──> Child Text Chunks
                                              │
                    ┌─────────────────────────┴──────────────────────────┐
                    ▼                                                    ▼
         [ Document DB (MongoMock) ]                    [ Vertex AI Embeddings ]
           Stores: full parent text                       text-embedding-004
           key: parent_id  ◄──── parent_id ────────────────────┤
                                                               ▼
                                                    [ Vector DB (LanceDB) ]
                                                      Stores: dense vectors
                                                      Metadata: parent_id

╔══════════════════════════════════════════════════════════════════════╗
║               RETRIEVAL & GENERATION PIPELINE                       ║
╚══════════════════════════════════════════════════════════════════════╝

  User Query ──────────────────────────────────┐
      │                                         │ (multi-turn)
      │                              [ Session Memory ]
      │                              conversation_id
      │                              history → context budget
      │                              _build_query_with_history()
      │                                         │
      ▼                                         ▼
[ LanceDB Vector Search ]  ←──── enriched query (with history)
   Top K=10 child chunks
      │
      ▼
[ Small-to-Big Expansion ]
   child_id → parent_id → full parent text
      │
      ▼
[ Jina Cross-Encoder Reranker ]
   Top K=3 high-precision parent docs
      │
      ▼
[ Query Type Detector ]   ← NEW: _is_analysis_query()
      │
      ├─────── Factual Query ──────────────────────────────────┐
      │        (lookup, definition, specific facts)            │
      │                                                        ▼
      │                                           [ GRADE_PROMPT (strict) ]
      │                                           "Does context contain
      │                                            a direct answer?"
      │                                                        │
      │                                            NO ──► [ Block / Fallback ]
      │                                            YES ──► [ GEN_PROMPT ]
      │                                                    "Strictly use context."
      │
      └─────── Analysis Query ────────────────────────────────┐
               (分析/影响/如何/为什么/规划/评估…)             │
               (analyze/impact/why/how/plan/risk…)            ▼
                                                 [ GRADE_ANALYSIS_PROMPT (relaxed) ]
                                                 "Does context contain ANY
                                                  topic-related background fact?"
                                                              │
                                                  NO ──► [ Block / Fallback ]
                                                  YES ──► [ GEN_ANALYSIS_PROMPT ]
                                                          "Ground facts + domain
                                                           reasoning. Label:
                                                           【文档事实】【分析推理】"
                                                              │
                                                              ▼
                                                   Final Synthesized Answer
```

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

---

## 🛠️ Tech Stack
* **Orchestration**: `LangGraph` & `LangChain`
* **Embedding Model**: Google Vertex AI `text-embedding-004`
* **LLM**: Google Vertex AI `gemini-2.5-pro`
* **Vector Database**: `LanceDB`
* **Semantic Chunking**: `Jina Segmenter API`
* **Reranker**: `jina-reranker-v2-base-multilingual`
* **Session Store**: In-memory `dict` with TTL (upgradeable to Redis)

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
export JINA_API_KEY="your-jina-api-key"

# 4. Start the web server
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
| [docs/dual-track-query-routing.md](./docs/dual-track-query-routing.md) | **Dual-Track Query Routing** — analysis vs. factual, grader/generator policies, demo Q&A |
| [docs/chat_test_memory_design.md](./docs/chat_test_memory_design.md) | Server-side multi-turn memory, `conversation_id` session design |
| [docs/doc-small-to-big-retrieval.md](./docs/doc-small-to-big-retrieval.md) | Parent-child chunk expansion (Small-to-Big retrieval) |
| [docs/doc-feauture-v1.md](./docs/doc-feauture-v1.md) | Initial Architecture RFC |
| [docs/doc-future.md](./docs/doc-future.md) | Enterprise principles for preventing bad answers |
