---
name: map-rag-api
description: Use when interacting with the MAP-RAG agent API — uploading documents, querying the knowledge base, managing sessions, or configuring retrieval behavior (hybrid RAG, graph search, agentic loop, MMR, query expansion).
---

# MAP-RAG Agent API Reference

## Overview

MAP-RAG is a Hybrid RAG service (FastAPI + PostgreSQL pgvector/AGE + LangGraph + Gemini).
It answers questions from an uploaded knowledge base via vector search, full-text search, and knowledge graph retrieval.
Base URL: `http://localhost:8000` (default local).

## Quick Start

```bash
# 1. Copy env template and fill in required vars
cp .env.example .env

# 2. Start full stack
docker compose up --build

# 3. Verify
curl http://localhost:8000/
```

**Required env vars:**
| Var | Purpose |
|-----|---------|
| `DATABASE_URL` | `postgresql://user:pass@localhost:5432/map_rag` |
| `GOOGLE_CLOUD_PROJECT` | GCP project (Vertex AI) |
| `GCP_SA_KEY_FILE` | Path to GCP service account JSON |
| `JINA_API_KEY` | Jina reranking API key |

## Core API Endpoints

### 1. Upload a Document
```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@contract.pdf"
# Returns: {"task_id": "uuid"}

# Poll for completion
curl http://localhost:8000/api/tasks/{task_id}
# Returns: {"status": "completed|processing|failed", ...}
```

Supported formats: `.pdf`, `.docx`, `.xlsx`, `.txt`

### 2. Ask a Question (Single-turn)
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the IP ownership clauses in contract A?"}'
```

Response:
```json
{
  "answer": "...",
  "context_used": ["...passage 1...", "...passage 2..."],
  "citations": [{"source": "contract_a.pdf", "page": 3}],
  "route": "hybrid|vector|graph|agentic",
  "logs": ["...breadcrumb trace..."]
}
```

### 3. Streaming Chat (SSE)
```python
import httpx, json

with httpx.stream("POST", "http://localhost:8000/api/chat/stream",
                  json={"query": "Summarize the employment contract"}) as r:
    for line in r.iter_lines():
        if line.startswith("data:"):
            event = json.loads(line[5:])
            if event["type"] == "token":
                print(event["content"], end="", flush=True)
            elif event["type"] == "done":
                break
```

### 4. Multi-turn Session Chat
```bash
# Start / continue a conversation
curl -X POST http://localhost:8000/api/chat/session \
  -H "Content-Type: application/json" \
  -d '{"query": "Who invented patent P1?", "conversation_id": "my-session-1"}'

# Returns same conversation_id — reuse it for follow-ups
```

### 5. Evaluation (Recall@K / NDCG@K)
```bash
curl -X POST http://localhost:8000/api/evaluate \
  -H "Content-Type: application/json" \
  -d '{"query": "patent ownership", "expected_sources": ["contract_a.pdf"]}'
```

### 6. Feedback
```bash
curl -X POST http://localhost:8000/api/feedback \
  -H "Content-Type: application/json" \
  -d '{"query": "...", "rating": -1, "comment": "wrong answer", "conversation_id": "..."}'
# rating: 1=thumbs up, 0=neutral, -1=thumbs down

# View negative feedback
curl http://localhost:8000/api/feedback/negative
```

### 7. Analytics (CSV/XLSX Profiling)
```bash
curl -X POST http://localhost:8000/api/analytics/profile \
  -F "file=@data.csv"
```

## Feature Flags (`.env`)

| Flag | Default | Effect |
|------|---------|--------|
| `HYBRID_RAG_ENABLED` | `true` | Router: VECTOR/GRAPH/GLOBAL/HYBRID |
| `AGENTIC_RETRIEVAL_ENABLED` | `false` | ReAct multi-hop loop (~2× latency) |
| `QUERY_EXPANSION_ENABLED` | `false` | LLM generates query rephrasings |
| `MMR_ENABLED` | `false` | Post-rerank diversity filter |
| `IMAGE_CAPTION_ENABLED` | `false` | Gemini Vision captions PDF images |
| `SKIP_GRAPH_INGEST` | `false` | Skip triple extraction on upload |
| `GRAPH_BACKEND` | `age` | `age` (Cypher) or `relational` |
| `INFERENCE_PROVIDER` | `vertex` | `vertex` / `openai_compat` / `anthropic` |
| `INFERENCE_LLM_MODEL` | `gemini-2.5-flash` | Model name |

## Retrieval Tuning

```bash
RAG_VECTOR_RERANK_TOP_N=5      # Final passages fed to LLM
HYBRID_DENSE_K=10              # Dense vector candidates
HYBRID_BM25_TOP_CHILD=12       # FTS candidates
HYBRID_RERANK_POOL_SIZE=40     # Reranker pool size
MMR_LAMBDA=0.7                 # 1.0=pure relevance, 0.0=pure diversity
```

## Debug Browser (read-only, `DEBUG_DATA_API=true`)

```bash
GET /api/pg/summary            # Document counts
GET /api/pg/kb_embedding/rows  # Embedding table rows
GET /api/graph/summary         # KG triple summary
```

## Response `route` Field Values

| Route | Triggered When |
|-------|---------------|
| `vector` | Hybrid disabled OR fallback |
| `hybrid` | Multi-signal fusion (dense+FTS+graph) |
| `graph` | Query detected as graph/relationship lookup |
| `global` | Query needs full-corpus summary |
| `agentic` | `AGENTIC_RETRIEVAL_ENABLED=true` |

## Architecture Layers (don't break these in new code)

```
api/           ← HTTP routers                        (top)
rag_core.py    ← LangGraph pipeline
hybrid_rag.py  ← hybrid router + multi-path retrieval
agentic_rag.py ← ReAct loop
core/          ← config, exceptions (no upward imports)
infrastructure/persistence/ ← PostgreSQL adapters    (bottom)
domain/        ← abstract interfaces only
```

Run `make lint` to enforce import rules before committing.

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Patching `src.hybrid_rag.get_pool` in tests | Patch `src.infrastructure.persistence.postgres_connection.get_pool` |
| Session breaks under multi-worker | `state.py` is in-process; use Redis for production |
| XLSX with special chars uploads 0 rows | Re-upload via `POST /api/upload` |
| Circular import at module level in rag_core | Use local imports inside functions for hybrid_rag/agentic_rag |
