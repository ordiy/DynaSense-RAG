# MVP: Routing + Hybrid RAG (Dual Recall)

This document implements the architecture described in **`readme-v2-1.md`** at **MVP** scope: a query router, dense + BM25 lexical retrieval, PostgreSQL-backed graph recall with linearization, optional global graph summary, and a **unified cross-encoder rerank** (Top‑K) before the existing anti-hallucination grader + generator.

---

## Goals (MVP)

1. **Query Router** — classify each user question into one of:
   - `VECTOR` — passage-level facts, definitions, troubleshooting, or lexical detail in documents.
   - `GRAPH` — multi-hop relationships (ownership, related parties, subsidiaries, dependencies).
   - `GLOBAL` — whole-corpus / high-level overview.
   - `HYBRID` — needs both narrative evidence and structured relations.

2. **Dual-engine indexing (offline)**
   - **Vector**: PostgreSQL `kb_embedding` + Vertex embeddings (child chunks).
   - **Graph**: LLM extracts `(subject, predicate, object)` triples per chunk + merge into PostgreSQL (Apache AGE or relational `kg_triple`) with **`chunk_id`** provenance.

3. **Online retrieval**
   - **VECTOR path**: dense retrieval (Small-to-Big parent expansion) + **BM25** over child chunks → parent expansion → **fusion rerank** (Top‑5 by default).
   - **GRAPH path**: LLM extracts **keywords** from the question → bounded keyword match on entity names / relation type (SQL or AGE) → **linearized** triples as text context.
   - **GLOBAL path**: graph summary (entity/relationship counts + sample names) + **small dense anchor** (top‑2 parents) for grounding.
   - **HYBRID path**: union vector + graph candidates → **single rerank** pool.

4. **Unified fusion & rerank** — Jina reranker scores all candidates (vector, BM25, graph) as plain text passages; **hard cut** to `HYBRID_FUSION_TOP_N` (default **5**). **Before** this step, optional **query anchor filtering** (`QUERY_ANCHOR_FILTER`, default on) can remove candidates that contain none of the detected anchors (e.g. institution names, A-share codes); see `src/core/query_anchors.py` (no matches → fail-open, keep the original list).

5. **Downstream** — unchanged **grader** + **generator** (including analysis vs factual dual-track prompts).

---

## Components (code map)

| Module | Responsibility |
|--------|-------------------|
| `src/graph_store.py` | PostgreSQL graph: `merge_triple`, `query_relationships_by_keywords`, `global_graph_summary`, `linearize_rows`. |
| `src/hybrid_rag.py` | Router (`RouteDecision`), keyword extraction, triple extraction for ingest, BM25 pool, `collect_vector_path`, `run_hybrid_chat_pipeline`. |
| `src/rag_core.py` | `retrieve_parent_documents_expanded` (shared dense path), `process_document_task` → calls `ingest_chunks_to_graph`, `run_chat_pipeline` → hybrid by default with env fallback. |

---

## Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `DATABASE_URL` | *(none)* | PostgreSQL URL (required for storage) |
| `GRAPH_BACKEND` / `AGE_GRAPH_NAME` | see `src/core/config.py` | AGE vs relational `kg_triple` |
| `HYBRID_RAG_ENABLED` | `true` | `false` → legacy LangGraph vector-only pipeline |
| `HYBRID_FUSION_TOP_N` | `5` | Final rerank cut |
| `HYBRID_BM25_TOP_CHILD` | `12` | BM25 child hits before parent expansion |
| `HYBRID_DENSE_K` | `10` | Dense child `k` |

---

## Local PostgreSQL (Docker)

```bash
docker compose -f docker-compose.postgres.yml up -d
export DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:5433/map_rag
```

---

## Design rationale (step-by-step)

### 1) Why a router?

Running graph + vector + global on every query is **slow and expensive**. The router sends most “simple” questions to the **VECTOR** path only; relationship-heavy questions use **GRAPH** or **HYBRID**.

### 2) Why BM25 + dense?

Dense vectors excel at **semantic similarity**; BM25 excels at **lexical / rare token** overlap (e.g. stock codes, exact product names). **Fusion rerank** lets the cross-encoder pick the best of both.

### 3) Why graph linearization?

Neo4j returns **nodes and edges**. The LLM consumes **text**. We convert each row to a line:

`EntityA —[predicate]→ EntityB (chunk_id=..., source=...)`

So graph evidence participates in the **same reranker** as document chunks.

### 4) Why LLM triple extraction (MVP)?

Production systems often use **NER + rule-based** or **schema-guided** extraction. For MVP we use **structured-output LLM** triples per chunk to populate Neo4j quickly; `chunk_id` is always attached for **traceability**.

### 5) Fail‑closed behavior

If Neo4j is down, `GRAPH` / `GLOBAL` / `HYBRID` **fall back** to `VECTOR` retrieval. If retrieval is empty, the answer path returns the **same blocking message** as the rest of the pipeline.

---

## Demo Q&A (Recall)

Use a domain document (e.g. related-party disclosure). After uploading `data/demo_related_party.txt`:

| Question (intent) | Expected route (typical) |
|-------------------|---------------------------|
| 中信银行 2025 关联交易报告的主要内容是什么？ | `VECTOR` |
| 中国中信银行的关联方有哪些？ | `GRAPH` or `HYBRID` |
| 知识库整体涵盖哪些实体？ | `GLOBAL` |

*(Exact route labels depend on the router LLM; check API response fields `route` / `effective_route`.)*

---

## API response shape

`POST /api/chat` returns the usual `answer`, `context_used`, `logs`, plus:

- `route` — original router label  
- `effective_route` — after Neo4j fallbacks  
- `router_reason` — short rationale from the router model  

---

## References

- `readme-v2-1.md` — product architecture spec  
- [Neo4j RAG cookbook (Hugging Face)](https://huggingface.co/learn/cookbook/zh-CN/rag_with_knowledge_graphs_neo4j)  
