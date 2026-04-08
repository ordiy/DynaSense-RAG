# MAP-RAG — Agent Entry Point

Hybrid RAG API: FastAPI + PostgreSQL (pgvector + JSONB + Apache AGE) + LangGraph + Google Vertex AI Gemini.
Answers questions from an uploaded knowledge base via vector search, full-text search, and knowledge graph retrieval.
Architecture: Small-to-Big retrieval (child chunk → parent expansion) with Jina reranking and LLM grading.

## Quick Start

```bash
docker compose up --build   # full stack (app + postgres); copy .env.example → .env first
pytest tests/ -x -q         # unit tests (most run without a live DB)
make test                   # lint-imports + pytest (recommended for CI)
```

## Architecture Layers (dependency direction: top → bottom only)

```
src/api/           ← HTTP routers, schemas, guardrails, session state    (top)
src/rag_core.py    ← LangGraph RAG pipeline, embeddings, Jina segmenter/reranker
src/hybrid_rag.py  ← hybrid router (VECTOR/GRAPH/GLOBAL/HYBRID), FTS, graph
src/agentic_rag.py ← ReAct agentic retrieval loop (feature-flagged, off by default)
src/retrieval_tools.py ← LangChain @tool wrappers for agentic loop
src/core/          ← cross-cutting: config, exceptions, citations, query anchors
src/infrastructure/persistence/ ← PostgreSQL adapters (pgvector, JSONB, AGE, FTS)
src/domain/        ← abstract interfaces only — no implementations          (bottom)
```

## Critical Rules

1. `src/rag_core.py` must NOT import `hybrid_rag` or `agentic_rag` at module level —
   use local imports *inside functions* to prevent circular dependencies.
2. `src/infrastructure/` must NOT import `api/`, `rag_core`, `hybrid_rag`, or `agentic_rag`.
3. `src/core/` must NOT import `api/`, `rag_core`, `hybrid_rag`, or `agentic_rag`.
4. New storage backends must implement `src/domain/interfaces/` — do not wire persistence
   modules directly into `src/api/routers/`.
5. In tests: `patch` at the **source** module path (e.g. `src.infrastructure.persistence.postgres_connection.get_pool`),
   not the importing module (e.g. NOT `src.hybrid_rag.get_pool`).

Rules 2–3 enforced by `lint-imports` (see `.importlinter`).
Rule 1 enforced by `scripts/check_rag_core_imports.py` (AST check — import-linter cannot
distinguish module-level from function-level imports).
Run `make lint` to check all three.

## Key Docs

- [`docs/architecture.md`](docs/architecture.md) — full layer diagram, all workflows, design patterns
- [`docs/TODO.md`](docs/TODO.md) — responsibility boundary: MAP-RAG vs OpenClaw
- [`docs/testing.md`](docs/testing.md) — conftest fixtures, `no_db_settings`, `clean_settings_cache`
- [`docs/bitter_lesson_roadmap.md`](docs/bitter_lesson_roadmap.md) — metrics-driven retrieval improvement roadmap
- [`docs/mvp_hybrid_rag.md`](docs/mvp_hybrid_rag.md) — hybrid router design (VECTOR/GRAPH/GLOBAL/HYBRID)
- [`docs/recall_evaluation.md`](docs/recall_evaluation.md) — Recall@K / NDCG@K evaluation methodology

## Common Pitfalls

- `fts_parent_documents` imports `get_pool` **inside the function body**, not at module level.
  Patch `src.infrastructure.persistence.postgres_connection.get_pool`, NOT `src.hybrid_rag.get_pool`.
- Session state (`src/api/state.py`) is in-process memory — breaks under multi-instance deployment.
  Production needs Redis or a DB-backed session store.
- xlsx files with special characters in filenames may land with 0 rows in `kb_embedding`.
  Fix: re-upload the file via `POST /api/upload`.
