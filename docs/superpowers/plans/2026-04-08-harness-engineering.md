# Harness Engineering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a CLAUDE.md agent entry-point map and import-linter dependency contracts so AI agents can orient quickly and layer violations are caught by CI.

**Architecture:** Two config artifacts — `.importlinter` defines four forbidden-import contracts enforced by `lint-imports`; `CLAUDE.md` at the project root is a ~70-line navigational map pointing to `docs/` rather than duplicating it. A `Makefile` wires `lint-imports` before `pytest` so violations surface before unit tests run.

**Tech Stack:** `import-linter` (PyPI), `make`, Markdown.

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `requirements-ci.txt` | Modify | Add `import-linter` dev dependency |
| `.importlinter` | Create | Four forbidden-import contracts |
| `Makefile` | Create | `lint` and `test` targets |
| `CLAUDE.md` | Create | Agent entry-point map (~70 lines) |

---

## Task 1: Add import-linter and create `.importlinter`

**Files:**
- Modify: `requirements-ci.txt`
- Create: `.importlinter`

- [ ] **Step 1.1: Add import-linter to requirements-ci.txt**

Open `requirements-ci.txt`. Append one line at the end:

```
import-linter
```

Full file should look like:
```
# CI: unit tests + FastAPI-only tests (no Vertex / LangChain for full RAG).
pytest>=8.0
pypdf>=4.0.0
fastapi>=0.110
httpx>=0.27
pydantic>=2.0
pydantic-settings>=2.0
pandas>=2.0
numpy>=1.26
openpyxl>=3.1.0
psycopg[binary,pool]>=3.1.0
pgvector>=0.3.0
import-linter
```

- [ ] **Step 1.2: Install it**

```bash
pip install import-linter
```

Expected: `Successfully installed import-linter-X.X.X` (plus `grimp` dependency).

- [ ] **Step 1.3: Create `.importlinter`**

Create file at project root:

```ini
[importlinter]
root_packages =
    src

[importlinter:contract:1]
name = Infrastructure must not import Application or API
type = forbidden
source_modules =
    src.infrastructure
forbidden_modules =
    src.rag_core
    src.hybrid_rag
    src.agentic_rag
    src.retrieval_tools
    src.api

[importlinter:contract:2]
name = Core must not import Application or API
type = forbidden
source_modules =
    src.core
forbidden_modules =
    src.rag_core
    src.hybrid_rag
    src.agentic_rag
    src.api

[importlinter:contract:3]
name = Domain must not import any other src layer
type = forbidden
source_modules =
    src.domain
forbidden_modules =
    src.rag_core
    src.hybrid_rag
    src.agentic_rag
    src.api
    src.infrastructure
    src.core

[importlinter:contract:4]
name = rag_core must not module-level import hybrid_rag or agentic_rag
type = forbidden
source_modules =
    src.rag_core
forbidden_modules =
    src.hybrid_rag
    src.agentic_rag
ignore_imports =
    src.rag_core -> src.hybrid_rag
    src.rag_core -> src.agentic_rag
```

- [ ] **Step 1.4: Run lint-imports and verify all 4 contracts pass**

```bash
lint-imports
```

Expected output:
```
Contracts
---------
Infrastructure must not import Application or API KEPT
Core must not import Application or API KEPT
Domain must not import any other src layer KEPT
rag_core must not module-level import hybrid_rag or agentic_rag KEPT

Contracts: 0 broken, 4 kept
```

If any contract is BROKEN, read the violation path and fix the offending import before continuing.

- [ ] **Step 1.5: Commit**

```bash
git add requirements-ci.txt .importlinter
git commit -m "feat: add import-linter with 4 layer dependency contracts"
```

---

## Task 2: Verify contracts catch violations (deliberate violation test)

**Files:**
- Temporarily modify: `src/infrastructure/persistence/postgres_fts.py`

This task proves the linter actually works — not just that it runs. Do not skip.

- [ ] **Step 2.1: Add a deliberate violation**

Open `src/infrastructure/persistence/postgres_fts.py`. Add this import at the top of the file (after existing imports):

```python
import src.rag_core  # DELIBERATE VIOLATION — remove after test
```

- [ ] **Step 2.2: Run lint-imports and verify it fails**

```bash
lint-imports
```

Expected output (contract 1 must be BROKEN):
```
Contracts
---------
Infrastructure must not import Application or API BROKEN
...
  src.infrastructure.persistence.postgres_fts -> src.rag_core

Contracts: 1 broken, 3 kept
```

If the output says `0 broken, 4 kept` instead, the linter is not detecting the violation — stop and investigate (check `.importlinter` root_packages setting, or whether `src` is on the Python path).

- [ ] **Step 2.3: Remove the deliberate violation**

Delete the line you added in Step 2.1. `postgres_fts.py` must be restored to its original state.

- [ ] **Step 2.4: Confirm clean again**

```bash
lint-imports
```

Expected: `Contracts: 0 broken, 4 kept`

No commit needed — nothing changed permanently.

---

## Task 3: Create `Makefile`

**Files:**
- Create: `Makefile`

- [ ] **Step 3.1: Create Makefile at project root**

```makefile
.PHONY: lint test

lint:
	lint-imports

test:
	lint-imports && pytest tests/ -x -q
```

**Important:** The indentation on `lint-imports` and `pytest` lines must be a **tab character**, not spaces. Makefile targets require tabs.

- [ ] **Step 3.2: Verify `make lint` works**

```bash
make lint
```

Expected: same clean output as Task 1 Step 1.4 (`0 broken, 4 kept`).

- [ ] **Step 3.3: Verify `make test` works**

```bash
make test
```

Expected: `lint-imports` runs first (clean), then `pytest` runs. Final line should show passing tests, e.g.:
```
Contracts: 0 broken, 4 kept
143 passed, 3 skipped in X.XXs
```

If `make test` fails because `pytest` can't find modules, ensure you're running from the project root with the virtualenv active.

- [ ] **Step 3.4: Commit**

```bash
git add Makefile
git commit -m "feat: add Makefile with lint and test targets (lint-imports before pytest)"
```

---

## Task 4: Create `CLAUDE.md`

**Files:**
- Create: `CLAUDE.md`

- [ ] **Step 4.1: Create CLAUDE.md at project root**

```markdown
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

Rules 1–3 are enforced by `lint-imports` (see `.importlinter`). Run `make lint` to check.

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
```

- [ ] **Step 4.2: Verify line count stays within limit**

```bash
wc -l CLAUDE.md
```

Expected: 70 or fewer lines. If over 80, move content to `docs/` and replace with a one-line pointer.

- [ ] **Step 4.3: Cold-start check (manual)**

Read `CLAUDE.md` as if you have no other context. Confirm:
- You know how to start the app (1 command)
- You know which layer owns which responsibility
- You know the 5 rules without opening any other file
- You know where to look for more detail (6 doc pointers)

- [ ] **Step 4.4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add CLAUDE.md agent entry-point map (harness engineering)"
```

---

## Final Verification

- [ ] **Step 5.1: Run full suite**

```bash
make test
```

Expected:
```
Contracts: 0 broken, 4 kept
143 passed, 3 skipped in X.XXs
```

- [ ] **Step 5.2: Confirm all 4 files exist**

```bash
ls -1 CLAUDE.md .importlinter Makefile requirements-ci.txt
```

Expected: all 4 listed without errors.
