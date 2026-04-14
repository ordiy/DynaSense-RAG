# Harness Engineering Learnings — Design Spec

**Date**: 2026-04-08  
**Source**: [OpenAI — Harness engineering: leveraging Codex in an agent-first world](https://openai.com/index/harness-engineering/)  
**Scope**: Apply two highest-value harness engineering principles to MAP-RAG

---

## Background

OpenAI's harness engineering article describes how a 3–7 person team shipped ~1M lines of AI-generated code over 5 months by designing the *environment* around the agent, not just the prompts. The core insight: **the bottleneck in agent performance is environment design, not model intelligence.**

Three pillars were identified:
1. **Context Engineering** — docs as infrastructure; a short map file pointing to structured `docs/`, not an encyclopedia
2. **Architectural Constraint Enforcement** — layer rules mechanically enforced by linters; error messages inject corrective instructions directly into agent context
3. **Entropy Governance** — background agents continuously refactoring

This spec covers pillars 1 and 2, which offer the highest ROI for our current project size and maturity.

---

## What We're Building

### A — CLAUDE.md (Agent Entry Point Map)

A ~70-line file at the project root that serves as the agent's first read. It does not duplicate `docs/`; it is a navigational index with critical rules surfaced at the top.

**File**: `/CLAUDE.md`

**Structure**:

```
# MAP-RAG — Agent Entry Point
[3-line project summary]

## Quick Start
[2 commands: docker compose up / pytest]

## Architecture Layers
API → Application → Core → Infrastructure → Domain
(dependency direction: top imports bottom only)

## Critical Rules (5)
1. rag_core: no module-level import of hybrid_rag / agentic_rag — use local imports inside functions
2. infrastructure/: forbidden from importing api/ or application layer
3. core/: forbidden from importing api/ or application layer
4. New storage backends must implement domain/interfaces/, not be called directly by routers/
5. In tests: patch at source module path (e.g. src.hybrid_rag.fts_parent_documents), not the importing module

## Key Docs (6 pointers)
- docs/architecture.md       — full architecture (layers, workflows, design patterns)
- docs/TODO.md               — RAG vs OpenClaw responsibility boundary
- docs/testing.md            — conftest fixtures, no_db_settings usage
- docs/bitter_lesson_roadmap.md — metrics-driven retrieval improvement roadmap
- docs/mvp_hybrid_rag.md     — hybrid router design (VECTOR/GRAPH/GLOBAL/HYBRID)
- docs/recall_evaluation.md  — Recall@K / NDCG@K evaluation methodology

## Common Pitfalls (3)
- fts_parent_documents imports get_pool locally (inside the function) — patch src.infrastructure.persistence.postgres_connection.get_pool, not src.hybrid_rag.get_pool
- Chat session state is in-process memory — multi-instance deployment requires Redis
- xlsx files with special characters in filename may have 0 embedding rows — re-upload to fix
```

**Design constraints**:
- Total length must not exceed 80 lines. If it grows, move content to `docs/` and add a pointer.
- Critical Rules must match the import-linter contracts exactly — one source of truth for humans, one for machines.
- Common Pitfalls records only non-obvious facts (things not derivable from reading the code).

---

### B — import-linter (Mechanical Dependency Enforcement)

Static analysis tool that enforces the layer model at CI time. Uses `grimp` under the hood (AST-based import graph). Zero runtime cost.

**Install**: add `import-linter` to `requirements-dev.txt`

**Config file**: `/.importlinter` (project root)

#### Layer Model

```
src.api                                              ← top (can import anything)
src.rag_core / src.hybrid_rag / src.agentic_rag     ← application
src.core                                             ← cross-cutting
src.infrastructure                                   ← persistence
src.domain                                           ← base interfaces
```

#### Four Contracts

**Contract 1 — Infrastructure cannot import upward**
```ini
[importlinter:contract:1]
name = Infrastructure must not import Application or API
type = forbidden
source_modules = src.infrastructure
forbidden_modules =
    src.rag_core
    src.hybrid_rag
    src.agentic_rag
    src.retrieval_tools
    src.api
```

**Contract 2 — Core cannot import upward**
```ini
[importlinter:contract:2]
name = Core must not import Application or API
type = forbidden
source_modules = src.core
forbidden_modules =
    src.rag_core
    src.hybrid_rag
    src.agentic_rag
    src.api
```

**Contract 3 — Domain is fully isolated**
```ini
[importlinter:contract:3]
name = Domain must not import any other src layer
type = forbidden
source_modules = src.domain
forbidden_modules =
    src.rag_core
    src.hybrid_rag
    src.agentic_rag
    src.api
    src.infrastructure
    src.core
```

**Contract 4 — rag_core horizontal isolation (documented)**
```ini
[importlinter:contract:4]
name = rag_core must not module-level import hybrid_rag or agentic_rag
type = forbidden
source_modules = src.rag_core
forbidden_modules =
    src.hybrid_rag
    src.agentic_rag
ignore_imports =
    src.rag_core -> src.hybrid_rag
    src.rag_core -> src.agentic_rag
```

> Contract 4 note: existing local imports (inside `run_chat_pipeline`) are whitelisted via `ignore_imports`. The contract still catches any future *new* module-level imports of `hybrid_rag` or `agentic_rag` from `rag_core`, because `ignore_imports` whitelists exact module pairs, not wildcards. This is primarily a documentary contract — the pattern is enforced by CLAUDE.md rule #1 for humans and partially by the linter for machines.

#### CI Integration

Add to `Makefile`:

```makefile
.PHONY: lint test

lint:
	lint-imports

test:
	lint-imports && pytest tests/ -x -q
```

Add to `requirements-dev.txt`:
```
import-linter
```

#### Error message design

import-linter does not support custom message bodies. The corrective instruction is carried by the **contract name itself** (which always appears in the output). When a contract breaks, the agent sees:

```
ImportLinter: Contract 'Infrastructure must not import Application or API' BROKEN

  src.infrastructure.persistence.postgres_fts
    imports src.rag_core

Contracts: 1 broken, 3 kept
```

The contract name is the corrective instruction. This is why contract names are written as imperatives ("must not import"), not descriptions ("layer boundary"). The agent reads the name and knows exactly what rule was violated and which direction to fix it. For deeper context, `docs/architecture.md §4.1` is the lookup target — referenced in CLAUDE.md so the agent already knows where to look.

---

## What We Are NOT Doing

- **No entropy governance agents** (Pillar 3): deferred. Requires a more mature harness before background refactor agents add value rather than noise.
- **No tiered per-directory CLAUDE.md files**: over-engineering for 60 .py files. Revisit if the codebase grows past ~150 files.
- **No changes to retrieval logic, scoring, or pipelines**: this spec is environment-only.

---

## Success Criteria

1. `CLAUDE.md` exists at project root, passes a "cold start" test: a new agent session can locate key docs and rules without reading any other file first.
2. `lint-imports` passes cleanly on current codebase (all 4 contracts green).
3. A deliberate test violation (add `import src.rag_core` to `src/infrastructure/persistence/postgres_fts.py`) causes `lint-imports` to fail with a clear error message.
4. `make test` runs `lint-imports` before `pytest`, so CI catches layer violations before unit tests run.

---

## Files Changed

| File | Action |
|------|--------|
| `/CLAUDE.md` | Create |
| `/.importlinter` | Create |
| `/requirements-dev.txt` | Add `import-linter` |
| `/Makefile` | Add `lint` and update `test` targets |
