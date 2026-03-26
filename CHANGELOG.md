# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v1.6.0-mvp-ui-performance] - 2026-03-24
### Added
- Implemented Cyberpunk/Glassmorphism theme for the Web GUI.
- Added `marked.js` in the frontend to render LLM responses in formatted Markdown instead of raw text.
- Added visual loading spinners and status indicators during chat synthesis.
- Python standard `logging` configured in `rag_core.py` for better debug visibility.

### Changed
- **Performance**: Solved the N+1 LLM query bottleneck in the LangGraph `grade_documents_node` by batching retrieved contexts into a single evaluation prompt.
- **Performance**: Downgraded the final generation node model from `gemini-2.5-pro` to `gemini-1.5-flash` to drastically reduce answer synthesis latency. Chat API response time significantly improved from ~20s to < 5s.

## [v1.5.0-mvp-refactored] - 2026-03-24
### Changed
- Completely refactored project directory structure into `src/`, `scripts/`, `docs/`, and `reports/` for enterprise maintainability.

## [v1.4.0-mvp-final] - 2026-03-24
### Added
- Created comprehensive `README.md` containing architectural philosophy, installation instructions, and benchmark summaries.
- Configured strict `.gitignore` and `.dockerignore` files to prevent large artifacts (like local `.venv` or LanceDB binaries) from cluttering source control.
- Dockerized the application with a `python:3.12-slim` based `Dockerfile`.

## [v1.3.0-mvp-benchmark] - 2026-03-24
### Added
- Fully-fledged evaluation test suite running against the public HuggingFace `SciQ` dataset.
- Added NDCG@K and Recall@K mathematical implementations for rigorous search relevance scoring.
- Web GUI now contains a dedicated "Evaluation" tab to dynamically calculate NDCG scores for specific queries and substrings.

### Fixed
- Fixed LanceDB insertion bug where passing `mode="overwrite"` and proper batch chunking was required to bypass Vertex AI's strict 250 instance per-request payload limit.

## [v1.2.1-mvp-stable] - 2026-03-24
### Changed
- Architectural pivot: Officially drafted and committed `doc-future.md` explaining the strict rejection of real-time "Query Rewrite" in favor of strict anti-hallucination policies.

## [v1.2.0-mvp] - 2026-03-24
### Added
- Replaced the naive keyword mock with the actual `jina-reranker-v2-base-multilingual` Cross-Encoder API.
- Expanded initial vector search K-value from 5 to 10 to give the Reranker a larger candidate pool, heavily improving final hit rates.

## [v1.1.0-mvp] - 2026-03-24
### Added
- Integrated `jina-segmenter` API for intelligent, semantic-aware document chunking (handling `\n\n` boundary splits).
- Simulated a Rerank node fetching original metadata-linked texts from an async local `mongomock` collection.

## [v1.0.0-mvp] - 2026-03-24
### Added
- Initial project scaffold created based on `doc-feauture-v1.md`.
- Basic `LangGraph` state machine implementation (Retrieve -> Grade -> Rewrite/Generate).
- LanceDB vector store integration using Google `text-embedding-004`.
- Setup a REST API backend with `FastAPI`.
