"""
Central settings (Pydantic Settings). Prefer this over scattered ``os.environ`` in new code.

Existing modules (e.g. ``rag_core``) may still read env directly; migrate incrementally.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """HTTP / session limits and feature flags for the API layer."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    debug_data_api: bool = Field(default=True, description="Enable /api/debug/* read-only browser")
    max_upload_bytes: int = Field(default=2 * 1024 * 1024)
    max_analytics_upload_bytes: int = Field(
        default=10 * 1024 * 1024,
        description="CSV/TSV/XLSX profiling endpoint body limit.",
    )
    max_analytics_rows: int = Field(default=100_000, ge=1, le=1_000_000)
    max_analytics_cols: int = Field(default=256, ge=1, le=2048)
    max_query_len: int = Field(default=2048)
    max_expected_substring_len: int = Field(default=512)
    task_ttl_seconds: int = Field(default=3600)
    chat_session_ttl_seconds: int = Field(default=3600)
    chat_memory_query_budget: int = Field(default=1800)
    chat_max_stored_history_chars: int = Field(default=20000)
    chat_memory_assistant_clip_chars: int = Field(default=280)

    # --- Storage (PostgreSQL + pgvector + JSONB; see docs/postgresql_storage_roadmap.md) ---
    database_url: str | None = Field(
        default=None,
        description="Required at runtime for RAG: postgresql://user:pass@localhost:5432/map_rag",
    )
    graph_backend: Literal["age", "relational"] = Field(
        default="age",
        description="age = Apache AGE Cypher graph; relational = kg_triple SQL table (fallback).",
    )
    age_graph_name: str = Field(
        default="map_rag_kg",
        description="AGE graph name for cypher('graph_name', ...).",
    )

    # --- PostgreSQL pool ---
    pg_pool_max_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum connections in the psycopg ConnectionPool.",
    )

    # --- Retrieval / generation (vector-only LangGraph path) ---
    rag_vector_rerank_top_n: int = Field(
        default=5,
        ge=2,
        le=20,
        description=(
            "Jina rerank: how many parent-level passages feed grader+generator on the vector LangGraph path. "
            "Higher uses more compute (rerank + LLM context) but lets the model fuse more evidence (Bitter Lesson)."
        ),
    )

    # --- Hybrid RAG feature flag ---
    hybrid_rag_enabled: bool = Field(
        default=True,
        description="Enable Hybrid RAG (router + BM25 + graph). False = vector-only LangGraph path.",
    )

    # --- Hybrid RAG tuning knobs ---
    mmr_enabled: bool = Field(default=False, description='Apply MMR after Jina reranking.')
    mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0, description='MMR tradeoff: 0=diversity, 1=relevance.')
    hybrid_fusion_top_n: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of passages returned by Jina rerank in the hybrid fusion step.",
    )
    hybrid_bm25_top_child: int = Field(
        default=12,
        ge=1,
        le=100,
        description="Number of top BM25 child-chunk hits before parent expansion.",
    )
    hybrid_dense_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of dense (pgvector) child chunks retrieved for parent expansion.",
    )
    hybrid_rerank_pool_size: int = Field(
        default=40,
        ge=1,
        le=200,
        description="Maximum candidate documents passed to Jina cross-encoder reranker in hybrid fusion.",
    )

    # --- Query Expansion ---
    query_expansion_enabled: bool = Field(
        default=False,
        description='Enable LLM query expansion before retrieval. Adds ~1 LLM call but improves recall on synonym/terminology-heavy queries.',
    )

    # --- MMR Diversification ---
    mmr_enabled: bool = Field(
        default=False,
        description="Apply MMR diversity filter after Jina reranking. Reduces redundant passages; most useful for overview queries.",
    )
    mmr_lambda: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="MMR trade-off: 1.0 = pure relevance (rank order), 0.0 = pure diversity.",
    )

    # --- Agentic (ReAct) retrieval ---
    agentic_retrieval_enabled: bool = Field(
        default=False,
        description=(
            "Enable ReAct agentic retrieval loop (LangGraph create_react_agent). "
            "When True, run_chat_pipeline dispatches here before hybrid/vector fallbacks. "
            "Disabled by default: higher latency (~2× hybrid) but better multi-hop recall."
        ),
    )
    agentic_max_steps: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum tool-call iterations for the agentic ReAct loop.",
    )

    # --- Ingestion ---
    image_caption_enabled: bool = Field(
        default=False,
        description='Enable Gemini Vision captioning of PDF-embedded images during ingestion. Adds LLM calls per image.'
    )
    skip_graph_ingest: bool = Field(
        default=False,
        description="Skip LLM triple extraction during document ingestion (e.g. for benchmarks).",
    )

    # --- Inference ---
    inference_provider: Literal["vertex", "openai_compat", "anthropic"] = Field(
        default="vertex",
        description="The inference provider for LLMs and embeddings. Use openai_compat for NIM/Ollama/vLLM."
    )
    inference_llm_model: str = Field(
        default="gemini-2.5-flash",
        description="The model name for the LLM."
    )
    inference_embedding_model: str = Field(
        default="text-embedding-004",
        description="The model name for embeddings."
    )
    inference_base_url: str | None = Field(
        default=None,
        description="Optional base URL for OpenAI-compatible endpoints."
    )
    inference_api_key: str | None = Field(
        default=None,
        description="Optional API key for OpenAI-compatible or Anthropic providers."
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
