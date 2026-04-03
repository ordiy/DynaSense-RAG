"""
Central settings (Pydantic Settings). Prefer this over scattered ``os.environ`` in new code.

Existing modules (e.g. ``rag_core``) may still read env directly; migrate incrementally.
"""
from __future__ import annotations

from functools import lru_cache

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
    max_query_len: int = Field(default=2048)
    max_expected_substring_len: int = Field(default=512)
    task_ttl_seconds: int = Field(default=3600)
    chat_session_ttl_seconds: int = Field(default=3600)
    chat_memory_query_budget: int = Field(default=1800)
    chat_max_stored_history_chars: int = Field(default=20000)
    chat_memory_assistant_clip_chars: int = Field(default=280)


@lru_cache
def get_settings() -> Settings:
    return Settings()
