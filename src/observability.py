"""Backward-compatible re-export; prefer ``src.core.langsmith_tracing`` in new code."""

from src.core.langsmith_tracing import init_langsmith_tracing

__all__ = ["init_langsmith_tracing"]
