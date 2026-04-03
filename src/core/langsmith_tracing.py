"""
LangSmith observability for LangChain / LangGraph.

Docs: https://docs.langchain.com/langsmith/observability
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def init_langsmith_tracing() -> bool:
    """
    Configure process environment for LangSmith tracing.

    Call **once** at process startup, **before** importing modules that construct
    LangChain chat models or LangGraph graphs (e.g. ``src.rag_core``).

    Returns True if tracing is turned on, False otherwise.
    """
    key = (os.environ.get("LANGCHAIN_API_KEY") or os.environ.get("LANGSMITH_API_KEY") or "").strip()
    if not key:
        logger.info("LangSmith: no LANGCHAIN_API_KEY (or LANGSMITH_API_KEY); tracing disabled.")
        return False

    tracing = os.environ.get("LANGCHAIN_TRACING_V2", "true").lower() in ("1", "true", "yes")
    if not tracing:
        logger.info("LangSmith: LANGCHAIN_TRACING_V2 is false; tracing disabled.")
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = key

    project = (
        os.environ.get("LANGCHAIN_PROJECT")
        or os.environ.get("LANGSMITH_PROJECT")
        or os.environ.get("LANGSMITH_PROJECT_NAME")
    )
    if project:
        os.environ["LANGCHAIN_PROJECT"] = project.strip()

    endpoint = os.environ.get("LANGCHAIN_ENDPOINT") or os.environ.get("LANGSMITH_ENDPOINT")
    if endpoint:
        os.environ["LANGCHAIN_ENDPOINT"] = endpoint.rstrip("/")

    logger.info(
        "LangSmith tracing enabled (project=%s).",
        project or "(default project)",
    )
    return True
