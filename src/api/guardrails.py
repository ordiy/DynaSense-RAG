"""
Lightweight request guardrails (MVP): length is enforced by Pydantic; this module
adds optional pattern checks for obviously sensitive tokens.

Design: fail-open for product iteration (log + flag) unless ``BLOCK_SUSPECT_PII=true``.
"""

from __future__ import annotations

import logging
import os
import re

from src.core.exceptions import QueryGuardrailError

logger = logging.getLogger(__name__)

# Credit card–like runs (16 digits with optional separators); CN ID 18-digit
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("credit_card_like", re.compile(r"\b(?:\d[ -]*?){15,19}\d\b")),
    ("cn_id_18", re.compile(r"\b[1-9]\d{5}(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b")),
)


def scan_query_for_sensitive_patterns(text: str) -> list[str]:
    """
    Return match type keys for known patterns (may overlap). Used for warnings
    and optional blocking.
    """
    if not (text or "").strip():
        return []
    hits: list[str] = []
    for name, pat in _PATTERNS:
        if pat.search(text):
            hits.append(name)
    return hits


def should_block_on_hits(hits: list[str]) -> bool:
    """When env BLOCK_SUSPECT_PII is truthy, block requests that matched patterns."""
    if not hits:
        return False
    return os.environ.get("BLOCK_SUSPECT_PII", "").lower() in ("1", "true", "yes")


def guard_query_or_raise(text: str) -> None:
    """
    Inspect user query; log hits; optionally raise so API returns 400.

    Raises ``QueryGuardrailError`` when blocking is enabled and a pattern matched.
    """
    hits = scan_query_for_sensitive_patterns(text)
    if not hits:
        return
    logger.info("Guardrail pattern hits: %s", hits)
    if should_block_on_hits(hits):
        raise QueryGuardrailError(
            "Request rejected: suspected sensitive pattern. Remove or redact and retry."
        )
