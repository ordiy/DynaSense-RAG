"""Guardrail pattern scan (no FastAPI import)."""

import pytest

from src.api.guardrails import (
    guard_query_or_raise,
    scan_query_for_sensitive_patterns,
    should_block_on_hits,
)
from src.core.exceptions import QueryGuardrailError


def test_scan_finds_long_digit_run():
    hits = scan_query_for_sensitive_patterns("pay with 4532015112830366")
    assert "credit_card_like" in hits


def test_guard_does_not_block_by_default():
    hits = scan_query_for_sensitive_patterns("4532015112830366")
    assert hits
    assert should_block_on_hits(hits) is False
    guard_query_or_raise("4532015112830366")  # no raise


def test_guard_blocks_when_env_set(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BLOCK_SUSPECT_PII", "true")
    with pytest.raises(QueryGuardrailError):
        guard_query_or_raise("card 4532015112830366")
