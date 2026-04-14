"""
Unit tests for src/core/faithfulness.py

All tests are pure unit tests — no live LLM, no database, no GCP credentials.
The LLM is replaced with a lightweight mock that returns pre-built FaithfulnessVerdict objects.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.core.faithfulness import (
    FaithfulnessVerdict,
    judge_faithfulness,
    verdict_to_score,
)


# ---------------------------------------------------------------------------
# verdict_to_score
# ---------------------------------------------------------------------------

def test_verdict_to_score_full():
    assert verdict_to_score("full") == 1.0


def test_verdict_to_score_partial():
    assert verdict_to_score("partial") == 0.5


def test_verdict_to_score_none():
    assert verdict_to_score("none") == 0.0


def test_verdict_to_score_unknown_defaults_zero():
    """Unknown verdicts from a malfunctioning judge should not raise — default to 0."""
    assert verdict_to_score("random_garbage") == 0.0


# ---------------------------------------------------------------------------
# judge_faithfulness — mocked LLM, happy paths
# ---------------------------------------------------------------------------

def _make_llm_mock(verdict: str, reasoning: str = "test") -> MagicMock:
    """Return a mock LLM whose with_structured_output().invoke() returns FaithfulnessVerdict."""
    mock_llm = MagicMock()
    verdict_obj = FaithfulnessVerdict(verdict=verdict, reasoning=reasoning)
    mock_llm.with_structured_output.return_value.invoke.return_value = verdict_obj
    return mock_llm


def test_judge_faithfulness_full_verdict():
    llm = _make_llm_mock("full", "All claims are directly in Passage 1.")
    passages = ["The company revenue was 100M in 2024."]
    answer = "The company earned 100M in 2024."

    result = judge_faithfulness(answer, passages, llm)

    assert result["faithfulness_score"] == 1.0
    assert result["faithfulness_verdict"] == "full"
    assert "All claims" in result["faithfulness_reasoning"]


def test_judge_faithfulness_partial_verdict():
    llm = _make_llm_mock("partial", "Most facts are present; minor inference detected.")
    passages = ["Revenue grew by 20%."]
    answer = "Revenue grew strongly, likely due to market expansion."

    result = judge_faithfulness(answer, passages, llm)

    assert result["faithfulness_score"] == 0.5
    assert result["faithfulness_verdict"] == "partial"


def test_judge_faithfulness_none_verdict():
    llm = _make_llm_mock("none", "Answer fabricates specific figures not in passages.")
    passages = ["The project was approved."]
    answer = "The project was approved with a budget of $5M in Q3."

    result = judge_faithfulness(answer, passages, llm)

    assert result["faithfulness_score"] == 0.0
    assert result["faithfulness_verdict"] == "none"


# ---------------------------------------------------------------------------
# judge_faithfulness — error / edge cases
# ---------------------------------------------------------------------------

def test_judge_faithfulness_llm_exception_returns_safe_fallback():
    """If the judge LLM raises an exception, the function must not propagate it."""
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("network timeout")

    result = judge_faithfulness("some answer", ["some passage"], mock_llm)

    assert result["faithfulness_score"] == 0.0
    assert result["faithfulness_verdict"] == "judge_error"
    assert "network timeout" in result["faithfulness_reasoning"]


def test_judge_faithfulness_empty_answer():
    """An empty answer should short-circuit without calling the LLM."""
    mock_llm = MagicMock()

    result = judge_faithfulness("", ["passage text"], mock_llm)

    mock_llm.with_structured_output.assert_not_called()
    assert result["faithfulness_score"] == 0.0
    assert result["faithfulness_verdict"] == "none"


def test_judge_faithfulness_whitespace_only_answer():
    """Whitespace-only answer treated as empty."""
    mock_llm = MagicMock()
    result = judge_faithfulness("   \n  ", ["passage text"], mock_llm)
    assert result["faithfulness_score"] == 0.0


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def test_faithfulness_verdict_model_validates_literal():
    v = FaithfulnessVerdict(verdict="partial", reasoning="ok")
    assert v.verdict == "partial"
    assert v.reasoning == "ok"


def test_faithfulness_verdict_model_rejects_invalid_verdict():
    with pytest.raises(Exception):
        FaithfulnessVerdict(verdict="maybe", reasoning="ok")


# ---------------------------------------------------------------------------
# EvalRequest schema — new compute_faithfulness field
# ---------------------------------------------------------------------------

def test_eval_request_faithfulness_field_defaults_false():
    from src.api.schemas import EvalRequest

    req = EvalRequest(query="test query", expected_substring="test")
    assert req.compute_faithfulness is False


def test_eval_request_faithfulness_field_accepts_true():
    from src.api.schemas import EvalRequest

    req = EvalRequest(query="test query", expected_substring="test", compute_faithfulness=True)
    assert req.compute_faithfulness is True


def test_eval_batch_request_faithfulness_field():
    from src.api.schemas import EvalBatchCase, EvalBatchRequest

    req = EvalBatchRequest(
        cases=[EvalBatchCase(id="q1", query="hello", expected_substring="world")],
        compute_faithfulness=True,
    )
    assert req.compute_faithfulness is True


# ---------------------------------------------------------------------------
# aggregate_mean — faithfulness key included when present
# ---------------------------------------------------------------------------

def test_aggregate_mean_includes_faithfulness():
    from src.recall_metrics import aggregate_mean, metrics_for_hit

    m1 = {**metrics_for_hit(0), "faithfulness_score": 1.0}
    m2 = {**metrics_for_hit(2), "faithfulness_score": 0.5}

    mean = aggregate_mean([m1, m2])

    assert "faithfulness_score" in mean
    assert abs(mean["faithfulness_score"] - 0.75) < 1e-9


def test_aggregate_mean_without_faithfulness_unchanged():
    """Existing test sets without faithfulness_score should still work fine."""
    from src.recall_metrics import aggregate_mean, metrics_for_hit

    mean = aggregate_mean([metrics_for_hit(0), metrics_for_hit(-1)])
    assert "faithfulness_score" not in mean
    assert "recall@10" in mean


def test_aggregate_mean_partial_faithfulness_entries():
    """Only cases that ran faithfulness should contribute to its mean."""
    from src.recall_metrics import aggregate_mean, metrics_for_hit

    m1 = {**metrics_for_hit(0), "faithfulness_score": 0.5}
    m2 = metrics_for_hit(1)  # no faithfulness

    mean = aggregate_mean([m1, m2])
    # faithfulness_score not in m2, so key may be absent or only from m1
    if "faithfulness_score" in mean:
        assert abs(mean["faithfulness_score"] - 0.5) < 1e-9
