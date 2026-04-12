"""
Faithfulness LLM-as-a-Judge for RAG evaluation.

Faithfulness measures whether the generated answer is *grounded* in the retrieved
passages — complementing Recall@K / NDCG@K which only measure retrieval quality.

Design decisions:
- 3-level verdict (full / partial / none) rather than a continuous scale: judge LLMs
  have poor calibration on fine-grained numeric scores without domain fine-tuning;
  coarse verdicts are more reproducible. Normalized to float post-hoc.
- Dependency injection for the LLM: callers (rag_core) pass their existing ChatVertexAI
  instance, keeping this module free of singleton initialization and easily mockable in tests.
- Safe fallback on judge failure: returns score=0.0 with verdict="judge_error" rather than
  raising, so a judge outage does not break batch evaluation loops.

Lesson from DoTA-RAG (arXiv 2506.12571): Faithfulness collapsed from 0.702 → 0.336 after
300-word response truncation, completely invisible to Recall@K metrics. This module closes
that measurement gap.
"""
from __future__ import annotations

import logging
from typing import Any, Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

logger = logging.getLogger(__name__)

VERDICT_SCORES: dict[str, float] = {
    "full": 1.0,
    "partial": 0.5,
    "none": 0.0,
}

_FAITHFULNESS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a strict faithfulness auditor for a RAG system.\n"
        "You will be given:\n"
        "  - ANSWER: the system's generated answer\n"
        "  - PASSAGES: the retrieved context passages that were given to the system\n\n"
        "Your task: decide whether every material claim in ANSWER is directly supported "
        "by at least one of the PASSAGES.\n\n"
        "Verdict options:\n"
        "  full    — every material claim has explicit support in the passages.\n"
        "  partial — most claims are supported, but minor unsupported inferences exist.\n"
        "  none    — the answer contains significant claims not found in the passages "
        "(hallucination risk).\n\n"
        "Return ONLY a JSON object with fields `verdict` and `reasoning` (≤60 words).",
    ),
    (
        "human",
        "ANSWER:\n{answer}\n\n"
        "PASSAGES:\n{passages}",
    ),
])


class FaithfulnessVerdict(BaseModel):
    """Structured output from the faithfulness judge."""

    verdict: Literal["full", "partial", "none"]
    reasoning: str


def verdict_to_score(verdict: str) -> float:
    """Map a verdict string to a normalized [0.0, 1.0] score."""
    return VERDICT_SCORES.get(verdict, 0.0)


def judge_faithfulness(
    answer: str,
    passages: list[str],
    llm: Any,
) -> dict[str, Any]:
    """
    Estimate how faithfully `answer` is grounded in `passages`.

    Args:
        answer:   The generated answer string to audit.
        passages: The retrieved context passages that were presented to the generator.
        llm:      A LangChain chat model instance (e.g. ChatVertexAI). Must support
                  ``with_structured_output(FaithfulnessVerdict)``.

    Returns:
        dict with keys:
          faithfulness_score    (float  0.0–1.0)
          faithfulness_verdict  (str    "full" | "partial" | "none" | "judge_error")
          faithfulness_reasoning (str)
    """
    if not answer or not answer.strip():
        return {
            "faithfulness_score": 0.0,
            "faithfulness_verdict": "none",
            "faithfulness_reasoning": "Empty answer — nothing to verify.",
        }

    passages_text = "\n\n".join(
        f"[Passage {i}]\n{p.strip()}" for i, p in enumerate(passages, start=1)
    )

    try:
        judge_llm = llm.with_structured_output(FaithfulnessVerdict)
        result: FaithfulnessVerdict = judge_llm.invoke(
            _FAITHFULNESS_PROMPT.format_messages(
                answer=answer.strip(),
                passages=passages_text,
            )
        )
        score = verdict_to_score(result.verdict)
        return {
            "faithfulness_score": score,
            "faithfulness_verdict": result.verdict,
            "faithfulness_reasoning": result.reasoning,
        }
    except Exception as exc:
        logger.warning("Faithfulness judge failed: %s", exc)
        return {
            "faithfulness_score": 0.0,
            "faithfulness_verdict": "judge_error",
            "faithfulness_reasoning": f"Judge unavailable: {exc}",
        }
