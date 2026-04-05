"""
Format reranked retrieval blocks for grader / generator prompts.

Numbering makes multi-passage fusion explicit (aligned with Bitter Lesson: rely on
cross-encoder ranking + LLM integration over hand-picked single snippets).
"""
from __future__ import annotations


def format_numbered_passages(passages: list[str]) -> str:
    """
    Join plain text blocks as ``[Passage 1]`` … ``[Passage N]`` for model consumption.

    ``passages`` are typically parent document texts (Small-to-Big) or hybrid fusion strings.
    """
    if not passages:
        return ""
    parts: list[str] = []
    for i, text in enumerate(passages, start=1):
        body = (text or "").strip()
        parts.append(f"[Passage {i}]\n{body}")
    return "\n\n".join(parts)
