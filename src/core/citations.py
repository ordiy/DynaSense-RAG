"""
Build structured citation entries from retrieval context strings.

Design: ``context_used`` in the RAG pipeline is a list of plain text blocks (often
parent documents with a ``[Source: filename]`` prefix from Small-to-Big expansion,
or synthetic blocks like ``[Graph retrieval — linearized triples]``). We derive
stable labels and short previews for UI / audit without changing the LLM prompts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Leading line like: [Source: report.pdf] or [Source: Unknown]
_SOURCE_LINE = re.compile(
    r"^\[Source:\s*([^\]]+)\]\s*",
    re.MULTILINE,
)


@dataclass(frozen=True)
class Citation:
    """One retrievable snippet shown to the client for transparency."""

    index: int
    label: str
    preview: str
    source: str | None


def _preview(text: str, max_chars: int = 220) -> str:
    """Single-line excerpt for tables; avoids huge payloads in JSON."""
    one = " ".join((text or "").split())
    if len(one) <= max_chars:
        return one
    return one[: max_chars - 1] + "…"


def _label_for_block(index: int, block: str) -> tuple[str, str | None]:
    """
    Derive a human-readable label and optional source filename.

    Vector parents are prefixed with ``[Source: name]``; graph/global use markers
    in the first line.
    """
    raw = (block or "").strip()
    if not raw:
        return f"[{index}] (empty)", None

    first_line = raw.split("\n", 1)[0].strip()
    m = _SOURCE_LINE.match(raw)
    if m:
        name = m.group(1).strip()
        body = _SOURCE_LINE.sub("", raw, count=1).strip()
        return f"[{index}] {name}", name or None

    if first_line.startswith("[Graph retrieval"):
        return f"[{index}] graph", "graph"
    if "graph summary" in first_line.lower() or first_line.startswith("Global:"):
        return f"[{index}] global_summary", "global"
    return f"[{index}] context", None


def build_citations_from_context(context_used: list[str] | None) -> list[dict]:
    """
    Convert pipeline ``context_used`` strings into JSON-serializable citation dicts.

    Each dict: index, label, preview, source (nullable).
    """
    if not context_used:
        return []
    out: list[dict] = []
    for i, block in enumerate(context_used, start=1):
        label, source = _label_for_block(i, block)
        out.append(
            {
                "index": i,
                "label": label,
                "preview": _preview(block),
                "source": source,
            }
        )
    return out
