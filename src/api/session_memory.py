"""Build multi-turn query strings and trim stored history (API session layer)."""

from __future__ import annotations

from typing import Literal

from src.core.config import get_settings


def build_query_with_history(
    messages: list[dict],
    budget: int | None = None,
    mode: Literal["prioritized", "legacy"] = "prioritized",
) -> str:
    s = get_settings()
    if budget is None:
        budget = s.chat_memory_query_budget

    def _normalize_text(text: str) -> str:
        return " ".join(text.split())

    def _clip_text(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[:limit] + "..."

    if mode == "legacy":
        instruction = "You are answering the latest user question using the conversation history.\n"
        remaining = max(300, budget - len(instruction))
        included_lines: list[str] = []
        total_len = 0
        for m in reversed(messages):
            role = m.get("role", "user")
            content = _normalize_text(str(m.get("content", "")))
            if not content:
                continue
            prefix = "User: " if role == "user" else "Assistant: "
            line = prefix + content
            sep_len = 1 if included_lines else 0
            next_total = total_len + sep_len + len(line)
            if included_lines and next_total > remaining:
                break
            included_lines.insert(0, line)
            total_len = next_total
        history = "\n".join(included_lines)
        full_query = instruction + "\nConversation History:\n" + history + "\n"
        return full_query[:budget]

    user_msgs = [
        _normalize_text(str(m.get("content", "")))
        for m in messages
        if m.get("role") == "user" and str(m.get("content", "")).strip()
    ]
    assistant_msgs = [
        _normalize_text(str(m.get("content", "")))
        for m in messages
        if m.get("role") == "assistant" and str(m.get("content", "")).strip()
    ]

    latest_user = user_msgs[-1] if user_msgs else ""
    topic_anchor = user_msgs[0] if user_msgs else ""

    header = (
        "You are answering the current user question using conversation history.\n"
        "Prioritize concrete project/entity names from earlier user messages.\n"
        "If previous assistant content is long, treat it as secondary hints.\n\n"
        f"Current User Question:\n{latest_user}\n\n"
    )

    lines: list[str] = []
    if topic_anchor and topic_anchor != latest_user:
        lines.append("Topic Anchor (first user intent): " + topic_anchor)

    recent_user_lines: list[str] = []
    for text in reversed(user_msgs[:-1]):
        recent_user_lines.append("User: " + text)
    recent_user_lines.reverse()
    lines.extend(recent_user_lines)

    clip = s.chat_memory_assistant_clip_chars
    recent_assistant_lines: list[str] = []
    for text in reversed(assistant_msgs[-3:]):
        recent_assistant_lines.append("Assistant (brief): " + _clip_text(text, clip))
    recent_assistant_lines.reverse()
    lines.extend(recent_assistant_lines)

    body = "Conversation Memory:\n" + ("\n".join(lines) if lines else "(none)") + "\n"
    full_query = header + body

    if len(full_query) <= budget:
        return full_query

    room_for_body = max(200, budget - len(header))
    trimmed_body = body[:room_for_body]
    return (header + trimmed_body)[:budget]


def history_chars(messages: list[dict]) -> int:
    return sum(len(str(m.get("content", ""))) for m in messages)


def trim_session_history(messages: list[dict]) -> list[dict]:
    s = get_settings()
    max_chars = s.chat_max_stored_history_chars
    trimmed = list(messages)
    while len(trimmed) > 2 and history_chars(trimmed) > max_chars:
        trimmed.pop(0)
    return trimmed
