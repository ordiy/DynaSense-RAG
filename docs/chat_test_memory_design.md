# Chat Test Page: Memory + Context-Length Design

## Goal
Add a new “Chat Test (Memory)” web page to the existing MAP-RAG GUI with a backend
conversation session mechanism so we can:
1) test multi-turn chatting behavior,
2) persist conversation history by `conversation_id`,
3) handle context-length control on the server side,
4) show per-turn request/response details (logs + contexts) for debugging.

## Backend Session Design (conversation_id + history)
### 1. Session Storage
In `src/app.py`, an in-memory store is added:
`chat_sessions: dict[str, dict]`

Session record shape:
- `created_at`
- `updated_at`
- `messages`: list of `{ role: "user" | "assistant", content: string }`

TTL cleanup:
- `CHAT_SESSION_TTL_SECONDS` (default `3600`)
- `_cleanup_chat_sessions()` removes stale sessions

### 2. New APIs
- `POST /api/chat/session`
  - Request: `{ conversation_id?: string, message: string }`
  - Behavior:
    1. create new session if `conversation_id` missing or not found
    2. append user message into session history
    3. build backend query from history under budget
    4. call `run_chat_pipeline(...)`
    5. append assistant answer to history
    6. trim overly long stored history
  - Response:
    - `conversation_id`
    - `request_query` (actual query sent to RAG)
    - `answer`, `logs`, `context_used`
    - `history`

- `GET /api/chat/session/{conversation_id}`
  - Returns session history + metadata

- `DELETE /api/chat/session/{conversation_id}`
  - Deletes a session (used by UI Reset)

### 3. Context-Length Strategy (Server Side)
`_build_query_with_history(messages, budget)` in `src/app.py`:
- Default budget: `CHAT_MEMORY_QUERY_BUDGET=1800`
- Adds instruction header + serialized recent turns:
  - `User: ...`
  - `Assistant: ...`
- Iterates from newest to oldest, keeps recent turns while within budget
- final query is hard-capped before sending to pipeline
- also bounded by backend `MAX_QUERY_LEN`

Stored history protection:
- `CHAT_MAX_STORED_HISTORY_CHARS` (default `20000`)
- oldest turns are removed when total stored chars exceed threshold

## Frontend Integration
In `src/static/index.html` Chat Test tab:
- keeps `chatTestConversationId` from backend
- sends new messages to `POST /api/chat/session`
- renders left panel from backend-returned `history`
- calls `DELETE /api/chat/session/{id}` on Reset

## Debug/Observability (Right Column)
The “Last Interaction Details” panel shows:
- request query built by backend (with length)
- graph logs (`data.logs`)
- final answer (`data.answer`)
- top retrieved contexts (`data.context_used`, first 6 items)

This makes it easier to analyze:
- whether memory helps retrieval
- whether the grader approved context or blocked it

## Security Considerations
- LLM answers may contain HTML. To reduce XSS risk, the page uses:
  `marked.Renderer()` with `renderer.html = () => ''`
- All other user-visible raw text (logs, request query, contexts) is escaped via `escapeHtml()`.

## Security Considerations
- LLM answers may contain HTML. To reduce XSS risk, the page uses:
  `marked.Renderer()` with `renderer.html = () => ''`
- All other user-visible raw text (logs, request query, contexts) is escaped via `escapeHtml()`.

## Testing Notes
Manual validation:
1. Open `/` and switch to “4. Chat Test (Memory)”.
2. Send first message; verify response includes non-empty `conversation_id`.
3. Send follow-up messages and confirm the same `conversation_id` is reused.
4. Confirm:
   - conversation panel updates from backend `history`,
   - right panel shows backend `request_query` length under budget,
   - reset action deletes the current session.
