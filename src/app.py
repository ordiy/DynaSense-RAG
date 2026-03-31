import os
import logging
import time
from typing import Literal
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uuid
import uvicorn
from src.rag_core import process_document_task, run_chat_pipeline, run_evaluation

app = FastAPI(title="MAP-RAG MVP API")

logger = logging.getLogger(__name__)

# Mount static files for Web UI
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --- Limits / Security defaults ---
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(2 * 1024 * 1024)))  # 2MiB
MAX_QUERY_LEN = int(os.environ.get("MAX_QUERY_LEN", "2048"))
MAX_EXPECTED_SUBSTRING_LEN = int(os.environ.get("MAX_EXPECTED_SUBSTRING_LEN", "512"))
TASK_TTL_SECONDS = int(os.environ.get("TASK_TTL_SECONDS", "3600"))
CHAT_SESSION_TTL_SECONDS = int(os.environ.get("CHAT_SESSION_TTL_SECONDS", "3600"))
CHAT_MEMORY_QUERY_BUDGET = int(os.environ.get("CHAT_MEMORY_QUERY_BUDGET", "1800"))
CHAT_MAX_STORED_HISTORY_CHARS = int(os.environ.get("CHAT_MAX_STORED_HISTORY_CHARS", "20000"))
CHAT_MEMORY_ASSISTANT_CLIP_CHARS = int(os.environ.get("CHAT_MEMORY_ASSISTANT_CLIP_CHARS", "280"))


# Task tracking dictionary
tasks: dict[str, dict] = {}
chat_sessions: dict[str, dict] = {}

def _cleanup_tasks() -> None:
    """清理过期任务，避免内存无限增长。"""
    now = time.time()
    for task_id, info in list(tasks.items()):
        created_at = info.get("created_at")
        if created_at is None:
            continue
        if now - float(created_at) > TASK_TTL_SECONDS:
            tasks.pop(task_id, None)

def _cleanup_chat_sessions() -> None:
    now = time.time()
    for conversation_id, info in list(chat_sessions.items()):
        updated_at = info.get("updated_at", info.get("created_at", now))
        if now - float(updated_at) > CHAT_SESSION_TTL_SECONDS:
            chat_sessions.pop(conversation_id, None)

def _build_query_with_history(
    messages: list[dict],
    budget: int = CHAT_MEMORY_QUERY_BUDGET,
    mode: Literal["prioritized", "legacy"] = "prioritized",
) -> str:
    """
    构造传给 RAG 的 query（把多轮对话拼进来），并做长度预算控制。
    """
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

    user_msgs = [_normalize_text(str(m.get("content", ""))) for m in messages if m.get("role") == "user" and str(m.get("content", "")).strip()]
    assistant_msgs = [_normalize_text(str(m.get("content", ""))) for m in messages if m.get("role") == "assistant" and str(m.get("content", "")).strip()]

    latest_user = user_msgs[-1] if user_msgs else ""
    topic_anchor = user_msgs[0] if user_msgs else ""

    header = (
        "You are answering the current user question using conversation history.\n"
        "Prioritize concrete project/entity names from earlier user messages.\n"
        "If previous assistant content is long, treat it as secondary hints.\n\n"
        f"Current User Question:\n{latest_user}\n\n"
    )

    # 用户信息优先：先放主题锚点和近期 user turns，再补少量 assistant 摘要
    lines: list[str] = []
    if topic_anchor and topic_anchor != latest_user:
        lines.append("Topic Anchor (first user intent): " + topic_anchor)

    # Add recent user messages (excluding latest already in header), newest first then reverse back.
    recent_user_lines: list[str] = []
    for text in reversed(user_msgs[:-1]):
        recent_user_lines.append("User: " + text)
    recent_user_lines.reverse()
    lines.extend(recent_user_lines)

    # Add short assistant hints only (to avoid drowning key user entities).
    recent_assistant_lines: list[str] = []
    for text in reversed(assistant_msgs[-3:]):
        recent_assistant_lines.append("Assistant (brief): " + _clip_text(text, CHAT_MEMORY_ASSISTANT_CLIP_CHARS))
    recent_assistant_lines.reverse()
    lines.extend(recent_assistant_lines)

    body = "Conversation Memory:\n" + ("\n".join(lines) if lines else "(none)") + "\n"
    full_query = header + body

    # Hard cap to budget; ensure latest question always present.
    if len(full_query) <= budget:
        return full_query

    # Trim body first; keep header with latest question.
    room_for_body = max(200, budget - len(header))
    trimmed_body = body[:room_for_body]
    return (header + trimmed_body)[:budget]

def _history_chars(messages: list[dict]) -> int:
    return sum(len(str(m.get("content", ""))) for m in messages)

def _trim_session_history(messages: list[dict]) -> list[dict]:
    """
    控制服务端会话内存。只在总字符超限时，从最早轮次开始裁剪。
    """
    trimmed = list(messages)
    while len(trimmed) > 2 and _history_chars(trimmed) > CHAT_MAX_STORED_HISTORY_CHARS:
        trimmed.pop(0)
    return trimmed

# Pydantic models for request bodies
class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_LEN)

class ChatSessionRequest(BaseModel):
    conversation_id: str | None = None
    message: str = Field(min_length=1, max_length=MAX_QUERY_LEN)
    memory_mode: Literal["prioritized", "legacy"] = "prioritized"

class ChatSessionABRequest(BaseModel):
    conversation_id: str | None = None
    message: str = Field(min_length=1, max_length=MAX_QUERY_LEN)

class EvalRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_LEN)
    expected_substring: str = Field(min_length=1, max_length=MAX_EXPECTED_SUBSTRING_LEN)

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

@app.get("/", response_class=HTMLResponse)
async def read_root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    _cleanup_tasks()

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded file is too large.")

    text_content = content.decode("utf-8", errors="replace")
    safe_filename = os.path.basename(file.filename) if file.filename else "upload.txt"
    
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "pending", "progress": 0, "created_at": time.time()}
    
    # Run async processing in background
    background_tasks.add_task(process_document_task, text_content, safe_filename, tasks[task_id])
    
    return {"task_id": task_id, "message": "Document uploaded and processing started."}

@app.get("/api/tasks/{task_id}")
async def get_task_status(task_id: str):
    _cleanup_tasks()
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        result = run_chat_pipeline(request.query)
        return result
    except Exception as e:
        logger.exception("Chat pipeline failed.")
        raise HTTPException(status_code=500, detail="Internal server error.")

@app.post("/api/chat/session")
async def chat_session(request: ChatSessionRequest):
    _cleanup_chat_sessions()
    conversation_id = request.conversation_id or str(uuid.uuid4())
    now = time.time()

    session = chat_sessions.get(conversation_id)
    if session is None:
        session = {"created_at": now, "updated_at": now, "messages": []}
        chat_sessions[conversation_id] = session

    messages = session["messages"]
    messages.append({"role": "user", "content": request.message})
    session["updated_at"] = now

    request_query = _build_query_with_history(
        messages,
        budget=CHAT_MEMORY_QUERY_BUDGET,
        mode=request.memory_mode,
    )
    request_query = request_query[:MAX_QUERY_LEN]

    try:
        result = run_chat_pipeline(request_query)
        answer = result.get("answer", "")
        messages.append({"role": "assistant", "content": answer})
        session["messages"] = _trim_session_history(messages)
        session["updated_at"] = time.time()
        return {
            "conversation_id": conversation_id,
            "memory_mode": request.memory_mode,
            "request_query": request_query,
            "answer": answer,
            "logs": result.get("logs", []),
            "context_used": result.get("context_used", []),
            "history": [ChatMessage(**m).dict() for m in session["messages"]],
        }
    except Exception:
        logger.exception("Session chat pipeline failed.")
        raise HTTPException(status_code=500, detail="Internal server error.")

@app.post("/api/chat/session/ab")
async def chat_session_ab(request: ChatSessionABRequest):
    """
    A/B 对比：同一个会话上下文 + 同一条新消息，分别用两种 memory 构造策略测试。
    该接口不写入会话历史，只返回对比结果。
    """
    _cleanup_chat_sessions()
    conversation_id = request.conversation_id or str(uuid.uuid4())
    session = chat_sessions.get(conversation_id)
    base_messages = list(session["messages"]) if session else []
    probe_messages = base_messages + [{"role": "user", "content": request.message}]

    mode_queries = {
        "prioritized": _build_query_with_history(probe_messages, budget=CHAT_MEMORY_QUERY_BUDGET, mode="prioritized")[:MAX_QUERY_LEN],
        "legacy": _build_query_with_history(probe_messages, budget=CHAT_MEMORY_QUERY_BUDGET, mode="legacy")[:MAX_QUERY_LEN],
    }

    results: dict[str, dict] = {}
    for mode in ("prioritized", "legacy"):
        try:
            out = run_chat_pipeline(mode_queries[mode])
            answer = out.get("answer", "")
            blocked = "未能找到与您问题高度相关的信息" in answer
            results[mode] = {
                "request_query": mode_queries[mode],
                "answer": answer,
                "logs": out.get("logs", []),
                "context_used": out.get("context_used", []),
                "blocked": blocked,
            }
        except Exception:
            logger.exception("A/B chat pipeline failed for mode=%s", mode)
            results[mode] = {
                "request_query": mode_queries[mode],
                "answer": "Internal server error.",
                "logs": [],
                "context_used": [],
                "blocked": True,
            }

    recommended_mode = "prioritized"
    if results["prioritized"]["blocked"] and not results["legacy"]["blocked"]:
        recommended_mode = "legacy"
    elif results["legacy"]["blocked"] and not results["prioritized"]["blocked"]:
        recommended_mode = "prioritized"

    return {
        "conversation_id": conversation_id,
        "message": request.message,
        "recommended_mode": recommended_mode,
        "results": results,
    }

@app.get("/api/chat/session/{conversation_id}")
async def get_chat_session(conversation_id: str):
    _cleanup_chat_sessions()
    session = chat_sessions.get(conversation_id)
    if not session:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {
        "conversation_id": conversation_id,
        "history": [ChatMessage(**m).dict() for m in session["messages"]],
        "updated_at": session.get("updated_at"),
    }

@app.delete("/api/chat/session/{conversation_id}")
async def delete_chat_session(conversation_id: str):
    _cleanup_chat_sessions()
    if conversation_id in chat_sessions:
        chat_sessions.pop(conversation_id, None)
        return {"conversation_id": conversation_id, "deleted": True}
    raise HTTPException(status_code=404, detail="Conversation not found")

@app.post("/api/evaluate")
async def evaluate(request: EvalRequest):
    try:
        result = run_evaluation(request.query, request.expected_substring)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        logger.exception("Evaluation pipeline failed.")
        raise HTTPException(status_code=500, detail="Internal server error.")

if __name__ == "__main__":
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)
