# LangSmith observability

This project integrates [LangSmith](https://smith.langchain.com) tracing for LangChain / LangGraph runs (LLM calls, chains, and graph steps visible in the LangSmith UI).

**Official guide:** [LangSmith Observability](https://docs.langchain.com/langsmith/observability)

---

## How it works in this repo

1. **`src/observability.py`** defines `init_langsmith_tracing()`, which reads environment variables and sets `LANGCHAIN_TRACING_V2` / `LANGCHAIN_API_KEY` (and optional project name) **before** any LangChain objects are constructed.

2. **`src/app.py`** calls `init_langsmith_tracing()` **immediately before** `from src.rag_core import ...`.  
   This order is required because `rag_core` creates `VertexAIEmbeddings`, `ChatVertexAI`, and compiles the LangGraph workflow at import time.

3. **Standalone scripts** (e.g. `scripts/*.py`) that import `rag_core` directly will **not** run `init_langsmith_tracing()` unless you call it yourself at the top of the script.

---

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `LANGCHAIN_API_KEY` | Yes, to enable tracing | LangSmith API key ([Settings → API Keys](https://smith.langchain.com/settings)). Alias: `LANGSMITH_API_KEY`. |
| `LANGCHAIN_TRACING_V2` | No | Default `true` when a key is set. Set to `false` to disable tracing even if a key is present. |
| `LANGCHAIN_PROJECT` | No | Project name in the LangSmith UI. Aliases: `LANGSMITH_PROJECT`, `LANGSMITH_PROJECT_NAME`. |
| `LANGCHAIN_ENDPOINT` | No | Only for self-hosted / custom endpoints. Alias: `LANGSMITH_ENDPOINT`. |

Copy `.env.example` to `.env` and fill in values (`.env` is gitignored).

Example:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY="lsv2_pt_..."   # never commit this value
export LANGCHAIN_PROJECT=DynaSense-RAG
```

Then start the API:

```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

---

## Security

- **Do not commit** API keys. Use environment variables or a secret manager.
- If a key was exposed (e.g. in chat or a ticket), **rotate** it in LangSmith and revoke the old key.

---

## Verification

1. Set `LANGCHAIN_API_KEY` and restart the server.
2. Call `/api/chat` or the customer portal `/demo` so LangChain / LangGraph runs.
3. Open [smith.langchain.com](https://smith.langchain.com) → **Tracing** and look for runs under the configured project (or default project).

---

## 中文摘要

- **作用**：将 LangChain / LangGraph 调用链上报到 LangSmith，便于调试与监控。  
- **接入位置**：`app.py` 在导入 `rag_core` **之前**调用 `init_langsmith_tracing()`（见 `src/observability.py`）。  
- **配置**：在环境变量中设置 `LANGCHAIN_API_KEY`（及可选 `LANGCHAIN_PROJECT`），勿将密钥写入仓库。  
- **文档**：官方说明见 [LangSmith Observability](https://docs.langchain.com/langsmith/observability)。
