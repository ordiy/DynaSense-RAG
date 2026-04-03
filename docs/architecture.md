# 项目架构（整洁架构 / Clean Architecture）

本仓库在持续演进中采用 **依赖倒置**：领域与用例不直接依赖 LanceDB、Neo4j、Vertex 等具体驱动；HTTP 与静态资源与 Agent/RAG 逻辑分离。

参考思想：整洁架构、端口与适配器、DDD（轻量）。

---

## 目录结构（`src/`）

| 路径 | 职责 |
|------|------|
| **`src/api/`** | **展现层 / HTTP**：`main.py` 组装 FastAPI；`routers/` 按功能拆分路由；`schemas.py` 为请求/响应 DTO；`state.py` 为进程内任务与会话状态；`session_memory.py` 为多轮 query 拼接。 |
| **`src/core/`** | **核心横切**：`config.py`（Pydantic Settings，统一读环境变量）、`exceptions.py`（领域异常）、`langsmith_tracing.py`（LangSmith 初始化）。 |
| **`src/domain/`** | **领域层**：`interfaces/` 定义 **端口**（如 `IGraphRepository`），不依赖任何数据库 SDK。 |
| **`src/infrastructure/`** | **基础设施层**：`persistence/` 等 **适配器**，实现 `domain` 接口并委托给现有实现（如 `graph_store`）。 |
| **`src/rag_core.py` / `hybrid_rag.py` / `graph_store.py`** | **Agent / RAG 管线**（仍在逐步迁移中）：LangGraph、检索、打分、生成；后续可迁入 `agents/` 与 `infrastructure/` 并仅通过端口交互。 |
| **`src/static/`** | 工程师控制台与客户演示 **HTML**（非独立前端工程）。 |

---

## 依赖方向

```
HTTP (api/routers)
    → 应用服务 / RAG 管线 (rag_core, hybrid_rag)
        → domain interfaces（抽象）
            ← infrastructure adapters（实现）
                → LanceDB / Neo4j / Vertex（外部）
```

- **新代码**：优先通过 `core.config.get_settings()` 读配置；通过 `IGraphRepository` 等接口访问图数据（调试 API 可逐步改为注入实现）。
- **存量代码**：`rag_core` 等仍可直接读 `os.environ`；迁移是渐进的，不必一次性大改。

---

## 入口与运行

- **ASGI 应用**：`src.api.main:app` 或兼容入口 **`src.app:app`**（`src/app.py` 再导出）。
- **LangSmith**：必须在导入 `rag_core` 之前执行 `init_langsmith_tracing()`，已在 `api/main.py` 顶部处理。

```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

---

## 后续可演进方向

1. **`agents/`**：将 LangGraph 工作流、Prompt 模板（YAML/Jinja）从 `rag_core` 迁出。
2. **向量端口**：`IVectorIndex` + LanceDB 适配器，与 `IGraphRepository` 对称。
3. **全局异常处理**：已实现 — `src/api/error_handlers.py` 注册 `DomainError` / `KnowledgeBaseError` / `QueryGuardrailError` 等映射。
4. **SSE 流式**：已实现 — `POST /api/chat/stream`，事件由 `rag_core.iter_chat_stream_events` 产出（与 `run_chat_pipeline` 同路径）。
5. **引用透明**：`src/core/citations.py` 从 `context_used` 文本生成结构化 `citations`。
6. **反馈与护栏**：`POST /api/feedback`（内存环形缓冲）；可选 `BLOCK_SUSPECT_PII` + `guard_query_or_raise`。
7. **批量评测脚本**：`scripts/eval_regression.py`（需完整依赖与凭证）。

---

## 相关文档

- [LangSmith 可观测性](./langsmith_observability.md)
- [LangGraph 流式日志](./langgraph_stream_log.md)
