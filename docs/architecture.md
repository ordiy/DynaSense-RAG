# 项目架构（整洁架构 / Clean Architecture）

本仓库在持续演进中采用 **依赖倒置**：领域与用例不直接依赖 LanceDB、Neo4j、Vertex 等具体驱动；HTTP 与静态资源与 Agent/RAG 逻辑分离。

参考思想：整洁架构、端口与适配器、DDD（轻量）。

---

## 目录结构（`src/`）

| 路径 | 职责 |
|------|------|
| **`src/api/`** | **展现层 / HTTP**：`main.py` 组装 FastAPI；`routers/` 按功能拆分路由；`schemas.py` 为请求/响应 DTO；`state.py` 为进程内任务与会话状态；`session_memory.py` 为多轮 query 拼接；`error_handlers.py` 注册全局异常映射；`whatif_pipeline.py` 编排无 RAG 的 What-If DAG；`upload_validation.py` 为上传文件名/类型纯函数（便于单测）。 |
| **`src/core/`** | **核心横切**：`config.py`（Pydantic Settings）、`exceptions.py`（领域异常）、`langsmith_tracing.py`（LangSmith）；`citations.py`（从 `context_used` 生成结构化引用）；`query_anchors.py`（融合重排前按锚点过滤候选）；`graph_constrained_queries.py`（受控 Cypher 模板）；`guardrails.py`（可选查询侧 PII 拦截）；`analytics_profile.py`（**受控报表分析**：CSV/TSV/XLSX 描述性统计，无任意代码执行）。 |
| **`src/tools/`** | **可组合业务工具**（无 RAG）：如 `loan_whatif.py`，由 `whatif_pipeline` 或后续 Agent 调用。 |
| **`src/domain/`** | **领域层**：`interfaces/` 定义 **端口**（如 `IGraphRepository`），不依赖任何数据库 SDK。 |
| **`src/infrastructure/`** | **基础设施层**：`persistence/` — `postgres_connection.py`（连接池 + AGE 连接预处理）、`postgres_schema.py`（`kb_doc` JSONB、`kb_embedding`、AGE 或 `kg_triple`）、`postgres_jsonb_collection.py`（替代 MongoMock）、`postgres_vectorstore.py`（pgvector）、`postgres_age_setup.py` / `postgres_age_graph.py`（Cypher）、`postgres_graph.py`（关系型三元组回退）；`Neo4jGraphRepository` 委托 `graph_store`。 |
| **`src/rag_core.py` / `hybrid_rag.py` / `graph_store.py`** | **Agent / RAG 管线**（仍在逐步迁移中）：LangGraph、检索、打分、生成；当前向量与父/子块为 **LanceDB + MongoMock**，图为 **Neo4j**；**GB 级部署** 可收敛到 **PostgreSQL + pgvector**（见 [postgresql_storage_roadmap.md](./postgresql_storage_roadmap.md)）。 |
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
8. **查询锚点过滤**：`src/core/query_anchors.py` — 当问题中含机构名、A 股代码等锚点时，在融合重排前剔除不含任一锚点的候选（`QUERY_ANCHOR_FILTER`，默认开启；无匹配时 fail-open 保留原列表）。
9. **What-If 工具（无 RAG）**：`POST /api/whatif/loan/compare` — 等额本息利率对比；工具在 `src/tools/loan_whatif.py`，DAG 编排在 `src/api/whatif_pipeline.py`。详见 [What-If 工具案例](./whatif_tools.md)。
10. **报表分析（受控）**：`POST /api/analytics/profile` — 上传 `.csv` / `.tsv` / `.txt`（逗号分隔）/ `.xlsx`，返回行列数、缺失率、数值列统计、类别 Top 频率。实现见 `src/core/analytics_profile.py`、`src/api/routers/analytics.py`。与 [HF 数据分析智能体](https://huggingface.co/learn/cookbook/zh-CN/agent_data_analyst) 不同：此处 **不** 使用 LLM 执行任意 `pandas` 代码，便于审计与合规；后续若需「自然语言问数」，可在网关层将问题 **映射到白名单聚合** 或对接 BI，而非开放 `ReactCodeAgent`。

---

## The Bitter Lesson 与有计划调整

[The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) 的核心主张是：长期看 **通用方法 + 算力与数据** 往往胜过 **手工领域技巧** 的堆叠。对本仓库而言，这意味着：**优先投资可扩展的检索与评测闭环**（embedding / reranker / 回归集 / 反馈），而不是无限增加不可验证的启发式分支；同时 **不** 把合规所要求的 **可审计接口**（引用、受控 Cypher、确定性 What‑If）换成黑箱。

**分阶段路线图（含与现有模块的对照）** 见 **[bitter_lesson_roadmap.md](./bitter_lesson_roadmap.md)**（阶段 A：度量与版本化；阶段 B：有数据后再用学习替代部分启发式；阶段 C：明确不做的反模式）。

### 存储层收敛（PostgreSQL）

为简化大规模部署，计划以 **PostgreSQL + pgvector**（+ 关系表承载三元组与父/子文档）替代 **LanceDB + Neo4j + MongoMock** 三件套；GB 级语料下单库更易运维。详见 **[postgresql_storage_roadmap.md](./postgresql_storage_roadmap.md)**（表设计示意、BM25/全文检索选项、分阶段迁移、风险）。

---

## 外部参考：HF《多智能体 RAG》Cookbook 与本仓库映射

[Hugging Face Cookbook · 多智能体 RAG 系统](https://huggingface.co/learn/cookbook/zh-CN/multiagent_rag_system) 使用 `transformers.agents`（`ManagedAgent`、中央 `ReactCodeAgent`、多路检索/搜索/生图）。**本仓库未采用该框架**；下列对照仅说明 **概念上** 的对应关系，便于阅读外部教程时定位到本仓库代码。

| Cookbook 概念 | 本仓库对应 | 说明 |
|---------------|------------|------|
| **中央协调智能体（Manager）** | `src/hybrid_rag.py`：`route_query()` → `RouteDecision`；`prepare_hybrid_chat()`；`src/rag_core.py`：LangGraph 状态机 `invoke` / 流式 | 用 **结构化路由 LLM + 固定 LangGraph 流水线** 代替「多轮委派子 Agent」；可观测、易测，延迟更可预期。 |
| **检索智能体 + 多个 RetrieverTool（多 KB）** | `gather_route_candidates()`、`retrieve_hybrid_ranked_documents()`、`fusion_rerank_docs()` / `fusion_rerank_all()` | 向量、BM25、Neo4j 等候选 **汇入同一重排池**（`HYBRID_FUSION_TOP_N`），而非让每个子 Agent 独立多轮检索。 |
| **单库语义检索（FAISS + embedding）** | LanceDB + Vertex：`src/rag_core.py`（如 `retrieve_parent_documents_expanded` 等）；子块 BM25：`hybrid_rag.bm25_parent_documents` | 栈不同，**职责类似**：从语料中取 Top‑K 上下文。 |
| **网络搜索智能体** | — | **未实现**；企业文档 / 图检索为主。若未来需要开放域，宜单独评估工具边界与安全。 |
| **图像生成智能体** | — | **未实现**。 |
| **代码解释器 / 沙箱工具** | — | **未实现**。What-If 为 **确定性工具** + DAG：`src/tools/*.py`、`src/api/whatif_pipeline.py`。 |
| **用户请求 → 选哪个子系统** | 显式 HTTP：`src/api/main.py` 挂载 `chat`、`whatif`、`ingest`、`debug_routes` 等路由 | 当前由 **客户端选端点**；Cookbook 由中央 Agent 选型。后续若要做「单入口自动分流」，可在网关层增加意图分类，**不必**引入整套 HF Multi-Agent。 |

**路由与入口速查（HTTP）**

| 能力 | 路由模块 | 典型路径 |
|------|----------|----------|
| 上传与 ingest 任务 | `src/api/routers/ingest.py` | `POST /api/upload`，`GET /api/tasks/{task_id}` |
| 对话 / SSE | `src/api/routers/chat.py` | `POST /api/chat`，`POST /api/chat/stream` |
| 会话与记忆 | `src/api/routers/session.py` | `POST /api/chat/session` 等 |
| What-If（无 RAG） | `src/api/routers/whatif.py` | `POST /api/whatif/loan/compare` |
| 报表分析（表格画像） | `src/api/routers/analytics.py` | `POST /api/analytics/profile` |
| 调试图 / LanceDB / 受控 Cypher | `src/api/routers/debug_routes.py` | `GET/POST /api/debug/...` |
| 评测 | `src/api/routers/eval.py` | `POST /api/...`（见 OpenAPI） |

---

## 相关文档

- [The Bitter Lesson 与架构演进路线](./bitter_lesson_roadmap.md)
- [PostgreSQL 统一存储路线图](./postgresql_storage_roadmap.md)
- [Hugging Face · 多智能体 RAG（Cookbook，中文）](https://huggingface.co/learn/cookbook/zh-CN/multiagent_rag_system)
- [受控图查询（白名单模板）](./graph_constrained_queries.md)
- [后续迭代 TODO（OpenClaw 与 RAG 边界）](./TODO.md)
- [LangSmith 可观测性](./langsmith_observability.md)
- [LangGraph 流式日志](./langgraph_stream_log.md)
