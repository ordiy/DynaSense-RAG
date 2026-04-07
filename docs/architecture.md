# 系统架构文档

> **版本**：逆向工程重写，基于当前代码库实际状态（2026-04）。  
> **原则**：依赖倒置（Dependency Inversion）+ 端口与适配器（Ports & Adapters）。  
> **存储**：已收敛为 **PostgreSQL 单库**（pgvector + JSONB + `kg_triple` + Apache AGE 图扩展），原 LanceDB / Neo4j / MongoMock 均已移除。

---

## 目录

1. [系统全景图](#1-系统全景图)
2. [核心工作流](#2-核心工作流)
3. [关键设计模式](#3-关键设计模式)
4. [依赖关系与风险](#4-依赖关系与风险)
5. [外部服务与环境依赖](#5-外部服务与环境依赖)
6. [相关文档](#6-相关文档)

---

## 1. 系统全景图

### 1.1 分层架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│  展现层 · Presentation                                           │
│  src/static/index.html  src/static/portal.html                  │
│  (工程师控制台)           (客户演示 /demo)                        │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP / SSE / multipart
┌────────────────────────────▼────────────────────────────────────┐
│  API 层 · HTTP                                                   │
│  src/api/main.py  (FastAPI 工厂 + lifespan)                     │
│  src/api/routers/  (按职责拆分的路由模块，见 §1.2)               │
│  src/api/schemas.py        (Pydantic 请求/响应 DTO)              │
│  src/api/state.py          (进程内共享状态：tasks / sessions)     │
│  src/api/error_handlers.py (全局异常→HTTP 状态码映射)            │
│  src/api/guardrails.py     (可选 PII 查询拦截)                   │
│  src/api/upload_validation.py (文件名/MIME 纯函数校验)           │
└────────────────────────────┬────────────────────────────────────┘
                             │ 函数调用
┌────────────────────────────▼────────────────────────────────────┐
│  应用服务 / RAG 管线层 · Application                             │
│  src/rag_core.py           (向量 RAG + LangGraph 状态机)         │
│  src/hybrid_rag.py         (路由器 + BM25 + 图检索 + 融合重排)   │
│  src/api/session_memory.py (多轮历史 query 拼接)                 │
│  src/api/whatif_pipeline.py (What-If DAG 编排)                  │
│  src/tools/loan_whatif.py  (无 RAG 确定性工具)                   │
└──────────┬─────────────────┬───────────────────────────────────┘
           │                 │ 通过 domain interfaces（端口）
┌──────────▼──────┐  ┌───────▼──────────────────────────────────┐
│  核心横切层      │  │  基础设施层 · Infrastructure               │
│  src/core/      │  │  src/infrastructure/persistence/           │
│  config.py      │  │  postgres_connection.py                    │
│  exceptions.py  │  │  postgres_schema.py                        │
│  citations.py   │  │  postgres_jsonb_collection.py              │
│  query_anchors  │  │  postgres_vectorstore.py                   │
│  rag_context_   │  │  postgres_graph.py (关系型三元组)           │
│  format.py      │  │  postgres_age_graph.py (AGE Cypher)        │
│  analytics_     │  │  postgres_age_setup.py (AGE 初始化)        │
│  profile.py     │  │  postgres_graph_repository.py              │
│  graph_const_   │  └──────────────────────┬────────────────────┘
│  rained_queries │                         │
│  langsmith_     │  ┌──────────────────────▼────────────────────┐
│  tracing.py     │  │  数据库 · PostgreSQL 14+                   │
└─────────────────┘  │  表: kb_doc (JSONB) · kb_embedding (pgvec)│
                     │  表: kg_triple (关系型三元组)               │
                     │  AGE 图: map_rag_kg (Entity/REL)           │
                     └────────────────────────────────────────────┘
```

### 1.2 路由模块清单

| 模块 | 前缀 | 核心端点 | 职责 |
|---|---|---|---|
| `routers/pages.py` | — | `GET /`  `GET /demo`  `GET /portal` | 静态页面服务 |
| `routers/chat.py` | `/api` | `POST /api/chat`  `POST /api/chat/stream` | 单轮对话（无会话）、SSE 流式 |
| `routers/session.py` | `/api` | `POST /api/chat/session`  `POST /api/chat/session/multimodal`  `GET/DELETE /api/chat/session/{id}` | 多轮会话 + 多模态（图片/文档） |
| `routers/ingest.py` | `/api` | `POST /api/upload`  `GET /api/tasks/{task_id}` | 文档上传 + 异步入库任务状态 |
| `routers/eval.py` | `/api` | `POST /api/evaluate`  `POST /api/evaluate/batch` | 检索精度评测（Recall@K, NDCG@K） |
| `routers/feedback.py` | `/api` | `POST /api/feedback`  `GET /api/feedback/summary` | 人工反馈收集 |
| `routers/whatif.py` | `/api` | `POST /api/whatif/loan/compare` | 确定性 What-If 工具（无 RAG） |
| `routers/analytics.py` | `/api` | `POST /api/analytics/profile` | 表格文件描述性统计画像 |
| `routers/debug_routes.py` | `/api` | `GET /api/debug/pg/*`  `POST /api/debug/graph/*` | 调试：PostgreSQL 摘要、图搜索、受控模板 |

### 1.3 PostgreSQL 表结构

| 表名 | 用途 | 关键列 |
|---|---|---|
| `kb_doc` | JSONB 文档存储（父/子块） | `id TEXT PK`, `doc JSONB`（含 `type`, `source`, `parent_id`, `content`/`full_content`） |
| `kb_embedding` | pgvector 向量索引 | `id TEXT PK`, `content TEXT`, `meta JSONB`, `embedding vector(768)` |
| `kg_triple` | 关系型知识图谱三元组（AGE 不可用时的回退） | `subject_norm`, `predicate`, `object_norm`, `chunk_id`, `source` |
| AGE 图 `map_rag_kg` | Apache AGE 图（Entity 顶点 + REL 边） | 由 `postgres_age_graph.py` 管理 |

---

## 2. 核心工作流

### 2.1 文档入库（Ingestion）

```
客户端
  POST /api/upload (multipart)
       │
       ▼
routers/ingest.py · upload_document()
  1. upload_validation.{is_pdf_upload, is_docx_upload, is_xlsx_upload, is_allowed_text_upload}
  2. extract_text_from_{pdf,docx,xlsx}_bytes() → 纯文本
  3. 生成 task_id，写入 state.tasks["pending"]
  4. BackgroundTasks.add_task(process_document_task)
       │
       ▼  （后台线程）
rag_core.process_document_task(content, filename, task_state)
  1. chunk_text_jina(content)
     └─ POST https://segment.jina.ai/ → chunks[]
     └─ 无 key 时 fallback: [content]
  2. 写父文档 → PostgresJsonbDocCollection.insert_one({type:"parent", ...})
  3. 生成子块元数据 → PostgresJsonbDocCollection.insert_many([{type:"child",...}])
  4. _embed_in_batches(chunk_texts) → VertexAI text-embedding-004（批次≤250）
  5. PostgresVectorStore.add_embedding_rows(rows)
  6. if NOT skip_graph_ingest:
       hybrid_rag.ingest_chunks_to_graph(chunks, chunk_ids, source)
       └─ LLM 提取三元组 → TripleExtraction
       └─ postgres_age_graph.merge_triple_age() 或 postgres_graph.merge_triple()
       └─ hybrid_rag.invalidate_bm25_cache()
  7. task_state["status"] = "completed"
       │
       ▼
GET /api/tasks/{task_id} → {status, progress, result, graph_triples_ingested}
```

**Small-to-Big 存储模式**

```
父文档 kb_doc {id:"parent_xxx", type:"parent", full_content:"...", source:"file.pdf"}
子块  kb_doc {id:"chunk_parent_xxx_0", type:"child", parent_id:"parent_xxx", content:"..."}
子块  kb_embedding {id:"chunk_parent_xxx_0", embedding:vector(768)}
```
检索时先取 top-k 子块，再扩展到父文档全文，传入 LLM 窗口。

---

### 2.2 标准问答请求（Hybrid RAG 路径）

```
客户端
  POST /api/chat/session  {conversation_id, message, memory_mode}
       │
       ▼
routers/session.py · chat_session()
  1. guardrails.guard_query_or_raise(message)  ← 可选 PII 拦截
  2. session_memory.build_query_with_history(messages, mode=memory_mode)
     └─ "prioritized": topic anchor(首轮) + 最新问题
     └─ "legacy":      顺序拼接历史
  3. request_query[:max_query_len]
  4. rag_core.run_chat_pipeline(request_query)
       │
       ▼
rag_core.run_chat_pipeline(query)
  if hybrid_rag_enabled:
    hybrid_rag.run_hybrid_chat_pipeline(query)
    else fallback: invoke_rag_app (LangGraph 向量路径)

hybrid_rag.run_hybrid_chat_pipeline(query)
  ┌─ prepare_hybrid_chat(query)
  │    1. route_query(query) → RouteDecision{route, reason}
  │       LLM 结构化输出 → Literal["VECTOR","GRAPH","GLOBAL","HYBRID"]
  │    2. gather_route_candidates(query, effective_route)
  │       VECTOR: collect_vector_path()
  │          retrieve_parent_documents_expanded(dense_k=10)
  │          filter_documents_by_query_anchors()
  │       GRAPH:  graph_context_documents()
  │          extract_graph_keywords() → KeywordList
  │          postgres_{age_}graph.query_relationships_by_keywords()
  │          线性化三元组 → Document 列表
  │       GLOBAL: global_context_documents()
  │          postgres_{age_}graph.global_graph_summary()
  │       HYBRID: VECTOR + BM25 合并
  │          bm25_parent_documents() (线程安全缓存 BM25Index)
  │    3. fusion_rerank_docs(query, candidates, top_n)
  │       filter_documents_by_query_anchors()
  │       截断至 hybrid_rerank_pool_size
  │       jina_rerank (jina-reranker-v2-base-multilingual)
  │    4. grade_documents_node(state) ← 幻觉防护
  │       GRADE_PROMPT / GRADE_ANALYSIS_PROMPT → GradeDocuments{binary_score}
  │       "yes" → 通过；"no" → 清空（_is_analysis_followup 兜底）
  │    5. 返回 HybridPrepared(state, effective_route, router_reason)
  └─ generate_node(state)
       GEN_PROMPT / GEN_ANALYSIS_PROMPT → llm.invoke → answer
  └─ build_citations_from_context(context_used) → [Citation]
  └─ return {answer, context_used, logs, citations, route, effective_route, router_reason}
       │
       ▼
session.py 继续
  5. session["messages"].append({role:"assistant", content:answer})
  6. session_memory.trim_session_history(messages)
  7. return {conversation_id, answer, citations, route, history, ...}
```

---

### 2.3 多模态请求（截图 / 附件）

```
客户端
  POST /api/chat/session/multimodal  (multipart/form-data)
  fields: message, conversation_id, memory_mode
  files:  图片(JPEG/PNG/WEBP/GIF/HEIC) 或 文档(PDF/DOCX/XLSX/TXT/MD)
       │
       ▼
routers/session.py · chat_session_multimodal()
  1. 按 MIME/扩展名分流：
     图片  → base64.b64encode → image_parts[(mime,b64)]
     PDF   → extract_text_from_pdf_bytes()   → doc_texts
     DOCX  → extract_text_from_docx_bytes()  → doc_texts
     XLSX  → extract_text_from_xlsx_bytes()  → doc_texts
     TXT/MD→ decode UTF-8                    → doc_texts
  2. 若有 doc_texts，拼入 message 构建 augmented_message
  3. 会话历史拼接（同标准路径）
  4. if image_parts:
       rag_core.run_chat_pipeline_multimodal(request_query, image_parts)
         1. retrieve_parent_documents_expanded() + jina_rerank (文本检索部分)
         2. format_numbered_passages(context_texts)
         3. HumanMessage(content=[
              {"type":"image_url", "image_url":{"url":"data:{mime};base64,{b64}"}},
              {"type":"text", "text": "KB context:\n{ctx}\n---\n{question}"}
            ])
         4. llm.invoke([msg]) → answer
     else:
       rag_core.run_chat_pipeline(request_query)  （纯文档路径）
```

---

### 2.4 流式问答（SSE）

```
客户端（EventSource / fetch + ReadableStream）
  POST /api/chat/stream  {query}
       │
       ▼
routers/chat.py · chat_stream()
  → StreamingResponse(event_generator(), media_type="text/event-stream")
  → rag_core.iter_chat_stream_events(query)
    if hybrid_rag_enabled:
      hybrid_rag.iter_hybrid_chat_stream_events(query)
        prepare_hybrid_chat() → HybridPrepared
        stream_generation_chunks(question, graded_docs, is_analysis)
          llm.stream(GEN_PROMPT / GEN_ANALYSIS_PROMPT)
          yield text fragment
      yield data: {"type":"meta",  "citations":[...], "route":"...", ...}
      yield data: {"type":"token", "content":"..."}  (多次)
      yield data: {"type":"done"}
    else: iter_vector_chat_stream_events(query)
```

---

### 2.5 What-If（确定性工具，无 RAG）

```
POST /api/whatif/loan/compare  {principal, annual_rate_percent_before/after, loan_months}
  → routers/whatif.py
  → whatif_pipeline.run_loan_rate_compare_pipeline(request)
    → tools/loan_whatif.LoanCompareTool.__call__(params)
       等额本息公式计算 monthly_payment / total_paid / total_interest
    → 返回 LoanCompareResponse{before, after, deltas}
```

---

## 3. 关键设计模式

### 3.1 单例 + 延迟初始化（Singleton + Lazy Init）

**文件**：`src/infrastructure/persistence/postgres_connection.py`

```python
_pool: ConnectionPool | None = None  # 模块级单例

def init_pool(url: str) -> None:     # 幂等，调用多次安全
def get_pool() -> ConnectionPool:    # 断言已初始化后返回
def close_pool() -> None:
```

**触发时机**：`src/api/main.py` 的 lifespan 上下文管理器（`@asynccontextmanager async def lifespan`）在应用启动时调用 `setup_storage()`，再由 `_init_postgres_storage()` 调用 `init_pool(url)`。

**类似单例**：
- `rag_core.py` 模块级 `collection: PostgresJsonbDocCollection | None` 和 `vectorstore: PostgresVectorStore | None`，由 `setup_storage()` 填充。
- `hybrid_rag.py` 模块级 `_bm25_index: BM25Index | None`（另有 `_bm25_lock: threading.Lock` 保护并发）。

---

### 3.2 端口与适配器（Ports & Adapters）

**端口（接口）**：`src/domain/interfaces/graph_repository.py`  
**适配器（实现）**：`src/infrastructure/persistence/postgres_graph_repository.py`

图存储后端在运行时由 `config.graph_backend` 决定（`"age"` 或 `"relational"`），对上层 `hybrid_rag.py` 透明。

```
hybrid_rag.py
    └─ postgres_age_graph.merge_triple_age()   if age_is_ready()
    └─ postgres_graph.merge_triple()            fallback
```

`postgres_age_setup.age_is_ready()` 返回一个进程内布尔标志，避免每次 I/O 重探 AGE 可用性。

---

### 3.3 状态机（State Machine）— LangGraph

**文件**：`src/rag_core.py`

```python
class AgentState(TypedDict, total=False):
    question: str
    documents: List[str]
    generation: str
    loop_count: int
    logs: List[str]
    is_analysis: bool     # 一次性计算，节点间复用
    skip_retrieval: bool  # hybrid 路径预填文档时置 True，跳过向量检索

# 图定义
graph = StateGraph(AgentState)
graph.add_node("retrieve", retrieve_and_rerank_node)
graph.add_node("grade",    grade_documents_node)
graph.add_node("generate", generate_node)
graph.add_edge(START,      "retrieve")
graph.add_edge("retrieve", "grade")
graph.add_edge("grade",    "generate")
graph.add_edge("generate", END)

rag_app = graph.compile()
```

`hybrid_rag.prepare_hybrid_chat()` 通过设置 `skip_retrieval=True` 将预检索的文档直接注入 `AgentState`，复用 `grade` 和 `generate` 节点，而不重复向量检索。

---

### 3.4 策略模式（Strategy）— 动态提示选择

**文件**：`src/rag_core.py`

```python
# 运行时选择 Prompt 策略
is_analysis = _is_analysis_query(question)  # 关键词匹配

grade_prompt = GRADE_ANALYSIS_PROMPT if is_analysis else GRADE_PROMPT
gen_prompt   = GEN_ANALYSIS_PROMPT   if is_analysis else GEN_PROMPT
```

**`_is_analysis_query()`** 扫描 `_ANALYSIS_INTENT_KEYWORDS`（含中英文 34 个词）。`is_analysis` 标志在 `retrieve_and_rerank_node` 中计算一次，通过 `AgentState` 传递给后续节点，避免重复计算。

---

### 3.5 模板方法（Template Method）— 受控图查询

**文件**：`src/core/graph_constrained_queries.py`

```python
# 白名单模板注册表（dict），避免自由 Cypher 注入
_TEMPLATES: dict[str, Callable] = {
    "edges_from_entity":      _tpl_edges_from_entity,
    "multi_keyword_edges":    _tpl_multi_keyword_edges,
    "graph_global_summary":   _tpl_graph_global_summary,
}

def execute_constrained_template(template_id: str, params: dict) -> ConstrainedGraphResult:
    fn = _TEMPLATES.get(template_id)
    if fn is None: raise ...
    return fn(params)
```

模板 ID 通过 `ConstrainedGraphRequest.template: Literal[...]` 在 schema 层强制限定，无法传入任意 Cypher。

---

### 3.6 工厂方法（Factory）— FastAPI 应用组装

**文件**：`src/api/main.py`

```python
def create_app() -> FastAPI:
    application = FastAPI(title="MAP-RAG MVP API", lifespan=lifespan)
    register_exception_handlers(application)
    application.mount("/static", StaticFiles(directory=STATIC_DIR))
    application.include_router(pages.router)
    application.include_router(ingest.router)
    # ... 其余 8 个路由
    return application

app = create_app()
```

工厂函数使得测试可以独立实例化应用（无全局副作用），也便于后续多实例部署。

---

### 3.7 重试装饰器模式（Retry）— 外部 API 弹性

**文件**：`src/rag_core.py`

```python
def _post_json_with_retries(url, headers, payload) -> dict:
    for attempt in range(JINA_MAX_RETRIES):   # JINA_MAX_RETRIES = 3
        resp = _http_session.post(url, ..., timeout=JINA_REQUEST_TIMEOUT)
        if resp.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** attempt)           # 指数退避
            continue
        return resp.json()
    raise RuntimeError(...)
```

供 `chunk_text_jina()` 和 `jina_rerank()` 共用，保护 Jina AI 的两个外部端点（`segment.jina.ai` 和 `api.jina.ai/v1/rerank`）。

---

### 3.8 进程内共享状态（In-Process State Store）

**文件**：`src/api/state.py`

```python
tasks: dict[str, dict[str, Any]] = {}         # 上传任务状态，TTL 清理
chat_sessions: dict[str, dict[str, Any]] = {} # 会话历史，TTL 清理
feedback_log: list[dict[str, Any]] = []        # 反馈环形缓冲，MAX=1000

def cleanup_tasks() -> None          # 每次请求前惰性清理过期任务
def cleanup_chat_sessions() -> None  # 每次请求前惰性清理过期会话
```

> ⚠️ 此设计为 **单进程内存存储**，多进程/多实例部署时会话不共享。生产级别需替换为 Redis 或数据库持久化。

---

## 4. 依赖关系与风险

### 4.1 模块依赖图

```
src/api/routers/chat.py
    → src/rag_core.py
    → src/api/guardrails.py
    → src/api/schemas.py

src/api/routers/session.py
    → src/rag_core.py  (run_chat_pipeline, run_chat_pipeline_multimodal)
    → src/hybrid_rag.py  (通过 rag_core.run_chat_pipeline 间接)
    → src/api/session_memory.py
    → src/api/state.py
    → src/api/guardrails.py
    → src/api/upload_validation.py
    → src/pdf_extract.py / src/docx_extract.py / src/xlsx_extract.py

src/api/routers/ingest.py
    → src/rag_core.py  (process_document_task)
    → src/api/state.py
    → src/api/upload_validation.py
    → src/pdf_extract.py / src/docx_extract.py / src/xlsx_extract.py

src/rag_core.py
    → src/infrastructure/persistence/postgres_connection.py
    → src/infrastructure/persistence/postgres_jsonb_collection.py
    → src/infrastructure/persistence/postgres_vectorstore.py
    → src/infrastructure/persistence/postgres_schema.py
    → src/core/config.py
    → src/core/rag_context_format.py
    → src/core/citations.py
    → src/core/query_anchors.py
    → langchain_google_vertexai  (ChatVertexAI, VertexAIEmbeddings)
    → langgraph

src/hybrid_rag.py
    → src/rag_core.py  (retrieve_parent_documents_expanded, jina_rerank,
                        grade_documents_node, generate_node, invoke_rag_app,
                        stream_generation_chunks, llm, vectorstore)
    → src/infrastructure/persistence/postgres_graph.py
    → src/infrastructure/persistence/postgres_age_graph.py
    → src/infrastructure/persistence/postgres_age_setup.py
    → src/infrastructure/persistence/postgres_connection.py
    → src/core/config.py
    → src/core/query_anchors.py
    → src/core/citations.py
    → rank_bm25

src/core/graph_constrained_queries.py
    → src/infrastructure/persistence/postgres_graph.py
    → src/infrastructure/persistence/postgres_age_graph.py
    → src/infrastructure/persistence/postgres_age_setup.py
    → src/infrastructure/persistence/postgres_connection.py

src/infrastructure/persistence/postgres_vectorstore.py
    → src/infrastructure/persistence/postgres_connection.py

src/infrastructure/persistence/postgres_jsonb_collection.py
    → src/infrastructure/persistence/postgres_connection.py
```

### 4.2 循环依赖风险

| 风险点 | 描述 | 缓解方式 |
|---|---|---|
| `hybrid_rag` → `rag_core` | `hybrid_rag.py` 直接导入 `rag_core` 中的函数（`retrieve_parent_documents_expanded`, `jina_rerank`, `grade_documents_node`, `generate_node`, `invoke_rag_app`, `stream_generation_chunks`, `llm`, `vectorstore`） | 目前单向，无循环。若 `rag_core` 反向导入 `hybrid_rag`，则产生循环依赖。**守则：`rag_core` 禁止导入 `hybrid_rag`**，已在 `run_chat_pipeline` 中以局部 `import` 保护 |
| `session.py` 导入两个提取器 | `session.py` 在函数体内以局部 `import` 引入 `pdf_extract`、`docx_extract`、`xlsx_extract`，避免循环 | 局部导入是正确做法，维持此风格 |
| `state.py` 被多处读写 | `tasks`、`chat_sessions`、`feedback_log` 为模块级可变状态，`ingest`/`session`/`feedback`/`eval` 路由均直接操作 | 单进程安全（GIL 保护 dict 操作）；多进程需外化状态 |

### 4.3 `rag_core` 中的局部导入防护

```python
# run_chat_pipeline 中保护循环导入
def run_chat_pipeline(query: str):
    if get_settings().hybrid_rag_enabled:
        try:
            from src.hybrid_rag import run_hybrid_chat_pipeline  # ← 局部导入
            ...
```

此模式在 `process_document_task` 中对 `hybrid_rag.ingest_chunks_to_graph`、`graph_constrained_queries.py` 中对持久层同样使用，属于刻意设计。

---

## 5. 外部服务与环境依赖

| 服务 | 用途 | 配置 | 不可用时的 fallback |
|---|---|---|---|
| **Google Vertex AI** | LLM `gemini-2.5-flash` + Embeddings `text-embedding-004` | GCP 应用默认凭证（ADC）或 `GOOGLE_APPLICATION_CREDENTIALS` | 无 fallback，请求失败抛异常 |
| **Jina AI Segmenter** | 文档语义分块 `segment.jina.ai` | 环境变量 `JINA_API_KEY` | fallback: 返回 `[full_text]`，不分块 |
| **Jina AI Reranker** | 交叉编码器重排 `api.jina.ai/v1/rerank`（`jina-reranker-v2-base-multilingual`） | 同上 | fallback: 返回 top-n 截断，不重排 |
| **PostgreSQL 14+** | 所有持久化存储（向量、文档、图） | `DATABASE_URL` 环境变量 | 缺失则 `setup_storage()` 跳过（警告），API 可启动但检索报错 |
| **Apache AGE** | PostgreSQL 图扩展（Cypher 支持） | 同上，由 `ensure_age_extension_and_graph()` 探测 | fallback: `postgres_graph.py` 关系型三元组路径 |
| **LangSmith** | LangChain 链路追踪 | `LANGCHAIN_API_KEY` + `LANGCHAIN_PROJECT` | 无 key 时静默跳过，不影响功能 |

---

## 6. 相关文档

| 文档 | 内容 |
|---|---|
| [bitter_lesson_roadmap.md](./bitter_lesson_roadmap.md) | 可扩展性路线图（度量→学习→反模式） |
| [postgresql_storage_roadmap.md](./postgresql_storage_roadmap.md) | PostgreSQL 统一存储迁移详情与表设计 |
| [mvp_hybrid_rag.md](./mvp_hybrid_rag.md) | Hybrid RAG 路由器设计（VECTOR/GRAPH/GLOBAL/HYBRID） |
| [doc-small-to-big-retrieval.md](./doc-small-to-big-retrieval.md) | Small-to-Big 父子块检索策略 |
| [dual-track-query-routing.md](./dual-track-query-routing.md) | 查询路由双轨设计 |
| [graph_constrained_queries.md](./graph_constrained_queries.md) | 受控 Cypher 白名单模板 |
| [whatif_tools.md](./whatif_tools.md) | What-If 确定性工具案例 |
| [recall_evaluation.md](./recall_evaluation.md) | Recall@K / NDCG@K 评测方法 |
| [langsmith_observability.md](./langsmith_observability.md) | LangSmith 集成与追踪 |
| [testing.md](./testing.md) | 测试策略与 conftest 配置 |
| [TODO.md](./TODO.md) | OpenClaw 与 RAG 边界、后续迭代项 |
