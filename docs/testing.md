# 测试与验证说明

本文记录本仓库 **pytest** 覆盖范围、推荐命令、环境依赖，以及验证过程中遇到的问题与处理方式。

## 测试分层

| 层级 | 说明 | 依赖 |
|------|------|------|
| **纯单元 / 无 DB** | 大部分 `tests/test_*.py`（analytics、citations、guardrails、what-if 等） | `requirements-ci.txt` 即可 |
| **PostgreSQL 集成** | `tests/test_postgres_integration.py`、`test_debug_data.test_kb_embedding_summary_with_pool` | 运行中的 PostgreSQL（`DATABASE_URL`） |
| **全栈 HTTP** | `tests/test_api_smoke.py` | `DATABASE_URL` + `GOOGLE_APPLICATION_CREDENTIALS`，且 **`CI` 不能为 `true`**（否则跳过） |

## 推荐命令

```bash
# 虚拟环境（仓库根目录）
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-ci.txt    # 仅 CI 类测试
pip install -r requirements.txt         # 全量（含 Vertex / LangChain）

# 1) 默认 CI 场景：无数据库
pytest tests/ -q

# 2) 本地/CI 带数据库：先启动 PostgreSQL（见 docker-compose.postgres.yml）
export DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:5433/map_rag
pytest tests/ -q

# 3) 仅数据库相关
pytest tests/test_postgres_integration.py tests/test_debug_data.py -v
```

## 验证结果（记录于 2026-04）

- **无 `DATABASE_URL`**：`56 passed, 3 skipped`（跳过 PG 集成、`kb_embedding` 池测试、以及条件不满足时的 `test_api_smoke`）。
- **设置 `DATABASE_URL` 且本机具备 Vertex 凭证**：`59 passed`（含 PG 集成与 `test_api_smoke`）。
- **应用导入**：`from src.api.main import create_app` 在设置 `DATABASE_URL` 后可成功构建应用；首次导入可能较慢（见下文「已知现象」）。

## 已知现象与处理

### 1. 首次导入 `rag_core` 较慢（数十秒）

**现象**：执行 `create_app()` 或首次导入 `src.rag_core` 时，进程在日志中出现 `google.auth` 相关调试信息后停顿。

**原因**：模块级初始化会构造 Vertex AI 的 `VertexAIEmbeddings` / `ChatVertexAI`，默认凭据解析与 gRPC 初始化会占用时间。

**处理**：属预期行为；自动化测试里若需缩短时间，可保持会话内只导入一次应用，或使用已缓存的凭据环境。长期可考虑将重型客户端改为懒加载（另起重构任务）。

### 2. LangChain 弃用告警

**现象**：pytest 对 `VertexAIEmbeddings` / `ChatVertexAI` 报告 `LangChainDeprecationWarning`。

**原因**：LangChain 提示迁移到 `langchain-google-genai`。

**处理**：不影响当前功能；迁移到 `GoogleGenerativeAIEmbeddings` 等可作为独立工单。

### 3. `test_api_smoke` 被跳过

**条件**（见 `tests/test_api_smoke.py`）：`CI=true` **或** 未设置 `GOOGLE_APPLICATION_CREDENTIALS` **或** 未设置 `DATABASE_URL`。

**处理**：在本地全量验证时导出上述变量；GitHub Actions 若未配 GCP 密钥，该测试保持跳过是合理的。

### 4. 无 `DATABASE_URL` 时无法启动 API

**现象**：导入 `rag_core` 时 `RuntimeError: DATABASE_URL is required`。

**处理**：启动 `uvicorn` 前设置 `DATABASE_URL`（见 `README.md` 与 `docker-compose.postgres.yml`）。

## 与存储相关的历史说明

存储已统一为 **PostgreSQL**（JSONB + pgvector + 可选 AGE）。旧版 LanceDB / MongoMock / Neo4j 路径已移除；回归以本文命令与 `DATABASE_URL` 为准。
