# PostgreSQL 统一存储路线图（替代 LanceDB / Neo4j / MongoMock）

## 动机

- **部署简化**：生产环境只维护 **一个** 有状态服务（PostgreSQL）及备份/高可用方案，而不是 LanceDB 文件目录 + Neo4j 集群 + 进程内 MongoMock。
- **规模匹配**：语料与向量在 **GB 量级** 时，单机或主备 PostgreSQL + 合适索引通常是 **甜蜜点**；无需为向量/图各上一套运维手册。
- **与整洁架构一致**：已有 `IGraphRepository` 端口，可自然增加 **向量索引端口**，将 `rag_core` / `hybrid_rag` 从具体存储解耦。

本文是 **迁移计划与表设计方向**，实施可按阶段推进，**不要求一次性切换**。

---

## 当前存储职责（代码映射）

| 能力 | 现状 | 用途 |
|------|------|------|
| **LanceDB** | `rag_core.py`：`db_lance`、LangChain `LanceDB`、手动 `add` 向量行 | 稠密检索、Small-to-Big 父文档展开 |
| **MongoMock** | `rag_core` 全局 `collection`：parent/child 文档 | **BM25** 在 `hybrid_rag.bm25_parent_documents` 中 `find({"type":"child"})` 拉全量子块建索引 |
| **Neo4j** | `graph_store.py`：`Entity`-`REL`->`Entity`，`query_relationships_by_keywords` 等 | 混合路由的图路径、全局摘要 |

---

## 目标形态：PostgreSQL + 插件

### 必选：`pgvector`

- 存储 **embedding** 与 **chunk 元数据**（`text`、`metadata` JSONB），用 `<=>` 或 `<#>` 做 **近似最近邻**（IVFFlat / HNSW，视版本与数据量选择）。
- 生态：`langchain_community.vectorstores.PGVector`（或自写轻量查询）均可对接。

###  lexical / BM25：二选一

| 方案 | 说明 |
|------|------|
| **A. PostgreSQL `tsvector` + GIN** | 对 `content` 建全文检索；查询用 `plainto_tsquery` / `websearch_to_tsquery`。与 Python `rank_bm25` 分数分布不同，需 **重跑评测** 调参。 |
| **B. 仍用 `rank_bm25`，数据从 PG 读出** | 逻辑与现网接近：子块表 `SELECT` → 内存建 BM25；GB 级子块数需评估 **单次加载内存**（可分页或按 `source` 分片）。 |

**建议**：先 **B** 降低行为变化；数据再大再切 **A** 或引入外部检索服务。

### 图数据：优先 **关系表**，不必默认上 Apache AGE

当前 Neo4j 查询以 **关键词命中实体/关系类型** 为主（见 `graph_store.query_relationships_by_keywords`），并非复杂图算法。

推荐 **单表即可**：

```sql
-- 示意：三元组 + 溯源（与 merge_triple 语义对齐）
CREATE TABLE kg_triple (
  id            bigserial PRIMARY KEY,
  subject_norm  text NOT NULL,
  subject_name  text NOT NULL,
  predicate     text NOT NULL,
  object_norm   text NOT NULL,
  object_name   text NOT NULL,
  chunk_id      text NOT NULL,
  source        text,
  created_at    timestamptz DEFAULT now()
);
CREATE INDEX kg_triple_subj_gin ON kg_triple USING gin (subject_name gin_trgm_ops);  -- 需 pg_trgm
CREATE INDEX kg_triple_obj_gin  ON kg_triple USING gin (object_name gin_trgm_ops);
CREATE INDEX kg_triple_pred     ON kg_triple (predicate);
```

用 `ILIKE` / `pg_trgm` 模拟现有 `CONTAINS` 行为；`global_graph_summary` 可用聚合 SQL 替代。

**何时考虑 [Apache AGE](https://age.apache.org/)**：需要 **多跳 Cypher**、路径查询、与 Neo4j 语法对齐迁移时；否则增加运维与扩展复杂度，**不**作为默认选项。

### MongoMock 替代

- **parent / child** 与向量行可 **合并建模** 或 **分表**：
  - `document_parent(id, source, full_content, …)`
  - `document_chunk(id, parent_id, source, content, embedding vector(768), …)`（维度与 Vertex 模型一致）

进程内不再使用 `mongomock`；BM25 路径改为读 PG。

---

## 迁移阶段（建议）

### 阶段 0 — 抽象与开关

- [ ] 定义 `IVectorIndex`（或沿用 LangChain VectorStore 接口但实现换为 PG）。
- [ ] `IGraphRepository` 增加 **PostgreSQL 实现**（与 `Neo4jGraphRepository` 并存）。
- [ ] 环境变量：`STORAGE_BACKEND=legacy|postgresql`（或分项开关）。

### 阶段 1 — 双写或离线迁移

- [ ] 新 ingest：**写入 PG**（chunk + embedding + triple）。
- [ ] 读路径：灰度 **只读 PG** 或 **Neo4j/Lance 对照** 验证一致性。
- [ ] `debug_routes`：增加 `/debug/pg/summary`（可选），与旧 debug 并存。

### 阶段 2 — 读切换与退役

- [ ] 默认检索走 PG；监控延迟与召回。
- [ ] 移除 LanceDB / Neo4j 驱动依赖（或标为 optional）；更新 `requirements.txt`、`docker-compose`、文档。

### 阶段 3 — 清理

- [ ] 删除 `mongomock`、本地 Lance 路径假设；脚本 `benchmark_recall_ndcg.py` 等改为 `DATABASE_URL`。

---

## 运维与规模（GB 级）

| 主题 | 建议 |
|------|------|
| 连接 | 连接池（PgBouncer）、`asyncpg` / SQLAlchemy pool |
| 向量索引 | 数据量上来后调 `lists`（IVFFlat）或 HNSW `m`/`ef_construction` |
| 备份 | `pg_dump` / 连续归档；单库比「文件型 Lance + 图库」易做 **一致快照**（若双写需注意顺序） |
| 多租户 | `tenant_id` 列 + RLS（行级安全）比多库多实例更易扩展 |

---

## 风险与缓解

| 风险 | 缓解 |
|------|------|
| 向量检索延迟高于 LanceDB 嵌入式 | 索引调参、限制 `probes`、缓存热门 query embedding |
| FTS 与旧 BM25 排序不一致 | 保留评测集，对比 Recall@K；必要时仍用应用侧 BM25 |
| 大迁移停机 | 双写 + 后台回填 + 读开关 |

---

## 相关代码入口（迁移时重点改）

- `src/rag_core.py`：LanceDB / MongoMock 初始化、`process_document_task`、`reset_knowledge_base`、`vectorstore` 全局。
- `src/hybrid_rag.py`：`collection` 依赖 → 改为仓储接口；`ingest_chunks_to_neo4j` → 写 `kg_triple`。
- `src/graph_store.py`：Neo4j → PG 实现或委托给 `PostgresGraphRepository`。
- `src/debug_data.py`：LanceDB/Neo4j 调试 → PG SQL。
- `scripts/benchmark_recall_ndcg.py`：环境变量与存储路径。

---

## 环境变量（启用 PostgreSQL 后端）

| 变量 | 说明 |
|------|------|
| `STORAGE_BACKEND` | 设为 `postgresql` 启用统一 PG 存储 |
| `DATABASE_URL` | 连接串，如 `postgresql://postgres:postgres@127.0.0.1:5433/map_rag`（见 `docker-compose.postgres.yml`） |
| `GRAPH_BACKEND` | `age`（默认）：**Apache AGE** + Cypher；`relational`：仅用 `kg_triple` 表（无 AGE 扩展时也会自动回退） |
| `AGE_GRAPH_NAME` | AGE 图名，默认 `map_rag_kg` |

**文档与父/子块**：使用单表 **`kb_doc`**（`id` + **`doc` JSONB**），不再使用 MongoMock 或 `doc_parent`/`doc_child` 双表。

**图数据**：优先在 AGE 中存储 `Entity` 与 `REL` 边；若 `CREATE EXTENSION age` 失败，则创建 **`kg_triple`** 关系表（与旧版行为一致）。

未设置或连接失败时，`rag_core` 会 **回退** 到 LanceDB + MongoMock，并打 error 级日志，避免进程无法启动。

---

## 相关文档

- [架构说明](./architecture.md)
- [受控图查询](./graph_constrained_queries.md)（迁移后模板仍适用，仅底层从 Neo4j 换为 SQL）
- [后续迭代 TODO](./TODO.md)
