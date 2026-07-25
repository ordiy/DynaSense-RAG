# PostgreSQL 统一存储（已完成：替代 LanceDB / Neo4j / MongoMock）

> **状态（2026）**：迁移已落地。运行时存储为 **PostgreSQL 单库**（`pgvector` + JSONB `kb_doc` + Apache AGE 或关系表 `kg_triple`）。LanceDB / Neo4j / MongoMock **已移除**，不再作为回退路径。  
> 本文保留动机与表设计说明，供运维与后续演进参考；阶段清单中的勾选项视为历史计划。

## 动机（历史）

- **部署简化**：生产环境只维护 **一个** 有状态服务（PostgreSQL）及备份/高可用方案，而不是 LanceDB 文件目录 + Neo4j 集群 + 进程内 MongoMock。
- **规模匹配**：语料与向量在 **GB 量级** 时，单机或主备 PostgreSQL + 合适索引通常是 **甜蜜点**。
- **与整洁架构一致**：通过 `IGraphRepository` 等端口，将 `rag_core` / `hybrid_rag` 从具体存储解耦。

---

## 当前存储职责（代码映射）

| 能力 | 实现 | 用途 |
|------|------|------|
| **向量** | `postgres_vectorstore.py` → `kb_embedding`（pgvector） | 稠密检索、Small-to-Big 父文档展开 |
| **文档** | `postgres_jsonb_collection.py` → `kb_doc`（JSONB parent/child） | 父全文 / 子块内容；FTS 读子块 |
| **图** | `graph_store.py` → AGE（`postgres_age_graph.py`）或 `kg_triple`（`postgres_graph.py`） | 混合路由 GRAPH/GLOBAL、全局摘要 |
| **词法** | `postgres_fts.py`（`tsvector` + GIN） | 替代进程内 BM25 缓存 |

原 Neo4j 路径（Bolt + `neo4j` 驱动、`docker-compose.neo4j.yml`）已退役；图语义由 **Apache AGE Cypher**（默认 `GRAPH_BACKEND=age`）或 **`kg_triple` SQL**（`relational` / AGE 不可用时回退）承担。

---

## 目标形态：PostgreSQL + 插件（已实现）

### 必选：`pgvector`

- 存储 **embedding** 与 **chunk 元数据**（`content`、`meta` JSONB），ANN 检索（IVFFlat / HNSW，视版本与数据量）。

### 词法检索：PostgreSQL FTS

- 对子块 `content` 建 `to_tsvector` + GIN；查询走 `postgres_fts.fulltext_search_children`。
- 分数分布与旧 `rank_bm25` 不同；召回对比见评测文档。

### 图数据：Apache AGE（默认）+ `kg_triple` 回退

关键词命中实体/关系类型（`query_relationships_by_keywords`）仍是主路径；需要多跳 / 受控 Cypher 时用 AGE。

关系表示意（与 `merge_triple` 语义对齐，AGE 不可用时使用）：

```sql
CREATE TABLE kg_triple (
  id            bigserial PRIMARY KEY,
  subject_norm  text NOT NULL,
  subject_name  text NOT NULL,
  predicate     text NOT NULL,
  object_norm   text NOT NULL,
  object_name   text NOT NULL,
  chunk_id      text NOT NULL,
  source        text,
  created_at    timestamptz DEFAULT now(),
  UNIQUE (subject_norm, object_norm, predicate, chunk_id)
);
```

`global_graph_summary` 由 AGE 聚合或 SQL 聚合实现。

### 文档存储（原 MongoMock）

- 单表 **`kb_doc`**（`id` + `doc` JSONB）：`type` = `parent` | `child`，含 `source`、`parent_id`、`content` / `full_content`。

---

## 迁移阶段（历史清单 — 已完成）

### 阶段 0 — 抽象与开关

- [x] PostgreSQL 向量 / JSONB / 图适配器落地。
- [x] `IGraphRepository` + `PostgresGraphRepository`（及 AGE 路径）。
- [x] 以 `DATABASE_URL` + `GRAPH_BACKEND` 配置存储（无双后端热切换）。

### 阶段 1 — 写入与读路径

- [x] Ingest 写入 PG（chunk + embedding + triple）。
- [x] 检索默认读 PG；`debug_routes` 提供 `/api/debug/pg/*` 与图调试 API。

### 阶段 2 — 退役旧组件

- [x] 移除 LanceDB / Neo4j / MongoMock 驱动与 compose；文档改为 PostgreSQL + AGE。

### 阶段 3 — 清理

- [x] 基准脚本等改为 `DATABASE_URL`；图入库跳过使用 `SKIP_GRAPH_INGEST`（原 `SKIP_NEO4J_INGEST` 已废弃）。

---

## 运维与规模（GB 级）

| 主题 | 建议 |
|------|------|
| 连接 | 连接池（PgBouncer）、应用侧 `psycopg` pool |
| 向量索引 | 数据量上来后调 `lists`（IVFFlat）或 HNSW `m`/`ef_construction` |
| 备份 | `pg_dump` / 连续归档；单库易做一致快照 |
| 多租户 | `tenant_id` 列 + RLS（行级安全）比多库多实例更易扩展 |
| AGE | 镜像需安装 `age` 扩展；失败时自动回退 `kg_triple` |

---

## 风险与缓解

| 风险 | 缓解 |
|------|------|
| 向量检索延迟 | 索引调参、限制 `probes`、缓存热门 query embedding |
| FTS 与旧 BM25 排序不一致 | 保留评测集，对比 Recall@K |
| AGE 扩展不可用 | `GRAPH_BACKEND=relational` 或自动回退 `kg_triple` |

---

## 相关代码入口

- `src/rag_core.py`：`setup_storage`、`process_document_task`、向量检索。
- `src/hybrid_rag.py`：路由、FTS、`ingest_chunks_to_graph`。
- `src/graph_store.py`：对 AGE / relational 的门面。
- `src/infrastructure/persistence/`：连接池、schema、vector、JSONB、AGE、FTS、feedback。
- `scripts/benchmark_recall_ndcg.py`：`DATABASE_URL` + `SKIP_GRAPH_INGEST`。

---

## 环境变量

| 变量 | 说明 |
|------|------|
| `DATABASE_URL` | 连接串，如 `postgresql://postgres:postgres@127.0.0.1:5433/map_rag`（见 `docker-compose.postgres.yml` / `docker-compose.yml`） |
| `GRAPH_BACKEND` | `age`（默认）：**Apache AGE** + Cypher；`relational`：仅用 `kg_triple` |
| `AGE_GRAPH_NAME` | AGE 图名，默认 `map_rag_kg` |
| `SKIP_GRAPH_INGEST` | `true` / `1`：跳过入库时的 LLM 三元组抽取（基准常用） |

未设置 `DATABASE_URL` 时应用可启动，但存储相关能力不可用（见启动日志）；**不再**回退到 LanceDB / Neo4j / MongoMock。

---

## 相关文档

- [架构说明](./architecture.md)
- [Hybrid RAG MVP](./mvp_hybrid_rag.md)
- [受控图查询](./graph_constrained_queries.md)（白名单模板；底层为 AGE Cypher 或 SQL）
- [后续迭代 TODO](./TODO.md)
