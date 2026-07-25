# Demo Console VNext — 方案说明（供 review）

> **版本**：2026-07 · 意向客户演示迭代  
> **范围**：左侧商务布局、演示登录、PostgreSQL 会话历史、测试覆盖

## 1. 动机

为意向客户演示提供：

1. 专业商务 UI（左侧导航，非顶部 Tab）
2. 可回放的多轮会话历史（跨刷新 / 重启仍在）
3. 登录门槛（避免演示环境裸奔）
4. 可回归的自动化测试

## 2. 架构决策

| 项 | 选择 | 理由 |
|----|------|------|
| 登录 | 环境变量演示账号 + HttpOnly 签名 Session Cookie | 实现快、适合闭门演示，无需 SSO |
| 密码 | bcrypt（启动时对明文配置哈希） | 不落库明文；演示账号仍可在 `.env` 配置 |
| 会话存储 | PostgreSQL `chat_conversation` / `chat_message` write-through | 解决进程内存会话多实例丢失问题 |
| 无 DB 降级 | `MemoryConversationStore` | 单测与降级模式 |
| 鉴权开关 | `AUTH_ENABLED`（默认 `true`） | CI / 本地可关 |

```text
Browser → /login → POST /api/auth/login → Session Cookie
Browser → / (index) → AuthGate → left sidebar console
RAG Chat → GET /api/chat/sessions + POST /api/chat/session → ConversationStore → PostgreSQL
```

## 3. 数据模型

```sql
chat_conversation (
  id TEXT PK,
  user_id TEXT NOT NULL,       -- username
  title TEXT NOT NULL,
  created_at TIMESTAMPTZ,
  updated_at TIMESTAMPTZ
)

chat_message (
  id TEXT PK,
  conversation_id TEXT REFERENCES chat_conversation(id) ON DELETE CASCADE,
  role TEXT CHECK (role IN ('user','assistant')),
  content TEXT NOT NULL,
  meta JSONB DEFAULT '{}',     -- route / citations
  created_at TIMESTAMPTZ
)
```

归属校验：所有读写均带 `user_id = 当前登录用户`。

## 4. API

| Method | Path | 说明 |
|--------|------|------|
| POST | `/api/auth/login` | 登录，写 session cookie |
| POST | `/api/auth/logout` | 登出 |
| GET | `/api/auth/me` | 当前用户 |
| GET | `/api/chat/sessions` | 当前用户会话列表 |
| POST | `/api/chat/session` | 多轮对话（持久化） |
| GET | `/api/chat/session/{id}` | 历史 |
| DELETE | `/api/chat/session/{id}` | 删除（级联消息） |

页面：`GET /login`；`AUTH_ENABLED` 时未登录访问 `/`、`/demo` → `302 /login?next=...`。

## 5. 环境变量

见 [`.env.example`](../.env.example)：

```bash
AUTH_ENABLED=true
AUTH_SECRET_KEY=replace-with-a-long-random-string
AUTH_USERS=[{"username":"demo","password":"demo123","display_name":"Demo User"}]
# AUTH_SESSION_HTTPS_ONLY=true   # 仅 HTTPS 部署时开启
```

本地开发可设 `AUTH_ENABLED=false` 跳过登录。

## 6. UI

- [`src/static/login.html`](../src/static/login.html) — 居中登录卡
- [`src/static/index.html`](../src/static/index.html) — 海军蓝侧栏 + 浅灰主区；RAG Chat 双栏（会话列表 | 对话）
- [`src/static/portal.html`](../src/static/portal.html) — 布局未大改，仅受登录门禁保护

## 7. 代码入口

| 模块 | 职责 |
|------|------|
| `src/api/routers/auth.py` | 登录 API |
| `src/api/auth_middleware.py` | AuthGate |
| `src/api/deps.py` | `require_user` |
| `src/api/auth_users.py` | 用户解析 / bcrypt |
| `src/infrastructure/persistence/postgres_conversations.py` | 会话仓储 |
| `src/infrastructure/persistence/postgres_schema.py` | DDL |
| `src/api/routers/session.py` | 会话 API（PG write-through） |

## 8. 测试矩阵

| 文件 | 覆盖 |
|------|------|
| `tests/test_auth_api.py` | 登录/登出/me、401、AUTH 开关、页面重定向 |
| `tests/test_conversation_store.py` | Memory + mock PG CRUD / 归属 |
| `tests/test_session_persistence_api.py` | list/chat/get/delete（mock pipeline） |
| `tests/test_console_static.py` | 侧栏、会话面板、login.html、DDL |
| `tests/test_portal_static.py` | portal 存在、/demo→login、/login 可访问 |

```bash
pytest tests/test_auth_api.py tests/test_conversation_store.py \
  tests/test_session_persistence_api.py tests/test_console_static.py \
  tests/test_portal_static.py -q
```

## 9. 演示剧本（建议）

1. 打开控制台 → 自动跳转登录页  
2. 使用发放的演示账号登录  
3. Data Ingestion：上传 `data/demo_related_party.txt`  
4. RAG Chat：提问「中国中信银行的关联方有哪些？」→ 左侧出现会话  
5. 刷新页面 → 会话仍在；点开可回放历史  
6. Sign out → 再访问 `/` 应回到登录页  

## 10. 已知边界

- 旧进程内存会话不迁移  
- `portal.html` 未做左侧改版  
- 无 OAuth / SSO  
- 演示账号密码在 env 明文配置（仅限闭门演示；生产应换 SSO 或密钥管理）
