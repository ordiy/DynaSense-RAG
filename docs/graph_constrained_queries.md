# 受控图查询（替代 GraphCypherQAChain 的方案）

## 目标

在 **不生成任意 Cypher** 的前提下，提供可测试、可审计的图查询能力：

- **白名单模板** → 仅调用 `graph_store` 中已有、参数化的查询，或等价逻辑。
- **参数校验**（长度、类型、limit 上限），字符串 **仅作查询参数绑定**，不拼进查询字符串。
- **可选启发式** `suggest_template_from_question`：用正则把自然语言映射到 `(template_id, params)`，便于联调；**生产可替换为 LLM 结构化输出**，但仍只产出 **JSON 槽位**，不产出 Cypher。

## API（需 `DEBUG_DATA_API=true`）

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/debug/graph/constrained/run` | 执行模板：`template` + `params` |
| POST | `/api/debug/graph/constrained/suggest` | 启发式建议；`?execute=true` 时顺带执行 |

### 模板 id

| `template` | `params` | 行为 |
|------------|----------|------|
| `edges_from_entity` | `name_substring`, `limit?` | 单关键词，复用 `query_relationships_by_keywords` |
| `multi_keyword_edges` | `keywords` (list), `limit?` | 多关键词 |
| `graph_global_summary` | `{}` | 调用 `global_graph_summary()` |

## 与 Hugging Face Cookbook 的差异

- **不做** NL → 自由 Cypher → 执行。
- **做** NL（或结构化体）→ **固定模板 id + 参数** → 安全执行。

## 测试

见 `tests/test_graph_constrained_queries.py`、`tests/test_graph_constrained_api.py`。
