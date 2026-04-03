# LangGraph 运行过程日志（stream）

在本地或服务器日志中查看 **LangGraph 每一步执行后的状态**，无需打开 LangSmith。

参考：[LangGraph streaming](https://docs.langchain.com/oss/python/langgraph/streaming)

---

## 启用方式

```bash
export LANGGRAPH_STREAM_LOG=true
```

或在 `.env` 中设置（与 `docs/langsmith_observability.md` 中的变量可并存）。

重启 `uvicorn` 后生效。

---

## 行为说明

### 向量-only LangGraph（`retrieve → grade → generate`）

在 `src/rag_core.py` 中，当 `LANGGRAPH_STREAM_LOG=true` 时，使用 **`rag_app.stream(..., stream_mode="values")`** 代替单次 `invoke`：每完成一步，**当前完整 state** 会打一条 **INFO** 日志，例如：

```text
[LangGraph stream] step=0 | documents=... | generation_chars=0 | log_lines=...
```

- **step**：从 0 递增，对应图中各节点执行后的状态快照。
- 最后一步即为最终状态，与原先 `invoke` 结果一致。

### Hybrid RAG 主路径

Hybrid 在检索与融合之后，对 **grade / generate** 是直接调用节点函数，**不是**整图 `invoke`。开启同一环境变量时，会额外打 **Hybrid pipeline** 的阶段性日志（候选数、grade 后文档数、生成后答案长度），便于对照「检索 → 打分 → 生成」。

---

## 日志级别

使用 Python `logging` 的 **INFO**。若需看到输出，请保证根日志级别为 INFO（`rag_core` 中已 `logging.basicConfig(level=logging.DEBUG)`，默认可见）。

---

## 关闭

```bash
export LANGGRAPH_STREAM_LOG=false
```

或取消该环境变量。
