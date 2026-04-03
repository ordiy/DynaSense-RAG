# What-If 工具案例：贷款利率敏感性（无 ReAct）

## 业务场景（案例）

**角色**：企业财务 / FP&A 分析员。  
**问题**：在 **本金与期限不变** 的前提下，若 **名义年利率** 从 **3.5%** 上调到 **4.2%**，**月供** 与 **总利息** 如何变化？  

该问题 **不依赖知识库检索**，而是通过 **确定性金融公式**（等额本息）完成，适合作为 **「调用 tools、但不必 ReAct」** 的示范：固定 **DAG** —— 计算情景 A → 计算情景 B → 求差。

## 架构位置

| 组件 | 路径 | 说明 |
|------|------|------|
| 工具（纯函数） | `src/tools/loan_whatif.py` | `monthly_payment`、`loan_snapshot`、`compare_rate_scenarios` |
| DAG 编排 | `src/api/whatif_pipeline.py` | `run_loan_rate_compare_pipeline` |
| HTTP | `POST /api/whatif/loan/compare` | `src/api/routers/whatif.py` |

与 **RAG 管线解耦**：不导入 `rag_core`；CI 可用轻量依赖单独测路由。

## API

`POST /api/whatif/loan/compare`

请求体（JSON）：

```json
{
  "principal": 1000000,
  "annual_rate_percent_before": 3.5,
  "annual_rate_percent_after": 4.2,
  "loan_months": 360
}
```

响应：`before` / `after` 各含 `monthly_payment`、`total_paid`、`total_interest`；`deltas` 含 `delta_monthly_payment`、`delta_total_interest`、`delta_total_paid`。

## 测试用例矩阵（摘要）

| 用例 | 断言 |
|------|------|
| 零利率 12 期 | 月供 = 本金/12 |
| 6%×30 年×10 万本金 | 月供 ≈ 599.55（公式自检） |
| 利率上升 | `delta_monthly_payment > 0` 且总利息增加 |
| 非法本金 | HTTP 422 |
| API 层 | `deltas` 与 `after - before` 一致 |

实现见：`tests/test_loan_whatif_math.py`、`tests/test_whatif_api.py`。

## 与 RAG / Agent 的关系

- **本案例**：**工具 + 固定 DAG**，无 LLM、无多轮工具循环。  
- **未来**：若要在自然语言里触发本逻辑，可增加 **意图路由** 或 **一次结构化抽取**，再调用同一 `run_loan_rate_compare_pipeline`，仍不必引入 ReAct。
