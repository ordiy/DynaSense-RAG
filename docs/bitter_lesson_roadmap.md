# 架构演进：The Bitter Lesson 与 MAP‑RAG

原文：[The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)（Rich Sutton, 2019）

---

## 文章在说什么（一句话）

长期来看，**能随算力与数据规模扩展的通用方法**（搜索、学习）往往胜过**人工堆砌的领域技巧**；历史案例包括博弈、视觉、语音等——**人类直觉的边际收益会递减**，而 **计算与学习** 的边际收益持续上升。

---

## 对企业 RAG 的适用边界（必须先承认张力）

| Bitter Lesson 倾向 | 本仓库（金融 / 合规）必须保留的 |
|--------------------|----------------------------------|
| 少写规则、多让模型与数据说话 | **可审计**：`citations`、受控 Cypher、确定性 What‑If |
| 端到端学习若更强则替代手工特征 | **可解释路由**：`RouteDecision`、阻断原因需可追溯 |
| 通用检索 + 大算力 | **数据主权**：内网语料、Neo4j 白名单、出站最小化 |

结论：**不是「删掉所有手工设计」**，而是——**把「可随数据与算力扩展」的部分做成主线**；把**监管与审计**显式化成**接口与契约**，而不是用更多 if‑else 冒充「智能」。

---

## 与当前模块的对照（何处偏「Lesson」，何处偏「治理」）

| 组件 | 偏 Lesson（宜长期加注算力与数据） | 偏治理（不宜轻易「端到端黑箱化」） |
|------|-----------------------------------|-------------------------------------|
| 向量召回 + **Jina 重排** | 升级 embedding / reranker、扩大语料、调 `HYBRID_FUSION_TOP_N` | — |
| **LLM 路由** `route_query` | 用离线评测集做 A/B、少改 prompt 大改结构 | 路由理由需可记录（已有结构化输出） |
| **Grader / 双轨生成** | 用评测集与反馈迭代 prompt，避免无指标微调 | 阻断策略变更需可回归 |
| **`query_anchors` 锚点过滤** | 长期：用 **点击/反馈** 学习「何时过滤」而非加规则 | 短期：规则可解释，适合合规解释 |
| **`guardrails` PII** | 可引入小分类器 + 数据，减少纯正则 | 必须可配置、可审计 |
| **Neo4j 受控查询** | — | **保持白名单模板**；不改为任意 Cypher |
| **`analytics_profile`** | 保持「固定统计」；若上 NL→SQL 需白名单或 BI | 禁止开放任意 `pandas` 执行 |

---

## 有计划的调整（分阶段）

### 阶段 A — 度量与算力基线（优先，低成本）

**目标**：让「改进」由 **指标** 驱动，而不是由会议驱动。

#### 已落地（检索 → 生成）：多段落参与决策

与 *Bitter Lesson* 一致，将 **算力** 用在更强的 **重排与更长上下文融合**，而不是过早压成单段摘要：

| 机制 | 说明 |
|------|------|
| **向量 LangGraph 路径** | `rag_vector_rerank_top_n`（默认 **5**，环境变量 `RAG_VECTOR_RERANK_TOP_N`，范围 2–20）：Jina 重排后保留更多父文档块进入 grader / generator。 |
| **编号上下文** | `src/core/rag_context_format.py` 的 `format_numbered_passages`：将重排结果格式化为 `[Passage 1]`…，grader / 双轨生成 prompt 明确要求 **通读各段、综合多段证据**，避免只盯第一段。 |
| **Hybrid 路径** | 仍由 `HYBRID_FUSION_TOP_N`（默认 5）控制融合重排宽度；与向量路径默认对齐，便于 A/B。 |

验证：`tests/test_rag_context_format.py`；回归仍跑 `pytest tests/`。

#### 已落地（生成 → 忠实度度量）：Faithfulness LLM-as-a-Judge

受 DoTA-RAG（arXiv 2506.12571）启发：该论文发现其系统 Faithfulness 因 300 词截断从 0.702 降至 0.336，但 Recall@K 完全看不出来。本项目同样存在此盲区。

| 机制 | 说明 |
|------|------|
| **`src/core/faithfulness.py`** | `judge_faithfulness(answer, passages, llm)`：3 级裁定（full=1.0 / partial=0.5 / none=0.0），结构化输出，异常安全回退。|
| **`/api/evaluate` 新字段** | `compute_faithfulness: bool = False`（opt-in，避免加默认延迟）；返回 `faithfulness_score`、`faithfulness_verdict`、`faithfulness_reasoning`。|
| **`/api/evaluate/batch`** | 批量评测时，`mean_metrics` 包含 `faithfulness_score` 均值。|
| **Generator 结论前置** | `GEN_PROMPT` / `GEN_ANALYSIS_PROMPT` 增加"先给结论再展开"指令，防止截断导致关键信息丢失（DoTA-RAG 核心教训）。|

验证：`tests/test_faithfulness.py`（18 个纯 mock 测试，无 LLM/DB 依赖），`make test` 161 passed。

- [ ] **固定评测资产**：维护 `scripts/` 与 `docs/recall_evaluation.md` 中提到的回归集；关键变更必须跑 **Recall@K / nDCG** 或等价批测。
- [ ] **显式模型版本**：embedding / reranker / 路由 LLM 在配置或部署中 **可版本化**，便于对比「换模型 vs 改规则」。
- [ ] **反馈闭环**：`POST /api/feedback` 数据若仅内存，则文档写明 **上限**；规划 **导出或落库** 以便后续学习（不要求本阶段实现训练）。
- [ ] **缓存与批处理**：对重复查询、热门检索路径评估 **缓存**（算力换延迟），与 Lesson 一致。

**交付物**：评测跑通记录在 CI 或发布 checklist；架构文档中「模型版本」一节。

---

### 阶段 B — 用数据替代「可替代的手工层」（需数据就绪后）

**目标**：在 **离线评测** 证明收益后，把部分 **启发式** 换成 **可学习或可校准的组件**。

- [ ] **锚点过滤**：在有关键词标注或用户反馈时，评估 **学习式门控**（例如：是否启用过滤、权重）是否优于纯规则；**规则** 作为安全兜底保留。
- [ ] **路由**：用离线集比较 **少分支 + 更强检索** vs **多分支 + 弱检索**；避免无评测增加 `GRAPH`/`HYBRID` 特殊分支。
- [ ] **护栏**：在误报/漏报统计后，用 **小型分类器** 辅助或替代部分正则（仍保留审计开关）。

**交付物**：每项替换前有 **AB 报告**（旧 vs 新，同一评测集）。

---

### 阶段 C — 刻意不做的（避免伪 Lesson）

- [ ] **不**为「显得智能」增加 **无评测覆盖** 的 prompt 分支或路由。
- [ ] **不**把 **合规必需的** 白名单 Cypher、引用结构、What‑If 确定性改为不可审计的黑盒。
- [ ] **不**在缺少沙箱与审计时引入 **任意代码** 数据分析 Agent。

---

## 与 OpenClaw / 编排层的分工（Reminder）

业务流程、通知、出站集成仍按 [TODO.md](./TODO.md)：本仓库专注 **可测检索 + 可引用生成 + 确定性工具**；**动作** 在外层。这与 Bitter Lesson 不矛盾：**Lesson 管的是「知识与检索」如何扩展**，**编排与合规动作** 仍是显式系统。

---

## 相关文档

- [架构说明](./architecture.md)
- [后续迭代 TODO](./TODO.md)
- [Hybrid RAG MVP](./mvp_hybrid_rag.md)
