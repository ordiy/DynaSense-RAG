# Dual-Track Query Routing: Analysis vs. Factual

## Overview

MAP-RAG uses a **dual-track routing strategy** to handle two fundamentally different query types with appropriate grading and generation policies. This design allows the system to be strictly anti-hallucination for factual lookups, while unlocking professional analytical reasoning for analysis-type questions—without sacrificing safety.

---

## The Core Problem

A single-policy grader-generator pair cannot serve both query types well:

| Query Type | User Expectation | Single-Policy Failure |
|---|---|---|
| **Factual** ("What is the yield of X?") | Exact answer from document | A too-lenient policy risks hallucination |
| **Analysis** ("How does weather affect X futures trading?") | Reasoned analysis grounded in available facts | A too-strict policy says "no answer found" even when relevant facts exist |

The key insight: **analysis questions do not require the retrieved context to contain a pre-formed answer**—they only require relevant background facts that serve as a reasoning foundation.

---

## Architecture: Dual-Track Routing

```text
                        User Query
                             │
                             ▼
                   [ Query Type Detector ]
                   _is_analysis_query()
                   (keyword matching on ~25
                    intent signals, e.g.
                    分析/影响/如何/为什么/
                    analyze/impact/why/how…)
                             │
             ┌───────────────┴───────────────┐
             │                               │
     [Factual Track]               [Analysis Track]
             │                               │
             ▼                               ▼
    GRADE_PROMPT                  GRADE_ANALYSIS_PROMPT
    "Does context contain         "Does context contain ANY
     a direct answer?"             topic-related background fact?"
    (strict)                      (relaxed)
             │                               │
             ▼                               ▼
    GEN_PROMPT                    GEN_ANALYSIS_PROMPT
    "Strictly use context.        "Use context as factual basis.
     No fabrication."              Apply domain knowledge for
                                   reasoning. Label facts vs.
                                   inference clearly."
             │                               │
             └───────────────┬───────────────┘
                             ▼
                    Final Answer to User
```

---

## Implementation Details

### 1. Query Type Detection (`rag_core.py`)

```python
_ANALYSIS_INTENT_KEYWORDS = frozenset([
    # Chinese
    "分析", "影响", "推断", "预测", "建议", "方案", "规划", "策略",
    "如何", "为什么", "原因", "趋势", "对比", "评估", "风险", "机会",
    "挑战", "可行性", "展望", "下一步", "优化", "改进", "判断",
    # English
    "analyze", "analysis", "predict", "suggest", "plan", "why", "how",
    "impact", "feasibility", "risk", "opportunity", "challenge",
])

def _is_analysis_query(question: str) -> bool:
    return any(k in question.lower() for k in _ANALYSIS_INTENT_KEYWORDS)
```

### 2. Grader Routing (`grade_documents_node`)

```python
grade_prompt = GRADE_ANALYSIS_PROMPT if is_analysis else GRADE_PROMPT
```

- **`GRADE_PROMPT`** (strict): "Do these paragraphs contain the key factual basis needed to answer the question?"
- **`GRADE_ANALYSIS_PROMPT`** (relaxed): "Do these paragraphs contain any background facts related to the topic that could serve as a foundation for analysis?"

### 3. Generator Routing (`generate_node`)

```python
prompt = GEN_ANALYSIS_PROMPT if _is_analysis_query(question) else GEN_PROMPT
```

- **`GEN_PROMPT`** (strict): "Strictly use the retrieved context. No fabrication."
- **`GEN_ANALYSIS_PROMPT`** (structured analysis): Instructs the LLM to produce two clearly labelled sections:
  - **【文档事实】**: Facts directly from the retrieved context
  - **【分析推理】**: Reasoned analysis applying domain expertise

---

## Pipeline Execution Logs

For an analysis query, the pipeline logs show:
```
ℹ️ Analysis query detected: using relaxed grader.
✅ Contexts approved for generation.
ℹ️ Analysis query: using GEN_ANALYSIS_PROMPT.
Generating answer based on verified context...
```

For a factual query, the pipeline logs show:
```
Grading context to prevent hallucination...
✅ Contexts approved for generation.
Generating answer based on verified context...
```

---

## Demo Q&A

### Case 1 — Analysis Query (Weather Impact on Futures)

**User Input:**
> 介绍一下"豌豆苗期货"，分析天气对该期货交易的影响

**Retrieved Context (knowledge base facts):**
```
豌豆苗期货：
作物生长周期：3个月
种植地区：东海岸城市周边农场
产量：10吨/日
```

**System Response (abridged):**

> **【文档事实】**
> 关于豌豆苗期货，知识库中记载：作物生长周期为3个月，种植于东海岸城市周边农场，日产量10吨。
>
> **【分析推理】**
> 基于行业经验分析，天气因素对豌豆苗期货交易的影响主要体现在三个维度：
>
> 1. **对产量的影响**：豌豆苗生长周期仅3个月，温湿度波动（持续高温、霜冻、干旱）或极端天气事件（冰雹、洪涝）可直接导致减产，现货供应收紧推动期货价格上涨。
> 2. **对品质的影响**：持续阴雨导致光照不足，影响豌豆苗色泽与营养；高温高湿利于病虫害滋生，降低可交割品质，进而影响期货合约价格预期。
> 3. **对运输与仓储的影响**：恶劣天气阻断运输，增加物流成本，加速生鲜损耗，推高现货价格并反映至期货端。
>
> **总结**：鉴于豌豆苗3个月的短生长周期，市场对天气预期反应极为迅速，交易者需密切关注种植区域的天气变化以做出合理交易决策。

**Pipeline logs:**
```
ℹ️ Analysis query detected: using relaxed grader.
✅ Contexts approved for generation.
ℹ️ Analysis query: using GEN_ANALYSIS_PROMPT.
```

---

### Case 2 — Factual Query (Strict Mode, No Change)

**User Input:**
> 豌豆苗期货的日产量是多少？

**System Response:**
> 根据知识库，豌豆苗期货的日产量为 **10吨/日**。

**Pipeline logs:**
```
Grading context to prevent hallucination...
✅ Contexts approved for generation.
```

---

### Case 3 — Analysis Query with Insufficient Context (Graceful Decline)

**User Input:**
> 分析"火星矿石期货"的价格走势

**Retrieved Context:** *(no relevant documents found)*

**System Response:**
> 抱歉，知识库中未能找到与您问题高度相关的信息，为避免产生误导（幻觉），系统已阻断回答。

**Pipeline logs:**
```
ℹ️ Analysis query detected: using relaxed grader.
⚠️ All contexts rejected. Hallucination blocked.
```

> This case shows the system is not unconditionally permissive: even with the relaxed analysis grader, if the context contains **zero** topic-related facts, the pipeline still blocks the answer.

---

### Case 4 — Multi-Turn Analysis Follow-up

**Turn 1 (User):** 介绍胖冬瓜项目的基本信息  
**Turn 1 (Assistant):** *(returns project facts from knowledge base)*

**Turn 2 (User):** 下一步项目重点工作规划（草拟几个可行的方案）

The multi-turn query sent to the RAG pipeline includes the session history (topic anchor + prior messages). The `_is_analysis_query` function detects "规划" and "方案", routing to the relaxed track so the retrieved project facts can serve as the basis for planning suggestions.

---

## Safety Boundaries

The dual-track design preserves the system's anti-hallucination guarantees through three mechanisms:

| Mechanism | Applies To | Behaviour |
|---|---|---|
| Strict grader | Factual queries | Blocks if context has no direct answer |
| Relaxed grader | Analysis queries | Blocks if context has **zero** topic-related facts |
| Structured output label | Analysis queries | LLM must explicitly label `【分析推理】` sections, making AI-generated inference transparent to users |

The system **never fabricates specific numbers, dates, or named entities** that are not present in the retrieved context—this constraint is retained in `GEN_ANALYSIS_PROMPT` regardless of query type.

---

## Relation to Other Design Docs

| Document | Topic |
|---|---|
| [chat_test_memory_design.md](./chat_test_memory_design.md) | Server-side multi-turn memory, `conversation_id` session management |
| [doc-small-to-big-retrieval.md](./doc-small-to-big-retrieval.md) | Parent-child chunk expansion strategy |
| [doc-feauture-v1.md](./doc-feauture-v1.md) | Initial architecture RFC |
