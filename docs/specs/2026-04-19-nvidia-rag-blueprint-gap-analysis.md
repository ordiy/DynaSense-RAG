# MAP-RAG vs NVIDIA RAG Blueprint — 技术差距分析与演进规范

**版本**: 1.0 · **日期**: 2026-04-19  
**受众**: 技术负责人 / 架构师  
**定位**: 决策参考文档（非直接实施规范）  
**对标策略**: 模式借鉴 — 保留 PostgreSQL + Vertex AI 技术栈，采纳 NVIDIA Blueprint 的架构模式

---

## 1. 总体差距速览

| 维度 | MAP-RAG 现状 | NVIDIA Blueprint 标准 | 差距等级 |
|------|-------------|----------------------|---------|
| **文档解析** | pypdf / python-docx / openpyxl；无 OCR | Unstructured.io；OCR + 表格结构提取 | 🔴 较大 |
| **多模态索引** | 图片走视觉 API（会话级），不入向量库 | 图片 caption → 嵌入，ColPali 多模态检索 | 🔴 较大 |
| **混合检索** | pgvector + PG FTS + AGE 图 | Dense + Sparse (SPLADE) + Graph | 🟡 部分 |
| **重排序** | Jina v2 Cross-Encoder（单路径） | BGE-Reranker / ColBERT，Query 扩展 | 🟡 部分 |
| **推理接口抽象** | Vertex AI SDK 散落 rag_core.py | NIM 标准 REST/gRPC，模型无关 | 🔴 较大 |
| **评估框架** | 自定义 Faithfulness + Recall@K/NDCG@K | RAGAS 全套指标 + TruLens 持续反馈 | 🟡 部分 |
| **可观测性** | LangSmith 追踪（手动） | 自动化 pipeline 级延迟 / 吞吐监控 | 🟡 部分 |

> **🟢 已对齐**: Small-to-Big 检索、引用溯源、图谱三元组提取、What-If 确定性工具。这些无需改动。

---

## 2. 数据流 (Ingestion Pipeline)

### 2.1 现状诊断

```
当前链路:
文件上传 → pypdf/docx/xlsx 文本提取 → Jina Segment API → Vertex AI Embedding → pgvector + JSONB
```

**三处结构性缺陷**:

1. **扫描件盲区**: `pdf_extract.py` 用 `pypdf.PdfReader` 提取文字层。扫描版 PDF（无文字层）返回空文本，静默失败，上游无感知。
2. **表格降维**: DOCX / XLSX 中的表格被展平为纯文本，失去行列结构语义。LLM 在回答"第 3 行第 2 列是什么"类问题时准确率显著下降。
3. **图片不入索引**: PDF 内嵌图表（流程图、数据图）完全跳过，无法被检索命中。会话级图片理解（Gemini Vision）与知识库检索是两个独立路径，不互通。

### 2.2 NVIDIA Blueprint 模式

NVIDIA 的 [Multimodal RAG Pipeline](https://github.com/NVIDIA/GenerativeAIExamples) 核心逻辑：

```
文件 → Unstructured.io 解析器
         ├─ 文字块 → 分块 → 嵌入 → 向量库
         ├─ 表格 → Markdown/HTML 序列化 → 分块 → 嵌入
         └─ 图片 → VLM Caption（LLaVA/GPT-4V）→ 文字描述 → 嵌入
                → 可选: ColPali 多向量图片嵌入
```

关键设计决策：**表格和图片都被"翻译"成自然语言文本后，走统一的文本嵌入通道**。这避免了引入额外的多模态向量库。

### 2.3 对 MAP-RAG 的推荐模式

保持现有存储架构，仅改造解析层：

```
文件上传
  ↓
[IngestParser — 新增]
  ├─ 纯文本 PDF/DOCX → 现有路径不变
  ├─ 扫描 PDF（无文字层检测）→ Tesseract OCR → 文字层 → 现有路径
  ├─ PDF 内嵌表格 → pdfplumber 结构提取 → Markdown 序列化 → 现有路径
  └─ PDF 内嵌图片 → 调用 Gemini Vision caption → 文字描述 → 现有路径（带 `[图片描述]` 前缀）
  ↓
现有 Jina Segment → Vertex AI Embedding → pgvector + JSONB（不变）
```

**为什么这样做**: 解析层改造对下游零侵入；Gemini Vision 已在会话侧使用，复用现有 API 调用模式；不引入新的向量库类型。

**预期提升**:
- 扫描件查询命中率：0% → ~70%（取决于 OCR 质量）
- 表格相关问答准确率：+30~40%（LLM 能正确理解行列关系）
- 图表/流程图问答：从完全无法命中 → 有基础召回能力

---

## 3. 检索增强 (Retrieval Strategy)

### 3.1 现状诊断

MAP-RAG 的混合检索已是行业主流水准：

```
路由决策（LLM）→ VECTOR | GRAPH | GLOBAL | HYBRID
  VECTOR: pgvector 余弦相似度 top-k
  GRAPH:  AGE Cypher 图遍历
  GLOBAL: PG FTS tsvector GIN 全文检索
  HYBRID: VECTOR + GLOBAL 结果合并 → Jina Cross-Encoder 重排序
```

**两处可精进点**（非缺陷，是天花板）:

1. **Query 表达力**: 当用户问题与文档用词差异较大时（同义词、缩写、行业术语），Dense 向量和 Sparse BM25 都会漏召。NVIDIA Blueprint 用 **Query Expansion** 或 **HyDE** 来提升词汇覆盖。

2. **重排序多样性**: 当前 Jina 重排序按相关性降序，可能返回 k 个语义高度相似的文档片段。**MMR (Maximal Marginal Relevance)** 在相关性和多样性之间取平衡，对"综述类"问题效果更好。

### 3.2 NVIDIA Blueprint 模式

```
原始 Query
  ↓
Query Expansion: LLM 生成 3 个改写版本（同义词扩展、假设文档生成 HyDE）
  ↓
多路并行检索（原始 + 3 个扩展）
  ↓
结果去重合并（Union + deduplicate）
  ↓
Cross-Encoder 重排序（BGE-Reranker-v2 或 NIM 端点）
  ↓
MMR 多样性筛选（λ=0.5，top-k）
  ↓
LLM 生成
```

### 3.3 对 MAP-RAG 的推荐模式

**优先级一（高性价比）: Query Expansion**

```python
# 新增 query_expansion_node，插入 LangGraph retrieve 节点之前
# 调用现有 LLM（gemini-2.5-flash）生成 2 个改写版本
# 三路并行 pgvector 查询，结果 Union 后去重，其余下游不变
```

代价：+1 次 LLM 调用（~100ms）。适合知识库术语密度高的场景（金融、医疗、法律）。

**优先级二（低优先）: MMR**

```python
# 在 jina_rerank 之后、grade_documents_node 之前插入 mmr_filter
# 纯 Python 实现，无外部依赖
# λ 参数可配置，默认 0.7（偏相关性）
```

**为什么这样做**: Query Expansion 解决的是"用户说 A，文档写 B"的召回问题，这是知识库问答的核心痛点。MMR 优化体验，但在精准问答场景收益有限。

**预期提升**:
- Query Expansion: Recall@10 提升 +15~25%（跨术语/同义词场景）
- MMR: 综述类问题用户满意度 +10~15%，精准问题无显著影响

---

## 4. 微服务化 (Inference Abstraction)

### 4.1 现状诊断

```python
# src/rag_core.py — 现状（问题所在）
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI

embeddings = VertexAIEmbeddings(model_name="text-embedding-004", ...)
llm = ChatVertexAI(model_name="gemini-2.5-flash", ...)
```

**根本问题**: Vertex AI SDK 以模块级单例形式绑定在 `rag_core.py`。替换模型提供商（如切换到 Anthropic Claude、本地 Ollama、或 NVIDIA NIM 端点）需要修改核心代码，而非配置。

**影响范围**:
- 无法 A/B 测试不同 Embedding 模型的检索质量
- 无法在开发环境用本地模型替代 Vertex AI（节省 API 费用）
- 合规要求"数据不出云"时无法快速切换到私有化部署模型

### 4.2 NVIDIA NIM 的核心模式

NIM (NVIDIA Inference Microservice) 不只是模型服务，它是一个**接口规范**：

```
POST /v1/embeddings          ← OpenAI 兼容
POST /v1/chat/completions    ← OpenAI 兼容
GET  /v1/models              ← 健康检查 + 版本信息
GET  /metrics                ← Prometheus 指标
```

**关键洞察**: NIM 使用 OpenAI 兼容 API。这意味着任何支持 OpenAI 格式的客户端，都能 0 代码修改接入 NIM。MAP-RAG 当前用 LangChain，而 LangChain 支持 `ChatOpenAI(base_url=..., api_key=...)` 指向任意 OpenAI 兼容端点。

### 4.3 对 MAP-RAG 的推荐模式

**不需要微服务拆分，需要接口抽象层**：

```python
# src/core/inference.py — 新增（约 60 行）

from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings

class InferenceConfig:
    """从 settings 读取，决定用哪个后端"""
    provider: Literal["vertex", "openai_compat", "anthropic"]
    llm_model: str
    embedding_model: str
    base_url: Optional[str]  # NIM / Ollama / vLLM 端点

def get_llm(cfg: InferenceConfig) -> BaseChatModel:
    match cfg.provider:
        case "vertex":       return ChatVertexAI(model_name=cfg.llm_model, ...)
        case "openai_compat": return ChatOpenAI(base_url=cfg.base_url, model=cfg.llm_model, ...)
        case "anthropic":    return ChatAnthropic(model=cfg.llm_model, ...)

def get_embeddings(cfg: InferenceConfig) -> Embeddings:
    # 同上，按 provider 分发
    ...
```

`rag_core.py` 将模块级单例替换为函数参数注入，依赖注入由 `app.py` 启动时完成。

**环境变量驱动切换**:
```bash
# .env — 接入 NVIDIA NIM
INFERENCE_PROVIDER=openai_compat
INFERENCE_BASE_URL=https://integrate.api.nvidia.com/v1
INFERENCE_LLM_MODEL=meta/llama-3.1-70b-instruct
INFERENCE_EMBEDDING_MODEL=nvidia/nv-embedqa-e5-v5
```

**为什么这样做**: LangChain 的 `BaseChatModel` 接口已经是事实标准抽象层，无需重新发明。额外 60 行代码换来模型提供商可配置化，边际成本极低。

**预期提升**:
- 模型替换成本：从"修改核心代码 + 测试" → "改 .env + 重启"（节省 2~4 小时/次）
- 本地开发成本：可切换 Ollama 本地模型，月均 Vertex AI API 费用降低 ~60%
- A/B 测试可行性：可同时运行两套 embedding 模型，对比 Recall@K

---

## 5. 评估与监控 (Evaluation & Monitoring)

### 5.1 现状诊断

MAP-RAG 的评估体系已有良好基础：

| 指标 | 实现位置 | 方式 |
|------|---------|------|
| Recall@K / NDCG@K | `src/recall_metrics.py` | 离线，有标注集 |
| Faithfulness | `src/core/faithfulness.py` | LLM-as-a-Judge（Gemini） |
| LangSmith 追踪 | `src/core/langsmith_tracing.py` | 手动触发 |

**三处结构性缺口**:

1. **无端到端自动化评估**: 现有指标需手动运行，未集成到 CI/CD。每次 retrieval 策略变更后，无法自动验证质量未回退。
2. **缺 Answer Relevancy 指标**: Faithfulness 衡量"回答是否忠于上下文"，但不衡量"回答是否真正回答了问题"。RAGAS 的 `answer_relevancy` 填补这个盲区。
3. **无生产追踪闭环**: LangSmith 追踪需手动触发，且用户反馈（👍/👎）未与具体 trace 关联，无法形成数据飞轮。

### 5.2 NVIDIA Blueprint 模式

NVIDIA 推荐的评估栈：

```
离线评估（CI）: RAGAS — faithfulness, answer_relevancy, context_precision, context_recall
在线监控（生产）: TruLens — 实时记录每次 RAG 调用的三项核心指标，可视化 dashboard
反馈闭环: 用户 👍/👎 → 写回 TruLens → 标注数据集扩充 → 下次 RAGAS eval 使用
```

### 5.3 对 MAP-RAG 的推荐模式

**第一步（低成本高收益）: RAGAS 集成到 CI**

```python
# tests/eval/test_ragas_regression.py
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

# 使用现有 tests/fixtures/ 中的标注问答对
result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])

assert result["faithfulness"] >= 0.80      # 回归防护线
assert result["answer_relevancy"] >= 0.75
```

`make test` 中加入 `pytest tests/eval/` 步骤。每次 PR 自动运行，结果写入 CI artifact。

**第二步（中等成本）: 生产追踪闭环**

```
用户反馈（src/api/routers/feedback.py 已存在）
  ↓
写入 PostgreSQL feedback 表（已存在）+ LangSmith trace_id 关联（新增字段）
  ↓
每周 cron: 提取负面反馈对应的 trace → 扩充标注集 → 下次 RAGAS eval
```

TruLens 可选引入，但鉴于 MAP-RAG 已有 LangSmith，**优先复用现有追踪基础设施**，避免双维护负担。

**为什么这样做**: RAGAS CI 集成是最小可行的质量安全网；反馈闭环是"让数据驱动改进"的前提，也是 `docs/bitter_lesson_roadmap.md` Phase B 的核心依赖。

**预期提升**:
- 检索质量回退发现时间：从"发布后用户投诉" → "PR 阶段自动拦截"
- 标注数据集增长：生产每月可积累 ~50~200 条高质量负面样本（依流量）
- 可量化的 Answer Relevancy 基线：当前该指标未测量，预计首次测量在 0.65~0.75，优化目标 0.85

---

## 6. 优先级矩阵

| 项目 | 实现难度 | 预期收益 | 推荐优先级 |
|------|---------|---------|-----------|
| OCR 支持（扫描 PDF） | 低（+Tesseract，解析层） | 高（覆盖盲区） | **P0** |
| RAGAS CI 集成 | 低（+ragas 包，1 个测试文件） | 高（质量安全网） | **P0** |
| 推理接口抽象层 | 中（重构 rag_core.py 单例） | 高（A/B 测试、成本优化） | **P1** |
| Query Expansion | 中（新增 LangGraph 节点） | 中（术语密集场景） | **P1** |
| 表格结构保留 | 中（+pdfplumber） | 中（金融文档场景） | **P1** |
| 图片 Caption 索引 | 中（复用 Gemini Vision） | 中（图表密集文档） | **P2** |
| 生产反馈闭环 | 高（跨系统数据流） | 高（长期数据飞轮） | **P2** |
| MMR 多样性筛选 | 低（纯 Python） | 低（体验优化） | **P3** |

---

## 7. 不推荐的方向

以下 NVIDIA Blueprint 特性在当前阶段**不建议引入**，原因如下：

| 特性 | 不引入原因 |
|------|-----------|
| 替换 pgvector → Milvus/Weaviate | MAP-RAG 的 PostgreSQL 一体化存储是合规优势（单数据库审计），Milvus 引入运维复杂度且无明显检索质量提升 |
| NIM 端点替换 Vertex AI | 成本更高，且推理接口抽象层（§4.3）已解决提供商锁定问题，无需迁移 |
| TruLens 全量引入 | LangSmith 已覆盖追踪需求；TruLens 和 LangSmith 双维护是负担，优先复用 |
| ColPali 多模态向量检索 | 需要独立向量存储或 pgvector 扩展，当前图片量不足以支撑额外基础设施成本 |

---

## 附录：关键参考

- [NVIDIA Generative AI Examples (GitHub)](https://github.com/NVIDIA/GenerativeAIExamples)
- [RAGAS 官方文档](https://docs.ragas.io)
- [LangChain OpenAI-compatible endpoints](https://python.langchain.com/docs/integrations/chat/openai)
- [`docs/bitter_lesson_roadmap.md`](../bitter_lesson_roadmap.md) — Phase B 数据驱动改进前提条件
- [`docs/recall_evaluation.md`](../recall_evaluation.md) — 现有 Recall@K 方法论
