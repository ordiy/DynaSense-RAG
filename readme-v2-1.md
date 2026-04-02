# 需求升级架构-Routing + Hybrid RAG (双路召回)
提升结构化数据的数据源是高度结构化的 Recall能力

你的直觉非常敏锐，且完全契合当前业界最前沿的 **Adaptive RAG（自适应RAG）** 或 **Agentic RAG（智能体化RAG）** 的设计哲学。

单纯依赖单一检索方式已经无法满足复杂的业务需求。一个合理的、达到企业级生产标准的混合架构，核心就在于你提到的 **“问题分类器（Query Router / Classifier）”**，辅以 **“双引擎索引”** 和 **“统一重排（Reranker）”**。

作为资深架构师，我为你设计一套标准的 **“动态意图路由双轨RAG架构”**。

---

### 🏛️ 企业级双轨 RAG 架构设计图 (逻辑拓扑)

```text
[ 阶段一：离线知识构建 (Dual-Engine Indexing) ]
原始文档 ──┬──> 分块整理 ──> Embedding模型 ──> [ 向量数据库 (DynaSense路线) ]
           └──> LLM信息抽取 ──> 实体/关系/属性构建 ──> [ 图数据库 Neo4j (GraphRAG路线) ]
                    (注: 实体节点需保留指向原文档Chunk的ID，实现图文互指)

================================================================================

[ 阶段二：在线检索与生成 (Online Retrieval & Generation) ]

用户提问 (Query)
   │
   ▼
[ 智能问题分类器 (Query Router) ] ──(意图识别与路由)
   │
   ├─> 意图 A: 实体事实、模糊语义、长尾细节 
   │     └─> 触发 [ DynaSense RAG 引擎 ] ──> 混合检索 (Dense+BM25) ──> Top-K 文本块
   │
   ├─> 意图 B: 多跳关系、复杂拓扑、图谱推理
   │     └─> 触发 [ Neo4j RAG 引擎 ] ──> Text2Cypher/实体子图检索 ──> Top-K 图拓扑文本/节点集合
   │
   ├─> 意图 C: 宏观总结、全局特征 (全局摘要)
   │     └─> 触发 [ 图谱社区检索 (Graph Community) ] ──> 提取图谱高层级聚合信息
   │
   └─> 意图 D: 复杂复合问题 (无法单一界定)
         └─> [ 双路并发召回 (Concurrent Retrieval) ]

   ▼ (收集所有召回的上下文)
[ 统一融合与重排模块 (Context Fusion & Reranking) ]
   │  (利用 Cross-Encoder 将图谱转化来的文本与Chunk结合，重新打分，严格截取 Top-5)
   ▼
[ 大语言模型 (LLM) ] ──(附带Top5高优Context)
   │
   ▼
最终精准回答 (附带来源溯源 Citation)
```

---

### 核心模块技术落地方案

为了让这个架构不仅停留在纸面，以下是各个关键节点的工程实现细节：

#### 1. 智能问题分类器 (Query Router) —— 架构的大脑
这是你提到的核心。在工程上，分类器通常有三种实现方式，按需取舍：
*   **方案一：基于轻量级 LLM (如 GPT-4o-mini / Qwen-Max-Fast)** *【推荐：准确率高，实现快】*
    *   **实现：** 写一个 System Prompt，给 LLM 几个 Few-shot examples，让它输出分类枚举值。
    *   *Prompt 示例：“你是一个意图识别引擎。如果用户询问具体的段落细节、长尾语义（如‘屏幕闪烁’），请输出 `VECTOR`。如果询问组织关系、持股结构、多级关联（如‘谁投资了A的子公司’），请输出 `GRAPH`。如果是全局总结，输出 `GLOBAL`。”*
*   **方案二：基于轻量级 NLP 模型 (如 BERT / FastText分类器)** *【推荐：极低延迟】*
    *   **实现：** 标注几千条历史 Query，训练一个文本分类小模型。响应时间在 10ms 级别，不消耗大模型 Token。
*   **方案三：启发式规则 (Rule-based) 兜底**
    *   **实现：** 检测问题中是否包含图谱中存在的高频实体名。如果包含多个图谱实体名（如同时出现“马斯克”和“OpenAI”），直接触发图谱并发检索。

#### 2. 双轨检索层 (Dual-Engine Retrieval) 的协同
*   **当路由到 DynaSense (向量层) 时**：执行标准的 Query 重写 -> 向量检索 -> BM25 检索 -> 召回 Chunk。
*   **当路由到 Neo4j (图谱层) 时**：这是难点。图谱查出来的是 JSON 格式的节点和边，LLM 不太好理解。需要将其 **线性化 (Linearization)**。
    *   *例如，Neo4j 查出的路径转化为文本段落：“张三 (CEO) 管理 公司A；公司A 投资了 公司B。” 将这些转化后的关系文本作为 Context 参与后续步骤。*

#### 3. 统一融合与重排 (Fusion & Reranking) —— 保证 Top-5 质量的关卡
如果分类器选择了“双路并发（意图D）”，两边可能会召回 10 个 Chunk 和 5 个关系段落。你怎么选出最终的 Top-5？
*   **操作方法**：引入一个专门的 Reranker 模型（如 BAAI/bge-reranker-v2-m3 或 Cohere Rerank）。
*   将用户 Query 与这 15 个候选内容（无论它原本是文档块还是图谱文本）一一配对，让 Reranker 给它们打分。
*   根据分数绝对排序，**硬截断取 Top-5**。这能有效过滤掉图谱召回的低相关度噪音，或者向量召回的字面相似但语义不符的废料。

---

### 💡 这种架构的压倒性优势

1.  **极高的健壮性 (Robustness)**：无论用户是像查字典一样查事实（走向量），还是像做侦探一样查关系（走图谱），系统都不会“抓瞎”。
2.  **成本与延迟可控 (Cost & Latency)**：如果不用分类器，每个问题都跑一次 GraphRAG 的 Text2Cypher 和多路向量召回，系统延迟极高且费钱。有了路由机制，80% 的简单问题可以直接走低成本的 DynaSense 向量通道，只有 20% 的复杂关系推理才去调用厚重的 Neo4j 引擎。
3.  **突破“大满贯”瓶颈**：在我的第一份评估中，我们受限于 Top-5 窗口。这种架构通过“重排（Rerank）”做最后把关，确保喂给大模型的 Top-5 永远是“信息密度最高、对回答最有帮助”的上下文组合。

**技术负责人视角总结：**
你提出的这个思路就是目前大型企业构建知识库问答的“终极形态”。不需要去纠结 GraphRAG 和 纯向量 RAG 谁好谁坏，**小孩子才做选择，成熟的工程架构是通过路由（Router）将它们变成底层的基础算子，按需调度。**

# 测试数据-demo 

https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?stockid=601998&id=12008291
这是一个关联交易报告：
```
TODO:请自行从这个文档中提取更多Q&A测试Recall

Q：中国中信银行的关联方有哪些
A: 在中信银行：2025年度关联交易专项报告中的关联有中信集团（占本行资本净额比例：10.76%），xxx
```


# reference doc 
https://huggingface.co/learn/cookbook/zh-CN/rag_with_knowledge_graphs_neo4j
https://neo4j.com/docs/graph-data-science/current/machine-learning/node-embeddings/
