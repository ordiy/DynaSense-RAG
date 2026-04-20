# 技术对话笔记 — 2026-04-20

> 涵盖主题：法律合同知识图谱设计 · Palantir AIP Ontology · 数据架构演进 · AI 时代的技术护城河

---

## 一、法律合同知识图谱设计

### 核心实体（Node / Vertex）

| 实体类型 | 关键属性 | 示例 |
|---|---|---|
| `Company` | name, type, jurisdiction | X公司, 甲方 |
| `Contract` | id, title, type, effective_date, expiry_date | 合同A (IP专利协议) |
| `Person` | name, role, employee_id | 张三, 发明人/员工 |
| `Patent` | patent_no, title, filing_date, status | CN202310001X |
| `IPRight` | right_type, scope, territory | 商标权/著作权/专利权 |
| `Product` | name, sku, category | 产品P1 |
| `Clause` | clause_id, title, type | 第7条, IP归属条款 |

### 核心关系（Edge / Predicate）

```
PARTY_TO        Company/Person → Contract      主体关联
EMPLOYS         Company → Person               雇佣关系
INVENTED_BY     Patent → Person                发明人
OWNED_BY        Patent/IPRight → Company       归属权
ASSIGNED_TO     IPRight → Company              让渡（员工→公司）
INCLUDES_CLAUSE Contract → Clause             合同包含条款
GOVERNS         Clause → Person/IPRight        条款约束对象
PROTECTED_BY    Product → Patent               专利保护产品
SOLD_UNDER      Product → Contract             产品在合同下出售
REFERENCES      Contract → Contract            跨合同引用
```

### 三份合同的关键跨合同关系

```
合同B（劳动合同）--REFERENCES--> 合同A（IP协议）
  └─ 员工IP让渡条款 指向 专利归属协议

张三（员工）--INVENTED_BY--> 专利P1 --OWNED_BY--> X公司 --PARTY_TO--> 合同C（销售）
  └─ 完整追溯链：员工发明 → 公司所有 → 产品销售

合同C --REFERENCES--> 合同A
  └─ 销售合同"IP不侵权声明"锚定到专利协议
```

**关键洞察**：`Company` 和 `Person` 是跨合同的枢纽实体。图谱的价值在于回答"这个员工的发明最终通过哪条合同链卖给了哪个客户"这类跨文档多跳问题。

### 知识图谱数据的两种填充方式

| 方式 | 适用场景 | 质量 |
|---|---|---|
| 上传文档，LLM 自动提取三元组 | 大量文档，容忍噪声 | 中（依赖 LLM 能力） |
| 手动 SQL 插入 `kg_triple` 表 | 少量核心合同，精确要求高 | 高 |

**最佳实践**：自动提取后人工审核，用 SQL 补充缺失的关键关系边。

### 重要认知：图数据库是"先写后查"

> Cypher 查询只能找到已经存在于图中的节点和边，不会动态推断不存在的关系。
> "动态"只体现在查询模式由 LLM 在运行时生成，数据本身必须预先入库。

---

## 二、Palantir AIP 的 Ontology 思想

### 核心架构对比

```
传统 ETL 思路：
  原始数据 → 清洗转换 → 写入目标表 → 查询
  （数据关系在物理层预先固化）

Palantir Ontology 思路：
  原始数据 ──映射──► Ontology 语义层 ──► 查询/AI Action
  （数据关系在语义层动态定义，源数据不动）
```

### Ontology 层定义的四个要素

| 要素 | 说明 | 类比 |
|---|---|---|
| **Object Type** | 业务实体类型 | 数据库表/类 |
| **Property** | 属性映射到源表列 | `Employee.name → hr_table.full_name` |
| **Link Type** | 跨数据源的虚拟关联 | 外键，但跨系统 |
| **Action** | 可作用于对象的业务操作 | `approve_contract(Contract)` |

### 关键洞察

> Palantir 的核心贡献是：**把"定义数据关系"这件事从 ETL 工程师的代码里，搬到了业务人员可操作的语义层**。
>
> 代价没有消除——仍然需要做语义映射——但这个工作比传统 ETL 低一个数量级，且可以随业务变化随时调整。

**没有消除的前提**：必须有人定义 Ontology 映射（哪个数据源的哪列对应哪个 Object Property）。

---

## 三、Palantir vs Databricks 技术差距分析

### 定位根本不同

```
Databricks：面向数据工程师/数据科学家
  "帮你更好地处理数据、训练模型"
  工具链：Delta Lake → Spark → MLflow → Unity Catalog

Palantir：面向业务决策者/运营人员
  "帮你用数据做出决策并执行动作"
  工具链：Ontology → AIP → AI Action → 业务系统
```

### 能力对比评分

| 维度 | Palantir | Databricks |
|---|---|---|
| 语义层 / Ontology | ★★★★★ | ★★☆ |
| 数据工程 / ETL | ★★★ | ★★★★★ |
| AI Action（操作型AI） | ★★★★★ | ★★★ |
| ML / 模型训练 | ★★☆ | ★★★★★ |
| 开放生态 | ★★☆ | ★★★★★ |
| 政府 / 国防部署 | ★★★★★ | ★★☆ |

### 业务价值差异

- Databricks：降低数据处理成本，加速模型上线（可量化：ETL 效率提升 3-5x）
- Palantir：让业务人员直接操作数据驱动的决策（案例：洛克希德马丁 F-35 维修决策从数天→分钟）

**结论**：Palantir 卖的不是数据处理能力，而是"数据→决策→行动"的完整闭环。在高价值、高风险决策场景中价值远超 Databricks；通用分析场景性价比极低。

---

## 四、AI 时代 Palantir 的发展逻辑

### 有利因素

1. **Ontology 是 LLM 的天然护栏**：LLM 幻觉在 Ontology 约束下可控，每个 AI Action 可追溯到具体数据源，满足审计要求
2. **政府/国防护城河在 AI 时代加深**：分类环境不能用商业云 API，Palantir 的气隙部署能力不可替代
3. **Bootcamp 模式创造强客户粘性**：驻场 2 周解决真实业务问题，Ontology 定义越积越多，迁移成本越来越高

### 风险因素

1. **开源平替压力**：LangGraph + Neo4j + dbt Semantic Layer 可以用 1/10 成本拼出类似能力
2. **超大云厂商挤压**：微软/Google/AWS 有天然的企业客户分发渠道
3. **高价格限制渗透率**：中型企业几乎无法承担 AIP 合同

### 发展阶段判断

| 阶段 | 判断 |
|---|---|
| 短期（1-3年） | 强势增长：AI 热潮推动预算扩张，政府合同持续增长 |
| 中期（3-7年） | 分化加剧：政府护城河加深；商业市场面临双重压力 |
| 长期（7年+） | 关键变量：可信/可审计 AI 是否成为监管要求 |

---

## 五、百亿/千亿 Token 上下文对 Palantir 的影响

### 会被冲击的部分

```
当前 RAG 的存在理由：上下文有限 → 必须先检索再生成
100M token 时：整个企业文档库可能直接塞进上下文
→ Ontology 作为"数据导航层"的价值大幅缩水
→ RAG pipeline 的工程复杂度壁垒消失
```

### 三堵长上下文打不穿的墙

**墙1：经济学不允许**
> 100M token × $7/1M = $700/次查询
> 企业每天 1 万次查询 = $700万/天
> 达到可负担级别，需要推理成本再下降 1000x

**墙2：注意力机制的物理限制**
> "Lost in the Middle" 研究：1M token 时，中间位置信息召回率降至 40-60%
> 精确事实检索的可靠性低于当前 RAG + 重排方案
> 法律/军事等高精度场景不可接受

**墙3：Palantir 真正的护城河不是 RAG**

| 核心能力 | 长上下文能解决吗 |
|---|---|
| AI 决策的审计追溯 | ✗ |
| 机密数据气隙部署 | ✗ |
| AI Action 的责任归属 | ✗ |
| 实时数据流更新 | ✗ |
| 操作权限隔离 | ✗ |

### 价值重心的迁移

```
现在（2025）：
  Ontology 价值 = 数据导航（60%）+ 可信行动层（40%）

100M context 时代：
  Ontology 价值 = 数据导航（10%）+ 可信行动层（90%）
```

**类比**：Google Maps 出现后导航被商品化，但"调度执行层"（Uber/滴滴）的价值反而爆发。长上下文商品化"找数据"，会让"用数据做什么、谁能做、怎么追责"这一层的价值放大。

### 最终判断

> 长上下文会迫使 Palantir 放弃"数据导航"卖点，但同时推动整个行业更迫切地需要"可信 AI 行动层"——这恰好是 Palantir 花了 20 年在政府客户中建立的另一个护城河。
>
> 这是一次**强迫升级**，不是终局。

---

## 核心方法论总结

1. **"先写后查"原则**：任何知识图谱系统，数据必须预先填充，查询只是模式匹配
2. **语义层 vs 物理层**：关系定义在语义层（低成本、可变）比固化在物理层（ETL）灵活得多
3. **护城河分层分析**：技术护城河要区分"基于当前约束的优势"（会随技术演进消失）和"基于业务信任的优势"（长期稳固）
4. **价值重心迁移**：技术能力被商品化时，价值向更高层的"可信执行层"迁移，而非消失
