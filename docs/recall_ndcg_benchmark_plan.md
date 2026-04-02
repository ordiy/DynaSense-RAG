# Recall / NDCG 测试方案（SciQ 公开数据集）

## 1. 目标

在**可复现**条件下，对 DynaSense-RAG 检索链路（Dense → Small-to-Big → Jina 重排 Top-10；可选 Hybrid 路由+融合）进行 **Recall@1、@3、@5、@10** 与 **NDCG@K（K=1,3,5,10）** 的均值评估，并输出 Markdown + JSON 报告。

## 2. 数据集选型：SciQ

| 项目 | 说明 |
|------|------|
| **名称** | SciQ（`allenai/sciq`） |
| **来源** | Hugging Face `datasets` |
| **特点** | 科学问答；每条含 `question` 与 `support`（支持段落），适合段落级检索评测 |
| **与业务语料差异** | 领域为通识科学，非金融/法律；用于**回归与算法对比**，不替代领域验收 |

**不采用 MS MARCO / BEIR 全量**：需大规模索引与更长运行时间；MVP 以 SciQ 与仓库内 `demo_related_party` 互补（后者见 `data/recall_eval_cases.json`）。

## 3. 相关性标注（无人工标注成本）

每条 `support` 在入库时于文末追加**唯一标记**：

```text
[benchmark_doc_id=N]
```

评测时 `expected_substring` 即为该标记；命中即表示**正确 support 对应的父文档**进入检索结果且排在某位置，避免自然语言子串碰撞。

## 4. 测试流程

```text
1. 配置环境变量（Vertex、Jina、独立 LanceDB 路径）
2. 从 SciQ train 抽取 N 个不同 support → 建 N 个父文档（带标记）
3. 从同一数据流中抽取 M 条 question（仅对应已建库 doc_id）
4. 对每条 question 调用 run_evaluation(query, marker, use_hybrid=...)
5. 聚合 metrics（recall@* / ndcg@*）取算术平均
6. 写出 reports/recall_ndcg_benchmark_<UTC>.md + .json，并更新 recall_ndcg_benchmark_latest.md
```

### 4.1 前置条件

- `GOOGLE_CLOUD_PROJECT`、`GOOGLE_APPLICATION_CREDENTIALS`（`text-embedding-004`）
- `JINA_API_KEY`（强烈建议；否则重排退化，指标偏低）
- 网络：首次需下载 SciQ

### 4.2 命令

```bash
cd /path/to/DynaSense-RAG
source .venv/bin/activate
export GOOGLE_CLOUD_PROJECT=...
export GOOGLE_APPLICATION_CREDENTIALS=...
export JINA_API_KEY=...

# 独立向量库，避免覆盖业务数据
export LANCEDB_URI=./data/lancedb_recall_benchmark
export SKIP_NEO4J_INGEST=1

python scripts/benchmark_recall_ndcg.py --num-docs 100 --num-queries 50 --seed 42

# Hybrid 路径（与线上一致的路由+融合 Top-10）
python scripts/benchmark_recall_ndcg.py --hybrid --num-docs 80 --num-queries 40
```

### 4.3 干跑（仅生成报告模板）

```bash
python scripts/benchmark_recall_ndcg.py --dry-run
```

## 5. 指标说明

与 [`recall_evaluation.md`](./recall_evaluation.md) 一致：二元相关性、单文档命中、NDCG 以 rank-1 为理想。

## 6. 产物

| 文件 | 内容 |
|------|------|
| `reports/recall_ndcg_benchmark_<UTC>.md` | 人类可读报告 |
| `reports/recall_ndcg_benchmark_<UTC>.json` | 逐条 query 结果 + mean_metrics |
| `reports/recall_ndcg_benchmark_latest.md` | 最近一次完整运行的副本 |

## 7. 风险与限制

- **API 费用与耗时**：嵌入 + 重排 × 查询数；脚本默认 `--sleep 0.35` 缓解 Jina 限流。
- **指标含义**：标记在父文档末尾；若分块策略将标记切到单独块且未进 Top-K 子块检索，可能影响结果（一般单 support 较短，风险低）。
- **Hybrid**：依赖 Neo4j 时部分路由会降级为 VECTOR（评测仍有效）。
