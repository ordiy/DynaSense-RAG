# 🧠 DynaSense-RAG（MAP-RAG アーキテクチャ）

> **MAP-RAG**: Multi-resolution Agentic Perception Retrieval-Augmented Generation

厳格なハルシネーション対策、インテリジェントな意味チャンク分割、クロスエンコーダによる再ランキングに重点を置いた、エンタープライズ向け RAG（検索拡張生成）アーキテクチャのプロトタイプです。

## 🌐 他言語
[English 🇺🇸](README.md) · [简体中文](README-cn.md) · [繁體中文](README-ch.md) · [Deutsch 🇩🇪](README-de.md)

## 🖥️ Web プレビュー

| チャット画面 | RAG 管理パネル |
|:-:|:-:|
| ![チャット画面](https://raw.githubusercontent.com/ordiy/study_notes/master/res/image/2026/20260412111147128.png) | ![RAG 管理パネル](https://raw.githubusercontent.com/ordiy/study_notes/master/res/image/2026/20260412111104002.png) |

## 🎯 コア哲学
**「悪い／有害な回答より、答えない方がまし。」**

法務、金融、社内人事ポリシーなどのエンタープライズ環境では、LLM のハルシネーションは許容できません。本 MVP は、メインパイプライン上でのリアルタイムな汎用クエリ書き換えを**意図的に採用しません**。これにより「意図のドリフト」（専門用語が汎用語に書き換えられ意味が失われる）を防ぎ、不要な LLM レイテンシも避けます。

代わりに、次の手段で高精度を実現します。
1. **インテリジェント・チャンキング**（Jina Segmenter）
2. **高次元ベクトル検索**（Google Vertex AI `text-embedding-004` + LanceDB）
3. **クロスエンコーダによる意味再ランキング**（Jina Multilingual Reranker）
4. **二系統の Grader + Generator**（LangGraph ステートマシン — 事実クエリは厳格、推論クエリは分析可能）
5. **サーバー側マルチターン記憶**（文脈長制御付き会話セッション）
6. **Hybrid RAG（MVP）** — **Query Router** + **Dense + BM25** + **Neo4j グラフ検索** + 採点前の統一 **Top‑K 再ランク**（`docs/mvp_hybrid_rag.md` 参照）



## 🏗️ アーキテクチャ設計（MAP-RAG）

```text
╔══════════════════════════════════════════════════════════════════════╗
║                     データ取り込みパイプライン                         ║
╚══════════════════════════════════════════════════════════════════════╝

生ドキュメント（TXT/MD）
      │
      ▼
[ Jina 意味セグメンター ] ──(チャンク分割)──> 子テキストチャンク
                                              │
                    ┌─────────────────────────┴──────────────────────────┐
                    ▼                                                    ▼
         [ ドキュメント DB（MongoMock） ]                    [ Vertex AI Embeddings ]
           保存: 親テキスト全文                       text-embedding-004
           キー: parent_id  ◄──── parent_id ────────────────────┤
                                                               ▼
                                                    [ ベクトル DB（LanceDB） ]
                                                      保存: 密ベクトル
                                                      メタデータ: parent_id

╔══════════════════════════════════════════════════════════════════════╗
║               検索・生成パイプライン                                   ║
╚══════════════════════════════════════════════════════════════════════╝

  ユーザークエリ ──────────────────────────────────┐
      │                                         │ （マルチターン）
      │                              [ セッション記憶 ]
      │                              conversation_id
      │                              履歴 → コンテキスト予算
      │                              _build_query_with_history()
      │                                         │
      ▼                                         ▼
[ LanceDB ベクトル検索 ]  ←──── 履歴付き拡張クエリ
   Top K=10 子チャンク
      │
      ▼
[ Small-to-Big 展開 ]
   child_id → parent_id → 親テキスト全文
      │
      ▼
[ Jina クロスエンコーダ再ランカー ]
   Top K=3 高精度親ドキュメント
      │
      ▼
[ クエリ種別検出 ]   ← NEW: _is_analysis_query()
      │
      ├─────── 事実クエリ ──────────────────────────────────┐
      │        （照会、定義、具体的な事実）                  │
      │                                                        ▼
      │                                           [ GRADE_PROMPT（厳格） ]
      │                                           「文脈に直接の答えが
      │                                            含まれるか？」
      │                                                        │
      │                                            NO ──► [ ブロック / フォールバック ]
      │                                            YES ──► [ GEN_PROMPT ]
      │                                                    「文脈を厳密に使用」
      │
      └─────── 分析クエリ ────────────────────────────────┐
               （分析/影響/方法/理由/計画/評価…）             │
               （analyze/impact/why/how/plan/risk…）          ▼
                                                 [ GRADE_ANALYSIS_PROMPT（緩和） ]
                                                 「文脈にトピック関連の
                                                  背景事実が**何か**あるか？」
                                                              │
                                                  NO ──► [ ブロック / フォールバック ]
                                                  YES ──► [ GEN_ANALYSIS_PROMPT ]
                                                          「事実の根拠 + ドメイン推論。
                                                           ラベル:
                                                           【文書事実】【分析推論】」
                                                              │
                                                              ▼
                                                   最終的な統合回答
```

本システムは有向 LangGraph ステートマシンを使用します。主な設計判断は次のとおりです。
- **クリティカルパスでのクエリ書き換えなし** — 意図のドリフトを防ぎ、レイテンシを削減
- **二系統ルーティング** — 分析クエリは厳格な事実用 Grader でブロックされない。LLM に推論と取得事実の区別を明示的にラベル付けさせる
- **デフォルトでフェイルクローズ** — Grader がエラーを返した場合、未検証の文脈を通すのではなく回答をブロック



## 📊 ベンチマーク結果（SciQ データセット）
HuggingFace の `sciq` データセットのサブセット（1000 文書、100 問）で本パイプラインを評価しました。

| メトリクス | ベースベクトル検索（Vertex AI） | パイプライン（ベクトル + Jina Reranker） | 改善 |
|---|---|---|---|
| **Recall@1** | 86.0% | **96.0%** | 🚀 **+10.0%** |
| **Recall@3** | 96.0% | **100.0%** | 🚀 **+4.0%** |
| **Recall@5** | 99.0% | **100.0%** | +1.0% |
| **Recall@10** | 100.0% | 100.0% | 上限到達 |

*結論*: 再ランカーは事実上、精度の「スナイパー」として機能し、LLM が正しい文脈を得るのに 1〜3 チャンクで十分となるケースが 100% になります。トークンコストの大幅削減、レイテンシの劇的な低減、ハルシネーションの余地の縮小につながります。

### Recall@K / NDCG@K（バッチスクリプト、SciQ）
`scripts/benchmark_recall_ndcg.py` による自動実行。評価と同じ検索スタック（`run_evaluation`）、**ベクトル経路のみ**（`use_hybrid=false`）。最新レポート: [`reports/recall_ndcg_benchmark_latest.md`](reports/recall_ndcg_benchmark_latest.md)。

| 設定 | 値 |
|--------|--------|
| コーパス | HuggingFace `allenai/sciq`（train）、一意の `support` 段落を親ドキュメントとして索引 |
| 索引ドキュメント数 | 60 |
| 評価クエリ数 | 30 |
| 検索モード | Dense → Small-to-Big → Jina 再ランク（hybrid ルーティング off） |

| 指標（平均） | 値 |
|---------------|-------|
| Recall@1,3,5,10 | 1.000 |
| NDCG@1,3,5,10 | 1.000 |

生 JSON とタイムスタンプ付きレポートは `reports/recall_ndcg_benchmark_*.{json,md}`。詳細は [`docs/recall_evaluation.md`](docs/recall_evaluation.md)。

## ✨ 機能ハイライト

### 二系統クエリルーティング（分析 vs 事実）
パイプラインは、クエリが**事実照会**か**分析的推論**かを自動検出し、適切な Grader とジェネレータ方針に振り分けます。

| | 事実トラック | 分析トラック |
|---|---|---|
| **トリガー** | デフォルト | キーワード: 分析/影響/方法/計画/evaluate/impact… |
| **Grader** | 厳格: 文脈に直接の答えが必要 | 緩和: トピック関連の事実があれば可 |
| **Generator** | `GEN_PROMPT`: 「文脈を厳密に使用」 | `GEN_ANALYSIS_PROMPT`: 事実 + ドメイン推論 |
| **出力形式** | 直接回答 | `【文書事実】` + `【分析推論】` のラベル付きセクション |

**デモ — 部分的な文脈での分析クエリ:**
> **ユーザー**: 「豌豆苗先物」を紹介し、天候が当該先物取引に与える影響を分析してください
>
> **取得文脈**: 生育サイクル 3 ヶ月、地域: 東海岸農場、収量 10 トン/日
>
> **応答** *(要約)*:
> **【文書事実】** 豌豆苗先物の作物生育サイクルは 3 ヶ月、日産量は 10 トン。
> **【分析推論】** 業界経験に基づく: ① 極端な天候（霜/高温）は減産を直接招き先物価格を押し上げる可能性; ② 高温多湿は病害虫を促進し、受渡し品質を低下; ③ 悪天候は輸送を妨げ物流コストを増加させ、先物に波及。

設計・実装詳細、4 件のデモケースは [docs/dual-track-query-routing.md](./docs/dual-track-query-routing.md) を参照してください。

### サーバー側マルチターン記憶
バックエンドで `conversation_id` により会話セッションを管理し、文脈長制御と TTL クリーンアップを行います。詳細は [docs/chat_test_memory_design.md](./docs/chat_test_memory_design.md) を参照。

### A/B メモリ戦略の比較
`POST /api/chat/session/ab` は同一メッセージに対して `prioritized` と `legacy` の両メモリモードを並列実行し、クエリ本文・回答・ブロック状態を並べて返します。メモリ戦略の効果を迅速に診断できます。

### Hybrid RAG — ルーティング + デュアルリコール + Neo4j（MVP）
**`readme-v2-1.md`** に基づく実装: LLM **意図ルータ**（`VECTOR` / `GRAPH` / `GLOBAL` / `HYBRID`）、**デュアル索引**（LanceDB + `chunk_id` 由来情報付き Neo4j トリプル）、オンライン **Dense + BM25** と **グラフ線形化**、既存 grader/generator の前に **単一 Jina 再ランク** で Top‑5 に截断。

```text
ユーザー Query
    │
    ▼
[ Query Router (LLM) ] ──► VECTOR | GRAPH | GLOBAL | HYBRID
    │
    ├─ VECTOR ──► Dense(Small-to-Big) + BM25(子→親) ──┐
    ├─ GRAPH ───► Neo4j 部分グラフ → 線形化トリプル ──┤──► [ Jina Rerank Top‑5 ]
    ├─ GLOBAL ──► グラフ要約 + 小さな dense アンカー ─┤
    └─ HYBRID ──► VECTOR + GRAPH 候補のマージ ────────┘
                                        │
                                        ▼
                           Grader（反ハルシネーション）→ Generator
```

```mermaid
flowchart TB
  subgraph ingest["オフライン：デュアルエンジン索引"]
    D[生ドキュメント] --> J[Jina チャンク]
    J --> E[Vertex 埋め込み + LanceDB]
    J --> T[LLM トリプル抽出]
    T --> N[(Neo4j + chunk_id)]
  end
  subgraph online["オンライン：ルータ + 統一再ランク"]
    Q[ユーザークエリ] --> R[Query Router]
    R --> V[VECTOR: dense + BM25]
    R --> G[GRAPH: Neo4j + 線形化]
    R --> GL[GLOBAL: グラフ要約 + dense アンカー]
    R --> HY[HYBRID: 候補マージ]
    V --> RR[Jina Cross-Encoder Top-K]
    G --> RR
    GL --> RR
    HY --> RR
    RR --> P[Grader + Generator]
  end
```

- **ローカル Neo4j**: `docker compose -f docker-compose.neo4j.yml up -d`（Bolt `7687`、既定パスワード `changeme`）。
- **デモコーパス**: `data/demo_related_party.txt` をアップロードし *「中国中信银行的关联方有哪些？」* など — ログで `GRAPH` または `HYBRID` とグラフ文脈が典型。
- **Hybrid 無効化**（ベクトルのみ LangGraph に戻す）: `export HYBRID_RAG_ENABLED=false`。

詳細は [`docs/mvp_hybrid_rag.md`](docs/mvp_hybrid_rag.md)。

---

## 🛠️ 技術スタック
* **オーケストレーション**: `LangGraph` & `LangChain`
* **埋め込みモデル**: Google Vertex AI `text-embedding-004`
* **LLM**: Google Vertex AI `gemini-2.5-pro`
* **ベクトル DB**: `LanceDB`
* **意味チャンク分割**: `Jina Segmenter API`
* **再ランカー**: `jina-reranker-v2-base-multilingual`
* **グラフ DB（Hybrid MVP）**: Neo4j Community（ローカル Docker）+ Python ドライバ `neo4j`
* **語彙検索**: `rank-bm25`（子チャンク上の BM25Okapi）
* **セッションストア**: TTL 付きインメモリ `dict`（Redis への拡張可能）

## 🚀 はじめに
```bash
# 1. 仮想環境のセットアップ
python3 -m venv .venv
source .venv/bin/activate

# 2. 依存関係のインストール
pip install -r requirements.txt

# 3. API キー、GCP、PostgreSQL
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/gcp-sa.json"
export JINA_API_KEY="your-jina-api-key"

# 4. PostgreSQL + pgvector（ストレージに必須）
docker compose -f docker-compose.postgres.yml up -d
export DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:5433/map_rag

# 5. Web サーバーの起動
.venv/bin/uvicorn src.app:app --host 0.0.0.0 --port 8000

# ブラウザで http://localhost:8000 を開く
# タブ 1: ドキュメントアップロード
# タブ 2: 単一ターン会話
# タブ 3: 評価
# タブ 4: マルチターン会話テスト（メモリ + A/B 比較）
```

## 📄 ドキュメント

| ドキュメント | 説明 |
|---|---|
| [docs/langsmith_observability.md](./docs/langsmith_observability.md) | **LangSmith オブザーバビリティ** — 環境変数、初期化順（`src/observability.py`）、[公式ドキュメント](https://docs.langchain.com/langsmith/observability) |
| [docs/langgraph_stream_log.md](./docs/langgraph_stream_log.md) | **LangGraph ストリームログ** — `LANGGRAPH_STREAM_LOG`、`invoke_rag_app` |
| [docs/architecture.md](./docs/architecture.md) | **クリーンアーキテクチャ** — `api/` / `core/` / `domain/` / `infrastructure/` |
| [docs/mvp_hybrid_rag.md](./docs/mvp_hybrid_rag.md) | **Hybrid RAG MVP** — ルータ、Dense+BM25、Neo4j、融合再ランク（`readme-v2-1.md`） |
| [docs/recall_evaluation.md](./docs/recall_evaluation.md) | **Recall@K / NDCG@K** — ケース、バッチ API、`scripts/run_recall_eval.py` |
| [docs/recall_ndcg_benchmark_plan.md](./docs/recall_ndcg_benchmark_plan.md) | **SciQ ベンチマーク計画** — `scripts/benchmark_recall_ndcg.py`、レポート `reports/recall_ndcg_benchmark_*.md` |
| [docs/dual-track-query-routing.md](./docs/dual-track-query-routing.md) | **二系統クエリルーティング** — 分析 vs 事実、Grader/Generator 方針、デモ Q&A |
| [docs/chat_test_memory_design.md](./docs/chat_test_memory_design.md) | サーバー側マルチターン記憶、`conversation_id` セッション設計 |
| [docs/doc-small-to-big-retrieval.md](./docs/doc-small-to-big-retrieval.md) | 親子チャンク展開（Small-to-Big 検索） |
| [docs/doc-feauture-v1.md](./docs/doc-feauture-v1.md) | 初期アーキテクチャ RFC |
| [docs/doc-future.md](./docs/doc-future.md) | 悪い回答を防ぐためのエンタープライズ原則 |
| [readme-v2-1.md](./readme-v2-1.md) | デュアルトラック Hybrid RAG の仕様と **Q&A テストデータ**（関連当事者デモリンク・例示質問） |
