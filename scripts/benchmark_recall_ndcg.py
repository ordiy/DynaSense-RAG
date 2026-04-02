#!/usr/bin/env python3
"""
SciQ-based Recall@K + NDCG@K benchmark (aligned with src/recall_metrics.py).

Prerequisites:
  - GOOGLE_CLOUD_PROJECT + GOOGLE_APPLICATION_CREDENTIALS (Vertex embeddings)
  - JINA_API_KEY (reranker; degraded without it)

Usage (from repo root):
  LANCEDB_URI=./data/lancedb_recall_benchmark SKIP_NEO4J_INGEST=1 \\
    .venv/bin/python scripts/benchmark_recall_ndcg.py --num-docs 100 --num-queries 50

  # Hybrid retrieval path (router + fusion rerank top-10):
  .venv/bin/python scripts/benchmark_recall_ndcg.py --hybrid --num-docs 80 --num-queries 40

Environment is applied inside main() before importing src.rag_core.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _marker(doc_id: int) -> str:
    return f"[benchmark_doc_id={doc_id}]"


def main() -> int:
    parser = argparse.ArgumentParser(description="SciQ Recall / NDCG benchmark")
    parser.add_argument("--num-docs", type=int, default=100, help="Unique support passages to index")
    parser.add_argument("--num-queries", type=int, default=50, help="Queries to evaluate (<= indexed QA pairs)")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for query subsample")
    parser.add_argument("--hybrid", action="store_true", help="Use run_evaluation(use_hybrid=True)")
    parser.add_argument(
        "--lancedb-uri",
        default=str(ROOT / "data" / "lancedb_recall_benchmark"),
        help="Isolated LanceDB directory (set LANCEDB_URI before import — this script sets it in main)",
    )
    parser.add_argument("--sleep", type=float, default=0.35, help="Seconds between API-heavy eval calls")
    parser.add_argument("--dry-run", action="store_true", help="Only write report skeleton + exit 0")
    args = parser.parse_args()

    report_dir = ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = report_dir / f"recall_ndcg_benchmark_{ts}.md"
    json_path = report_dir / f"recall_ndcg_benchmark_{ts}.json"

    if args.dry_run:
        body = _report_markdown(
            title_suffix="(dry-run)",
            dataset="sciq (HuggingFace)",
            num_docs=args.num_docs,
            num_queries=args.num_queries,
            use_hybrid=args.hybrid,
            mean_metrics={},
            n_ok=0,
            n_err=0,
            notes="Dry run: no indexing, no Vertex/Jina calls.",
        )
        report_path.write_text(body, encoding="utf-8")
        print(f"Wrote {report_path}")
        return 0

    # Critical: set paths before rag_core import
    os.environ.setdefault("LANCEDB_URI", args.lancedb_uri)
    os.makedirs(os.path.dirname(os.environ["LANCEDB_URI"]) or ".", exist_ok=True)
    os.environ["SKIP_NEO4J_INGEST"] = "1"

    sys.path.insert(0, str(ROOT))

    from datasets import load_dataset

    from src.recall_metrics import aggregate_mean
    from src.rag_core import process_document_task, reset_knowledge_base, run_evaluation

    print("Loading SciQ train split...")
    ds = load_dataset("sciq", split="train")

    unique_contexts: dict[str, int] = {}
    qa_pairs: list[dict] = []

    for row in ds:
        support = (row.get("support") or "").strip()
        question = (row.get("question") or "").strip()
        if not support or not question:
            continue
        if support not in unique_contexts:
            if len(unique_contexts) >= args.num_docs:
                continue
            unique_contexts[support] = len(unique_contexts) + 1
        doc_id = unique_contexts[support]
        qa_pairs.append({"question": question, "doc_id": doc_id, "support": support})

    if len(unique_contexts) < 5:
        print("ERROR: Not enough unique SciQ supports. Check dataset / network.")
        return 2

    reset_knowledge_base()

    print(f"Indexing {len(unique_contexts)} parent documents (markers + Jina chunking)...")
    for support, doc_id in unique_contexts.items():
        content = support + "\n\n" + _marker(doc_id)
        process_document_task(content, f"sciq_{doc_id}.txt", {})

    # Subsample queries that reference indexed docs only
    indexed_pairs = [q for q in qa_pairs if q["doc_id"] <= len(unique_contexts)]
    random.seed(args.seed)
    random.shuffle(indexed_pairs)
    test_pairs = indexed_pairs[: args.num_queries]

    print(f"Evaluating {len(test_pairs)} queries (hybrid={args.hybrid})...")
    metrics_list: list[dict] = []
    rows: list[dict] = []
    for i, item in enumerate(test_pairs):
        q = item["question"]
        doc_id = item["doc_id"]
        needle = _marker(doc_id)
        try:
            r = run_evaluation(q, needle, use_hybrid=args.hybrid)
            if "error" in r:
                rows.append({"question": q, "doc_id": doc_id, "error": r["error"]})
            else:
                m = r.get("metrics", {})
                if isinstance(m, dict):
                    metrics_list.append(m)
                rows.append(
                    {
                        "question": q,
                        "doc_id": doc_id,
                        "hit_rank": r.get("hit_rank"),
                        "metrics": m,
                    }
                )
        except Exception as e:
            rows.append({"question": q, "doc_id": doc_id, "error": str(e)})
        if args.sleep > 0 and i + 1 < len(test_pairs):
            time.sleep(args.sleep)
        if (i + 1) % 10 == 0:
            print(f"  ... {i + 1}/{len(test_pairs)}")

    mean_metrics = aggregate_mean(metrics_list) if metrics_list else {}
    n_ok = len(metrics_list)
    n_err = len(test_pairs) - n_ok

    payload = {
        "timestamp_utc": ts,
        "dataset": "sciq",
        "num_docs_indexed": len(unique_contexts),
        "num_queries": len(test_pairs),
        "use_hybrid": args.hybrid,
        "mean_metrics": mean_metrics,
        "cases": rows,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    body = _report_markdown(
        title_suffix=f"run {ts}",
        dataset="sciq (HuggingFace `allenai/sciq`, train split)",
        num_docs=len(unique_contexts),
        num_queries=len(test_pairs),
        use_hybrid=args.hybrid,
        mean_metrics=mean_metrics,
        n_ok=n_ok,
        n_err=n_err,
        notes=(
            "Ground truth: synthetic marker `[benchmark_doc_id=N]` appended to each support passage; "
            "Recall/NDCG use first hit in reranked top-10 contexts (see `docs/recall_evaluation.md`). "
            "Neo4j ingest skipped (`SKIP_NEO4J_INGEST=1`). "
            f"Raw JSON: `{json_path.name}`."
        ),
    )
    report_path.write_text(body, encoding="utf-8")
    latest = report_dir / "recall_ndcg_benchmark_latest.md"
    latest.write_text(body, encoding="utf-8")

    print("\n=== Mean metrics ===")
    for k in sorted(mean_metrics.keys()):
        print(f"  {k}: {mean_metrics[k]:.6f}")
    print(f"\nWrote {report_path}\nWrote {json_path}\nSymlink-style copy: {latest}")
    return 0


def _report_markdown(
    title_suffix: str,
    dataset: str,
    num_docs: int,
    num_queries: int,
    use_hybrid: bool,
    mean_metrics: dict,
    n_ok: int,
    n_err: int,
    notes: str,
) -> str:
    lines = [
        f"# Recall / NDCG Benchmark Report {title_suffix}",
        "",
        "## 1. Test objective",
        "",
        "Measure **mean Recall@1,3,5,10** and **mean NDCG@K** (K=1,3,5,10) on a public scientific-QA corpus, using the same retrieval + rerank stack as production evaluation (`run_evaluation`).",
        "",
        "## 2. Dataset",
        "",
        f"- **Corpus**: {dataset}",
        "- **Indexing**: each unique `support` paragraph is stored as one parent document; a deterministic marker `[benchmark_doc_id=N]` is appended so relevance is unambiguous (no accidental substring collisions).",
        f"- **Indexed documents**: {num_docs}",
        f"- **Evaluation queries**: {num_queries}",
        f"- **Retrieval mode**: `use_hybrid={'true' if use_hybrid else 'false'}` (see `docs/recall_evaluation.md`).",
        "",
        "## 3. Environment",
        "",
        "- `LANCEDB_URI`: isolated benchmark directory (default `./data/lancedb_recall_benchmark`)",
        "- `SKIP_NEO4J_INGEST=1`: skip graph extraction during ingest",
        "- Vertex AI embeddings + Jina reranker (if `JINA_API_KEY` set)",
        "",
        "## 4. Aggregated results",
        "",
        f"- Successful queries: **{n_ok}**, errors: **{n_err}**",
        "",
        "| Metric | Mean |",
        "|--------|------|",
    ]
    keys = [k for k in sorted(mean_metrics.keys()) if k.startswith("recall@") or k.startswith("ndcg@")]
    if not keys:
        lines.append("| *(no successful runs)* | — |")
    else:
        for k in keys:
            lines.append(f"| `{k}` | {mean_metrics[k]:.6f} |")
    lines.extend(
        [
            "",
            "## 5. Interpretation",
            "",
            "- **Recall@K**: fraction of queries where the relevant parent (identified by marker) appears in the **top-K** after dense retrieval → small-to-big → Jina rerank (and hybrid routing when enabled).",
            "- **NDCG@K**: binary relevance, single relevant document; normalized DCG vs ideal rank-1 (see `src/recall_metrics.py`).",
            "",
            "## 6. Notes",
            "",
            notes,
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
