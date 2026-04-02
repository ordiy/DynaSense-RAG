#!/usr/bin/env python3
"""
Offline batch evaluation: Recall@1,3,5,10 and NDCG@K (K=1,3,5,10).

Usage:
  cd repo && .venv/bin/python scripts/run_recall_eval.py [--hybrid] [--json path/to/cases.json]

Requires indexed LanceDB (upload documents first). Uses Vertex + Jina when enabled.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Ensure project root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Recall/NDCG batch eval.")
    parser.add_argument(
        "--json",
        default=os.path.join(ROOT, "data", "recall_eval_cases.json"),
        help="Path to JSON array of {id, query, expected_substring}",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Use Hybrid RAG retrieval (router + fusion); default is vector-only top-10.",
    )
    args = parser.parse_args()

    os.chdir(ROOT)

    with open(args.json, "r", encoding="utf-8") as f:
        cases = json.load(f)

    from src.recall_metrics import aggregate_mean
    from src.rag_core import run_evaluation

    metrics_list: list[dict] = []
    rows: list[dict] = []

    for c in cases:
        q = c["query"]
        exp = c["expected_substring"]
        cid = c.get("id")
        r = run_evaluation(q, exp, use_hybrid=args.hybrid)
        if "error" in r:
            print(f"[ERR] {cid}: {r['error']}")
            rows.append({"id": cid, "error": r["error"]})
            continue
        m = r.get("metrics", {})
        rows.append({"id": cid, **r})
        if isinstance(m, dict):
            metrics_list.append(m)
        print(f"[OK] {cid} hit_rank={r.get('hit_rank')} metrics={m}")

    mean = aggregate_mean(metrics_list) if metrics_list else {}
    print("\n=== Mean metrics (successful cases) ===")
    for k in sorted(mean.keys()):
        print(f"  {k}: {mean[k]:.6f}")

    out_path = os.path.join(ROOT, "reports", "recall_eval_last.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"mean_metrics": mean, "cases": rows, "use_hybrid": args.hybrid}, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
