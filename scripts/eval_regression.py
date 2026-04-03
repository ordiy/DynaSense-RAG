#!/usr/bin/env python3
"""
Batch retrieval evaluation from a JSON file (offline / CI with full stack).

Usage:
  export GOOGLE_APPLICATION_CREDENTIALS=...
  pip install -r requirements.txt
  python scripts/eval_regression.py path/to/cases.json

JSON format: { "cases": [ { "id": "optional", "query": "...", "expected_substring": "..." } ] }
Delegates to ``rag_core.run_evaluation`` (same metrics as ``/api/evaluate``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MAP-RAG batch eval from JSON cases.")
    parser.add_argument("cases_file", type=Path, help="JSON file with a 'cases' array")
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Use hybrid retrieval (same as API use_hybrid=true)",
    )
    args = parser.parse_args()
    raw = json.loads(args.cases_file.read_text(encoding="utf-8"))
    cases = raw.get("cases") or []
    if not cases:
        print("No cases in file.", file=sys.stderr)
        return 1

    # Import after argparse so ``python scripts/eval_regression.py --help`` works without Vertex.
    from src.rag_core import run_evaluation
    from src.recall_metrics import aggregate_mean

    rows: list[dict] = []
    metrics_list: list[dict] = []
    for c in cases:
        q = c.get("query", "")
        exp = c.get("expected_substring", "")
        cid = c.get("id")
        r = run_evaluation(q, exp, use_hybrid=args.hybrid)
        if "error" in r:
            rows.append({"id": cid, "error": r["error"]})
            continue
        rows.append({"id": cid, **r})
        m = r.get("metrics")
        if isinstance(m, dict):
            metrics_list.append(m)

    mean_metrics = aggregate_mean(metrics_list) if metrics_list else {}
    out = {
        "n_ok": len(metrics_list),
        "n_err": len(cases) - len(metrics_list),
        "mean_metrics": mean_metrics,
        "cases": rows,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if out["n_err"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
