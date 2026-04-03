"""Retrieval evaluation endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from src.api.schemas import EvalBatchRequest, EvalRequest
from src.recall_metrics import aggregate_mean
from src.rag_core import run_evaluation

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["eval"])


@router.post("/evaluate")
async def evaluate(request: EvalRequest):
    try:
        result = run_evaluation(
            request.query,
            request.expected_substring,
            use_hybrid=request.use_hybrid,
        )
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception:
        logger.exception("Evaluation pipeline failed.")
        raise HTTPException(status_code=500, detail="Internal server error.")


@router.post("/evaluate/batch")
async def evaluate_batch(request: EvalBatchRequest):
    try:
        rows: list[dict] = []
        metrics_list: list[dict] = []
        for c in request.cases:
            r = run_evaluation(
                c.query,
                c.expected_substring,
                use_hybrid=request.use_hybrid,
            )
            if "error" in r:
                rows.append({"id": c.id, "error": r["error"]})
                continue
            rows.append({"id": c.id, **r})
            m = r.get("metrics")
            if isinstance(m, dict):
                metrics_list.append(m)
        mean_metrics = aggregate_mean(metrics_list) if metrics_list else {}
        return {
            "use_hybrid": request.use_hybrid,
            "n_ok": len(metrics_list),
            "n_err": len(request.cases) - len(metrics_list),
            "mean_metrics": mean_metrics,
            "cases": rows,
        }
    except Exception:
        logger.exception("Batch evaluation failed.")
        raise HTTPException(status_code=500, detail="Internal server error.")
