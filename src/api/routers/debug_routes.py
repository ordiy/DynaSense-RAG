"""Read-only LanceDB / Neo4j debug API."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

from src import debug_data
from src.api.schemas import (
    ConstrainedGraphRequest,
    ConstrainedGraphResponse,
    ConstrainedGraphSuggestRequest,
    ConstrainedGraphSuggestResponse,
    Neo4jSearchRequest,
)
from src.core.graph_constrained_queries import (
    execute_constrained_template,
    suggest_template_from_question,
)
from src.core.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/debug", tags=["debug"])


def _require_debug_data_api() -> None:
    if not get_settings().debug_data_api:
        raise HTTPException(
            status_code=403,
            detail="Debug data API is disabled. Set environment variable DEBUG_DATA_API=true to enable.",
        )


@router.get("/lancedb/summary")
async def debug_lancedb_summary():
    _require_debug_data_api()
    try:
        return debug_data.lancedb_summary()
    except Exception as e:
        logger.exception("lancedb summary failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/lancedb/rows")
async def debug_lancedb_rows(
    table: str | None = None,
    limit: int = 40,
    offset: int = 0,
    source: str | None = None,
    parent_id: str | None = None,
):
    _require_debug_data_api()
    try:
        out = debug_data.lancedb_rows(
            table_name=table,
            limit=limit,
            offset=offset,
            source_substring=source,
            parent_id_substring=parent_id,
        )
        if out.get("error"):
            raise HTTPException(status_code=404, detail=out["error"])
        return out
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("lancedb rows failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/neo4j/summary")
async def debug_neo4j_summary():
    _require_debug_data_api()
    text, err = debug_data.neo4j_summary_text()
    if err:
        raise HTTPException(status_code=503, detail=err)
    return {"summary": text}


@router.post("/neo4j/search")
async def debug_neo4j_search(body: Neo4jSearchRequest):
    _require_debug_data_api()
    rows, err = debug_data.neo4j_keyword_search(body.keywords, body.limit)
    if err:
        raise HTTPException(status_code=400 if "Provide" in err else 503, detail=err)
    return {"rows": rows, "count": len(rows)}


@router.post("/graph/constrained/run", response_model=ConstrainedGraphResponse)
async def debug_graph_constrained_run(body: ConstrainedGraphRequest):
    """
    Run a **whitelisted** graph template with validated params (no LLM-generated Cypher).

    Prefer this over GraphCypherQAChain for production-adjacent debugging.
    """
    _require_debug_data_api()
    try:
        out = execute_constrained_template(body.template, body.params)
        return ConstrainedGraphResponse(template_id=out.template_id, rows=out.rows, logs=out.logs)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.post("/graph/constrained/suggest", response_model=ConstrainedGraphSuggestResponse)
async def debug_graph_constrained_suggest(
    body: ConstrainedGraphSuggestRequest,
    execute: bool = Query(False, description="If true, run the suggested template immediately."),
):
    """
    Heuristic mapping from natural language to ``(template, params)`` — **not** an LLM.

    Set query param ``execute=true`` to run the suggested template in one shot (debug only).
    """
    _require_debug_data_api()
    tid, params = suggest_template_from_question(body.question)
    executed = None
    if execute and tid:
        out = execute_constrained_template(tid, params)
        executed = ConstrainedGraphResponse(template_id=out.template_id, rows=out.rows, logs=out.logs)
    return ConstrainedGraphSuggestResponse(template_id=tid, params=params, executed=executed)
