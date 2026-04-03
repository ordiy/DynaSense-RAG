"""
Map domain exceptions to HTTP responses. Registered on the FastAPI app in ``main``.

Design: keep routers thin — raise ``DomainError`` subclasses where appropriate;
handlers translate to stable status codes and JSON bodies for clients.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.core.exceptions import (
    DomainError,
    IngestionError,
    KnowledgeBaseError,
    QueryGuardrailError,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    """Attach handlers so all included routers share consistent error shapes."""

    @app.exception_handler(KnowledgeBaseError)
    async def kb_err(_: Request, exc: KnowledgeBaseError) -> JSONResponse:
        logger.warning("KnowledgeBaseError: %s", exc)
        return JSONResponse(
            status_code=503,
            content={"detail": str(exc) or "Knowledge base unavailable.", "code": "knowledge_base"},
        )

    @app.exception_handler(IngestionError)
    async def ingest_err(_: Request, exc: IngestionError) -> JSONResponse:
        logger.warning("IngestionError: %s", exc)
        return JSONResponse(
            status_code=422,
            content={"detail": str(exc) or "Ingestion failed.", "code": "ingestion"},
        )

    @app.exception_handler(QueryGuardrailError)
    async def guard_err(_: Request, exc: QueryGuardrailError) -> JSONResponse:
        logger.info("QueryGuardrailError: %s", exc)
        return JSONResponse(
            status_code=400,
            content={"detail": str(exc), "code": "guardrail", "blocked": True},
        )

    @app.exception_handler(DomainError)
    async def domain_err(_: Request, exc: DomainError) -> JSONResponse:
        logger.warning("DomainError: %s", exc)
        return JSONResponse(
            status_code=400,
            content={"detail": str(exc) or "Bad request.", "code": "domain"},
        )
