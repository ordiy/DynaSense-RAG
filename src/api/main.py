"""
FastAPI application factory.

LangSmith **must** initialize before ``rag_core`` / LangChain imports used by routers.
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.core.langsmith_tracing import init_langsmith_tracing

init_langsmith_tracing()

from src.api.error_handlers import register_exception_handlers
from src.api.routers import (  # noqa: E402
    analytics,
    chat,
    debug_routes,
    eval,
    feedback,
    ingest,
    pages,
    session,
    whatif,
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")


def create_app() -> FastAPI:
    application = FastAPI(title="MAP-RAG MVP API")
    register_exception_handlers(application)
    os.makedirs(STATIC_DIR, exist_ok=True)
    application.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    application.include_router(pages.router)
    application.include_router(ingest.router)
    application.include_router(analytics.router)
    application.include_router(debug_routes.router)
    application.include_router(chat.router)
    application.include_router(session.router)
    application.include_router(eval.router)
    application.include_router(feedback.router)
    application.include_router(whatif.router)
    return application


app = create_app()
