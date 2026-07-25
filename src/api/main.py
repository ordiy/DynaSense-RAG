"""
FastAPI application factory.

LangSmith **must** initialize before ``rag_core`` / LangChain imports used by routers.
Storage (PostgreSQL pool) is initialised in the lifespan handler so the module can be
imported safely in unit-test environments that have no live database.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from src.core.langsmith_tracing import init_langsmith_tracing

init_langsmith_tracing()

from src.api.auth_middleware import AuthGateMiddleware  # noqa: E402
from src.api.error_handlers import register_exception_handlers  # noqa: E402
from src.api.routers import (  # noqa: E402
    analytics,
    auth,
    chat,
    debug_routes,
    eval,
    feedback,
    ingest,
    pages,
    session,
    whatif,
)
from src.core.config import get_settings  # noqa: E402

STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: initialise PostgreSQL pool + RAG storage. Shutdown: close pool."""
    import logging

    logger = logging.getLogger(__name__)
    try:
        from src.rag_core import setup_storage

        setup_storage()
    except Exception as exc:
        logger.warning("Storage init failed at startup: %s", exc)
    yield
    try:
        from src.infrastructure.persistence.postgres_connection import close_pool

        close_pool()
        logger.info("PostgreSQL connection pool closed.")
    except Exception as exc:
        logger.debug("Pool close on shutdown: %s", exc)


def create_app() -> FastAPI:
    application = FastAPI(title="MAP-RAG MVP API", lifespan=lifespan)
    register_exception_handlers(application)
    s = get_settings()
    # Starlette runs the *last* added middleware outermost. Session must wrap AuthGate
    # so ``request.session`` is available inside the auth gate.
    application.add_middleware(AuthGateMiddleware)
    application.add_middleware(
        SessionMiddleware,
        secret_key=s.auth_secret_key,
        session_cookie="map_rag_session",
        https_only=s.auth_session_https_only,
        same_site="lax",
        max_age=86400 * 7,
    )

    os.makedirs(STATIC_DIR, exist_ok=True)
    application.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    application.include_router(pages.router)
    application.include_router(auth.router)
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
