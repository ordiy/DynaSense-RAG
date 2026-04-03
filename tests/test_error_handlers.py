"""HTTP mapping for domain exceptions (minimal FastAPI app, no rag_core)."""

from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from src.api.error_handlers import register_exception_handlers
from src.core.exceptions import DomainError, KnowledgeBaseError, QueryGuardrailError

router = APIRouter()


@router.get("/kb")
def kb_fail():
    raise KnowledgeBaseError("Vector store unavailable.")


@router.get("/guard")
def guard_fail():
    raise QueryGuardrailError("blocked")


@router.get("/domain")
def domain_fail():
    raise DomainError("invalid request")


def _client() -> TestClient:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(router)
    return TestClient(app)


def test_knowledge_base_maps_to_503():
    r = _client().get("/kb")
    assert r.status_code == 503
    body = r.json()
    assert body.get("code") == "knowledge_base"


def test_guardrail_maps_to_400():
    r = _client().get("/guard")
    assert r.status_code == 400
    assert r.json().get("code") == "guardrail"
    assert r.json().get("blocked") is True


def test_domain_maps_to_400():
    r = _client().get("/domain")
    assert r.status_code == 400
    assert r.json().get("code") == "domain"
