"""
Optional HTTP smoke tests (imports full app → rag_core → Vertex).

Skipped on GitHub Actions and when GCP credentials are absent.
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("CI") == "true" or not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
    reason="Full stack: requires GOOGLE_APPLICATION_CREDENTIALS and Vertex (not in default CI).",
)


def test_openapi_and_pages_exist():
    from fastapi.testclient import TestClient

    from src.app import app

    c = TestClient(app)
    assert c.get("/").status_code == 200
    assert c.get("/demo").status_code == 200
    assert c.get("/portal", follow_redirects=False).status_code == 302
    spec = c.get("/openapi.json").json()
    paths = spec.get("paths", {})
    assert "/api/chat" in paths
    assert "/api/chat/stream" in paths
    assert "/api/upload" in paths
    assert "/api/feedback" in paths
    assert "/api/whatif/loan/compare" in paths
    assert "/api/debug/lancedb/summary" in paths
