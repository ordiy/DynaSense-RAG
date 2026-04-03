"""Ensure customer portal static asset exists (no FastAPI import)."""
from pathlib import Path


def test_portal_html_exists():
    root = Path(__file__).resolve().parent.parent
    p = root / "src" / "static" / "portal.html"
    assert p.is_file()
    text = p.read_text(encoding="utf-8")
    assert "/api/chat/session" in text
    assert "DynaSense" in text
