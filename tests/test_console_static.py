"""Static checks for demo console layout and login page."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "src" / "static"


def test_login_html_exists():
    p = STATIC / "login.html"
    assert p.is_file()
    text = p.read_text(encoding="utf-8")
    assert "/api/auth/login" in text
    assert "MAP-RAG" in text


def test_index_has_sidebar_not_top_pills():
    text = (STATIC / "index.html").read_text(encoding="utf-8")
    assert 'id="appSidebar"' in text or 'class="sidebar"' in text
    assert 'id="sessionPanel"' in text
    assert 'id="sessionList"' in text
    assert "nav-pills-custom" not in text
    assert "app-shell" in text
    assert "/api/chat/sessions" in text
    assert "/api/auth/logout" in text


def test_schema_has_conversation_tables():
    from src.infrastructure.persistence.postgres_schema import _conversation_tables_ddl

    ddl = "\n".join(_conversation_tables_ddl())
    assert "chat_conversation" in ddl
    assert "chat_message" in ddl
