"""Unit tests for debug_data helpers (no FastAPI / Vertex import)."""
import os

from src import debug_data


def test_lancedb_summary_local_path():
    uri = os.path.join(os.path.dirname(__file__), "..", "data", "lancedb_store")
    uri = os.path.abspath(uri)
    prev = os.environ.get("LANCEDB_URI")
    os.environ["LANCEDB_URI"] = uri
    try:
        out = debug_data.lancedb_summary()
        assert "tables" in out
        assert isinstance(out["tables"], list)
    finally:
        if prev is None:
            os.environ.pop("LANCEDB_URI", None)
        else:
            os.environ["LANCEDB_URI"] = prev


def test_neo4j_keyword_search_empty_keywords():
    rows, err = debug_data.neo4j_keyword_search([], limit=10)
    assert rows == []
    assert err
    el = err.lower()
    assert "keyword" in el or "neo4j" in el or "driver" in el
