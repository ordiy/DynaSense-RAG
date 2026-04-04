"""Tests for whitelisted graph templates (no arbitrary Cypher)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.core.graph_constrained_queries import (
    ConstrainedGraphResult,
    execute_constrained_template,
    suggest_template_from_question,
)


def test_unknown_template_raises():
    with pytest.raises(ValueError, match="unknown template"):
        execute_constrained_template("drop_database", {})


def test_edges_from_entity_validates_empty_string():
    with pytest.raises(ValueError):
        execute_constrained_template("edges_from_entity", {"name_substring": "   "})


@patch("src.core.graph_constrained_queries.get_driver")
@patch("src.core.graph_constrained_queries.query_relationships_by_keywords")
def test_edges_from_entity_calls_keyword_search(mock_q, mock_drv):
    mock_drv.return_value = MagicMock()
    mock_q.return_value = [{"subject": "A", "predicate": "x", "object": "B", "chunk_id": "c", "source": "s"}]
    out = execute_constrained_template(
        "edges_from_entity",
        {"name_substring": "中信银行", "limit": 10},
    )
    assert isinstance(out, ConstrainedGraphResult)
    assert out.template_id == "edges_from_entity"
    assert len(out.rows) == 1
    mock_q.assert_called_once_with(["中信银行"], limit=10)


@patch("src.core.graph_constrained_queries.get_driver")
def test_edges_from_entity_no_driver(mock_drv):
    mock_drv.return_value = None
    out = execute_constrained_template("edges_from_entity", {"name_substring": "test"})
    assert out.rows == []


def test_multi_keyword_requires_list():
    with pytest.raises(ValueError, match="list"):
        execute_constrained_template("multi_keyword_edges", {"keywords": "not-a-list"})


def test_suggest_global_summary():
    tid, p = suggest_template_from_question("请给出图谱摘要")
    assert tid == "graph_global_summary"
    assert p == {}


def test_suggest_entity_phrase():
    tid, p = suggest_template_from_question("与中信银行相关的边有哪些")
    assert tid == "edges_from_entity"
    assert p.get("name_substring") == "中信银行"


def test_suggest_keywords_line():
    tid, p = suggest_template_from_question("keywords: 中信, 关联交易")
    assert tid == "multi_keyword_edges"
    assert "中信" in p["keywords"]

