"""Unit tests for query anchor extraction and document filtering (no LanceDB)."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from src.core import query_anchors as qa


def test_extract_citic_bank_and_stock_code():
    q = "查询：中信银行2025年关联交易，分析风险；代码601998"
    anchors = qa.extract_anchor_keywords(q)
    assert "中信银行" in anchors
    assert "601998" in anchors


def test_extract_company_limited():
    q = "请说明招商银行股份有限公司2024年事项"
    anchors = qa.extract_anchor_keywords(q)
    assert any("招商银行" in a or "股份有限公司" in a for a in anchors)


def test_no_anchor_for_generic_question():
    q = "总结一下关联交易的一般原则"
    assert qa.extract_anchor_keywords(q) == []


def test_filter_keeps_matching_docs(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("QUERY_ANCHOR_FILTER", "true")
    docs = [
        Document(page_content="中信银行公告内容"),
        Document(page_content="胖冬瓜期货项目"),
    ]
    kept, logs = qa.filter_documents_by_query_anchors("中信银行2025年报告", docs)
    assert len(kept) == 1
    assert "中信银行" in kept[0].page_content
    assert any("Anchor filter" in line for line in logs)


def test_filter_fail_open_when_nothing_matches(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("QUERY_ANCHOR_FILTER", "true")
    docs = [
        Document(page_content="完全无关的文本"),
    ]
    kept, logs = qa.filter_documents_by_query_anchors("中信银行年报", docs)
    assert len(kept) == 1
    assert "fail-open" in "\n".join(logs).lower() or "fail-open" in " ".join(logs).lower()


def test_filter_disabled_passthrough(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("QUERY_ANCHOR_FILTER", "false")
    docs = [Document(page_content="B only")]
    kept, logs = qa.filter_documents_by_query_anchors("中信银行", docs)
    assert len(kept) == 1
    assert any("disabled" in x.lower() for x in logs)
