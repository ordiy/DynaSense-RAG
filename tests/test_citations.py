"""Unit tests for citation extraction from context strings (no Vertex)."""

from src.core.citations import build_citations_from_context


def test_empty_context():
    assert build_citations_from_context([]) == []
    assert build_citations_from_context(None) == []


def test_source_prefix_parsed():
    block = "[Source: policy.pdf]\nFirst line of content.\nMore text."
    cites = build_citations_from_context([block])
    assert len(cites) == 1
    assert cites[0]["index"] == 1
    assert cites[0]["source"] == "policy.pdf"
    assert "policy.pdf" in cites[0]["label"]
    assert "First line" in cites[0]["preview"]


def test_graph_marker_label():
    block = "[Graph retrieval — linearized triples]\nA -> B"
    cites = build_citations_from_context([block])
    assert cites[0]["source"] == "graph"


def test_multiple_blocks_indexed():
    cites = build_citations_from_context(["a", "b"])
    assert cites[0]["index"] == 1
    assert cites[1]["index"] == 2
