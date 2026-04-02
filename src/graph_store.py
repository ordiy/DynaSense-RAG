"""
Neo4j graph layer for Hybrid RAG (MVP).

Schema (simplified):
  (:Entity {norm, name})     — normalized key for MERGE
  (:Chunk {id, parent_id, source, preview})
  (:Entity)-[:REL {type, chunk_id, source}]->(:Entity)

Chunk nodes link extraction provenance; edges carry chunk_id for citation / traceability.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "changeme")

_driver = None
_schema_ready = False


def _norm_key(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s[:512]


def get_driver():
    global _driver
    if _driver is not None:
        return _driver
    try:
        from neo4j import GraphDatabase
    except ImportError:
        logger.warning("neo4j package not installed; graph features disabled.")
        return None
    try:
        _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        _driver.verify_connectivity()
        logger.info("Neo4j connected: %s", NEO4J_URI)
    except Exception as e:
        logger.warning("Neo4j unavailable (%s); graph features disabled.", e)
        _driver = None
    return _driver


def ensure_schema() -> None:
    global _schema_ready
    drv = get_driver()
    if not drv or _schema_ready:
        return
    with drv.session() as session:
        session.run(
            "CREATE CONSTRAINT entity_norm IF NOT EXISTS FOR (e:Entity) REQUIRE e.norm IS UNIQUE"
        )
    _schema_ready = True


def merge_triple(
    subject: str,
    predicate: str,
    obj: str,
    chunk_id: str,
    source: str,
) -> None:
    """MERGE two entities and a typed relationship with provenance."""
    drv = get_driver()
    if not drv:
        return
    ensure_schema()
    sn = _norm_key(subject)
    on = _norm_key(obj)
    if not sn or not on:
        return
    pred = (predicate or "RELATED_TO").strip()[:128] or "RELATED_TO"
    with drv.session() as session:
        session.run(
            """
            MERGE (a:Entity {norm: $sn})
            ON CREATE SET a.name = $sname
            MERGE (b:Entity {norm: $on})
            ON CREATE SET b.name = $oname
            MERGE (a)-[r:REL {type: $ptype, chunk_id: $cid}]->(b)
            ON CREATE SET r.source = $src
            SET r.source = $src
            """,
            sn=sn,
            sname=subject.strip()[:512],
            on=on,
            oname=obj.strip()[:512],
            cid=chunk_id,
            src=source[:512],
            ptype=pred,
        )


def query_relationships_by_keywords(keywords: list[str], limit: int = 40) -> list[dict[str, Any]]:
    """Return relationship rows matching any keyword in entity names or relation type."""
    drv = get_driver()
    if not drv:
        return []
    ensure_schema()
    kws = [k.strip() for k in keywords if k and len(k.strip()) > 1][:20]
    if not kws:
        return []
    rows: list[dict[str, Any]] = []
    with drv.session() as session:
        result = session.run(
            """
            UNWIND $kws AS kw
            MATCH (a:Entity)-[r:REL]->(b:Entity)
            WHERE toLower(a.name) CONTAINS toLower(kw)
               OR toLower(b.name) CONTAINS toLower(kw)
               OR toLower(r.type) CONTAINS toLower(kw)
            RETURN DISTINCT a.name AS an, r.type AS rt, b.name AS bn, r.chunk_id AS cid, r.source AS src
            LIMIT $lim
            """,
            kws=kws,
            lim=int(limit),
        )
        for rec in result:
            rows.append(
                {
                    "subject": rec["an"],
                    "predicate": rec["rt"],
                    "object": rec["bn"],
                    "chunk_id": rec["cid"],
                    "source": rec["src"],
                }
            )
    return rows


def global_graph_summary() -> str:
    """MVP 'community/global' signal: entity and relationship counts + sample entities."""
    drv = get_driver()
    if not drv:
        return ""
    ensure_schema()
    with drv.session() as session:
        n_ent = session.run("MATCH (e:Entity) RETURN count(e) AS c").single()["c"]
        n_rel = session.run("MATCH ()-[r:REL]->() RETURN count(r) AS c").single()["c"]
        names = session.run(
            "MATCH (e:Entity) RETURN e.name AS n ORDER BY e.name LIMIT 30"
        )
        sample = [r["n"] for r in names if r["n"]]
    lines = [
        f"[Graph summary] Entities: {n_ent}, Relationships: {n_rel}.",
        "Sample entity names: " + ", ".join(sample[:15]) + ("..." if len(sample) > 15 else ""),
    ]
    return "\n".join(lines)


def linearize_rows(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    parts = []
    for r in rows:
        parts.append(
            f"{r.get('subject','')} —[{r.get('predicate','REL')}]→ {r.get('object','')} "
            f"(chunk_id={r.get('chunk_id','')}, source={r.get('source','')})"
        )
    return "\n".join(parts)
