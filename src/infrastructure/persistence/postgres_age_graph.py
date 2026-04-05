"""
Apache AGE Cypher graph — replaces Neo4j and relational ``kg_triple`` when AGE is available.

Vertices: ``Entity`` with ``norm``, ``name``. Edges: ``REL`` with ``type``, ``chunk_id``, ``source``.
Matches the semantics of ``graph_store.merge_triple`` / ``query_relationships_by_keywords``.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.infrastructure.persistence.postgres_age_setup import prepare_connection, run_cypher

logger = logging.getLogger(__name__)


def _norm_key(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s[:512]


def _esc(s: str) -> str:
    """Escape single-quoted Cypher string literals."""
    return s.replace("\\", "\\\\").replace("'", "\\'")


def merge_triple_age(
    subject: str,
    predicate: str,
    obj: str,
    chunk_id: str,
    source: str,
    pool,
    graph_name: str,
) -> None:
    """MERGE ``Entity`` nodes and a ``REL`` edge (properties set after MERGE for AGE compatibility)."""
    sn = _norm_key(subject)
    on = _norm_key(obj)
    if not sn or not on:
        return
    pred = (predicate or "RELATED_TO").strip()[:128] or "RELATED_TO"
    sname = _esc(subject.strip()[:512])
    oname = _esc(obj.strip()[:512])
    esn = _esc(sn)
    eon = _esc(on)
    ep = _esc(pred)
    ecid = _esc(chunk_id)
    esrc = _esc(source[:512])
    cy = f"""
    MERGE (a:Entity {{norm: '{esn}'}})
    SET a.name = '{sname}'
    MERGE (b:Entity {{norm: '{eon}'}})
    SET b.name = '{oname}'
    MERGE (a)-[r:REL]->(b)
    SET r.type = '{ep}', r.chunk_id = '{ecid}', r.source = '{esrc}'
    RETURN true
    """
    run_cypher(pool, graph_name, cy.strip(), [("v", "agtype")])


def query_relationships_by_keywords_age(
    keywords: list[str],
    limit: int,
    pool,
    graph_name: str,
) -> list[dict[str, Any]]:
    kws = [k.strip() for k in keywords if k and len(k.strip()) > 1][:20]
    if not kws:
        return []
    parts = []
    for kw in kws:
        ek = _esc(kw.lower())
        parts.append(
            f"(toLower(a.name) CONTAINS '{ek}' OR toLower(b.name) CONTAINS '{ek}' OR toLower(r.type) CONTAINS '{ek}')"
        )
    where_or = " OR ".join(parts)
    cy = f"""
    MATCH (a:Entity)-[r:REL]->(b:Entity)
    WHERE {where_or}
    RETURN a.name, r.type, b.name, r.chunk_id, r.source
    LIMIT {int(limit)}
    """
    rows = run_cypher(
        pool,
        graph_name,
        cy.strip(),
        [
            ("an", "agtype"),
            ("rt", "agtype"),
            ("bn", "agtype"),
            ("cid", "agtype"),
            ("src", "agtype"),
        ],
    )
    out: list[dict[str, Any]] = []
    for row in rows:
        def _ag(v: Any) -> str:
            if v is None:
                return ""
            t = str(v)
            if t.startswith('"') and t.endswith('"'):
                return t[1:-1].replace('\\"', '"')
            return t

        out.append(
            {
                "subject": _ag(row[0]),
                "predicate": _ag(row[1]),
                "object": _ag(row[2]),
                "chunk_id": _ag(row[3]),
                "source": _ag(row[4]),
            }
        )
    return out


def _ag_to_int(v: Any) -> int:
    t = str(v).strip()
    if t.isdigit():
        return int(t)
    for ch in t:
        if ch.isdigit():
            s = "".join(x for x in t if x.isdigit())
            return int(s) if s else 0
    return 0


def global_graph_summary_age(pool, graph_name: str) -> str:
    """Entity count, edge count, sample names (Cypher)."""
    r_ec = run_cypher(pool, graph_name, "MATCH (e:Entity) RETURN count(e)", [("c", "agtype")])
    r_rc = run_cypher(pool, graph_name, "MATCH ()-[r:REL]->() RETURN count(r)", [("c", "agtype")])
    n_ent = _ag_to_int(r_ec[0][0]) if r_ec else 0
    n_rel = _ag_to_int(r_rc[0][0]) if r_rc else 0
    cy3 = "MATCH (e:Entity) RETURN e.name ORDER BY e.name LIMIT 30"
    r3 = run_cypher(pool, graph_name, cy3.strip(), [("n", "agtype")])
    sample = []
    for row in r3:
        t = str(row[0])
        if t.startswith('"') and t.endswith('"'):
            t = t[1:-1].replace('\\"', '"')
        if t:
            sample.append(t)
    lines = [
        f"[Graph summary] Entities: {n_ent}, Relationships: {n_rel}.",
        "Sample entity names: " + ", ".join(sample[:15]) + ("..." if len(sample) > 15 else ""),
    ]
    return "\n".join(lines)


def reset_age_graph_if_configured(pool, graph_name: str) -> None:
    """Drop and recreate an empty graph (used on TRUNCATE / benchmark reset)."""
    import re

    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", graph_name):
        raise ValueError(f"Invalid graph name: {graph_name!r}")
    with pool.connection() as conn:
        prepare_connection(conn)
        try:
            conn.execute(f"SELECT drop_graph('{graph_name}', true)")
        except Exception as e:
            logger.debug("drop_graph: %s", e)
        try:
            conn.execute(f"SELECT create_graph('{graph_name}')")
        except Exception as e:
            logger.exception("create_graph after drop failed: %s", e)
            raise
        conn.commit()


def ping_age(pool, graph_name: str) -> bool:
    try:
        run_cypher(pool, graph_name, "RETURN 1", [("x", "agtype")])
        return True
    except Exception as e:
        logger.debug("AGE ping failed: %s", e)
        return False
