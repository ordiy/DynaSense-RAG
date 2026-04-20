"""PostgreSQL feedback store."""

import json
from datetime import datetime, timezone

class FeedbackStore:
    def __init__(self, pool):
        self._pool = pool

    def insert(self, entry: dict) -> None:
        try:
            ts_dt = datetime.fromtimestamp(entry["ts"], tz=timezone.utc)
            tags_json = json.dumps(entry.get("tags", []))
            
            with self._pool.connection() as conn:
                conn.execute(
                    """
                    INSERT INTO feedback (id, ts, conversation_id, query, rating, comment, tags, trace_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        entry["id"],
                        ts_dt,
                        entry.get("conversation_id"),
                        entry["query"],
                        entry["rating"],
                        entry.get("comment"),
                        tags_json,
                        entry.get("trace_id"),
                    )
                )
                conn.commit()
        except Exception:
            pass

    def get_negative(self, limit: int = 200) -> list[dict]:
        with self._pool.connection() as conn:
            cur = conn.execute(
                """
                SELECT id, ts, conversation_id, query, rating, comment, tags, trace_id
                FROM feedback
                WHERE rating = -1
                ORDER BY ts DESC
                LIMIT %s
                """,
                (limit,)
            )
            rows = []
            for row in cur.fetchall():
                rows.append({
                    "id": row[0],
                    "ts": row[1].timestamp() if hasattr(row[1], 'timestamp') else None,
                    "conversation_id": row[2],
                    "query": row[3],
                    "rating": row[4],
                    "comment": row[5],
                    "tags": row[6] if isinstance(row[6], list) else json.loads(row[6]),
                    "trace_id": row[7],
                })
            return rows
