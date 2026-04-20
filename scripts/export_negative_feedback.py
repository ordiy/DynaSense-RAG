import argparse
import json
import os
import psycopg

def main():
    parser = argparse.ArgumentParser(description="Export negative feedback")
    parser.add_argument("--out", required=True, help="Output JSONL file")
    parser.add_argument("--limit", type=int, default=200, help="Maximum number of records to export")
    args = parser.parse_args()

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        from src.core.config import get_settings
        db_url = get_settings().database_url

    from psycopg.rows import dict_row
    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT query, trace_id, comment, ts
                FROM feedback
                WHERE rating = -1
                ORDER BY ts DESC
                LIMIT %s
                """,
                (args.limit,)
            )
            rows = cur.fetchall()

    with open(args.out, "w") as f:
        for row in rows:
            record = {
                "question": row["query"],
                "trace_id": row["trace_id"],
                "comment": row["comment"],
                "ts": str(row["ts"]),
            }
            f.write(json.dumps(record) + "\n")
            
    print(f"Exported {len(rows)} negative feedback records to {args.out}")

if __name__ == "__main__":
    main()
