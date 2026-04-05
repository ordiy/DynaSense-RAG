"""Quick embedding + similarity smoke test against PostgreSQL kb_embedding."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if not os.environ.get("DATABASE_URL"):
    raise SystemExit("Set DATABASE_URL (PostgreSQL + pgvector).")

from langchain_google_vertexai import VertexAIEmbeddings

from src.infrastructure.persistence.postgres_connection import init_pool, get_pool
from src.infrastructure.persistence.postgres_schema import ensure_schema
from src.infrastructure.persistence.postgres_vectorstore import PostgresVectorStore

doc_embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
init_pool(os.environ["DATABASE_URL"])
ensure_schema(get_pool())
vectorstore = PostgresVectorStore(get_pool(), doc_embeddings)

q = "What type of organism is commonly used in preparation of foods such as cheese and yogurt?"
docs = vectorstore.similarity_search(q, k=5)
for doc in docs:
    print(doc.metadata.get("doc_id"), doc.page_content[:80])
