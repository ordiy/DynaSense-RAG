import lancedb
from langchain_community.vectorstores import LanceDB
from langchain_google_vertexai import VertexAIEmbeddings

doc_embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
q = "What type of organism is commonly used in preparation of foods such as cheese and yogurt?"
vec = doc_embeddings.embed_query(q)
db_lance = lancedb.connect("/tmp/lancedb_benchmark")
vectorstore = LanceDB(connection=db_lance, embedding=doc_embeddings, table_name="benchmark_docs")

# Get raw query results without retriever
res = vectorstore.similarity_search_with_score(q, k=5)
for doc, score in res:
    print(doc.metadata.get('doc_id'), score)
