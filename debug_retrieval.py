import lancedb
from langchain_community.vectorstores import LanceDB
from langchain_google_vertexai import VertexAIEmbeddings

doc_embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
db_lance = lancedb.connect("/tmp/lancedb_benchmark")
vectorstore = LanceDB(connection=db_lance, embedding=doc_embeddings, table_name="benchmark_docs")
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

res = retriever.invoke("What type of organism is commonly used in preparation of foods such as cheese and yogurt?")
print(res)
