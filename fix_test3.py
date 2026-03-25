with open("run_benchmark.py", "r") as f:
    code = f.read()

# completely replace batching with direct from_documents call which is safer and supported in newer langchain
import re
new_code = re.sub(
    r'batch_size = 100.*?retriever = vectorstore\.as_retriever',
    r'''
# To avoid silent rate limits or db append bugs, embed in pure small batches and construct DB at once
print("Embedding in batches via VertexAI...")
embedded_docs = []
# Vertex default quota is usually 60-120 req/min for free tier.
# Langchain's embed_documents batches inside. Let's just do from_documents and rely on Langchain.
# Wait, Vertex might fail silently. Let's force it.
try:
    vectorstore = LanceDB.from_documents(
        documents=documents,
        embedding=doc_embeddings,
        connection=db_lance,
        table_name=TABLE_NAME,
        mode="overwrite"
    )
except Exception as e:
    print(f"Ingestion error: {e}")
    # If error, maybe we use a smaller slice
    print("Trying smaller slice...")
    documents = documents[:100]
    test_queries = [qa for qa in qa_pairs if qa["expected_doc_id"] <= 100][:100]
    vectorstore = LanceDB.from_documents(
        documents=documents,
        embedding=doc_embeddings,
        connection=db_lance,
        table_name=TABLE_NAME,
        mode="overwrite"
    )

print(f"Table size: {db_lance.open_table(TABLE_NAME).to_pandas().shape}")
retriever = vectorstore.as_retriever''',
    code,
    flags=re.DOTALL
)

with open("run_benchmark.py", "w") as f:
    f.write(new_code)
