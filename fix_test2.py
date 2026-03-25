with open("run_benchmark.py", "r") as f:
    code = f.read()

# Fix the bug in VectorStore population
code = code.replace("""    if vectorstore is None:
        vectorstore = LanceDB.from_documents(
            documents=batch,
            embedding=doc_embeddings,
            connection=db_lance,
            table_name=TABLE_NAME,
            mode="overwrite"
        )
    else:
        vectorstore.add_documents(batch)""", """    # Vertex AI might return exactly the same 0-vector if it hits quota silently, or LanceDB add_documents bug.
    # We create the LanceDB wrapper ONCE, then add.
    if vectorstore is None:
        vectorstore = LanceDB.from_documents(
            documents=batch,
            embedding=doc_embeddings,
            connection=db_lance,
            table_name=TABLE_NAME,
            mode="overwrite"
        )
    else:
        # LanceDB 0.5 wrapper can just take LanceDB instance
        vectorstore.add_documents(batch)""")
        
with open("run_benchmark.py", "w") as f:
    f.write(code)

# Actually, the simplest fix is to just pass all documents to from_documents. Vertex API client handles batching internally up to 250.
# Let's completely replace the ingestion block.
