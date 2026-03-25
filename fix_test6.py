import re

with open("run_benchmark.py", "r") as f:
    code = f.read()

# Langchain wrapper expects metadata in a specific format when constructing vectorstore manually.
# In `tbl = db_lance.create_table(TABLE_NAME, data=all_embeddings)` we just added doc_id to root. 
# Langchain LanceDB wrapper looks in `metadata` column for filtering/getting ids.
# Let's fix the ingestion shape for manual creation.

replacement = """
    for j, d in enumerate(batch_docs):
        all_embeddings.append({"vector": embeds[j], "text": d.page_content, "metadata": {"doc_id": d.metadata["doc_id"]}})
"""

code = code.replace("""    for j, d in enumerate(batch_docs):
        all_embeddings.append({"vector": embeds[j], "text": d.page_content, "doc_id": d.metadata["doc_id"]})""", replacement)

with open("run_benchmark.py", "w") as f:
    f.write(code)
