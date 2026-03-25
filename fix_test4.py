import re

with open("run_benchmark.py", "r") as f:
    code = f.read()

# It worked! Vertex limits payload to 250 instances.
# I will batch the documents and embed them using the correct manual batching array logic.

replacement = """
print("Embedding in batches via VertexAI...")
all_embeddings = []
from tqdm import tqdm
batch_size = 200
for i in tqdm(range(0, len(documents), batch_size)):
    batch_docs = documents[i:i+batch_size]
    # embed explicitly
    batch_texts = [d.page_content for d in batch_docs]
    embeds = doc_embeddings.embed_documents(batch_texts)
    for j, d in enumerate(batch_docs):
        all_embeddings.append({"vector": embeds[j], "text": d.page_content, "doc_id": d.metadata["doc_id"]})

# Create LanceDB table manually
try:
    db_lance.drop_table(TABLE_NAME)
except:
    pass

tbl = db_lance.create_table(TABLE_NAME, data=all_embeddings)
vectorstore = LanceDB(connection=db_lance, table_name=TABLE_NAME, embedding=doc_embeddings)

print(f"Table size: {db_lance.open_table(TABLE_NAME).to_pandas().shape}")
retriever = vectorstore.as_retriever"""

new_code = re.sub(
    r'print\("Embedding in batches via VertexAI\.\.\."\).*?retriever = vectorstore\.as_retriever',
    replacement,
    code,
    flags=re.DOTALL
)

# Also scale up testing 
new_code = new_code.replace("NUM_DOCS_TO_INDEX = 1000", "NUM_DOCS_TO_INDEX = 1000")

with open("run_benchmark.py", "w") as f:
    f.write(new_code)
