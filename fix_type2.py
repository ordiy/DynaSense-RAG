with open("run_benchmark.py", "r") as f:
    code = f.read()

code = code.replace("from langchain_community.vectorstores import LanceDB", "from langchain_community.vectorstores import LanceDB\nimport numpy as np")

# We will check if tests actually point to correct expected_id. Wait, expected_doc_id in run_benchmark is doc_id.
# Let's write a small print to verify expected_id vs retrieved_id in calc_recall

new_calc = """def calc_recall(retrieved_docs, expected_id, k):
    retrieved_ids = []
    for d in retrieved_docs[:k]:
        try:
            retrieved_ids.append(int(d.metadata.get("doc_id", -1)))
        except:
            pass
    if expected_id in retrieved_ids:
        return 1
    return 0"""

code = code.replace("""def calc_recall(retrieved_docs, expected_id, k):
    retrieved_ids = [int(d.metadata.get("doc_id", -1)) for d in retrieved_docs[:k]]
    return 1 if int(expected_id) in retrieved_ids else 0""", new_calc)

# Let's inspect test queries
code = code.replace("""    if idx % 10 == 0 and idx > 0:
        print(f"Processed {idx} queries...")""", """    if idx % 10 == 0 and idx > 0:
        print(f"Processed {idx} queries...")
    if idx == 0:
        print(f"DEBUG: Q: {query}, expected: {expected_id}")
        base_ids = [int(d.metadata.get("doc_id", -1)) for d in retriever.invoke(query)[:5]]
        print(f"DEBUG: base_ids: {base_ids}")""")

with open("run_benchmark.py", "w") as f:
    f.write(code)
