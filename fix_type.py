with open("run_benchmark.py", "r") as f:
    code = f.read()

old_calc = """def calc_recall(retrieved_docs, expected_id, k):
    # LanceDB inserts metadata correctly, doc_id should be in there
    retrieved_ids = [d.metadata.get("doc_id") for d in retrieved_docs[:k]]
    return 1 if expected_id in retrieved_ids else 0"""

new_calc = """def calc_recall(retrieved_docs, expected_id, k):
    retrieved_ids = [int(d.metadata.get("doc_id", -1)) for d in retrieved_docs[:k]]
    return 1 if int(expected_id) in retrieved_ids else 0"""

code = code.replace(old_calc, new_calc)

with open("run_benchmark.py", "w") as f:
    f.write(code)
