with open("run_benchmark.py", "r") as f:
    code = f.read()

# The issue is Vertex API batches limit or quota. Let's look at the database.
code = code.replace("""vectorstore = None

for i in range(0, len(documents), batch_size):
    batch = documents[i:i+batch_size]""", """vectorstore = None

print("Checking existing DB size...")
try:
    tbl = db_lance.open_table(TABLE_NAME)
    print("Table size:", tbl.to_pandas().shape)
except:
    pass

for i in range(0, len(documents), batch_size):
    batch = documents[i:i+batch_size]""")

with open("run_benchmark.py", "w") as f:
    f.write(code)
