import re

with open("run_benchmark.py", "r") as f:
    code = f.read()

# Vertex has a strict 20k token limit per request. 
# Batching 200 documents exceeds 20k tokens. Reduce batch_size to 50.

new_code = code.replace("batch_size = 200", "batch_size = 50")

with open("run_benchmark.py", "w") as f:
    f.write(new_code)
