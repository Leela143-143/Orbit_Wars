with open("HTMRL/temporal_memory.py", "r") as f:
    code = f.read()

code = code.replace("for ind in random.sample(unconnected, count):", "for ind in random.sample(list(unconnected), count):")

with open("HTMRL/temporal_memory.py", "w") as f:
    f.write(code)
