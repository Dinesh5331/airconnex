import json
with open('preprocessing1.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
for i, cell in enumerate(nb['cells']):
    ctype = cell['cell_type']
    src = ''.join(cell['source'])
    print(f"=== Cell {i} [{ctype}] ===")
    print(src[:800])
    print()
