import json

with open('pairs.jsonl', 'r') as f:
    for line in f:
        rec = json.loads(line)
        if '![](https' in rec['problem']:
            print(rec['id'])