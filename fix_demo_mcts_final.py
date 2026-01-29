#!/usr/bin/env python3
with open('demo_mcts_mdap.py', 'r') as f:
    content = f.read()
content = content.replace('""")', ')')
with open('demo_mcts_mdap.py', 'w') as f:
    f.write(content)
print("Fixed")
