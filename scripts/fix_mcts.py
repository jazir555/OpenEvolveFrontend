with open('demo_mcts_mdap.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

fixed = []
i = 0
while i < len(lines):
    line = lines[i]
    # Check for pattern ending with backslash
    if line.rstrip().endswith('\\'):
        # Remove backslash
        fixed_line = line.rstrip()[:-1] + '\n'
        fixed.append(fixed_line)
        i += 1
        # Process next line - remove trailing "" if present
        if i < len(lines):
            next_line = lines[i]
            if next_line.rstrip().endswith('""'):
                fixed_line = next_line.rstrip()[:-2] + '\n'
                fixed.append(fixed_line)
            else:
                fixed.append(next_line)
            i += 1
    else:
        fixed.append(line)
        i += 1

with open('demo_mcts_mdap.py', 'w', encoding='utf-8') as f:
    f.writelines(fixed)

print("Fixed all backslash patterns")
