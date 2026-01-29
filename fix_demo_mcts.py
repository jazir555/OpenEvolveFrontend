#!/usr/bin/env python3
"""Fix demo_mcts_mdap.py f-string issues."""

with open('demo_mcts_mdap.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

fixed_lines = []
i = 0

while i < len(lines):
    line = lines[i]

    # Check for pattern: print(f"\  followed by line ending with """)
    stripped = line.rstrip()
    if stripped.endswith('\\') and i < len(lines) - 1:
        next_line = lines[i + 1].rstrip()

        if next_line.endswith('""'):
            # Fix this multiline print
            # Remove backslash from first line
            fixed = stripped[:-1]
            # Remove two quotes from second line
            next_fixed = next_line[:-2]

            # Combine if it's a print statement
            if 'print(f"' in fixed:
                combined = fixed + next_fixed + '"'
                fixed_lines.append(combined + '\n')
                i += 2
                continue

    fixed_lines.append(line)
    i += 1

with open('demo_mcts_mdap.py', 'w', encoding='utf-8') as f:
    f.writelines(fixed_lines)

print("Fixed demo_mcts_mdap.py")
