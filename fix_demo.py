import re

with open('demo_mcts_mdap.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Split by lines
lines = content.split('\n')
fixed_lines = []

i = 0
while i < len(lines):
    line = lines[i]

    # Check for print(f" pattern
    if re.search(r'print\(f"', line):
        # Check if line ends with backslash
        stripped = line.strip()
        if stripped.endswith('\\\\'):
            # This is the problem pattern
            # Get indent
            indent = len(line) - len(line.lstrip())

            # Skip this line and fix next line
            i += 1
            if i < len(lines):
                next_line = lines[i].rstrip()
                # Remove trailing ""
                if next_line.endswith('""'):
                    next_line = next_line[:-2]
                # Add proper print with f-string
                fixed_lines.append(' ' * indent + f'print(f"{next_line}")')
                i += 1
                continue

    # Otherwise just add the line
    fixed_lines.append(line)
    i += 1

with open('demo_mcts_mdap.py', 'w', encoding='utf-8') as f:
    f.write('\n'.join(fixed_lines))

print("Fixed")
