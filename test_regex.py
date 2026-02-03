import sys
sys.stdout = open('test_output.txt', 'w', encoding='utf-8')

import re

code = 'theorem add_comm (n m : Nat) : n + m = m + n := by sorry'

# The pattern from the file
THEOREM_PATTERN = re.compile(
    r'^(theorem|lemma|def|class|structure|inductive)\s+(\w+)(?:\s*\[.*?\])?\s*(?::\s*(.+?))?\s*:=',
    re.MULTILINE | re.DOTALL
)

print('Code:', code)
print('Looking for theorems...')

matches = list(THEOREM_PATTERN.finditer(code))
print(f'Number of matches: {len(matches)}')

for match in matches:
    print(f'Match: {match.group()}')
    print(f'  Type: {match.group(1)}')
    print(f'  Name: {match.group(2)}')
    print(f'  Signature: {match.group(3)}')

# Try a simpler pattern
SIMPLE_PATTERN = re.compile(
    r'^(theorem|lemma|def)\s+(\w+)',
    re.MULTILINE
)

print('\nTrying simpler pattern...')
matches = list(SIMPLE_PATTERN.finditer(code))
print(f'Number of matches: {len(matches)}')
for match in matches:
    print(f'  Found: {match.group()}')

print('\nDone', file=sys.__stdout__)
