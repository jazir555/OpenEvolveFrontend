"""
Debug the source code to see exactly what Python is parsing
"""

import problem_decomposition
import inspect

# Get the class source
cls = problem_decomposition.ProblemDecomposer
source = inspect.getsource(cls)

# Print the end of the source to see where it stops
lines = source.split('\n')
print("Last 20 lines of parsed source:")
for i, line in enumerate(lines[-20:], len(lines) - 19):
    print(f"{i}: {line}")

print(f"\nTotal lines in parsed source: {len(lines)}")

# Check if the source ends abruptly
if not source.strip().endswith('"""') and not source.strip().endswith('pass'):
    print("⚠️  Source appears to end abruptly")
else:
    print("✅ Source appears to end properly")