#!/usr/bin/env python3
"""Analyze import errors and categorize them."""

import json

with open('import_error_scan_results.json') as f:
    errors = json.load(f)

# Filter to only real project files
real_errors = []
for e in errors:
    path = e['file']
    # Skip templates (contain {{ or {%)
    # Skip venv files
    # Skip core-projects sub-dependencies
    if 'openevolve_test_env' in path:
        continue
    if 'core-projects' in path and 'crewAI' in path and 'templates' in path:
        continue
    if '{{' in path or '{%' in path:
        continue
    real_errors.append(e)

print(f'Real project errors: {len(real_errors)}')
for e in real_errors:
    print(f"  {e['file']}:{e.get('line', '?')} - {e['error'][:80]}")

# Save filtered results
with open('filtered_errors.json', 'w') as f:
    json.dump(real_errors, f, indent=2)
