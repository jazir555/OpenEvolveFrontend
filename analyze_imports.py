#!/usr/bin/env python3
"""
Comprehensive Import Test Analysis and Fix Script
"""

import json
from collections import defaultdict

# Load all batch files
all_failures = []
for i in range(1, 10):
    try:
        with open(f'import_test_batch{i}.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        failed = data.get('failed', [])
        for item in failed:
            if isinstance(item, dict):
                all_failures.append({
                    'batch': i,
                    'file': item.get('file', ''),
                    'error': item.get('error', '')
                })
    except Exception as e:
        print(f'Error loading batch {i}: {e}')

print(f"Total failures: {len(all_failures)}")

# Group by error type
by_error = defaultdict(list)
for f in all_failures:
    error = f['error']
    if 'cannot import name' in error:
        cat = 'ImportError_cannot_import_name'
    elif 'No module named' in error:
        cat = 'ModuleNotFoundError'
    elif 'AttributeError' in error:
        cat = 'AttributeError'
    elif 'NameError' in error:
        cat = 'NameError'
    elif 'TypeError' in error:
        cat = 'TypeError'
    elif 'SKIPPED' in error or 'Timeout' in error:
        cat = 'Skipped/Timeout'
    elif 'KeyError' in error:
        cat = 'KeyError'
    else:
        cat = 'Other'
    by_error[cat].append(f)

# Extract 'cannot import name' failures
import_failures = by_error['ImportError_cannot_import_name']

# Group by the name being imported
name_pattern = defaultdict(list)
for f in import_failures:
    error = f['error']
    try:
        if "'" in error:
            parts = error.split("'")
            if len(parts) >= 2:
                name = parts[1]
                name_pattern[name].append(f)
    except:
        pass

print('\n=== Top Import Errors (cannot import name) ===\n')
for name, items in sorted(name_pattern.items(), key=lambda x: -len(x[1])):
    print(f'{name}: {len(items)} files')

# Extract ModuleNotFoundError
module_failures = by_error['ModuleNotFoundError']
module_pattern = defaultdict(list)
for f in module_failures:
    error = f['error']
    try:
        if "'" in error:
            parts = error.split("'")
            if len(parts) >= 2:
                name = parts[1]
                module_pattern[name].append(f)
    except:
        pass

print('\n=== ModuleNotFoundError ===\n')
for name, items in sorted(module_pattern.items(), key=lambda x: -len(x[1])):
    print(f'{name}: {len(items)} files')

# Create comprehensive report
report = {
    'total_failures': len(all_failures),
    'by_error_type': {k: len(v) for k, v in by_error.items()},
    'top_missing_names': {k: [item['file'] for item in v] for k, v in sorted(name_pattern.items(), key=lambda x: -len(x[1]))},
    'top_missing_modules': {k: [item['file'] for item in v] for k, v in sorted(module_pattern.items(), key=lambda x: -len(x[1]))},
    'all_failures': all_failures
}

with open('IMPORT_FIXES_FINAL_ROUND.json', 'w') as f:
    json.dump(report, f, indent=2)

print('\n\nReport saved to IMPORT_FIXES_FINAL_ROUND.json')
