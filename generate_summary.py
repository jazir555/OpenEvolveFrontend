#!/usr/bin/env python3
"""Generate summary of import errors."""
import json
from collections import Counter

with open('import_errors_batch_2.json', 'r') as f:
    data = json.load(f)

# Summary
print('='*70)
print('BATCH 2 IMPORT ERROR SCAN - SUMMARY')
print('='*70)
print(f'Total files scanned: {data["total_files"]}')
print(f'Total errors found: {data["errors_found"]}')
print()

# Count by error type
error_types = Counter(e['error_type'] for e in data['errors'])
print('Error breakdown:')
for etype, count in error_types.most_common():
    print(f'  {etype}: {count}')
print()

# Syntax errors
syntax_errors = [e for e in data['errors'] if e['error_type'] == 'syntax_error']
if syntax_errors:
    print('='*70)
    print('SYNTAX ERRORS:')
    print('='*70)
    for e in syntax_errors:
        fname = e['file'].split('\\')[-1]
        print(f'  File: {fname}')
        print(f'  Line: {e["line_number"]}')
        print(f'  Message: {e["message"]}')
        print()

# Circular imports
circular = [e for e in data['errors'] if e['error_type'] == 'circular_import']
if circular:
    print('='*70)
    print(f'CIRCULAR IMPORTS: {len(circular)} found')
    print('='*70)
    for e in circular:
        fname = e['file'].split('\\')[-1]
        print(f'  {fname}: {e["message"]}')
    print()

# Top missing modules
import_errors = [e for e in data['errors'] if e['error_type'] == 'import_error']
modules = [e['message'].replace("Module '", '').replace("' not found in project", '') for e in import_errors]
module_counts = Counter(modules)
print('='*70)
print('TOP MISSING MODULES (appearing most frequently):')
print('='*70)
for mod, count in module_counts.most_common(30):
    print(f'  {count:3d}  {mod}')
print()

# Files with most errors
file_errors = Counter(e['file'] for e in data['errors'])
print('='*70)
print('FILES WITH MOST ERRORS:')
print('='*70)
for filepath, count in file_errors.most_common(20):
    fname = filepath.split('\\')[-1]
    print(f'  {count:3d}  {fname}')
