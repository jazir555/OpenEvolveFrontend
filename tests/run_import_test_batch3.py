#!/usr/bin/env python3
"""Test imports for all openevolve Python files."""

import json
import sys
import importlib.util
import glob

# Get all files to test
files = set()
for pattern in ['openevolve*.py', 'openevolve/**/*.py']:
    for f in glob.glob(pattern, recursive=True):
        files.add(f)

files = sorted(files)

results = {
    'total_files': 0,
    'successful_imports': 0,
    'failed_imports': 0,
    'success_rate': '0%',
    'successful': [],
    'failed': []
}

for filepath in files:
    results['total_files'] += 1
    module_name = filepath.replace('/', '.').replace('\\', '.').replace('.py', '')
    
    try:
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        if spec is None or spec.loader is None:
            raise ImportError(f'Could not create module spec for {filepath}')
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        results['successful_imports'] += 1
        results['successful'].append(filepath)
        print(f'OK: {filepath}')
    except Exception as e:
        results['failed_imports'] += 1
        error_msg = f'{type(e).__name__}: {str(e)}'
        results['failed'].append({'file': filepath, 'error': error_msg})
        print(f'FAIL: {filepath}: {error_msg}')

# Calculate success rate
if results['total_files'] > 0:
    rate = (results['successful_imports'] / results['total_files']) * 100
    results['success_rate'] = f'{rate:.1f}%'

# Write report
with open('import_test_batch3.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSummary: {results['successful_imports']}/{results['total_files']} ({results['success_rate']})")
print("Report written to import_test_batch3.json")
