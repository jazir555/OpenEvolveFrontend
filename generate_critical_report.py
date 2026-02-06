#!/usr/bin/env python3
"""Generate consolidated critical import errors report."""
import json

batch_files = [
    'import_errors_batch_1.json', 'import_errors_batch_2.json', 
    'import_errors_batch_3.json', 'import_errors_batch_4.json',
    'import_errors_batch_8.json', 'import_errors_batch_9.json',
    'import_errors_batch_10.json', 'import_errors_batch_11_16.json',
    'import_errors_batch_17_22.json', 'import_errors_batch_23_28.json',
    'import_errors_batch_29_32.json'
]

all_errors = []
total_scanned = 0
for bf in batch_files:
    try:
        with open(bf, 'r', encoding='utf-8') as f:
            data = json.load(f)
            total_scanned += data.get('total_files', 0)
            all_errors.extend(data.get('errors', []))
    except Exception as e:
        print(f'Warning: Could not read {bf}: {e}')

print(f'Total files scanned: {total_scanned}')
print(f'Total errors found: {len(all_errors)}')

# Standard library modules (false positives)
stdlib_mods = {'argparse', 'subprocess', 'heapq', 'signal', 'atexit', 'unittest', 'unittest.mock',
    'gc', 'concurrent.futures', 'tracemalloc', 'weakref', 'array', 'json', 'os', 'sys',
    'typing', 'collections', 'dataclasses', 'datetime', 're', 'math', 'random',
    'string', 'hashlib', 'base64', 'io', 'pathlib', 'tempfile', 'shutil', 'fnmatch',
    'warnings', 'functools', 'itertools', 'inspect', 'textwrap', 'copy', 'pickle',
    'enum', 'decimal', 'fractions', 'statistics', 'csv', 'time', 'uuid', 'bisect',
    'email.mime.text', 'email.mime.multipart', 'mimetypes', 'wave', 'stat', 'atexit',
    'contextlib', 'types', 'numbers', 'operator', 'builtins', '__future__'}

def is_stdlib(msg):
    msg_l = str(msg).lower()
    return any(m in msg_l for m in stdlib_mods)

def is_critical(e):
    et = e.get('error_type', '').lower()
    msg = str(e.get('message', ''))
    fp = str(e.get('file', ''))
    
    # Skip virtual environment files
    if 'openevolve_test_env' in fp:
        return False
    
    # Skip Jinja2 templates
    if '{{' in msg or '{%' in msg or '{#' in msg:
        return False
    
    # Skip __future__ import placement errors in vendored code
    if 'from __future__ imports must occur at the beginning' in msg:
        return False
    
    # True syntax/compile errors are always critical
    if et in ['syntax_error', 'compile_error']:
        return True
    
    # Python 2 style print statements
    if 'python 2 style' in msg.lower():
        return True
    
    # Typos in imports (but not stdlib)
    if 'typo' in et or 'possible typo' in msg.lower():
        if not is_stdlib(msg):
            return True
    
    # Import errors - only non-stdlib, non-relative
    if et in ['import_error', 'unresolved_import']:
        if is_stdlib(msg):
            return False
        if 'relative import' in msg.lower() and 'may not resolve' in msg.lower():
            return False
        return True
    
    return False

critical = [e for e in all_errors if is_critical(e)]

# Deduplicate by file+line+message
seen = set()
unique = []
for e in critical:
    key = (e.get('file', ''), e.get('line_number', 0), e.get('message', ''))
    if key not in seen:
        seen.add(key)
        unique.append(e)

print(f'Critical errors after filtering: {len(unique)}')

# Create structured output
errors_list = []
syntax_files = set()
import_files = set()

for e in unique:
    et = e.get('error_type', '').lower()
    msg = e.get('message', '')
    fp = e.get('file', '')
    line = e.get('line_number', 0)
    
    # Determine category and severity
    if 'syntax' in et or 'compile' in et:
        severity = 'critical'
        error_category = 'syntax_error'
        syntax_files.add(fp)
    elif 'python 2' in msg.lower():
        severity = 'high'
        error_category = 'syntax_error'
        syntax_files.add(fp)
    else:
        severity = 'high'
        error_category = 'import_error'
        import_files.add(fp)
    
    errors_list.append({
        'file': fp,
        'error_type': error_category,
        'line_number': line,
        'message': msg,
        'severity': severity,
        'suggested_fix': e.get('suggested_fix', 'Fix the import or syntax error')
    })

# Sort errors by file path
errors_list.sort(key=lambda x: x['file'])

report = {
    'summary': {
        'total_scanned': total_scanned,
        'critical_errors': len(unique),
        'syntax_errors': len([e for e in errors_list if e['error_type'] == 'syntax_error']),
        'import_errors': len([e for e in errors_list if e['error_type'] == 'import_error']),
        'files_to_fix': len(set(e['file'] for e in errors_list))
    },
    'errors': errors_list,
    'files_by_fix_type': {
        'syntax_errors': sorted(list(syntax_files)),
        'missing_imports': sorted(list(import_files)),
        'circular_imports': []
    }
}

with open('critical_import_errors.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f'Report created: critical_import_errors.json')
print(f"Files to fix: {report['summary']['files_to_fix']}")
print(f"Syntax errors: {report['summary']['syntax_errors']}")
print(f"Import errors: {report['summary']['import_errors']}")
