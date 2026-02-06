#!/usr/bin/env python3
"""Scan for import and syntax errors across the codebase."""

import os
import sys
import json
from pathlib import Path
import ast

def scan_for_errors():
    """Scan Python files for syntax and import errors."""
    errors = []
    import_errors = []
    
    py_files = []
    for root, dirs, files in os.walk('.', topdown=True):
        # Skip problematic directories
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.venv', 'node_modules', '.git']]
        for file in files:
            if file.endswith('.py'):
                py_files.append(Path(root) / file)
    
    print(f"Scanning {len(py_files)} Python files...")
    
    for file_path in py_files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                source = f.read()
            
            # Try to parse with AST
            ast.parse(source)
            
        except SyntaxError as e:
            errors.append({
                'file': str(file_path), 
                'error': str(e), 
                'type': 'syntax',
                'line': e.lineno
            })
        except Exception as e:
            errors.append({
                'file': str(file_path), 
                'error': str(e), 
                'type': 'other'
            })
    
    # Summary
    print(f"\n=== SCAN RESULTS ===")
    print(f"Total files scanned: {len(py_files)}")
    print(f"Syntax errors found: {len([e for e in errors if e['type'] == 'syntax'])}")
    print(f"Other errors found: {len([e for e in errors if e['type'] == 'other'])}")
    
    if errors:
        print("\n=== ERRORS ===")
        for e in errors[:50]:  # Show first 50
            print(f"\n{e['file']}")
            print(f"  Type: {e['type']}")
            if 'line' in e:
                print(f"  Line: {e['line']}")
            print(f"  Error: {e['error'][:200]}")
    
    # Save to file
    with open('import_error_scan_results.json', 'w') as f:
        json.dump(errors, f, indent=2)
    
    print(f"\nResults saved to import_error_scan_results.json")
    return errors

if __name__ == "__main__":
    scan_for_errors()
