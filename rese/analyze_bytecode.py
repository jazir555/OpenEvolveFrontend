#!/usr/bin/env python
"""Analyze RESE bytecode files to extract structure information."""

import dis
import marshal
import os
import json
from pathlib import Path

def extract_full_bytecode_info(pyc_path):
    """Extract comprehensive information from .pyc file"""
    try:
        with open(pyc_path, 'rb') as f:
            # Skip header (16 bytes for Python 3.11)
            f.read(16)
            code = marshal.load(f)

            # Clean module name
            module_name = (pyc_path
                .replace('.cpython-311.pyc', '')
                .replace('__pycache__', '')
                .replace('/', '.')
                .replace('\\', '.')
                .lstrip('.'))

            info = {
                'module': module_name,
                'original_file': code.co_filename,
                'functions': [],
                'classes': [],
                'imports': list(code.co_names)[:50],
                'docstring': None,
                'instruction_count': len(list(dis.get_instructions(code)))
            }

            # Extract docstring
            if code.co_consts and len(code.co_consts) > 0:
                doc = code.co_consts[0]
                if isinstance(doc, str) and len(doc) > 50:
                    info['docstring'] = doc

            # Recursively extract functions and classes
            for const in code.co_consts:
                if isinstance(const, type(code)):
                    if const.co_name == '<module>':
                        continue

                    func_info = {
                        'name': const.co_name,
                        'argcount': const.co_argcount,
                        'locals': list(const.co_varnames)[:10],
                        'names': list(const.co_names)[:20]
                    }

                    # Simple heuristic for class detection
                    if const.co_name[0].isupper():
                        info['classes'].append(func_info)
                    else:
                        info['functions'].append(func_info)

            return info

    except Exception as e:
        return {'module': pyc_path, 'error': str(e)}

def main():
    # Find all .pyc files
    rese_dir = Path('.')
    pyc_files = list(rese_dir.rglob('*.cpython-311.pyc'))

    print(f"Found {len(pyc_files)} .pyc files\n")

    # Analyze all modules
    all_modules = []
    for pyc_file in sorted(pyc_files):
        info = extract_full_bytecode_info(str(pyc_file))
        all_modules.append(info)

        if 'error' not in info:
            print(f"{info['module']}")
            print(f"  Functions: {len(info['functions'])}, Classes: {len(info['classes'])}")
            if info['docstring']:
                try:
                    doc_preview = info['docstring'][:80]
                    print(f"  Doc: {doc_preview}...")
                except:
                    print(f"  Doc: [encoding error]")
            print()

    # Save to JSON
    output = {
        'total_pyc_files': len(pyc_files),
        'analyzed_modules': all_modules
    }

    output_file = 'bytecode_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved analysis to {output_file}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    modules_with_errors = [m for m in all_modules if 'error' in m]
    successful_modules = [m for m in all_modules if 'error' not in m]

    print(f"Total .pyc files: {len(pyc_files)}")
    print(f"Successfully analyzed: {len(successful_modules)}")
    print(f"Failed to analyze: {len(modules_with_errors)}")

    if modules_with_errors:
        print("\nErrors:")
        for m in modules_with_errors:
            print(f"  {m['module']}: {m['error']}")

if __name__ == '__main__':
    main()
