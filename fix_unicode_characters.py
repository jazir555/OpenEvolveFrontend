#!/usr/bin/env python3
"""
Script to fix Unicode encoding errors in Python files.
Replaces Unicode characters with ASCII-safe alternatives.
"""

import os
import re
from pathlib import Path

# Mapping of Unicode characters to ASCII replacements
UNICODE_REPLACEMENTS = {
    '\u2705': '[OK]',      # Check mark (U+2705)
    '\u2713': '[OK]',      # Check mark (light) (U+2713)
    '\u274C': '[FAIL]',    # Cross mark (U+274C)
    '\u2717': '[FAIL]',    # Cross mark (light) (U+2717)
    '\u26A0\uFE0F': '[WARN]',  # Warning sign with variation selector (U+26A0 U+FE0F)
    '\u26A0': '[WARN]',    # Warning sign without variation selector (U+26A0)
    '\u2605': '*',         # Star (U+2605)
    '\u2022': '*',         # Bullet (U+2022)
    '\u2192': '->',        # Right arrow (U+2192)
    '\u27A1': '->',        # Right arrow (heavy) (U+27A1)
    '\u2190': '<-',        # Left arrow (U+2190)
    '\u2191': '^',         # Up arrow (U+2191)
    '\u2193': 'v',         # Down arrow (U+2193)
}


def fix_unicode_in_file(filepath):
    """Fix Unicode characters in a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except (UnicodeDecodeError, IOError) as e:
        print(f"[SKIP] Cannot read {filepath}: {e}")
        return False
    
    # Check if file contains any of the Unicode characters
    has_unicode = False
    for unicode_char in UNICODE_REPLACEMENTS.keys():
        if unicode_char in content:
            has_unicode = True
            break
    
    if not has_unicode:
        return False
    
    # Replace each Unicode character
    new_content = content
    for unicode_char, replacement in UNICODE_REPLACEMENTS.items():
        if unicode_char in new_content:
            new_content = new_content.replace(unicode_char, replacement)
    
    # Write the fixed content back
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True
    except IOError as e:
        print(f"[ERROR] Cannot write {filepath}: {e}")
        return False


def main():
    """Main function to fix Unicode in all Python files."""
    fixed_files = []
    errors = []
    
    # Excluded directories
    skip_dirs = {'__pycache__', '.venv', 'node_modules', '.git', '.pytest_cache', 
                 '.cache', 'bubblelab-converted', 'checkpoints', '.claude', 
                 '.gemini', '.leanaide_cache', '.c2c_cache', '.openevolve',
                 '.steer', '.tdad'}
    
    # Excluded files
    skip_files = {'fix_unicode_characters.py', 'test_unicode_fix.py'}
    
    print("Scanning Python files for Unicode characters...\n")
    
    # Use os.walk for more robust directory traversal
    python_files = []
    for root, dirs, files in os.walk('.', topdown=True):
        # Modify dirs in-place to skip excluded directories
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for filename in files:
            if filename.endswith('.py') and filename not in skip_files:
                filepath = os.path.join(root, filename)
                python_files.append(filepath)
    
    print(f"Found {len(python_files)} Python files to check.\n")
    
    for filepath in python_files:
        try:
            if fix_unicode_in_file(filepath):
                fixed_files.append(filepath)
                print(f"[FIXED] {filepath}")
        except Exception as e:
            errors.append((filepath, str(e)))
            print(f"[ERROR] {filepath}: {e}")
    
    # Summary
    print("\n" + "="*60)
    print(f"SUMMARY: Fixed {len(fixed_files)} files")
    print(f"Errors: {len(errors)}")
    
    if fixed_files:
        print("\nFixed files:")
        for f in fixed_files:
            print(f"  - {f}")
    
    if errors:
        print("\nFiles with errors:")
        for f, e in errors:
            print(f"  - {f}: {e}")
    
    return len(fixed_files), errors


if __name__ == '__main__':
    fixed_count, errors = main()
    exit(0 if not errors else 1)
