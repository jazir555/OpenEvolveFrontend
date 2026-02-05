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
    '[OK]': '[OK]',      # Check mark
    '[OK]': '[OK]',      # Check mark (light)
    '[FAIL]': '[FAIL]',    # Cross mark
    '[FAIL]': '[FAIL]',    # Cross mark (light)
    '[WARN]': '[WARN]',    # Warning sign
    '*': '*',          # Star
    '*': '*',          # Bullet
    '->': '->',         # Right arrow
    '->': '->',         # Right arrow (heavy)
    '<-': '<-',         # Left arrow
    '^': '^',          # Up arrow
    'v': 'v',          # Down arrow
}

# Pattern to match any Unicode character we want to replace
UNICODE_PATTERN = re.compile('|'.join(map(re.escape, UNICODE_REPLACEMENTS.keys())))


def fix_unicode_in_file(filepath):
    """Fix Unicode characters in a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except (UnicodeDecodeError, IOError) as e:
        print(f"[SKIP] Cannot read {filepath}: {e}")
        return False
    
    # Check if file contains any of the Unicode characters
    if not UNICODE_PATTERN.search(content):
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
    
    print("Scanning Python files for Unicode characters...\n")
    
    # Use os.walk for more robust directory traversal
    python_files = []
    for root, dirs, files in os.walk('.', topdown=True):
        # Modify dirs in-place to skip excluded directories
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for filename in files:
            if filename.endswith('.py'):
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
