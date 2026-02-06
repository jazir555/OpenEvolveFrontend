#!/usr/bin/env python3
"""Fix all syntax errors across the codebase."""

import os
import re

def fix_bom_issue(filepath):
    """Fix BOM (Byte Order Mark) at start of file."""
    with open(filepath, 'rb') as f:
        content = f.read()
    
    # Check for BOM and remove it
    if content.startswith(b'\xef\xbb\xbf'):
        content = content[3:]  # Remove UTF-8 BOM
        with open(filepath, 'wb') as f:
            f.write(content)
        print(f"  Fixed BOM in: {filepath}")
        return True
    return False

def fix_verify_implementation():
    """Fix the duplicate function definition."""
    filepath = 'knowledge_engine/verify_implementation.py'
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # The issue is duplicate function definition at lines 141-144
    # Line 141: async def verify_functionality():
    # Line 142:     """Verify basic functionality of the knowledge engine."""
    # Line 143:     async def verify_functionality():  <- DUPLICATE!
    # Line 144:     """Verify basic functionality of the knowledge engine."""
    
    # We need to remove the duplicate lines 143-144
    # But we also need to fix the indentation issue
    
    # Let's read and fix the specific section
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Skip duplicate function definition at line 143 (0-indexed: 142)
        if i == 142 and 'async def verify_functionality():' in line:
            # Skip this line and the next (docstring)
            i += 2
            continue
        
        # Fix the second "try:" at line 149 (0-indexed: 148)
        if i == 148 and 'try:' in line:
            # This is a duplicate try - skip it
            i += 1
            continue
            
        new_lines.append(line)
        i += 1
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print(f"  Fixed duplicate function in: {filepath}")

def main():
    print("=== Fixing Syntax Errors ===\n")
    
    # Fix 1: BOM in glue/adapters/rese-sce/__init__.py
    print("1. Fixing BOM in glue/adapters/rese-sce/__init__.py")
    fix_bom_issue('glue/adapters/rese-sce/__init__.py')
    
    # Fix 2: BOM in unified/__init__.py
    print("2. Fixing BOM in unified/__init__.py")
    fix_bom_issue('unified/__init__.py')
    
    # Fix 3: Duplicate function in knowledge_engine/verify_implementation.py
    print("3. Fixing duplicate function in knowledge_engine/verify_implementation.py")
    fix_verify_implementation()
    
    # Fix 4: Databricks notebook syntax (shell command in Python)
    print("4. Databricks file has shell commands - this is a notebook file, not standard Python")
    print("   Skipping (it's a Databricks notebook with MAGIC commands)")
    
    print("\n=== Verifying Fixes ===")
    
    # Verify by trying to parse the fixed files
    import ast
    files_to_check = [
        'glue/adapters/rese-sce/__init__.py',
        'unified/__init__.py',
        'knowledge_engine/verify_implementation.py'
    ]
    
    all_good = True
    for filepath in files_to_check:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                source = f.read()
            ast.parse(source)
            print(f"  ✓ {filepath} - OK")
        except SyntaxError as e:
            print(f"  ✗ {filepath} - Still has error: {e}")
            all_good = False
    
    if all_good:
        print("\n✓ All fixable errors have been fixed!")
    else:
        print("\n✗ Some errors could not be fixed automatically.")

if __name__ == "__main__":
    main()
