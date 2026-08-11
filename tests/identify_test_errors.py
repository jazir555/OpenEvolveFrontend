#!/usr/bin/env python3
"""
Identify test files with import errors.
"""
import os
import sys
import importlib.util
from pathlib import Path

def check_test_file(filepath):
    """Check if a test file can be imported without errors."""
    try:
        spec = importlib.util.spec_from_file_location("test_module", filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return True, None
    except Exception as e:
        return False, str(e)

def main():
    """Find all test files with import errors."""
    test_files = list(Path('.').glob('test_*.py'))
    
    errors = []
    for test_file in test_files:
        success, error = check_test_file(test_file)
        if not success:
            errors.append((test_file.name, error))
            print(f"ERROR in {test_file.name}: {error[:100]}...")
    
    print(f"\n{'='*60}")
    print(f"Total test files: {len(test_files)}")
    print(f"Files with errors: {len(errors)}")
    
    return errors

if __name__ == "__main__":
    errors = main()
    sys.exit(0 if not errors else 1)
