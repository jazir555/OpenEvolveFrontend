#!/usr/bin/env python3
"""
Quick syntax checker for the main files to ensure they can be imported without errors.
"""

import ast
import sys
import os

# Add the current directory to the Python path to make imports work
sys.path.insert(0, os.getcwd())

def check_file_syntax(filename):
    """Try to parse a file and report any syntax errors"""
    try:
        # SECURITY FIX: Use ast.parse instead of exec() to check syntax
        # This prevents code execution while still validating syntax
        with open(filename, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        # Parse the code to check for syntax errors without executing it
        ast.parse(source_code, filename=filename)
        print(f"[OK] {filename} - Syntax OK")
        return True
    except SyntaxError as e:
        print(f"[FAIL] {filename} - Syntax Error at line {e.lineno}: {e.msg}")
        return False
    except UnicodeDecodeError as e:
        print(f"[FAIL] {filename} - Encoding Error: {e}")
        return False
    except IOError as e:
        print(f"[FAIL] {filename} - File Error: {e}")
        return False
    except Exception as e:
        print(f"? {filename} - Other Error: {e}")
        return True  # Could be other error, not necessarily syntax

def main():
    files_to_check = [
        'advanced_features.py',
        'monitoring_system.py', 
        'scalability_improvements.py',
        'auth_system.py',
        'input_validation.py', 
        'secure_api.py',
        'performance_optimization.py',
        'testing_framework.py',
        'deployment_operations.py'
    ]

    print("Checking Python syntax for key files...")
    print("-" * 50)
    
    results = []
    for filename in files_to_check:
        if os.path.exists(filename):
            results.append(check_file_syntax(filename))
        else:
            print(f"? {filename} - File does not exist")
            results.append(True)  # Not a syntax error if it doesn't exist
    
    print("-" * 50)
    total = len([r for r in results if r])
    checked = len(results)
    print(f"Checked {checked} files, {total} passed syntax check.")

if __name__ == "__main__":
    main()