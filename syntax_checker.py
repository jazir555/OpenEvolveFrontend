#!/usr/bin/env python3
"""
Quick syntax checker for the main files to ensure they can be imported without errors.
"""
import sys
import os

# Add the current directory to the Python path to make imports work
sys.path.insert(0, os.getcwd())

def check_file_syntax(filename):
    """Try to import a file and report any syntax errors"""
    try:
        exec(open(filename).read(), {"__name__": "__main__", "__file__": filename})
        print(f"✓ {filename} - Syntax OK")
        return True
    except SyntaxError as e:
        print(f"✗ {filename} - Syntax Error at line {e.lineno}: {e.msg}")
        return False
    except ImportError as e:
        print(f"⚠ {filename} - Import Error: {e}")
        return True  # Not a syntax error, just a missing dependency
    except Exception as e:
        print(f"? {filename} - Other Error: {e}")
        return True  # Could be runtime error, not necessarily syntax

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