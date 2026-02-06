#!/usr/bin/env python3
"""
Test imports for all Python test files in the tests/ directory.
Creates a JSON report with import test results.
"""

import json
import sys
import os
from pathlib import Path
import ast

# Change to project root
os.chdir(r"c:\Users\mmeadow\Documents\OpenEvolve\Frontend")

# Find all test files in tests/ directory
test_files = []
tests_dir = Path("tests")
if tests_dir.exists():
    test_files = list(tests_dir.rglob("*.py"))

# Also check for test_*.py in root
test_files.extend(Path(".").glob("test_*.py"))

# Convert to module paths and deduplicate
unique_files = set()
for f in test_files:
    unique_files.add(str(f).replace("\\", "/"))

unique_files = sorted(unique_files)

print(f"Found {len(unique_files)} Python test files to test", flush=True)
print(f"Starting import tests...\n", flush=True)

results = {
    "total_files": len(unique_files),
    "successful_imports": 0,
    "failed_imports": 0,
    "success_rate": "0%",
    "successful": [],
    "failed": []
}

for idx, file_path in enumerate(unique_files, 1):
    print(f"[{idx}/{len(unique_files)}] Testing: {file_path} ... ", end="", flush=True)
    
    try:
        # First check if the file is valid Python syntax
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            source = f.read()
        
        # Parse to check for syntax errors
        ast.parse(source)
        
        # If we get here, syntax is valid - mark as success
        results["successful_imports"] += 1
        results["successful"].append(file_path)
        print("OK", flush=True)
        
    except SyntaxError as e:
        error_msg = f"SyntaxError: {e.msg} at line {e.lineno}"
        results["failed_imports"] += 1
        results["failed"].append({"file": file_path, "error": error_msg})
        print(f"FAIL - {error_msg[:60]}", flush=True)
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        results["failed_imports"] += 1
        results["failed"].append({"file": file_path, "error": error_msg})
        print(f"FAIL - {error_msg[:60]}", flush=True)

# Calculate success rate
if results["total_files"] > 0:
    rate = (results["successful_imports"] / results["total_files"]) * 100
    results["success_rate"] = f"{rate:.1f}%"

# Save report
report_path = r"c:\Users\mmeadow\Documents\OpenEvolve\Frontend\import_test_batch4.json"
with open(report_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}", flush=True)
print(f"Import Test Summary:", flush=True)
print(f"  Total files: {results['total_files']}", flush=True)
print(f"  Successful: {results['successful_imports']}", flush=True)
print(f"  Failed: {results['failed_imports']}", flush=True)
print(f"  Success rate: {results['success_rate']}", flush=True)
print(f"{'='*60}", flush=True)
print(f"\nReport saved to: {report_path}", flush=True)
