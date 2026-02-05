#!/usr/bin/env python
"""Test runner for Group C tests"""
import sys
import subprocess
import os

# Set required environment variables
os.environ['OPENAI_API_KEY'] = 'test_key_for_testing'

tests = [
    ("tests/test_entanglement_matrix.py", "Entanglement Matrix"),
    ("tests/test_global_context_management.py", "Global Context Management"),
    ("tests/test_globalchem_integration.py", "GlobalChem Integration"),
    ("tests/test_integrated_functionality.py", "Integrated Functionality"),
]

results = []

for test_file, test_name in tests:
    print(f"\n{'='*60}")
    print(f"Running: {test_name}")
    print(f"File: {test_file}")
    print('='*60)

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"],
            capture_output=True,
            text=True,
            timeout=180,  # 3 minutes
            env=os.environ
        )

        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        if result.returncode == 0:
            print(f"\n[OK] {test_name} PASSED")
            results.append((test_name, True))
        else:
            print(f"\n[FAIL] {test_name} FAILED")
            results.append((test_name, False))

    except subprocess.TimeoutExpired:
        print(f"\n[TIMEOUT] {test_name} timed out")
        results.append((test_name, False))
    except Exception as e:
        print(f"\n[ERROR] {test_name} error: {e}")
        results.append((test_name, False))

print(f"\n{'='*60}")
print("SUMMARY")
print('='*60)
for test_name, passed in results:
    status = "[OK] PASSED" if passed else "[FAIL] FAILED"
    print(f"{status} {test_name}")

passed_count = sum(1 for _, p in results if p)
print(f"\nTotal: {passed_count}/{len(results)} tests passed")
