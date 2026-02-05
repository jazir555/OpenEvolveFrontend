#!/usr/bin/env python3
"""
Run all security and other tests and report results.
"""
import subprocess
import sys
from pathlib import Path

# Test files to run
TEST_FILES = [
    "tests/test_security.py",
    "tests/test_sovereign_workflow.py",
    "tests/test_unified_evolution_integration.py",
    "tests/test_universal_problem_solver_gauntlet_pipeline.py",
    "tests/test_adaptive_maker_evolution.py",
    "tests/test_fix.py",
    "tests/test_imports.py",
    "tests/test_main.py",
]

def run_test(test_file):
    """Run a single test file and return success status."""
    print(f"\n{'='*80}")
    print(f"Running: {test_file}")
    print('='*80)

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"],
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout per test
        )

        # Print output
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        # Check for passed tests
        if "passed" in result.stdout.lower():
            # Extract summary line
            for line in result.stdout.split('\n'):
                if 'passed' in line.lower() and ('==' in line or '%' in line):
                    print(f"\n✓ Summary: {line.strip()}")
                    break

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print(f"✗ TIMEOUT: {test_file} exceeded 120 seconds")
        return False
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False

def main():
    """Run all tests and report results."""
    print("="*80)
    print("SECURITY & OTHER TEST SUITE")
    print("="*80)

    results = {}
    for test_file in TEST_FILES:
        results[test_file] = run_test(test_file)

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    passed = sum(1 for v in results.values() if v)
    failed = len(results) - passed

    for test_file, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {test_file}")

    print(f"\nTotal: {len(results)} tests, {passed} passed, {failed} failed")

    if failed > 0:
        sys.exit(1)
    else:
        print("\n✓ All tests passed!")
        sys.exit(0)

if __name__ == "__main__":
    main()
