#!/usr/bin/env python3
"""
Quick test runner for security and other tests.
"""
import subprocess
import sys
from pathlib import Path

# Test files to run
TEST_FILES = [
    ("tests/test_main.py", "python"),
    ("tests/test_imports.py", "python"),
    ("tests/test_fix.py", "python"),
    ("tests/test_security.py", "pytest"),
    ("tests/test_sovereign_workflow.py", "pytest"),
    ("tests/test_unified_evolution_integration.py", "pytest"),
    ("tests/test_universal_problem_solver_gauntlet_pipeline.py", "pytest"),
    ("tests/test_adaptive_maker_evolution.py", "pytest"),
]

def run_test(test_file, runner):
    """Run a single test file."""
    print(f"\n{'='*60}")
    print(f"Testing: {test_file}")
    print('='*60)

    try:
        if runner == "python":
            result = subprocess.run(
                [sys.executable, test_file],
                capture_output=True,
                text=True,
                timeout=120
            )
        else:  # pytest
            result = subprocess.run(
                [sys.executable, "-m", "pytest", test_file, "-v", "--tb=line", "-x"],
                capture_output=True,
                text=True,
                timeout=180
            )

        # Print last 30 lines of output
        output_lines = result.stdout.split('\n')
        for line in output_lines[-30:]:
            if line.strip():
                print(line)

        if result.returncode == 0:
            print(f"\nPASSED: {test_file}")
            return True
        else:
            print(f"\nFAILED: {test_file} (exit code: {result.returncode})")
            return False

    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {test_file}")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("SECURITY & OTHER TEST SUITE")
    print("="*60)

    results = []
    for test_file, runner in TEST_FILES:
        results.append((test_file, run_test(test_file, runner)))

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    passed = sum(1 for _, success in results if success)
    failed = len(results) - passed

    for test_file, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{status}: {test_file}")

    print(f"\nTotal: {len(results)} tests, {passed} passed, {failed} failed")

    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
