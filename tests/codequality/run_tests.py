"""
Test Runner Script for Code Quality Tests

Runs all code quality tests and generates coverage report.
"""

import sys
import subprocess
from pathlib import Path

# Add paths
utils_path = Path(__file__).parent.parent.parent / "utils"
sys.path.insert(0, str(utils_path))

def run_tests():
    """Run all code quality tests."""
    print("=" * 80)
    print("Running Code Quality Tests")
    print("=" * 80)
    print()

    test_files = [
        "tests/codequality/test_logging.py",
        "tests/codequality/test_errors.py",
        "tests/codequality/test_timestamps.py",
        "tests/codequality/test_integration.py",
    ]

    results = {}

    for test_file in test_files:
        print(f"\n{'=' * 80}")
        print(f"Running: {test_file}")
        print('=' * 80)

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", test_file, "-v"],
                capture_output=True,
                text=True,
                timeout=120
            )

            results[test_file] = {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }

            print(result.stdout)

            if result.returncode != 0:
                print(f"❌ FAILED: {test_file}")
                if result.stderr:
                    print("STDERR:", result.stderr)
            else:
                print(f"✅ PASSED: {test_file}")

        except subprocess.TimeoutExpired:
            print(f"⏱️  TIMEOUT: {test_file}")
            results[test_file] = {"returncode": -1, "error": "timeout"}
        except Exception as e:
            print(f"❌ ERROR: {test_file} - {e}")
            results[test_file] = {"returncode": -1, "error": str(e)}

    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    passed = sum(1 for r in results.values() if r["returncode"] == 0)
    failed = sum(1 for r in results.values() if r["returncode"] != 0)

    print(f"\nTotal: {len(results)} tests")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")

    if failed > 0:
        print("\nFailed tests:")
        for test_file, result in results.items():
            if result["returncode"] != 0:
                print(f"  - {test_file}")

    return passed == len(results)


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
