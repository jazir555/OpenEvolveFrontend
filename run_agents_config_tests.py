#!/usr/bin/env python
"""Run the Agents & Config tests"""
import sys
import subprocess

test_files = [
    "tests/agents/test_compliance_monitor.py",
    "tests/agents/test_investment_committee.py",
    "tests/cli/test_cli.py",
    "tests/config/test_config_system.py"
]

print("Running Agents & Config tests...")
print("=" * 60)

all_passed = True
summary = []

for test_file in test_files:
    print(f"\n\nRunning: {test_file}")
    print("-" * 60)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"],
        capture_output=True,
        text=True,
        timeout=60
    )
    print(result.stdout)

    # Count passed, failed, skipped
    passed = result.stdout.count("PASSED")
    failed = result.stdout.count("FAILED")
    skipped = result.stdout.count("SKIPPED")

    summary.append({
        'file': test_file,
        'passed': passed,
        'failed': failed,
        'skipped': skipped,
        'exit_code': result.returncode
    })

    if result.returncode != 0:
        all_passed = False

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for s in summary:
    print(f"\n{s['file']}:")
    print(f"  Passed: {s['passed']}")
    print(f"  Failed: {s['failed']}")
    print(f"  Skipped: {s['skipped']}")
    print(f"  Exit Code: {s['exit_code']}")

print("\n" + "=" * 60)
if all_passed:
    print("✓ All tests passed or were appropriately skipped!")
else:
    print("✗ Some tests failed")
print("=" * 60)
