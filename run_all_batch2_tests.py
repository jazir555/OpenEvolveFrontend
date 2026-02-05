#!/usr/bin/env python3
"""
Run all Batch 2 validation scripts and provide a comprehensive report
"""

import subprocess
import sys
import time
from pathlib import Path

def run_command(cmd, description):
    """Run a command and capture its output"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {cmd}")
    print('='*60)

    start_time = time.time()

    try:
        # Run the command
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60
        )

        end_time = time.time()
        duration = end_time - start_time

        # Print output
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)

        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        # Print summary
        print(f"\nExecution time: {duration:.2f} seconds")
        print(f"Return code: {result.returncode}")

        return result.returncode == 0, duration

    except subprocess.TimeoutExpired:
        print(f"[ERROR] Command timed out after 60 seconds")
        return False, 60.0
    except Exception as e:  # TODO: Catch specific exception instead of Exception
        print(f"[ERROR] Failed to run command: {e}")
        return False, 0.0

def main():
    """Run all Batch 2 validation scripts"""
    print("BATCH 2 VALIDATION SUITE")
    print("=" * 80)
    print("\nThis script runs all three Batch 2 validation scripts")
    print("and provides a comprehensive report.")

    scripts = [
        ("python validate_batch2_adapters.py", "Batch 2 Adapter Validation"),
        ("python test_adapter_functionality.py", "Adapter Functionality Tests"),
        ("python compare_before_after.py", "Performance Comparison")
    ]

    results = []
    total_time = 0

    # Run each script
    for cmd, description in scripts:
        success, duration = run_command(cmd, description)
        results.append((description, success, duration))
        total_time += duration

    # Generate report
    print("\n" + "="*80)
    print("COMPREHENSIVE VALIDATION REPORT")
    print("="*80)

    print(f"\nTotal execution time: {total_time:.2f} seconds")

    # Summary
    passed = 0
    failed = 0

    print("\nIndividual Results:")
    for description, success, duration in results:
        status = "[OK] PASS" if success else "[FAIL] FAIL"
        print(f"  {description:30} {status} ({duration:.2f}s)")

        if success:
            passed += 1
        else:
            failed += 1

    # Overall assessment
    print(f"\nOverall Summary:")
    print(f"  Scripts passed: {passed}")
    print(f"  Scripts failed: {failed}")
    print(f"  Total scripts: {passed + failed}")

    if failed == 0:
        print(f"\n[SUCCESS] ALL SCRIPTS PASSED!")
        print("Batch 2 adapter validation is complete and successful.")
        return 0
    elif passed > 0:
        print(f"\n[WARN] PARTIAL SUCCESS: {passed}/{passed + failed} scripts passed.")
        print("Some validation failed. Check output above for details.")
        return 1
    else:
        print(f"\n[ERROR] ALL SCRIPTS FAILED!")
        print("Batch 2 adapter validation needs attention.")
        return 2

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)