#!/usr/bin/env python
"""
Quick Start Script for PES Test Suite

This script helps you get started with the PES test suite quickly.
Usage:
    python quickstart.py
"""

import sys
import subprocess
from pathlib import Path


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(text)
    print("="*60 + "\n")


def run_command(cmd, description):
    """Run a command and report results"""
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print("-"*60)

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        print(f"\n[OK] {description} - SUCCESS\n")
        return True
    else:
        print(f"\n[FAIL] {description} - FAILED\n")
        return False


def main():
    """Main quick start routine"""
    print_header("PES Test Suite - Quick Start")

    # Get to project root
    project_root = Path(__file__).parent.parent.parent
    print(f"Project root: {project_root}\n")

    # Step 1: Validate
    print_header("Step 1: Validating Test Infrastructure")
    validate_cmd = [sys.executable, "tests/pes/validate_tests.py"]
    if not run_command(validate_cmd, "Validation"):
        print("Validation failed. Please fix issues before continuing.")
        return 1

    # Step 2: Check if pytest is installed
    print_header("Step 2: Checking Dependencies")
    try:
        import pytest
        print(f"[OK] pytest installed (version {pytest.__version__})\n")
    except ImportError:
        print("[FAIL] pytest not installed")
        print("Installing test dependencies...")
        install_cmd = [
            sys.executable, "-m", "pip", "install",
            "-r", "tests/pes/requirements-test.txt"
        ]
        if not run_command(install_cmd, "Install dependencies"):
            print("Failed to install dependencies")
            return 1

    # Step 3: Run a small test
    print_header("Step 3: Running Sample Test")
    test_cmd = [
        sys.executable, "-m", "pytest",
        "tests/pes/test_database.py::TestProgramDatabase::test_database_initialization",
        "-v"
    ]

    if not run_command(test_cmd, "Sample test"):
        print("Sample test failed")
        return 1

    # Step 4: Show next steps
    print_header("Quick Start Complete!")
    print("Your test environment is ready!")
    print()
    print("Next steps:")
    print("  1. Run all tests:")
    print("     cd tests/pes && python run_tests.py")
    print()
    print("  2. Run with coverage:")
    print("     cd tests/pes && python run_tests.py --coverage")
    print()
    print("  3. Run specific test file:")
    print("     pytest tests/pes/test_controller.py -v")
    print()
    print("  4. Run specific test:")
    print("     pytest tests/pes/test_database.py::TestProgramDatabase::test_database_initialization -v")
    print()
    print("  5. View documentation:")
    print("     cat tests/pes/README.md")
    print()
    print("For more information, see:")
    print("  - tests/pes/README.md - Complete documentation")
    print("  - tests/pes/TEST_SUMMARY.md - Implementation summary")
    print()
    print("="*60)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
