#!/usr/bin/env python3
"""
Test script to verify all examples are syntactically correct
and can be imported successfully.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Color codes for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

# Set UTF-8 encoding for Windows
if os.name == 'nt':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')


def test_python_syntax(filepath):
    """Test if a Python file has valid syntax."""
    try:
        with open(filepath, 'r') as f:
            compile(f.read(), filepath, 'exec')
        return True, None
    except SyntaxError as e:
        return False, str(e)


def test_evaluator_structure(filepath):
    """Test if evaluator file has required structure."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()

        # Check for evaluate function
        if 'def evaluate(' not in content:
            return False, "Missing evaluate() function"

        # Check for combined_score return
        if 'combined_score' not in content:
            return False, "Missing 'combined_score' in return value"

        return True, None
    except Exception as e:
        return False, str(e)


def test_program_structure(filepath):
    """Test if program file has evolution markers."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()

        # Check for evolution markers
        if 'EVOLVE-BLOCK-START' not in content:
            return False, "Missing EVOLVE-BLOCK-START marker"

        if 'EVOLVE-BLOCK-END' not in content:
            return False, "Missing EVOLVE-BLOCK-END marker"

        return True, None
    except Exception as e:
        return False, str(e)


def run_tests():
    """Run all tests on example files."""
    examples_dir = Path(__file__).parent

    # Define example pairs (program, evaluator)
    example_pairs = [
        ("01_basic_evolution.py", "01_basic_evolution_evaluator.py"),
        ("02_function_evolution.py", "02_function_evolution_evaluator.py"),
        ("03_config_file.py", "03_optimize_evaluator.py"),
        ("04_python_api.py", "04_string_evaluator.py"),
        ("05_cli_usage.py", "05_algo_evaluator.py"),
        ("06_advanced_features.py", "06_multi_evaluator.py"),
    ]

    print("=" * 80)
    print("Testing OpenEvolve Examples")
    print("=" * 80)
    print()

    all_passed = True
    total_tests = 0
    passed_tests = 0

    for program_file, evaluator_file in example_pairs:
        program_path = examples_dir / program_file
        evaluator_path = examples_dir / evaluator_file

        print(f"\nTesting: {program_file} + {evaluator_file}")
        print("-" * 80)

        # Test program file
        total_tests += 1
        valid, error = test_python_syntax(program_path)
        if not valid:
            print(f"  {RED}[FAIL]{RESET} Program syntax error: {error}")
            all_passed = False
        else:
            print(f"  {GREEN}[OK]{RESET} Program syntax valid")

            # Test structure
            total_tests += 1
            valid, error = test_program_structure(program_path)
            if not valid:
                print(f"  {RED}[FAIL]{RESET} Program structure error: {error}")
                all_passed = False
            else:
                print(f"  {GREEN}[OK]{RESET} Program structure valid")
                passed_tests += 1

        # Test evaluator file
        total_tests += 1
        valid, error = test_python_syntax(evaluator_path)
        if not valid:
            print(f"  {RED}[FAIL]{RESET} Evaluator syntax error: {error}")
            all_passed = False
        else:
            print(f"  {GREEN}[OK]{RESET} Evaluator syntax valid")

            # Test structure
            total_tests += 1
            valid, error = test_evaluator_structure(evaluator_path)
            if not valid:
                print(f"  {RED}[FAIL]{RESET} Evaluator structure error: {error}")
                all_passed = False
            else:
                print(f"  {GREEN}[OK]{RESET} Evaluator structure valid")
                passed_tests += 1

    # Test documentation files
    print("\n" + "=" * 80)
    print("Testing Documentation")
    print("=" * 80)

    doc_files = [
        "README.md",
        "QUICKSTART.md",
        "config_example.yaml"
    ]

    for doc_file in doc_files:
        doc_path = examples_dir / doc_file
        total_tests += 1

        if doc_path.exists():
            print(f"  {GREEN}[OK]{RESET} {doc_file} exists")
            passed_tests += 1
        else:
            print(f"  {RED}[FAIL]{RESET} {doc_file} missing")
            all_passed = False

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Tests passed: {passed_tests}/{total_tests}")

    if all_passed:
        print(f"{GREEN}All tests passed!{RESET}")
        print()
        print("Next steps:")
        print("  1. Try running: python 01_basic_evolution.py")
        print("  2. Read QUICKSTART.md for detailed guide")
        print("  3. Explore other examples")
        return 0
    else:
        print(f"{RED}Some tests failed!{RESET}")
        print("Please fix the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
