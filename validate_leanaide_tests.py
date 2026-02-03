#!/usr/bin/env python3
"""
LeanAide Test Suite Validator

Validates that the test suite is properly configured and can run.

Usage:
    python validate_leanaide_tests.py
"""

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import List, Tuple


class Colors:
    """ANSI color codes."""
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def print_status(message: str, status: str):
    """Print status message with color."""
    # Use ASCII characters for Windows compatibility
    if status == "OK":
        print(f"{Colors.GREEN}[OK]{Colors.ENDC} {message}")
    elif status == "WARN":
        print(f"{Colors.YELLOW}[WARN]{Colors.ENDC} {message}")
    elif status == "ERROR":
        print(f"{Colors.RED}[ERROR]{Colors.ENDC} {message}")
    elif status == "INFO":
        print(f"{Colors.BLUE}[INFO]{Colors.ENDC} {message}")


def check_python_version() -> Tuple[bool, str]:
    """Check Python version."""
    version = sys.version_info
    if version >= (3, 8):
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)"


def check_module(module_name: str) -> Tuple[bool, str]:
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
        return True, f"{module_name} is installed"
    except ImportError:
        return False, f"{module_name} is NOT installed"


def check_file(filepath: str) -> Tuple[bool, str]:
    """Check if a file exists."""
    path = Path(filepath)
    if path.exists():
        return True, f"{filepath} exists"
    else:
        return False, f"{filepath} NOT found"


def check_test_file(filepath: str) -> Tuple[bool, str]:
    """Check if test file exists and is valid."""
    path = Path(filepath)
    if not path.exists():
        return False, f"{filepath} NOT found"

    # Try to read the file
    try:
        content = path.read_text()
        if "def test_" in content or "async def test_" in content:
            return True, f"{filepath} exists with tests"
        else:
            return False, f"{filepath} exists but contains no tests"
    except Exception as e:
        return False, f"{filepath} exists but cannot be read: {e}"


def validate_imports() -> List[Tuple[bool, str, str]]:
    """Validate LeanAide module imports."""
    results = []

    modules = [
        ("pytest", "Required"),
        ("pytest_asyncio", "Required"),
        ("pytest_cov", "Optional"),
        ("pytest_xdist", "Optional"),
    ]

    for module, level in modules:
        success, message = check_module(module)
        status = "OK" if success else "WARN" if level == "Optional" else "ERROR"
        results.append((success, message, status))

    return results


def validate_leanaide_modules() -> List[Tuple[bool, str, str]]:
    """Validate LeanAide-specific modules."""
    results = []

    modules = [
        ("leanaide_client", "Client"),
        ("leanaide_mcp_tools", "MCP Tools"),
        ("leanaide_crewai_bridge", "Bridge"),
    ]

    for module, name in modules:
        try:
            # Try to import from parent directory
            parent_dir = Path(__file__).parent
            if str(parent_dir) not in sys.path:
                sys.path.insert(0, str(parent_dir))

            importlib.import_module(module)
            results.append((True, f"{name} module available", "OK"))
        except ImportError:
            results.append((False, f"{name} module NOT available (offline mode)", "WARN"))

    return results


def validate_test_structure() -> List[Tuple[bool, str, str]]:
    """Validate test file structure."""
    results = []
    test_file = Path(__file__).parent / "test_leanaide_integration.py"

    if not test_file.exists():
        results.append((False, "test_leanaide_integration.py NOT found", "ERROR"))
        return results

    # Read and check for required test classes
    content = test_file.read_text()

    required_classes = [
        "TestLeanAideClientInitialization",
        "TestMCPToolRegistry",
        "TestMCPTool1_TranslateTheorem",
        "TestBridgePhase1_Analysis",
        "TestFullWorkflowIntegration",
        "TestErrorHandling",
        "TestPerformanceAndCaching",
    ]

    for cls in required_classes:
        if f"class {cls}" in content:
            results.append((True, f"Test class {cls} present", "OK"))
        else:
            results.append((False, f"Test class {cls} missing", "WARN"))

    return results


def validate_test_data() -> List[Tuple[bool, str, str]]:
    """Validate test data files."""
    results = []
    test_data_dir = Path(__file__).parent / "test_leanaide_data"

    if test_data_dir.exists():
        results.append((True, "test_leanaide_data directory exists", "OK"))

        # Check for test data files
        expected_files = [
            "sample_theorems.json",
            "sample_lean_code.lean",
        ]

        for filename in expected_files:
            filepath = test_data_dir / filename
            if filepath.exists():
                results.append((True, f"Test data file {filename} exists", "OK"))
            else:
                results.append((False, f"Test data file {filename} missing", "WARN"))
    else:
        results.append((False, "test_leanaide_data directory NOT found", "WARN"))

    return results


def count_tests() -> Tuple[int, int, int]:
    """Count tests in the test file."""
    test_file = Path(__file__).parent / "test_leanaide_integration.py"

    if not test_file.exists():
        return 0, 0, 0

    content = test_file.read_text()

    # Count test functions
    total = content.count("def test_") + content.count("async def test_")
    unit = content.count("@mark.unit")
    integration = content.count("@mark.integration")
    mock = content.count("@mark.mock")
    server = content.count("@mark.server")

    return total, unit + integration + mock + server


def main():
    """Run validation."""
    print()
    print(f"{Colors.BOLD}LeanAide Test Suite Validator{Colors.ENDC}")
    print("=" * 70)
    print()

    # Check Python version
    success, message = check_python_version()
    print_status(message, "OK" if success else "ERROR")
    print()

    # Validate dependencies
    print(f"{Colors.BOLD}Checking Dependencies:{Colors.ENDC}")
    for success, message, status in validate_imports():
        print_status(message, status)
    print()

    # Validate LeanAide modules
    print(f"{Colors.BOLD}Checking LeanAide Modules:{Colors.ENDC}")
    for success, message, status in validate_leanaide_modules():
        print_status(message, status)
    print()

    # Validate test structure
    print(f"{Colors.BOLD}Checking Test Structure:{Colors.ENDC}")
    for success, message, status in validate_test_structure():
        print_status(message, status)
    print()

    # Validate test data
    print(f"{Colors.BOLD}Checking Test Data:{Colors.ENDC}")
    for success, message, status in validate_test_data():
        print_status(message, status)
    print()

    # Count tests
    total_tests, marked_tests = count_tests()
    print_status(f"Total test functions: {total_tests}", "INFO")
    print_status(f"Marked test functions: {marked_tests}", "INFO")
    print()

    # Summary
    print("=" * 70)
    print()
    print(f"{Colors.BOLD}Next Steps:{Colors.ENDC}")
    print()

    # Check if can run tests
    can_run = True

    try:
        import pytest
        print_status("pytest is installed", "OK")
    except ImportError:
        print_status("pytest is NOT installed", "ERROR")
        can_run = False

    if can_run:
        print()
        print("Run all tests:")
        print(f"  {Colors.GREEN}python test_leanaide_integration.py{Colors.ENDC}")
        print()
        print("Or use the test runner:")
        print(f"  {Colors.GREEN}python run_leanaide_tests.py{Colors.ENDC}")
        print()
        print("Run specific test categories:")
        print(f"  {Colors.GREEN}python run_leanaide_tests.py --unit{Colors.ENDC}")
        print(f"  {Colors.GREEN}python run_leanaide_tests.py --integration{Colors.ENDC}")
        print(f"  {Colors.GREEN}python run_leanaide_tests.py --mock{Colors.ENDC}")
        print()
    else:
        print()
        print("Install required dependencies:")
        print(f"  {Colors.YELLOW}pip install pytest pytest-asyncio{Colors.ENDC}")
        print()
        print("Optional dependencies:")
        print(f"  {Colors.YELLOW}pip install pytest-cov pytest-xdist{Colors.ENDC}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
