#!/usr/bin/env python
"""
Test Environment Initialization Script

This script sets up the test environment, checks for common issues,
and validates that tests can run properly.

Usage:
    python tests/init_test_env.py              # Check and display status
    python tests/init_test_env.py --fix        # Auto-fix issues
    python tests/init_test_env.py --verbose    # Detailed output

Author: OpenEvolve Distinguished Engineer
Version: 1.0.0
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Any
import argparse


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
TESTS_DIR = PROJECT_ROOT / "tests"
REQUIRED_ENV_VARS = [
    "TESTING",
]
OPTIONAL_ENV_VARS = [
    "DATABASE_URL",
    "API_HOST",
    "API_PORT",
    "LOG_LEVEL",
]

REQUIRED_PYTHON_VERSION = (3, 8)
REQUIRED_PACKAGES = [
    "pytest",
    "pytest_asyncio",
    "pytest_mock",
]

OPTIONAL_PACKAGES = [
    "pytest_cov",
    "pytest_xdist",
    "responses",
    "freezegun",
]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_header(title: str, level: int = 1):
    """Print a formatted header."""
    if level == 1:
        print(f"\n{'=' * 70}")
        print(f"  {title}")
        print(f"{'=' * 70}\n")
    elif level == 2:
        print(f"\n{'-' * 70}")
        print(f"  {title}")
        print(f"{'-' * 70}\n")
    else:
        print(f"\n>>> {title}\n")


def print_success(message: str):
    """Print a success message."""
    print(f"✓ {message}")


def print_error(message: str):
    """Print an error message."""
    print(f"✗ {message}")


def print_warning(message: str):
    """Print a warning message."""
    print(f"⚠ {message}")


def print_info(message: str):
    """Print an info message."""
    print(f"  {message}")


# ============================================================================
# CHECK FUNCTIONS
# ============================================================================

def check_python_version() -> Tuple[bool, str]:
    """Check Python version."""
    version = sys.version_info[:2]
    if version >= REQUIRED_PYTHON_VERSION:
        return True, f"Python {version[0]}.{version[1]} (OK)"
    else:
        return False, f"Python {version[0]}.{version[1]} (Need {REQUIRED_PYTHON_VERSION[0]}.{REQUIRED_PYTHON_VERSION[1]}+)"


def check_project_structure() -> Tuple[bool, List[str]]:
    """Check that project structure is correct."""
    issues = []

    # Check for tests directory
    if not TESTS_DIR.exists():
        issues.append(f"Tests directory not found: {TESTS_DIR}")

    # Check for conftest.py
    conftest = TESTS_DIR / "conftest.py"
    if not conftest.exists():
        issues.append(f"Root conftest.py not found: {conftest}")

    # Check for test_helpers.py
    helpers = TESTS_DIR / "test_helpers.py"
    if not helpers.exists():
        issues.append(f"test_helpers.py not found: {helpers}")

    # Check for knowledge_engine directory
    ke_dir = PROJECT_ROOT / "knowledge_engine"
    if not ke_dir.exists():
        issues.append(f"Knowledge engine directory not found: {ke_dir}")

    return len(issues) == 0, issues


def check_environment_variables() -> Tuple[bool, Dict[str, Any]]:
    """Check environment variables."""
    status = {
        "required": {},
        "optional": {},
        "missing_required": [],
        "missing_optional": [],
    }

    # Check required variables
    for var in REQUIRED_ENV_VARS:
        if var in os.environ:
            status["required"][var] = os.environ[var]
        else:
            status["missing_required"].append(var)

    # Check optional variables
    for var in OPTIONAL_ENV_VARS:
        if var in os.environ:
            status["optional"][var] = os.environ[var]
        else:
            status["missing_optional"].append(var)

    all_ok = len(status["missing_required"]) == 0
    return all_ok, status


def check_python_packages() -> Tuple[bool, Dict[str, Any]]:
    """Check Python packages."""
    status = {
        "required": {},
        "missing_required": [],
        "optional": {},
        "missing_optional": [],
    }

    # Check required packages
    for package in REQUIRED_PACKAGES:
        try:
            __import__(package.replace("-", "_"))
            status["required"][package] = True
        except ImportError:
            status["missing_required"].append(package)

    # Check optional packages
    for package in OPTIONAL_PACKAGES:
        try:
            __import__(package.replace("-", "_"))
            status["optional"][package] = True
        except ImportError:
            status["missing_optional"].append(package)

    all_ok = len(status["missing_required"]) == 0
    return all_ok, status


def check_path_configuration() -> Tuple[bool, List[str]]:
    """Check Python path configuration."""
    issues = []

    # Check if project root is in sys.path
    if str(PROJECT_ROOT) not in sys.path:
        issues.append(f"Project root not in sys.path: {PROJECT_ROOT}")

    # Check if knowledge_engine is importable
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        import knowledge_engine
        print_info("knowledge_engine is importable")
    except ImportError as e:
        issues.append(f"Cannot import knowledge_engine: {e}")

    return len(issues) == 0, issues


def check_test_configuration() -> Tuple[bool, List[str]]:
    """Check test configuration files."""
    issues = []

    # Check conftest.py
    conftest = TESTS_DIR / "conftest.py"
    if conftest.exists():
        try:
            with open(conftest) as f:
                content = f.read()
                if "set_test_environment_defaults" not in content:
                    issues.append("conftest.py missing set_test_environment_defaults function")
        except Exception as e:
            issues.append(f"Error reading conftest.py: {e}")
    else:
        issues.append("conftest.py not found")

    # Check test_helpers.py
    helpers = TESTS_DIR / "test_helpers.py"
    if helpers.exists():
        try:
            with open(helpers) as f:
                content = f.read()
                if "safe_import" not in content:
                    issues.append("test_helpers.py missing safe_import function")
        except Exception as e:
            issues.append(f"Error reading test_helpers.py: {e}")
    else:
        issues.append("test_helpers.py not found")

    return len(issues) == 0, issues


# ============================================================================
# FIX FUNCTIONS
# ============================================================================

def fix_environment_variables() -> bool:
    """Set default environment variables."""
    defaults = {
        "TESTING": "true",
        "DATABASE_URL": "sqlite:///:memory:",
        "TEST_DATABASE_URL": "sqlite:///:memory:",
        "API_HOST": "localhost",
        "API_PORT": "8000",
        "API_TIMEOUT": "5",
        "LOG_LEVEL": "WARNING",
        "TEST_LOG_LEVEL": "WARNING",
        "TZ": "UTC",
    }

    for key, value in defaults.items():
        if key not in os.environ:
            os.environ[key] = value
            print_info(f"Set {key}={value}")

    return True


def fix_path_configuration() -> bool:
    """Add project root to sys.path."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
        print_info(f"Added {PROJECT_ROOT} to sys.path")
        return True
    return False


def install_missing_packages(packages: List[str]) -> bool:
    """Install missing Python packages."""
    if not packages:
        return True

    print_info(f"Installing packages: {', '.join(packages)}")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet"] + packages
        )
        print_success("Packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install packages: {e}")
        return False


# ============================================================================
# MAIN CHECK FUNCTION
# ============================================================================

def run_all_checks(fix: bool = False, verbose: bool = False) -> int:
    """Run all checks and return exit code."""
    exit_code = 0

    print_header("OpenEvolve Frontend Test Environment Check")

    # Check Python version
    print_header("Python Version", 2)
    ok, msg = check_python_version()
    if ok:
        print_success(msg)
    else:
        print_error(msg)
        exit_code = 1

    # Check project structure
    print_header("Project Structure", 2)
    ok, issues = check_project_structure()
    if ok:
        print_success("Project structure OK")
    else:
        print_error("Project structure issues found:")
        for issue in issues:
            print_info(f"  - {issue}")
        exit_code = 1

    # Check path configuration
    print_header("Path Configuration", 2)
    ok, issues = check_path_configuration()
    if ok:
        print_success("Path configuration OK")
    else:
        print_error("Path configuration issues found:")
        for issue in issues:
            print_info(f"  - {issue}")
        if fix:
            print_info("Attempting to fix...")
            fix_path_configuration()

    # Check environment variables
    print_header("Environment Variables", 2)
    ok, status = check_environment_variables()
    if ok:
        print_success("All required environment variables set")
        if verbose and status["optional"]:
            print_info("Optional variables set:")
            for var, value in status["optional"].items():
                print_info(f"  {var}={value}")
    else:
        print_error("Missing required environment variables:")
        for var in status["missing_required"]:
            print_info(f"  - {var}")
        if status["missing_optional"] and verbose:
            print_info("Missing optional variables:")
            for var in status["missing_optional"]:
                print_info(f"  - {var}")
        if fix:
            print_info("Setting default values...")
            fix_environment_variables()

    # Check Python packages
    print_header("Python Packages", 2)
    ok, status = check_python_packages()
    if ok:
        print_success("All required packages installed")
        if verbose and status["optional"]:
            print_info("Optional packages installed:")
            for pkg in status["optional"]:
                print_info(f"  - {pkg}")
    else:
        print_error("Missing required packages:")
        for pkg in status["missing_required"]:
            print_info(f"  - {pkg}")
        if status["missing_optional"] and verbose:
            print_info("Missing optional packages:")
            for pkg in status["missing_optional"]:
                print_info(f"  - {pkg}")
        if fix and status["missing_required"]:
            print_info("Attempting to install...")
            if not install_missing_packages(status["missing_required"]):
                exit_code = 1

    # Check test configuration
    print_header("Test Configuration", 2)
    ok, issues = check_test_configuration()
    if ok:
        print_success("Test configuration OK")
    else:
        print_error("Test configuration issues found:")
        for issue in issues:
            print_info(f"  - {issue}")
        exit_code = 1

    # Final summary
    print_header("Summary", 2)
    if exit_code == 0:
        print_success("All checks passed!")
        print_info("You can now run tests with: python -m pytest tests/")
    else:
        print_error("Some checks failed!")
        print_info("Run with --fix to attempt automatic fixes")
        print_info("Run with --verbose for detailed information")

    return exit_code


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check and initialize test environment"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Automatically fix issues where possible"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output"
    )

    args = parser.parse_args()

    exit_code = run_all_checks(fix=args.fix, verbose=args.verbose)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
