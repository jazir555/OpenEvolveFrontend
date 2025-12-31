"""
Pytest configuration and fixtures for BubbleLabs integration tests.
"""

import sys
import os
import tempfile
from typing import List
import pytest

# Add frontend directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestResult:
    """Track test results for custom test reporting."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.errors: List[str] = []

    def add_pass(self):
        self.passed += 1

    def add_fail(self, error: str):
        self.failed += 1
        self.errors.append(error)

    def add_warning(self, warning: str):
        self.warnings += 1
        print(f"[!] WARNING: {warning}")

    def print_summary(self):
        total = self.passed + self.failed
        print("\n" + "=" * 70)
        print("Test Summary")
        print("=" * 70)
        print(f"Total Tests: {total}")
        print(f"[OK] Passed: {self.passed}")
        if self.failed > 0:
            print(f"[FAIL] Failed: {self.failed}")
        if self.warnings > 0:
            print(f"[!] Warnings: {self.warnings}")

        if self.errors:
            print("\nFailed Tests:")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")

        return self.failed == 0


@pytest.fixture
def result():
    """Provide a TestResult instance for test tracking."""
    return TestResult()


@pytest.fixture
def temp_db_path():
    """Provide a temporary database path that gets cleaned up."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    yield db_path
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def temp_dir():
    """Provide a temporary directory that gets cleaned up."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture(autouse=True)
def cleanup_test_resources():
    """Automatically clean up resources after each test."""
    yield
    # Force garbage collection to help with resource cleanup
    import gc
    gc.collect()


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "slow: marks tests as slow running")
