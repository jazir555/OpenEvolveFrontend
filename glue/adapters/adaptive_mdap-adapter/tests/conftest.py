"""
Pytest Configuration for Adaptive MDAP/MAKER Adapter Contract Tests

Federation Constitution - Section 4: Contract Tests
These tests verify that the API returns the specific fields we rely on.
"""

import os
import sys
import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "contract: Mark test as a contract test (API validation)"
    )
    config.addinivalue_line(
        "markers", "acl: Mark test as an Anti-Corruption Layer test"
    )
    config.addinivalue_line(
        "markers", "integration: Mark test as an integration test"
    )


@pytest.fixture(scope="session")
def test_correlation_id():
    """Provide a test correlation ID."""
    return "test-correlation-123456789"


@pytest.fixture(scope="session")
def test_timeout_ms():
    """Provide test timeout in milliseconds."""
    return 5000
