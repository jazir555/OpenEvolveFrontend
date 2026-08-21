"""
Pytest configuration for OpenEvolve API tests
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Make the real OpenEvolve library importable as ``openevolve`` at collection
# time, BEFORE the test modules import it. The library source lives at
# ``core-projects/openevolve`` (its ``openevolve`` package directory), and it is
# also exposed via an editable install; prepending it guarantees
# ``import openevolve`` (and ``pytest.importorskip("openevolve")``) resolves here
# even in a clean environment without the editable install registered.
_OPENVOLVE_SRC = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "openevolve",
    )
)
if os.path.isdir(_OPENVOLVE_SRC) and _OPENVOLVE_SRC not in sys.path:
    sys.path.insert(0, _OPENVOLVE_SRC)


def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "integration: Integration tests (require running service)"
    )
    config.addinivalue_line(
        "markers", "unit: Unit tests (no external dependencies)"
    )
    config.addinivalue_line(
        "markers", "slow: Slow-running tests"
    )
