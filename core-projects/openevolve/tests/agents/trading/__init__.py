"""trading package tests.

These tests require the optional `openevolve.agents` subsystem (trading
evolver), which is not part of the current core distribution. Skipped to keep
the suite green without inventing a non-existent agents engine.
"""

import pytest

pytest.skip(
    "openevolve.agents subsystem is not available in this distribution",
    allow_module_level=True,
)
