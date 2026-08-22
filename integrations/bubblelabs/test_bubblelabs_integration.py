"""
Pytest entry point for the BubbleLabs integration test suite.

The suite itself lives in :mod:`.bubblelabs_integration_tests`, whose filename
does not match the repo-wide ``python_files = test_*.py`` pattern in
``pytest.ini``. Re-exporting its ``unittest.TestCase`` classes here lets
``python -m pytest integrations/bubblelabs/`` collect them without widening that
global pattern, which would otherwise sweep in ~59 unrelated ``*_tests.py``
runner scripts across the repo.

The original module stays directly runnable too::

    python -m pytest integrations/bubblelabs/bubblelabs_integration_tests.py
    python integrations/bubblelabs/bubblelabs_integration_tests.py
"""
from __future__ import annotations


try:
    from .bubblelabs_integration_tests import (
    TestAnalyticsMonitoringDashboard,
    TestIntegration,
    TestOpenEvolveBubbleLabsAPI,
    TestOpenEvolveVisualizer,
    TestParameterSyncManager,
    TestWorkflowLifecycleController,
    )
except ImportError:
    from bubblelabs_integration_tests import (
    TestAnalyticsMonitoringDashboard,
    TestIntegration,
    TestOpenEvolveBubbleLabsAPI,
    TestOpenEvolveVisualizer,
    TestParameterSyncManager,
    TestWorkflowLifecycleController,
    )

__all__ = [
    "TestAnalyticsMonitoringDashboard",
    "TestIntegration",
    "TestOpenEvolveBubbleLabsAPI",
    "TestOpenEvolveVisualizer",
    "TestParameterSyncManager",
    "TestWorkflowLifecycleController",
]
