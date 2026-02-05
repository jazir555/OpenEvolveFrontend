"""
Pytest configuration for gauntlet monitoring tests
"""

import pytest
import sys
from pathlib import Path
import importlib.util

# Add parent directories to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import monitoring modules directly and add to sys.modules
monitoring_path = project_root / 'glue' / 'adapters' / 'gauntlet-adapter' / 'monitoring'

# Load metrics module
metrics_spec = importlib.util.spec_from_file_location(
    "gauntlet_metrics",
    monitoring_path / "metrics.py"
)
metrics_module = importlib.util.module_from_spec(metrics_spec)
sys.modules["gauntlet_metrics"] = metrics_module
metrics_spec.loader.exec_module(metrics_module)

# Load health_checks module
health_spec = importlib.util.spec_from_file_location(
    "gauntlet_health_checks",
    monitoring_path / "health_checks.py"
)
health_module = importlib.util.module_from_spec(health_spec)
sys.modules["gauntlet_health_checks"] = health_module
health_spec.loader.exec_module(health_module)

# Load alerting module
alerting_spec = importlib.util.spec_from_file_location(
    "gauntlet_alerting",
    monitoring_path / "alerting.py"
)
alerting_module = importlib.util.module_from_spec(alerting_spec)
sys.modules["gauntlet_alerting"] = alerting_module
alerting_spec.loader.exec_module(alerting_module)

# Create a mock monitoring module that imports from all three
import types
monitoring_module = types.ModuleType("monitoring")
for name in dir(metrics_module):
    if not name.startswith("_"):
        setattr(monitoring_module, name, getattr(metrics_module, name))
for name in dir(health_module):
    if not name.startswith("_"):
        setattr(monitoring_module, name, getattr(health_module, name))
for name in dir(alerting_module):
    if not name.startswith("_"):
        setattr(monitoring_module, name, getattr(alerting_module, name))
sys.modules["monitoring"] = monitoring_module


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "monitoring: marks tests as monitoring tests"
    )
