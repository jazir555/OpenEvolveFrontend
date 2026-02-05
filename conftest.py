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

# Add schemas directory to path for RESE tests
_schemas_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "glue", "schemas")
if _schemas_dir not in sys.path:
    sys.path.insert(0, _schemas_dir)

# **ACTUAL INTEGRATION**: Alerting, knowledge, and adaptive for Test Configuration
try:
    from alerting_system import get_alert_manager, AlertSeverity
    ALERTING_AVAILABLE = True
except ImportError:
    ALERTING_AVAILABLE = False

try:
    from knowledge_engine.enterprise_knowledge_engine import enterprise_knowledge_engine, KnowledgeArtifact
    KNOWLEDGE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_AVAILABLE = False

try:
    from adaptive_strategy_selector import StrategyPerformanceTracker, StrategyPerformanceData
    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False


# **ACTUAL INTEGRATION HELPER METHODS**: Test Configuration
def _trigger_test_alerts(operation, success, test_id=None, error=None, metadata=None):
    """Trigger alerts for test configuration operations"""
    if not ALERTING_AVAILABLE:
        return

    try:
        alert_mgr = get_alert_manager()
        if success:
            return  # No alerts for successful operations

        alert_mgr.trigger_alert(
            title=f"Test Config {operation} Failed",
            message=f"Test configuration operation '{operation}' failed: {error}",
            severity=AlertSeverity.LOW,
            source="TestConfig",
            metadata=metadata or {"test_id": test_id, "operation": operation}
        )
    except Exception:
        pass  # Suppress errors during test configuration


def _extract_test_knowledge(operation, test_id, result):
    """Extract knowledge from test operations"""
    if not KNOWLEDGE_AVAILABLE:
        return

    try:
        from datetime import datetime
        artifact = KnowledgeArtifact(
            artifact_id=f"test_config_{operation}_{test_id}",
            artifact_type="test_execution",
            source_component="TestConfig",
            content={
                "operation": operation,
                "test_id": test_id,
                "success": getattr(result, 'failed', 0) == 0,
            },
            metadata={"timestamp": datetime.utcnow().isoformat()}
        )
        enterprise_knowledge_engine.store_artifact(artifact)
    except Exception:
        pass  # Suppress errors during test configuration


def _track_test_performance(operation, success, duration_seconds):
    """Track performance of test operations"""
    if not ADAPTIVE_AVAILABLE:
        return

    try:
        tracker = StrategyPerformanceTracker.get_instance()
        data = StrategyPerformanceData(
            strategy_name="test_configuration",
            component_name="TestConfig",
            operation_name=operation,
            success=success,
            duration_seconds=duration_seconds,
            metadata={}
        )
        tracker.record_execution(data)
    except Exception:
        pass  # Suppress errors during test configuration


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


# Platform detection for skip logic
import platform

def is_windows() -> bool:
    """Check if running on Windows."""
    return platform.system() == "Windows"

def has_cuda() -> bool:
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

# Skip markers for platform-specific tests
skip_on_windows = pytest.mark.skipif(is_windows(), reason="Test skipped on Windows")
skip_without_cuda = pytest.mark.skipif(not has_cuda(), reason="Test requires CUDA")
