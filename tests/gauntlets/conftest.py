"""
Pytest configuration for gauntlet tests

Provides shared fixtures and configuration for all gauntlet test modules.
"""

import pytest
import asyncio
import sys
from pathlib import Path
from typing import AsyncGenerator
import importlib.util

# Add parent directories to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'fixtures'))

# Import gauntlet-adapter src modules directly
src_path = project_root / 'glue' / 'adapters' / 'gauntlet-adapter' / 'src'

# Load adaptive_learner
adaptive_learner_spec = importlib.util.spec_from_file_location(
    "adaptive_learner",
    src_path / "adaptive_learner.py"
)
adaptive_learner_module = importlib.util.module_from_spec(adaptive_learner_spec)
sys.modules["adaptive_learner"] = adaptive_learner_module
adaptive_learner_spec.loader.exec_module(adaptive_learner_module)

# Load ml_optimizer
ml_optimizer_spec = importlib.util.spec_from_file_location(
    "ml_optimizer",
    src_path / "ml_optimizer.py"
)
ml_optimizer_module = importlib.util.module_from_spec(ml_optimizer_spec)
sys.modules["ml_optimizer"] = ml_optimizer_module
ml_optimizer_spec.loader.exec_module(ml_optimizer_module)

# Load predictive_gauntlet_executor
executor_spec = importlib.util.spec_from_file_location(
    "predictive_gauntlet_executor",
    src_path / "predictive_gauntlet_executor.py"
)
executor_module = importlib.util.module_from_spec(executor_spec)
sys.modules["predictive_gauntlet_executor"] = executor_module
executor_spec.loader.exec_module(executor_module)

# Note: intelligent_orchestrator is skipped due to syntax errors in the source file
# It can be added back once the source is fixed


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests (deselect with '-m \"not performance\"')"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def async_setup():
    """Setup for async tests"""
    # Setup code here
    yield
    # Teardown code here


@pytest.fixture(scope="session")
def test_data_dir():
    """Get test data directory"""
    return Path(__file__).parent.parent / 'data' / 'gauntlet_solutions'


@pytest.fixture(scope="session")
def fixtures_dir():
    """Get fixtures directory"""
    return Path(__file__).parent.parent / 'fixtures'


# Performance test configuration
@pytest.fixture
def perf_config():
    """Performance test configuration"""
    return {
        'round1_target': 30.0,
        'round2_target': 120.0,
        'round3_target': 300.0,
        'full_pipeline_target': 480.0,
        'iterations': 10
    }


# Quality metrics configuration
@pytest.fixture
def quality_config():
    """Quality metrics configuration"""
    return {
        'max_false_positive_rate': 0.05,
        'max_false_negative_rate': 0.10,
        'min_precision': 0.90,
        'min_recall': 0.85,
        'min_f1': 0.87
    }
