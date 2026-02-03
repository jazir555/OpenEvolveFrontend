"""
Pytest configuration and fixtures for Adaptive MDAP tests.
"""

import pytest
from adaptive_mdap.core.types import SubProblem
from adaptive_mdap.classifiers.task_complexity_classifier import (
    TaskComplexityClassifier,
    ClassifierConfig,
)
from adaptive_mdap.allocators.resource_allocator import AdaptiveMDAPAllocator
from adaptive_mdap.controllers.execution_controller import AdaptiveExecutionController


@pytest.fixture
def classifier():
    """Create a TaskComplexityClassifier for testing."""
    return TaskComplexityClassifier()


@pytest.fixture
def classifier_with_config():
    """Create a TaskComplexityClassifier with custom config."""
    config = ClassifierConfig(
        embedding_model="all-MiniLM-L6-v2",
        feature_weights={
            "text_length": 0.15,
            "domain_rarity": 0.20,
            "depth": 0.15,
            "historical_error": 0.20,
            "dependency": 0.10,
            "keyword_complexity": 0.10,
            "constraint_density": 0.10,
        },
    )
    return TaskComplexityClassifier(config=config)


@pytest.fixture
def allocator():
    """Create an AdaptiveMDAPAllocator for testing."""
    return AdaptiveMDAPAllocator()


@pytest.fixture
def context_aware_allocator():
    """Create a context-aware AdaptiveMDAPAllocator for testing."""
    return AdaptiveMDAPAllocator(
        enable_context_aware=True,
        enable_learning=False,
    )


@pytest.fixture
def controller():
    """Create an AdaptiveExecutionController for testing."""
    return AdaptiveExecutionController()


@pytest.fixture
def sample_subproblem():
    """Create a sample SubProblem for testing."""
    return SubProblem(
        id="test-sample",
        description="This is a sample problem for testing purposes.",
        domain="testing",
        depth=2,
        dependencies=["dep1", "dep2"],
        metadata={
            "constraints": ["must be fast"],
            "success_criteria": ["passes tests"],
        },
    )


@pytest.fixture
def simple_subproblem():
    """Create a simple SubProblem for testing."""
    return SubProblem(
        id="test-simple",
        description="Simple task",
        domain="basic",
        depth=0,
        dependencies=[],
        metadata={},
    )


@pytest.fixture
def complex_subproblem():
    """Create a complex SubProblem for testing."""
    return SubProblem(
        id="test-complex",
        description=(
            "This is an extremely complex problem involving distributed concurrency, "
            "security vulnerabilities, cryptographic protocols, and performance optimization. "
            "The solution must handle distributed consensus, implement secure communication "
            "channels, and optimize for high-throughput scenarios while maintaining correctness."
        ),
        domain="ultra_rare_quantum_biological_neural_encryption_domain",
        depth=10,
        dependencies=[f"dep{i}" for i in range(10)],
        metadata={
            "constraints": ["must be O(log n)", "must be thread-safe", "must be cryptographically secure"],
            "success_criteria": ["passes all tests", "no security leaks", "verified", "optimized"],
        },
    )


@pytest.fixture
def medium_subproblem():
    """Create a medium complexity SubProblem for testing."""
    return SubProblem(
        id="test-medium",
        description="Implement a REST API with authentication and database integration.",
        domain="backend_development",
        depth=3,
        dependencies=["user_model", "database"],
        metadata={
            "constraints": ["RESTful", "secure"],
        },
    )


@pytest.fixture(scope="session")
def test_cache_dir(tmp_path_factory):
    """Create a temporary cache directory for tests."""
    return tmp_path_factory.mktemp("adaptive_mdap_cache")


@pytest.fixture
def clean_allocator():
    """Create a fresh allocator with clean stats."""
    allocator = AdaptiveMDAPAllocator()
    allocator.reset_stats()
    return allocator
