"""Test fixtures for Adaptive MDAP."""

import pytest
from adaptive_mdap.core.types import SubProblem
from adaptive_mdap.classifiers.task_complexity_classifier import (
    TaskComplexityClassifier,
    ClassifierConfig,
)
from adaptive_mdap.allocators.resource_allocator import AdaptiveMDAPAllocator


@pytest.fixture
def sample_subproblem():
    """Create a sample sub-problem for testing."""
    return SubProblem(
        id="test-001",
        description="Test sub-problem description",
        domain="mathematics",
        depth=2,
        dependencies=["dep1", "dep2"],
        metadata={},
    )


@pytest.fixture
def complex_subproblem():
    """Create a complex sub-problem for testing."""
    return SubProblem(
        id="test-complex",
        description="A very complex problem involving distributed concurrency and security refactor. "
                   "This requires extensive analysis of cryptographic scaling and recursive bottlenecks.",
        domain="quantum_computing",
        depth=8,
        dependencies=["dep1", "dep2", "dep3", "dep4", "dep5", "dep6", "dep7"],
        metadata={
            "constraints": ["must be O(log n)", "must be thread-safe"],
            "success_criteria": ["passes all tests", "no security leaks", "verified"]
        },
    )


@pytest.fixture
def simple_subproblem():
    """Create a simple sub-problem for testing."""
    return SubProblem(
        id="test-simple",
        description="Simple task",
        domain="basic_math",
        depth=0,
        dependencies=[],
        metadata={},
    )


@pytest.fixture
def classifier():
    """Create a classifier for testing."""
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
    return TaskComplexityClassifier(config)


@pytest.fixture
def allocator():
    """Create an allocator for testing."""
    return AdaptiveMDAPAllocator(
        thresholds=[0.2, 0.4, 0.6, 0.8],
        enable_learning=False,
    )


@pytest.fixture
def conservative_allocator():
    """Create a conservative allocator for testing."""
    return AdaptiveMDAPAllocator(
        thresholds=[0.1, 0.3, 0.5, 0.7],
        enable_learning=False,
    )


@pytest.fixture
def aggressive_allocator():
    """Create an aggressive allocator for testing."""
    return AdaptiveMDAPAllocator(
        thresholds=[0.3, 0.5, 0.7, 0.9],
        enable_learning=False,
    )
