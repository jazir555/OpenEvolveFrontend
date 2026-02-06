"""
Test Helper Functions for OpenEvolve Frontend

Provides reusable helper functions for common testing scenarios:
- Mock creation helpers
- Test data generators
- Assertion helpers
- Configuration builders
- Import testing utilities

Author: OpenEvolve Distinguished Engineer
Version: 1.0.0
"""

import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar
from unittest.mock import Mock, AsyncMock, MagicMock
from datetime import datetime, timezone
import asyncio
import json
import importlib


# ============================================================================
# IMPORT TESTING UTILITIES
# ============================================================================

def safe_import(module_path: str) -> tuple[bool, Optional[Any], str]:
    """
    Safely attempt to import a module.

    Args:
        module_path: Dot-separated module path (e.g., 'knowledge_engine.integrations.roma_integration')

    Returns:
        Tuple of (success: bool, module: Optional[ModuleType], error_message: str)
    """
    try:
        module = importlib.import_module(module_path)
        return True, module, ""
    except ImportError as e:
        return False, None, f"ImportError: {str(e)}"
    except Exception as e:
        return False, None, f"Error: {type(e).__name__}: {str(e)}"


def check_integration_available(integration_name: str) -> bool:
    """
    Check if a specific integration is available for testing.

    Args:
        integration_name: Name of the integration (e.g., 'roma', 'dspy', 'deepke')

    Returns:
        True if integration can be imported, False otherwise
    """
    module_map = {
        'roma': 'knowledge_engine.integrations.roma_integration',
        'dspy': 'knowledge_engine.integrations.dspy_integration',
        'deepke': 'knowledge_engine.integrations.deepke_integration',
        'ragbits': 'knowledge_engine.integrations.ragbits_integration',
        'crewai': 'knowledge_engine.integrations.crewai_integration',
        'lagrange_mapper': 'knowledge_engine.integrations.lagrange_mapper_integration',
        'neuromancer': 'knowledge_engine.integrations.neuromancer_integration',
        'oneke': 'knowledge_engine.integrations.oneke_integration',
        'graphiti': 'knowledge_engine.integrations.graphiti_integration',
        'roma_dspy': 'knowledge_engine.integrations.roma_dspy_integration',
        'roma_deepke': 'knowledge_engine.integrations.roma_deepke_integration',
        'roma_ragbits': 'knowledge_engine.integrations.roma_ragbits_integration',
    }

    module_path = module_map.get(integration_name)
    if not module_path:
        return False

    success, _, _ = safe_import(module_path)
    return success


# ============================================================================
# MOCK CREATION HELPERS
# ============================================================================

def create_mock_entity(
    entity_id: str = "test-entity-001",
    entity_type: str = "test_type",
    name: str = "Test Entity",
    properties: Optional[Dict[str, Any]] = None
) -> Mock:
    """Create a mock entity with common attributes."""
    entity = Mock()
    entity.entity_id = entity_id
    entity.entity_type = entity_type
    entity.name = name
    entity.properties = properties or {
        "description": "A test entity",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "active"
    }
    entity.metadata = {"source": "test", "confidence": 0.95}
    entity.to_dict = Mock(return_value={
        "entity_id": entity_id,
        "entity_type": entity_type,
        "name": name,
        "properties": entity.properties,
        "metadata": entity.metadata
    })
    return entity


def create_mock_relationship(
    relationship_id: str = "test-rel-001",
    source_id: str = "entity-001",
    target_id: str = "entity-002",
    relationship_type: str = "test_relationship"
) -> Mock:
    """Create a mock relationship with common attributes."""
    rel = Mock()
    rel.relationship_id = relationship_id
    rel.source_id = source_id
    rel.target_id = target_id
    rel.relationship_type = relationship_type
    rel.properties = {
        "weight": 0.8,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    rel.metadata = {"source": "test", "confidence": 0.9}
    rel.to_dict = Mock(return_value={
        "relationship_id": relationship_id,
        "source_id": source_id,
        "target_id": target_id,
        "relationship_type": relationship_type,
        "properties": rel.properties,
        "metadata": rel.metadata
    })
    return rel


def create_mock_knowledge_artifact(
    artifact_id: str = "artifact-001",
    artifact_type: str = "test_artifact"
) -> Mock:
    """Create a mock knowledge artifact."""
    artifact = Mock()
    artifact.artifact_id = artifact_id
    artifact.artifact_type = artifact_type
    artifact.content = {"test": "data"}
    artifact.source = "test"
    artifact.created_at = datetime.now(timezone.utc).isoformat()
    artifact.metadata = {"test": "metadata"}
    artifact.to_dict = Mock(return_value={
        "artifact_id": artifact_id,
        "artifact_type": artifact_type,
        "content": artifact.content,
        "source": artifact.source,
        "created_at": artifact.created_at,
        "metadata": artifact.metadata
    })
    return artifact


def create_mock_roma_decomposition(
    decomposition_id: str = "decomp-001",
    problem: str = "Test problem",
    is_atomic: bool = True,
    depth: int = 0
) -> Mock:
    """Create a mock ROMA decomposition."""
    decomp = Mock()
    decomp.decomposition_id = decomposition_id
    decomp.problem = problem
    decomp.sub_problems = []
    decomp.is_atomic = is_atomic
    decomp.depth = depth
    decomp.parent_id = None
    decomp.metadata = {}
    decomp.created_at = datetime.now(timezone.utc).isoformat()
    return decomp


def create_mock_roma_solution(
    solution_id: str = "solution-001",
    problem_id: str = "prob-001",
    confidence: float = 0.9
) -> Mock:
    """Create a mock ROMA solution."""
    solution = Mock()
    solution.solution_id = solution_id
    solution.problem_id = problem_id
    solution.solution = "test solution"
    solution.confidence = confidence
    solution.reasoning = "test reasoning"
    solution.metadata = {}
    solution.created_at = datetime.now(timezone.utc).isoformat()
    return solution


def create_mock_roma_verification(
    verification_id: str = "verify-001",
    solution_id: str = "solution-001",
    passed: bool = True,
    score: float = 0.85
) -> Mock:
    """Create a mock ROMA verification."""
    verification = Mock()
    verification.verification_id = verification_id
    verification.solution_id = solution_id
    verification.passed = passed
    verification.score = score
    verification.feedback = "Test feedback"
    verification.requirements_met = {}
    verification.metadata = {}
    verification.created_at = datetime.now(timezone.utc).isoformat()
    return verification


# ============================================================================
# ASYNC MOCK HELPERS
# ============================================================================

def create_async_mock(return_value: Any = None, side_effect: Any = None) -> AsyncMock:
    """Create an AsyncMock with the specified return value or side effect."""
    mock = AsyncMock()
    if return_value is not None:
        mock.return_value = return_value
    if side_effect is not None:
        mock.side_effect = side_effect
    return mock


async def run_async(coro):
    """Run an async coroutine in tests."""
    return await coro


def run_sync(coro):
    """Run an async coroutine synchronously (for non-async test functions)."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


# ============================================================================
# TEST DATA GENERATORS
# ============================================================================

def generate_test_entities(count: int = 5) -> List[Mock]:
    """Generate a list of mock test entities."""
    return [
        create_mock_entity(
            entity_id=f"entity-{i:03d}",
            entity_type=f"type_{i % 3}",
            name=f"Entity {i}"
        )
        for i in range(count)
    ]


def generate_test_relationships(count: int = 5) -> List[Mock]:
    """Generate a list of mock test relationships."""
    return [
        create_mock_relationship(
            relationship_id=f"rel-{i:03d}",
            source_id=f"entity-{i:03d}",
            target_id=f"entity-{(i+1) % count:03d}"
        )
        for i in range(count)
    ]


def generate_test_problems(count: int = 3) -> List[Dict[str, Any]]:
    """Generate test problem data for ROMA-style testing."""
    return [
        {
            "problem_id": f"problem-{i:03d}",
            "problem": f"Test problem {i}",
            "domain": "test_domain",
            "complexity": ["low", "medium", "high"][i % 3],
            "metadata": {
                "priority": ["low", "medium", "high"][i % 3],
                "created_at": datetime.now(timezone.utc).isoformat()
            }
        }
        for i in range(count)
    ]


# ============================================================================
# CONFIGURATION BUILDERS
# ============================================================================

def build_test_config(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Build a test configuration with optional overrides.

    Args:
        overrides: Dictionary of configuration values to override

    Returns:
        Complete test configuration dictionary
    """
    base_config = {
        "api": {
            "host": "localhost",
            "port": 8000,
            "timeout": 5,
            "retry_attempts": 3,
            "retry_delay": 0.1
        },
        "database": {
            "url": "sqlite:///:memory:",
            "pool_size": 1,
            "max_overflow": 0
        },
        "knowledge_engine": {
            "enabled": False,
            "entity_extraction": False,
            "relationship_extraction": False,
            "artifact_storage": False
        },
        "roma": {
            "enabled": False,
            "max_decomposition_depth": 3,
            "max_parallel_problems": 5,
            "timeout_seconds": 30
        },
        "dspy": {
            "enabled": False,
            "max_iterations": 10
        },
        "deepke": {
            "enabled": False,
            "model_name": "test-model"
        },
        "testing": {
            "mock_external_services": True,
            "use_in_memory_db": True,
            "disable_async": False,
            "log_level": "WARNING"
        },
        "logging": {
            "level": "WARNING",
            "format": "json",
            "include_timestamp": True,
            "include_correlation_id": True
        }
    }

    if overrides:
        # Deep merge overrides
        for key, value in overrides.items():
            if isinstance(value, dict) and key in base_config and isinstance(base_config[key], dict):
                base_config[key].update(value)
            else:
                base_config[key] = value

    return base_config


# ============================================================================
# ASSERTION HELPERS
# ============================================================================

def assert_valid_entity(entity: Any, entity_type: Optional[str] = None):
    """
    Assert that an entity object has valid structure.

    Args:
        entity: Entity object to validate
        entity_type: Expected entity type (optional)

    Raises:
        AssertionError: If entity is invalid
    """
    assert hasattr(entity, 'entity_id'), "Entity missing entity_id"
    assert hasattr(entity, 'entity_type'), "Entity missing entity_type"
    assert hasattr(entity, 'properties'), "Entity missing properties"

    if entity_type:
        assert entity.entity_type == entity_type, f"Expected type {entity_type}, got {entity.entity_type}"

    assert entity.entity_id, "Entity ID cannot be empty"


def assert_valid_relationship(rel: Any, rel_type: Optional[str] = None):
    """
    Assert that a relationship object has valid structure.

    Args:
        rel: Relationship object to validate
        rel_type: Expected relationship type (optional)

    Raises:
        AssertionError: If relationship is invalid
    """
    assert hasattr(rel, 'relationship_id'), "Relationship missing relationship_id"
    assert hasattr(rel, 'source_id'), "Relationship missing source_id"
    assert hasattr(rel, 'target_id'), "Relationship missing target_id"
    assert hasattr(rel, 'relationship_type'), "Relationship missing relationship_type"

    if rel_type:
        assert rel.relationship_type == rel_type, f"Expected type {rel_type}, got {rel.relationship_type}"

    assert rel.source_id, "Source ID cannot be empty"
    assert rel.target_id, "Target ID cannot be empty"


def assert_valid_artifact(artifact: Any, artifact_type: Optional[str] = None):
    """
    Assert that a knowledge artifact has valid structure.

    Args:
        artifact: Artifact object to validate
        artifact_type: Expected artifact type (optional)

    Raises:
        AssertionError: If artifact is invalid
    """
    assert hasattr(artifact, 'artifact_id'), "Artifact missing artifact_id"
    assert hasattr(artifact, 'artifact_type'), "Artifact missing artifact_type"
    assert hasattr(artifact, 'content'), "Artifact missing content"

    if artifact_type:
        assert artifact.artifact_type == artifact_type, f"Expected type {artifact_type}, got {artifact.artifact_type}"

    assert artifact.artifact_id, "Artifact ID cannot be empty"


def assert_logs_contain(log_records: List[Any], expected_messages: List[str]):
    """
    Assert that log records contain expected messages.

    Args:
        log_records: List of log records
        expected_messages: List of message fragments that should be present

    Raises:
        AssertionError: If any expected message is not found
    """
    log_messages = [str(record) for record in log_records]

    for expected_msg in expected_messages:
        found = any(expected_msg in msg for msg in log_messages)
        assert found, f"Expected message not found in logs: {expected_msg}"


def assert_valid_timestamp(timestamp_str: str):
    """
    Assert that a timestamp string is valid ISO-8601 format.

    Args:
        timestamp_str: Timestamp string to validate

    Raises:
        AssertionError: If timestamp is invalid
    """
    try:
        datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
    except ValueError as e:
        raise AssertionError(f"Invalid timestamp format: {timestamp_str}") from e


# ============================================================================
# ENVIRONMENT HELPERS
# ============================================================================

def set_test_env_var(key: str, value: str):
    """Set an environment variable for testing."""
    os.environ[key] = value


def clear_test_env_var(key: str):
    """Clear an environment variable if it exists."""
    if key in os.environ:
        del os.environ[key]


def with_env_vars(env_vars: Dict[str, str]):
    """
    Context manager to temporarily set environment variables.

    Usage:
        with with_env_vars({"TEST_VAR": "value"}):
            # Test code here
            pass
    """
    class EnvVarContext:
        def __init__(self, vars_dict: Dict[str, str]):
            self.vars_dict = vars_dict
            self.old_values = {}

        def __enter__(self):
            for key, value in self.vars_dict.items():
                self.old_values[key] = os.environ.get(key)
                os.environ[key] = value
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            for key, old_value in self.old_values.items():
                if old_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old_value

    return EnvVarContext(env_vars)


# ============================================================================
# FIXTURE SKIP HELPERS
# ============================================================================

def skip_if_integration_missing(integration_name: str):
    """
    Return a pytest skip reason if integration is missing.

    Usage in tests:
        if skip_reason := skip_if_integration_missing('roma'):
            pytest.skip(skip_reason)
    """
    if not check_integration_available(integration_name):
        return f"Integration '{integration_name}' not available - install dependencies or skip with -m 'not {integration_name}'"
    return None


def skip_if_env_var_missing(env_var: str):
    """
    Return a pytest skip reason if environment variable is missing.

    Usage in tests:
        if skip_reason := skip_if_env_var_missing('API_KEY'):
            pytest.skip(skip_reason)
    """
    if env_var not in os.environ:
        return f"Environment variable '{env_var}' not set - set it or skip with -m 'not requires_env'"
    return None


# ============================================================================
# PROJECT PATH HELPERS
# ============================================================================

def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_tests_dir() -> Path:
    """Get the tests directory."""
    return Path(__file__).parent


def get_test_data_dir() -> Path:
    """Get the test data directory."""
    return get_tests_dir() / "data"


def get_fixtures_dir() -> Path:
    """Get the fixtures directory."""
    return get_tests_dir() / "fixtures"
