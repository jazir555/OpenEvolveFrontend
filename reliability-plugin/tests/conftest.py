"""
Pytest configuration and shared fixtures for Reliability Plugin tests.

This module provides shared fixtures for testing all reliability components:
- Mock configurations
- Mock adapters
- Test prompts and tasks
- Test constraints
- Shared utilities
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# =============================================================================
# SHARED FIXTURES
# =============================================================================

@pytest.fixture
def mock_config():
    """Create a mock ReliabilityConfig for testing."""
    return {
        "lmql": {
            "enabled": True,
            "model": "gpt-4",
            "max_tokens": 1000,
            "temperature": 0.7,
            "timeout": 30
        },
        "guardrails": {
            "enabled": True,
            "validators": [
                "toxic_language",
                "pii_detection",
                "json_structure",
                "roma_length"
            ],
            "on_fail": "fix",
            "max_retries": 3
        },
        "roma": {
            "enabled": True,
            "max_depth": 3,
            "execution_mode": "recursive",
            "enable_checkpoints": True,
            "core_integration": True
        },
        "mdap": {
            "enabled": True,
            "max_votes": 100,
            "validation_threshold": 0.8,
            "core_integration": True
        },
        "unified_bridge": {
            "enabled": True,
            "max_retries": 3,
            "retry_delay": 1.0,
            "batch_size": 5,
            "enable_coordination": True
        }
    }


@pytest.fixture
def mock_lmql_adapter():
    """Create a mock LMQL adapter for testing."""
    adapter = Mock()
    adapter.is_available.return_value = True
    adapter.get_status.return_value = {
        "lmql_available": True,
        "model": "gpt-4",
        "max_tokens": 1000,
        "temperature": 0.7
    }

    # Mock constraint creation
    mock_constraint = Mock()
    mock_constraint.type = "max_tokens"
    mock_constraint.value = 1000
    adapter.create_constraint.return_value = mock_constraint

    # Mock constrained generation
    mock_result = Mock()
    mock_result.success = True
    mock_result.text = "Test generated text"
    mock_result.tokens_used = 100
    mock_result.constraint_violations = []
    adapter.constrained_generation.return_value = mock_result

    # Mock structured generation
    mock_structured_result = Mock()
    mock_structured_result.success = True
    mock_structured_result.data = {"test": "data"}
    mock_structured_result.json_valid = True
    adapter.structured_generation.return_value = mock_structured_result

    # Mock availability check
    adapter.check_availability.return_value = True

    return adapter


@pytest.fixture
def mock_guardrails_adapter():
    """Create a mock Guardrails adapter for testing."""
    adapter = Mock()
    adapter.is_available.return_value = True
    adapter.get_status.return_value = {
        "guardrails_enabled": True,
        "validators": ["toxic_language", "pii_detection", "json_structure"],
        "validation_mode": "strict"
    }

    # Mock input validation
    mock_validation_result = Mock()
    mock_validation_result.is_valid = True
    mock_validation_result.failures = []
    mock_validation_result.remediation_applied = None
    adapter.validate_input.return_value = mock_validation_result

    # Mock output validation
    mock_output_result = Mock()
    mock_output_result.is_valid = True
    mock_output_result.failures = []
    mock_output_result.remediation_applied = None
    mock_output_result.output = None
    adapter.validate_output.return_value = mock_output_result

    # Mock validator registration
    adapter.register_validator.return_value = True

    # Mock batch validation
    mock_batch_result = Mock()
    mock_batch_result.results = [mock_validation_result]
    mock_batch_result.all_valid = True
    mock_batch_result.failures = []
    adapter.batch_validate.return_value = mock_batch_result

    # Mock statistics
    adapter.get_statistics.return_value = {
        "total_validations": 100,
        "successful_validations": 95,
        "failed_validations": 5,
        "remediations_applied": 3
    }

    return adapter


@pytest.fixture
def mock_roma_core():
    """Create a mock ROMA core for testing."""
    core = Mock()
    core.RecursiveSolver = Mock()
    core.solve = Mock()
    core.async_solve = Mock()
    core.event_solve = Mock()
    core.ROMAConfig = Mock()

    # Mock solver
    mock_solver = Mock()
    mock_solver.solve.return_value = Mock(
        result="Test solution",
        status=Mock(value="completed")
    )
    mock_solver.event_solve.return_value = Mock(
        result="Test solution",
        status=Mock(value="completed")
    )
    core.RecursiveSolver.return_value = mock_solver

    # Mock config
    mock_config = Mock()
    mock_config.max_depth = 3
    mock_config.enable_checkpoints = True
    core.ROMAConfig.return_value = mock_config

    return core


@pytest.fixture
def mock_roma_mcp_tools():
    """Create a mock ROMA MCP tools for testing."""
    tools = Mock()
    tools.solve_with_roma = Mock()
    tools.analyze_with_roma = Mock()
    tools.verify_with_roma = Mock()
    tools.critique_with_roma = Mock()
    tools.get_roma_status = Mock()

    # Mock solve function
    mock_result = {
        "result": "Test solution",
        "status": "completed",
        "token_usage": {"input": 100, "output": 200},
        "execution_time": 2.5
    }
    tools.solve_with_roma.return_value = mock_result

    # Mock analyze function
    mock_analysis = {
        "analysis": {"complexity": "medium", "decomposition": ["step1", "step2"]},
        "status": "completed"
    }
    tools.analyze_with_roma.return_value = mock_analysis

    # Mock status
    mock_status = {
        "available": True,
        "version": "1.0.0",
        "components": ["RecursiveSolver", "Planner", "Executor"]
    }
    tools.get_roma_status.return_value = mock_status

    return tools


@pytest.fixture
def mock_mdap_core():
    """Create a mock MDAP core for testing."""
    core = Mock()
    core.MDAPSolver = Mock()
    core.VoteValidator = Mock()
    core.StatisticsTracker = Mock()

    # Mock solver
    mock_solver = Mock()
    mock_solver.solve.return_value = Mock(
        votes=[Mock(content="vote1", score=0.9), Mock(content="vote2", score=0.8)],
        final_decision="Test decision"
    )
    core.MDAPSolver.return_value = mock_solver

    # Mock validator
    mock_validator = Mock()
    mock_validator.validate_vote.return_value = Mock(
        is_valid=True,
        failures=[],
        remediated=False
    )
    core.VoteValidator.return_value = mock_validator

    # Mock statistics
    mock_stats = Mock()
    mock_stats.get_statistics.return_value = {
        "total_votes": 100,
        "valid_votes": 95,
        "invalid_votes": 5,
        "average_score": 0.85
    }
    core.StatisticsTracker.return_value = mock_stats

    return core


@pytest.fixture
def test_prompts():
    """Provide common test prompts for testing."""
    return {
        "simple_task": "What is the capital of France?",
        "complex_task": "Design a scalable web application architecture for e-commerce with microservices",
        "math_problem": "Solve for x: 2x + 5 = 15",
        "code_generation": "Write a Python function to calculate fibonacci numbers",
        "analysis_task": "Analyze the time complexity of quicksort algorithm"
    }


@pytest.fixture
def test_constraints():
    """Provide common test constraints for testing."""
    return {
        "basic": {
            "max_depth": 3,
            "max_subtasks": 10,
            "subtask_token_limit": 500
        },
        "strict": {
            "max_depth": 2,
            "max_subtasks": 5,
            "subtask_token_limit": 200,
            "require_json": True
        },
        "lenient": {
            "max_depth": 5,
            "max_subtasks": 20,
            "subtask_token_limit": 1000
        }
    }


@pytest.fixture
def test_tasks():
    """Provide common test tasks for testing."""
    return [
        {
            "id": "task_1",
            "description": "Simple calculation task",
            "complexity": "low",
            "expected_type": "numerical"
        },
        {
            "id": "task_2",
            "description": "Complex reasoning task",
            "complexity": "high",
            "expected_type": "structured"
        },
        {
            "id": "task_3",
            "description": "Code generation task",
            "complexity": "medium",
            "expected_type": "code"
        }
    ]


@pytest.fixture
def mock_correlation_id():
    """Generate a correlation ID for testing."""
    return f"test_correlation_{datetime.utcnow().timestamp()}"


@pytest.fixture
def sample_validation_failures():
    """Provide sample validation failures for testing."""
    return [
        {
            "validator": "toxic_language",
            "message": "Toxic content detected",
            "severity": "high",
            "remediation": "Remove offensive content"
        },
        {
            "validator": "pii_detection",
            "message": "PII detected in output",
            "severity": "medium",
            "remediation": "Mask or remove PII"
        }
    ]


@pytest.fixture
def sample_constraint_violations():
    """Provide sample constraint violations for testing."""
    return [
        {
            "constraint": "max_depth",
            "violated": True,
            "actual_value": 5,
            "allowed_value": 3,
            "message": "Maximum depth exceeded"
        },
        {
            "constraint": "max_tokens",
            "violated": True,
            "actual_value": 1500,
            "allowed_value": 1000,
            "message": "Token limit exceeded"
        }
    ]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_mock_result(success: bool = True, data: Any = None, error: str = None):
    """Create a mock result object for testing."""
    result = Mock()
    result.success = success
    result.data = data
    result.error = error
    return result


def assert_adapter_basic_health(adapter, expected_available: bool = True):
    """Assert basic health of an adapter."""
    status = adapter.get_status()
    assert isinstance(status, dict)
    assert "available" in status
    assert status["available"] == expected_available


def assert_result_structure(result, expected_success: bool = True):
    """Assert the structure of a result object."""
    assert hasattr(result, 'success')
    assert result.success == expected_success
    if hasattr(result, 'error') and expected_success:
        assert result.error is None
    if hasattr(result, 'data') and expected_success:
        assert result.data is not None


# =============================================================================
# PARAMETERIZED FIXTURES
# =============================================================================

@pytest.fixture(params=[True, False])
def lmql_availability(request):
    """Parameterized fixture for LMQL availability."""
    return request.param


@pytest.fixture(params=[True, False])
def guardrails_availability(request):
    """Parameterized fixture for Guardrails availability."""
    return request.param


@pytest.fixture(params=["recursive", "event_driven"])
def execution_mode(request):
    """Parameterized fixture for execution modes."""
    return request.param


@pytest.fixture(params=[True, False])
def enable_constraints(request):
    """Parameterized fixture for constraint enabling."""
    return request.param


@pytest.fixture(params=["fix", "reask", "exception"])
def on_fail_strategy(request):
    """Parameterized fixture for on-fail strategies."""
    return request.param


# =============================================================================
# CONTEXT MANAGERS
# =============================================================================

@pytest.fixture
def patch_environment():
    """Context manager to temporarily patch environment variables."""
    with patch.dict(os.environ, {
        "OPENAI_API_KEY": "test_key",
        "LMQL_MODEL": "gpt-4",
        "GUARDRAILS_ENABLED": "true"
    }):
        yield


@pytest.fixture
def patch_unavailable_services():
    """Context manager to patch services as unavailable."""
    with patch.dict('sys.modules', {
        'lmql': None,
        'guardrails': None,
        'roma_dspy': None,
        'roma_mcp_tools': None
    }):
        yield


# =============================================================================
# SETUP/TEARDOWN
# =============================================================================

@pytest.fixture(scope="function", autouse=True)
def setup_logging():
    """Setup logging for tests."""
    import logging
    logging.basicConfig(level=logging.DEBUG)
    yield
    # Cleanup


@pytest.fixture(autouse=True)
def clear_imports():
    """Clear imports between tests to prevent module caching issues."""
    modules_to_remove = [k for k in sys.modules.keys()
                        if k.startswith('reliability')]
    for module in modules_to_remove:
        del sys.modules[module]