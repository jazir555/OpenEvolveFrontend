"""
Test Utilities Module

Provides common test helper functions, mocks, and utilities for OpenEvolve tests.
This module addresses common testing patterns and reduces code duplication.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from unittest.mock import Mock, MagicMock, AsyncMock, patch
import pytest
import json


# ============================================================================
# Import Helpers - Graceful handling of optional dependencies
# ============================================================================

def safe_import(module_name: str, attribute: str = None) -> tuple[bool, Any]:
    """
    Safely import a module or attribute, returning success status and module.

    Args:
        module_name: Name of the module to import
        attribute: Optional attribute to retrieve from module

    Returns:
        Tuple of (success: bool, module_or_attribute: Any)
    """
    try:
        if attribute:
            module = __import__(module_name, fromlist=[attribute])
            return True, getattr(module, attribute)
        else:
            return True, __import__(module_name)
    except ImportError:
        return False, None


def skip_if_not_available(module_name: str, reason: str = None) -> Callable:
    """
    Create a pytest skip marker for optional modules.

    Args:
        module_name: Name of the module to check
        reason: Optional custom skip reason

    Returns:
        pytest.mark.skipif object
    """
    available, _ = safe_import(module_name)
    if reason is None:
        reason = f"{module_name} not available"
    return pytest.mark.skipif(not available, reason=reason)


# ============================================================================
# Mock Factories - Common mock objects
# ============================================================================

def create_mock_team(name: str = "TestTeam", role: str = "Blue",
                    model_id: str = "test-model",
                    api_key: str = "test-key") -> Mock:
    """
    Create a mock Team object for testing.

    Args:
        name: Team name
        role: Team role (Blue, Red, Gold, etc.)
        model_id: Model ID for team members
        api_key: API key for team members

    Returns:
        Mock Team object
    """
    from openevolve_structures import Team, ModelConfig

    member = ModelConfig(model_id=model_id, api_key=api_key)
    return Team(name=name, role=role, members=[member])


def create_mock_gauntlet(name: str = "TestGauntlet",
                        team_name: str = "TestTeam",
                        is_red_team: bool = False) -> Mock:
    """
    Create a mock GauntletDefinition object for testing.

    Args:
        name: Gauntlet name
        team_name: Name of the team
        is_red_team: Whether this is a red team gauntlet

    Returns:
        Mock GauntletDefinition object
    """
    from openevolve_structures import GauntletDefinition, GauntletRoundRule

    rounds = [GauntletRoundRule(
        round_number=1,
        quorum_required_approvals=1,
        quorum_from_panel_size=1,
        min_overall_confidence=0.5
    )]

    return GauntletDefinition(
        name=name,
        team_name=team_name,
        rounds=rounds,
        generation_mode="single_candidate"
    )


def create_mock_workflow_state(problem_statement: str = "Test problem") -> Mock:
    """
    Create a mock WorkflowState for testing.

    Args:
        problem_statement: The problem to solve

    Returns:
        Mock WorkflowState object
    """
    from workflow_structures import WorkflowState, DecompositionPlan

    return WorkflowState(
        workflow_id="test-workflow-123",
        workflow_type="test_workflow",
        problem_statement=problem_statement,
        current_stage="INITIALIZING",
        decomposition_plan=DecompositionPlan(
            problem_statement=problem_statement,
            analyzed_context={},
            sub_problems=[]
        ),
        mdap_enabled=False,
        maker_enabled=False
    )


def create_mock_ace_components() -> Dict[str, Mock]:
    """
    Create mock ACE (Agentic Context Engine) components for testing.

    Returns:
        Dictionary of mock ACE components
    """
    mock_skillbook = MagicMock()
    mock_skillbook.get_skills.return_value = [
        {"name": "question_answering", "proficiency": 0.9},
        {"name": "context_analysis", "proficiency": 0.85}
    ]
    mock_skillbook.as_prompt.return_value = "Skills: question_answering, context_analysis"

    mock_agent = MagicMock()
    mock_agent.generate.return_value = MagicMock(
        final_answer="Test answer",
        reasoning="Test reasoning"
    )

    mock_reflector = MagicMock()
    mock_reflector.reflect.return_value = MagicMock(
        reflections=["Test reflection"],
        improved_output="Improved output"
    )

    mock_skill_manager = MagicMock()
    mock_skill_manager.update_skills.return_value = MagicMock(
        updated_skills=["question_answering"],
        new_skills=["test_skill"]
    )

    mock_offline_ace = MagicMock()
    mock_offline_ace.run.return_value = [
        MagicMock(agent_output=MagicMock(final_answer="Training completed"))
    ]

    mock_online_ace = MagicMock()
    mock_online_ace.run.return_value = [
        MagicMock(agent_output=MagicMock(final_answer="Online learning completed"))
    ]

    return {
        "skillbook": mock_skillbook,
        "agent": mock_agent,
        "reflector": mock_reflector,
        "skill_manager": mock_skill_manager,
        "offline_ace": mock_offline_ace,
        "online_ace": mock_online_ace
    }


def create_mock_knowledge_graph() -> Mock:
    """
    Create a mock knowledge graph for testing.

    Returns:
        Mock EntityKnowledgeGraph object
    """
    kg = Mock()
    kg.add_entity = Mock(return_value="entity-001")
    kg.add_relationship = Mock(return_value="rel-001")
    kg.get_entity = Mock(return_value=None)
    kg.get_related_entities = Mock(return_value=[])
    kg.query = Mock(return_value=[])
    kg.exists = Mock(return_value=False)
    kg.delete = Mock(return_value=True)
    return kg


# ============================================================================
# LLM Mocking Helpers
# ============================================================================

def mock_llm_response(text: str = "Test response",
                     is_json: bool = False) -> Any:
    """
    Create a mock LLM response.

    Args:
        text: Response text
        is_json: Whether the response is JSON formatted

    Returns:
        Mock response object
    """
    if is_json:
        return text  # Return raw JSON string

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = text
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20
    mock_response.usage.total_tokens = 30
    return mock_response


def setup_llm_mock(patch_target: str,
                   responses: List[str] = None) -> Mock:
    """
    Set up a mock for LLM API calls.

    Args:
        patch_target: Target to patch (e.g., 'workflow_engine._request_openai_compatible_chat')
        responses: List of response strings to return in sequence

    Returns:
        Mock object
    """
    if responses is None:
        responses = ["Test response"]

    mock_patch = patch(patch_target)
    mock = mock_patch.__enter__()
    mock.side_effect = responses
    return mock


# ============================================================================
# Manager Mocking Helpers
# ============================================================================

def setup_mock_team_manager() -> Mock:
    """
    Set up a mock TeamManager with common test data.

    Returns:
        Mock TeamManager
    """
    from team_manager import TeamManager

    mock_manager = Mock(spec=TeamManager)
    mock_manager.teams = {}

    # Add some default teams
    mock_manager.teams["Solver"] = create_mock_team("Solver", "Blue")
    mock_manager.teams["RedTeam"] = create_mock_team("RedTeam", "Red")
    mock_manager.teams["GoldTeam"] = create_mock_team("GoldTeam", "Gold")
    mock_manager.teams["Planner"] = create_mock_team("Planner", "Blue")
    mock_manager.teams["Patcher"] = create_mock_team("Patcher", "Blue")
    mock_manager.teams["Assembler"] = create_mock_team("Assembler", "Blue")
    mock_manager.teams["ContentAnalyzer"] = create_mock_team("ContentAnalyzer", "Blue")

    mock_manager.get_team = lambda name: mock_manager.teams.get(name)
    mock_manager.create_team = lambda team: mock_manager.teams.__setitem__(team.name, team)
    mock_manager.team_exists = lambda name: name in mock_manager.teams

    return mock_manager


def setup_mock_gauntlet_manager() -> Mock:
    """
    Set up a mock GauntletManager with common test data.

    Returns:
        Mock GauntletManager
    """
    from gauntlet_manager import GauntletManager

    mock_manager = Mock(spec=GauntletManager)
    mock_manager.gauntlets = {}

    # Add some default gauntlets
    mock_manager.gauntlets["SubProblemRed"] = create_mock_gauntlet("SubProblemRed", "RedTeam", True)
    mock_manager.gauntlets["SubProblemGold"] = create_mock_gauntlet("SubProblemGold", "GoldTeam", False)
    mock_manager.gauntlets["FinalRed"] = create_mock_gauntlet("FinalRed", "RedTeam", True)
    mock_manager.gauntlets["FinalGold"] = create_mock_gauntlet("FinalGold", "GoldTeam", False)
    mock_manager.gauntlets["SolverGen"] = create_mock_gauntlet("SolverGen", "Solver", False)

    mock_manager.get_gauntlet = lambda name: mock_manager.gauntlets.get(name)
    mock_manager.create_gauntlet = lambda gauntlet: mock_manager.gauntlets.__setitem__(gauntlet.name, gauntlet)
    mock_manager.gauntlet_exists = lambda name: name in mock_manager.gauntlets

    return mock_manager


# ============================================================================
# Data Generation Helpers
# ============================================================================

def generate_test_entities(count: int = 5) -> List[Dict[str, Any]]:
    """
    Generate test entity data.

    Args:
        count: Number of entities to generate

    Returns:
        List of entity dictionaries
    """
    entities = []
    for i in range(count):
        entities.append({
            "entity_id": f"entity-{i:03d}",
            "entity_type": "test_entity",
            "name": f"Test Entity {i}",
            "properties": {
                "description": f"Test entity {i}",
                "index": i
            }
        })
    return entities


def generate_test_relationships(count: int = 5) -> List[Dict[str, Any]]:
    """
    Generate test relationship data.

    Args:
        count: Number of relationships to generate

    Returns:
        List of relationship dictionaries
    """
    relationships = []
    for i in range(count):
        relationships.append({
            "relationship_id": f"rel-{i:03d}",
            "source_id": f"entity-{i:03d}",
            "target_id": f"entity-{(i+1) % count:03d}",
            "relationship_type": "test_relationship",
            "properties": {
                "weight": 0.5 + (i * 0.1)
            }
        })
    return relationships


def generate_test_subproblems(count: int = 3) -> List[Any]:
    """
    Generate test sub-problem data.

    Args:
        count: Number of sub-problems to generate

    Returns:
        List of SubProblem objects
    """
    from workflow_structures import SubProblem

    subproblems = []
    for i in range(count):
        deps = [f"sub_{i-1}"] if i > 0 else []
        subproblems.append(SubProblem(
            id=f"sub_{i}",
            description=f"Test sub-problem {i}",
            dependencies=deps
        ))
    return subproblems


# ============================================================================
# Assertion Helpers
# ============================================================================

def assert_valid_result(result: Any,
                       success: bool = True,
                       has_error: bool = False) -> None:
    """
    Assert that a result object is valid.

    Args:
        result: Result object to check
        success: Expected success status
        has_error: Whether result should have an error

    Raises:
        AssertionError: If result doesn't match expectations
    """
    assert hasattr(result, "success"), "Result must have 'success' attribute"
    assert result.success == success, f"Expected success={success}, got {result.success}"

    if has_error:
        assert hasattr(result, "error"), "Result must have 'error' attribute"
        assert result.error is not None, "Expected error to be set"
    else:
        if hasattr(result, "error"):
            assert result.error is None, f"Expected no error, got: {result.error}"


def assert_log_called(mock_logger: Mock,
                     level: str,
                     message: str = None) -> None:
    """
    Assert that a logger method was called.

    Args:
        mock_logger: Mock logger object
        level: Log level (info, warning, error, etc.)
        message: Optional message to check

    Raises:
        AssertionError: If log wasn't called as expected
    """
    log_method = getattr(mock_logger, level, None)
    assert log_method is not None, f"Logger has no {level} method"
    assert log_method.called, f"Logger.{level} was not called"

    if message:
        call_args = log_method.call_args
        if call_args and call_args[0]:
            assert message in str(call_args[0][0]), \
                f"Expected '{message}' in log call, got: {call_args}"


# ============================================================================
# Environment Helpers
# ============================================================================

def set_test_env_vars(vars: Dict[str, str] = None) -> None:
    """
    Set environment variables for testing.

    Args:
        vars: Dictionary of variables to set
    """
    defaults = {
        "OPENAI_API_KEY": "sk-test-key",
        "ANTHROPIC_API_KEY": "sk-ant-test-key",
        "TESTING": "true",
        "LOG_LEVEL": "WARNING"
    }

    if vars:
        defaults.update(vars)

    for key, value in defaults.items():
        os.environ[key] = value


def clear_test_env_vars(*vars: str) -> None:
    """
    Clear environment variables after testing.

    Args:
        *vars: Variable names to clear
    """
    for var in vars:
        os.environ.pop(var, None)
