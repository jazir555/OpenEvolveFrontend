"""
Unit tests for ContextGatherer
"""

import pytest
from unittest.mock import Mock, AsyncMock


@pytest.fixture
def mock_storage_manager():
    """Create mock storage manager"""
    storage = Mock()

    # Mock retrieve_context_for_stage
    storage.retrieve_context_for_stage = AsyncMock(
        return_value={
            "stage": "test_stage",
            "artifacts": {},
            "similar_historical": []
        }
    )

    # Mock _search_artifacts
    storage._search_artifacts = AsyncMock(return_value=[])

    # Mock get_artifacts_by_stage
    storage.get_artifacts_by_stage = AsyncMock(return_value=[])

    return storage


@pytest.fixture
def context_gatherer(mock_storage_manager):
    """Create context gatherer with mock storage"""
    from ragbits_integration.intermediary_storage.context_gatherer import ContextGatherer
    return ContextGatherer(mock_storage_manager)


@pytest.mark.asyncio
async def test_gather_for_blue_team(context_gatherer, mock_storage_manager):
    """Test gathering context for Blue Team"""
    context = await context_gatherer.gather_for_blue_team(
        sub_problem_id="sub_1",
        problem_description="Implement authentication"
    )

    assert context["agent_role"] == "blue_team"
    assert context["task"] == "generate_solution"
    assert context["sub_problem_id"] == "sub_1"


@pytest.mark.asyncio
async def test_gather_for_red_team(context_gatherer, mock_storage_manager):
    """Test gathering context for Red Team"""
    context = await context_gatherer.gather_for_red_team(
        sub_problem_id="sub_1"
    )

    assert context["agent_role"] == "red_team"
    assert context["task"] == "critique_solution"
    assert context["sub_problem_id"] == "sub_1"


@pytest.mark.asyncio
async def test_gather_for_gold_team(context_gatherer, mock_storage_manager):
    """Test gathering context for Gold Team"""
    context = await context_gatherer.gather_for_gold_team(
        sub_problem_id="sub_1"
    )

    assert context["agent_role"] == "gold_team"
    assert context["task"] == "verify_solution"
    assert context["sub_problem_id"] == "sub_1"


@pytest.mark.asyncio
async def test_gather_for_decomposition(context_gatherer, mock_storage_manager):
    """Test gathering context for decomposition"""
    context = await context_gatherer.gather_for_decomposition(
        problem_description="Build scalable system"
    )

    assert context["agent_role"] == "decomposer"
    assert context["task"] == "decompose_problem"


@pytest.mark.asyncio
async def test_gather_for_reassembly(context_gatherer, mock_storage_manager):
    """Test gathering context for reassembly"""
    context = await context_gatherer.gather_for_reassembly()

    assert context["agent_role"] == "assembler"
    assert context["task"] == "assemble_solution"


@pytest.mark.asyncio
async def test_gather_for_final_verification(context_gatherer, mock_storage_manager):
    """Test gathering context for final verification"""
    context = await context_gatherer.gather_for_final_verification(
        assembled_solution="Final solution content"
    )

    assert context["agent_role"] == "final_verifier"
    assert context["task"] == "final_verification"


@pytest.mark.asyncio
async def test_get_subproblem_summary(context_gatherer, mock_storage_manager):
    """Test getting sub-problem summary"""
    # Mock artifacts
    mock_storage_manager.get_artifacts_by_sub_problem = AsyncMock(
        return_value=[
            {
                "content": "Solution",
                "metadata": {
                    "artifact_id": "art1",
                    "type": "solution_draft",
                    "status": "verified",
                    "team": "blue",
                    "timestamp": 1234567890
                }
            }
        ]
    )

    summary = await context_gatherer.get_subproblem_summary("sub_1")

    assert summary["sub_problem_id"] == "sub_1"
    assert summary["total_artifacts"] >= 0
    assert "by_type" in summary
    assert "by_status" in summary
    assert "timeline" in summary
