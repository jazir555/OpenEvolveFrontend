import pytest

import workflow_stage_z3 as z3_stage
import z3_crewai_bridge as z3_crewai
import z3_cli


def test_z3_stage_registry_contains_web3_stage_types():
    registry = z3_stage.get_z3_stage_registry()
    assert z3_stage.Z3StageType.WEB3_INVARIANT_TRANSLATE.value in registry.stage_types
    assert z3_stage.Z3StageType.WEB3_EXPLOIT_WITNESS.value in registry.stage_types


def test_z3_coordinator_creates_web3_audit_agent():
    coordinator = z3_crewai.Z3AgentCoordinator()
    agent = coordinator.create_web3_audit_agent("web3_agent_1")
    assert agent.role == z3_crewai.AgentRole.WEB3_AUDITOR
    assert "smart_contract_audit" in agent.get_capabilities()


def test_z3_coordinator_web3_problem_detection():
    assert z3_crewai.Z3AgentCoordinator._is_web3_problem(
        "Audit this Solidity vault for flash-loan and reentrancy exploits."
    )
    assert not z3_crewai.Z3AgentCoordinator._is_web3_problem(
        "Compute shortest path in a weighted graph."
    )


def test_z3_cli_exposes_web3_commands():
    if not getattr(z3_cli, "CLICK_AVAILABLE", False):
        pytest.skip("click not installed")
    command_names = set(z3_cli.cli.commands.keys())
    assert "web3-translate-invariant" in command_names
    assert "web3-solve-witness" in command_names
