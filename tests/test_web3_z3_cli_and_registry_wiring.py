import pytest

import workflow_stage_z3 as z3_stage
import z3_crewai_bridge as z3_crewai
import z3_cli


def test_z3_stage_registry_contains_web3_stage_types():
    registry = z3_stage.get_z3_stage_registry()
    assert z3_stage.Z3StageType.WEB3_INVARIANT_TRANSLATE.value in registry.stage_types
    assert z3_stage.Z3StageType.WEB3_EXPLOIT_WITNESS.value in registry.stage_types
    assert z3_stage.Z3StageType.WEB3_AUDIT_EXPLOIT_VERIFICATION.value in registry.stage_types


def test_z3_stage_registry_status_exposes_web3_formal_schema():
    registry = z3_stage.get_z3_stage_registry()
    status = registry.get_status()
    assert "web3_formal_tools" in status
    assert "formal_capabilities" in status
    assert "registered_stage_types" in status


def test_z3_coordinator_creates_web3_audit_agent():
    coordinator = z3_crewai.Z3AgentCoordinator()
    agent = coordinator.create_web3_audit_agent("web3_agent_1")
    assert agent.role == z3_crewai.AgentRole.WEB3_AUDITOR
    assert "smart_contract_audit" in agent.get_capabilities()


def test_z3_coordinator_create_agent_role_factory():
    coordinator = z3_crewai.Z3AgentCoordinator()
    solver = coordinator.create_agent("solver_1", "solver")
    web3_agent = coordinator.create_agent("web3_1", "web3_auditor")
    assert solver.role == z3_crewai.AgentRole.SOLVER
    assert web3_agent.role == z3_crewai.AgentRole.WEB3_AUDITOR


def test_z3_coordinator_status_exposes_web3_formal_schema():
    coordinator = z3_crewai.Z3AgentCoordinator()
    coordinator.create_web3_audit_agent("web3_agent_2")
    status = coordinator.get_status()
    assert "web3_formal_tools" in status
    assert "formal_capabilities" in status
    assert status["registered_agents"] >= 1


def test_z3_crewai_bridge_module_web3_formal_status_shape():
    status = z3_crewai.get_web3_formal_status()
    assert "available" in status
    assert "web3_formal_tools" in status
    assert "formal_capabilities" in status


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
    assert "web3-audit-exploit-verification" in command_names
