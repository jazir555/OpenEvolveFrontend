import asyncio
import pytest

import workflow_stage_z3 as z3_stage
import z3_crewai_bridge as z3_crewai


def test_workflow_stage_web3_invariant_translate_wiring(monkeypatch):
    monkeypatch.setattr(z3_stage, "WEB3_FORMAL_AVAILABLE", True)
    monkeypatch.setattr(
        z3_stage,
        "translate_solidity_assignment_to_z3",
        lambda **kwargs: {"constraints": ["new_balance == old_balance - amount"], "invariants": ["new_balance >= 0"]},
    )
    monkeypatch.setattr(
        z3_stage,
        "verify_solidity_invariant_translation",
        lambda **kwargs: {"proven": True, "reason": "Constraints imply invariants"},
    )

    config = z3_stage.Z3StageConfig(
        stage_type=z3_stage.Z3StageType.WEB3_INVARIANT_TRANSLATE,
        statement="balance[msg.sender] -= amount;",
    )
    result = z3_stage.Z3WorkflowStage(config).execute({})

    assert result.success is True
    assert result.status == "translated"
    assert "translation" in result.metadata
    assert "verification" in result.metadata


def test_workflow_stage_web3_exploit_witness_wiring(monkeypatch):
    monkeypatch.setattr(z3_stage, "WEB3_FORMAL_AVAILABLE", True)
    monkeypatch.setattr(
        z3_stage,
        "solve_smart_contract_exploit_witness",
        lambda **kwargs: {"status": "sat", "satisfiable": True, "model": {"user_deposit": "0"}},
    )

    config = z3_stage.Z3StageConfig(
        stage_type=z3_stage.Z3StageType.WEB3_EXPLOIT_WITNESS,
        additional_constraints=["contract_balance_post < contract_balance_pre"],
    )
    result = z3_stage.Z3WorkflowStage(config).execute({})

    assert result.success is True
    assert result.status == "sat"
    assert result.model == {"user_deposit": "0"}


def test_workflow_stage_web3_audit_exploit_verification_wiring(monkeypatch):
    monkeypatch.setattr(z3_stage, "WEB3_FORMAL_AVAILABLE", True)
    monkeypatch.setattr(
        z3_stage,
        "translate_solidity_assignment_to_z3",
        lambda **kwargs: {"constraints": ["new_balance == old_balance - amount"], "invariants": ["new_balance >= 0"]},
    )
    monkeypatch.setattr(
        z3_stage,
        "verify_solidity_invariant_translation",
        lambda **kwargs: {"proven": True},
    )
    monkeypatch.setattr(
        z3_stage,
        "solve_smart_contract_exploit_witness",
        lambda **kwargs: {"status": "sat", "satisfiable": True, "model": {"user_deposit": "0"}},
    )

    config = z3_stage.Z3StageConfig(
        stage_type=z3_stage.Z3StageType.WEB3_AUDIT_EXPLOIT_VERIFICATION,
        statement="balance[msg.sender] -= amount;",
    )
    result = z3_stage.Z3WorkflowStage(config).execute({})

    assert result.success is True
    assert result.status == "verified_exploit"
    assert result.metadata.get("verified_exploit") is True


def test_workflow_stage_web3_formal_status_schema(monkeypatch):
    monkeypatch.setattr(
        z3_stage,
        "translate_solidity_assignment_to_z3",
        lambda **kwargs: {"constraints": ["new_balance == old_balance - amount"]},
    )
    monkeypatch.setattr(
        z3_stage,
        "verify_solidity_invariant_translation",
        lambda **kwargs: {"proven": True},
    )
    monkeypatch.setattr(
        z3_stage,
        "solve_smart_contract_exploit_witness",
        lambda **kwargs: {"status": "sat", "satisfiable": True},
    )
    status = z3_stage.get_web3_formal_status()
    assert "web3_formal_tools" in status
    assert "formal_capabilities" in status
    assert "z3_web3_audit_exploit_verification" in status["web3_formal_tools"]


def test_z3_crewai_web3_audit_agent_executes_full_audit(monkeypatch):
    monkeypatch.setattr(z3_crewai, "WEB3_FORMAL_AVAILABLE", True)
    monkeypatch.setattr(
        z3_crewai,
        "translate_solidity_assignment_to_z3",
        lambda **kwargs: {"constraints": ["new_balance == old_balance - amount"], "invariants": ["new_balance >= 0"]},
    )
    monkeypatch.setattr(
        z3_crewai,
        "verify_solidity_invariant_translation",
        lambda **kwargs: {"proven": True},
    )
    monkeypatch.setattr(
        z3_crewai,
        "solve_smart_contract_exploit_witness",
        lambda **kwargs: {"status": "sat", "satisfiable": True, "model": {"attacker_input": "1"}},
    )

    agent = z3_crewai.Z3Web3AuditAgent(agent_id="web3_auditor_1")
    task = z3_crewai.AgentTask(
        task_id="web3_task_1",
        role=z3_crewai.AgentRole.WEB3_AUDITOR,
        problem="balance[msg.sender] -= amount;",
        parameters={"action": "full_audit"},
    )
    result = asyncio.run(agent.execute(task))

    assert result.success is True
    assert "translation" in result.result_data
    assert "exploit_witness" in result.result_data
    assert result.result_data.get("verified_exploit") is True


def test_z3_crewai_web3_audit_agent_supports_composite_action_alias(monkeypatch):
    monkeypatch.setattr(z3_crewai, "WEB3_FORMAL_AVAILABLE", True)
    monkeypatch.setattr(
        z3_crewai,
        "translate_solidity_assignment_to_z3",
        lambda **kwargs: {"constraints": ["new_balance == old_balance - amount"], "invariants": ["new_balance >= 0"]},
    )
    monkeypatch.setattr(
        z3_crewai,
        "verify_solidity_invariant_translation",
        lambda **kwargs: {"proven": True},
    )
    monkeypatch.setattr(
        z3_crewai,
        "solve_smart_contract_exploit_witness",
        lambda **kwargs: {"status": "sat", "satisfiable": True, "model": {"attacker_input": "1"}},
    )

    agent = z3_crewai.Z3Web3AuditAgent(agent_id="web3_auditor_2")
    task = z3_crewai.AgentTask(
        task_id="web3_task_2",
        role=z3_crewai.AgentRole.WEB3_AUDITOR,
        problem="balance[msg.sender] -= amount;",
        parameters={"action": "audit_exploit_verification"},
    )
    result = asyncio.run(agent.execute(task))

    assert result.success is True
    assert result.result_data.get("verified_exploit") is True
    assert "composite_exploit_verification" in agent.get_capabilities()


def test_z3_bubblelabs_ui_exposes_web3_node_types():
    try:
        import z3_leanaide_bubblelabs_ui as z3_ui
    except Exception as exc:
        pytest.skip(f"z3_leanaide_bubblelabs_ui unavailable: {exc}")
    manager = z3_ui.Z3BubbleLabsUIManager(config={"use_cav_nlp": False})
    node_types = {node["type"] for node in manager.get_node_definitions()}
    assert {
        "z3_web3_invariant_translate",
        "z3_web3_exploit_witness",
        "z3_web3_audit_exploit_verification",
    }.issubset(node_types)


def test_z3_bubblelabs_ui_status_exposes_formal_capabilities(monkeypatch):
    try:
        import z3_leanaide_bubblelabs_ui as z3_ui
    except Exception as exc:
        pytest.skip(f"z3_leanaide_bubblelabs_ui unavailable: {exc}")
    monkeypatch.setattr(
        z3_ui,
        "translate_solidity_assignment_to_z3",
        lambda **kwargs: {"constraints": ["new_balance == old_balance - amount"]},
    )
    monkeypatch.setattr(
        z3_ui,
        "verify_solidity_invariant_translation",
        lambda **kwargs: {"proven": True},
    )
    monkeypatch.setattr(
        z3_ui,
        "solve_smart_contract_exploit_witness",
        lambda **kwargs: {"status": "sat", "satisfiable": True},
    )
    manager = z3_ui.Z3BubbleLabsUIManager(config={"use_cav_nlp": False})
    status = manager.get_status()
    assert "web3_formal_tools" in status
    assert "formal_capabilities" in status
    assert status["formal_capabilities"]["composite_exploit_verification"] is True
    assert "z3_web3_audit_exploit_verification" in status["web3_formal_tools"]


def test_z3_bubblelabs_ui_handles_web3_invariant_node(monkeypatch):
    try:
        import z3_leanaide_bubblelabs_ui as z3_ui
    except Exception as exc:
        pytest.skip(f"z3_leanaide_bubblelabs_ui unavailable: {exc}")
    manager = z3_ui.Z3BubbleLabsUIManager(config={"use_cav_nlp": False})
    monkeypatch.setattr(
        z3_ui,
        "translate_solidity_assignment_to_z3",
        lambda **kwargs: {"constraints": ["new_balance == old_balance - amount"], "invariants": ["new_balance >= 0"]},
    )
    monkeypatch.setattr(
        z3_ui,
        "verify_solidity_invariant_translation",
        lambda **kwargs: {"proven": True},
    )

    result = asyncio.run(
        manager.handle_node_execution(
            "z3_web3_invariant_translate",
            "node-1",
            {"statement": "balance[msg.sender] -= amount;", "verify_translation": True},
        )
    )

    assert result["status"] == "success"
    assert "translation" in result
    assert "verification" in result


def test_z3_bubblelabs_ui_handles_web3_composite_audit_node(monkeypatch):
    try:
        import z3_leanaide_bubblelabs_ui as z3_ui
    except Exception as exc:
        pytest.skip(f"z3_leanaide_bubblelabs_ui unavailable: {exc}")
    manager = z3_ui.Z3BubbleLabsUIManager(config={"use_cav_nlp": False})
    monkeypatch.setattr(
        z3_ui,
        "translate_solidity_assignment_to_z3",
        lambda **kwargs: {"constraints": ["new_balance == old_balance - amount"], "invariants": ["new_balance >= 0"]},
    )
    monkeypatch.setattr(
        z3_ui,
        "verify_solidity_invariant_translation",
        lambda **kwargs: {"proven": True},
    )
    monkeypatch.setattr(
        z3_ui,
        "solve_smart_contract_exploit_witness",
        lambda **kwargs: {"status": "sat", "satisfiable": True, "model": {"amount": 1}},
    )
    result = asyncio.run(
        manager.handle_node_execution(
            "z3_web3_audit_exploit_verification",
            "node-2",
            {"statement": "balance[msg.sender] -= amount;", "verify_translation": True},
        )
    )
    assert result["status"] == "success"
    assert result["verified_exploit"] is True
