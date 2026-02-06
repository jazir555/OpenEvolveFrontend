import asyncio

import pytest

z3_mcp_tools = pytest.importorskip("z3_mcp_tools")


def test_web3_inventory_includes_composite_exploit_verification_tool():
    inventory = z3_mcp_tools.get_web3_formal_tool_inventory()
    assert "z3_web3_audit_exploit_verification" in inventory.get("tools", [])
    assert "z3_web3_audit_exploit_verification" in inventory.get("web3_formal_tools", [])
    assert "composite_exploit_verification" in inventory.get("formal_capabilities", {})


def test_web3_composite_tool_orchestrates_translation_and_witness(monkeypatch):
    monkeypatch.setattr(
        z3_mcp_tools,
        "translate_solidity_assignment_to_z3",
        lambda **kwargs: {"constraints": ["x == y"], "invariants": ["x >= 0"]},
    )
    monkeypatch.setattr(
        z3_mcp_tools,
        "verify_solidity_invariant_translation",
        lambda **kwargs: {"proven": True},
    )
    monkeypatch.setattr(
        z3_mcp_tools,
        "solve_smart_contract_exploit_witness",
        lambda **kwargs: {"satisfiable": True, "model": {"amount": 1}},
    )
    result = asyncio.run(
        z3_mcp_tools.z3_web3_audit_exploit_verification(
            statement="balance[msg.sender] -= amount;"
        )
    )
    assert result["success"] is True
    assert result["verified_exploit"] is True
