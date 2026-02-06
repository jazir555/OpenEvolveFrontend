import pytest


decomp_crewai_tools = pytest.importorskip("decomposition_crewai_tools")
decomp_crewai_bridge = pytest.importorskip("decomposition_crewai_bridge")


def test_crewai_tools_status_infers_web3_formal_tools_from_capabilities(monkeypatch):
    monkeypatch.setattr(
        decomp_crewai_tools,
        "_get_mcp_tool_inventory",
        lambda: {
            "web3_tools": [],
            "web3_formal_tools": [],
            "formal_capabilities": {
                "solidity_invariant_translation": True,
                "symbolic_exploit_witness": True,
                "composite_exploit_verification": True,
            },
        },
    )
    status = decomp_crewai_tools.get_decomposition_status()
    assert {
        "z3_translate_solidity_invariant",
        "z3_solve_smart_contract_exploit_witness",
        "z3_web3_audit_exploit_verification",
    }.issubset(set(status["web3_formal_tools"]))
    assert status["formal_capabilities"]["composite_exploit_verification"] is True
    assert status["web3_formal_available"] is True
    assert status["audit_exploit_verification_available"] is True
    assert status["web3_domain_extension_available"] is True


def test_crewai_bridge_status_infers_web3_formal_tools_from_capabilities(monkeypatch):
    monkeypatch.setattr(
        decomp_crewai_bridge,
        "_get_mcp_tool_inventory",
        lambda: {
            "web3_tools": [],
            "web3_formal_tools": [],
            "formal_capabilities": {
                "solidity_invariant_translation": True,
                "symbolic_exploit_witness": True,
                "composite_exploit_verification": True,
            },
        },
    )
    status = decomp_crewai_bridge.get_decomposition_status()
    assert {
        "z3_translate_solidity_invariant",
        "z3_solve_smart_contract_exploit_witness",
        "z3_web3_audit_exploit_verification",
    }.issubset(set(status["web3_formal_tools"]))
    assert status["formal_capabilities"]["composite_exploit_verification"] is True
    assert status["web3_formal_available"] is True
    assert status["audit_exploit_verification_available"] is True
    assert status["web3_domain_extension_available"] is True
