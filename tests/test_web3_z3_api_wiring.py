import asyncio
import pytest


z3_api_server = pytest.importorskip("z3_api_server")


def test_z3_api_registers_web3_formal_routes():
    paths = {route.path for route in z3_api_server.app.routes}
    expected = {
        "/web3/status",
        "/web3/invariants/translate",
        "/web3/exploits/symbolic-witness",
        "/web3/audit/exploit-verification",
    }
    assert expected.issubset(paths)


def test_z3_api_web3_status_shape():
    status = asyncio.run(z3_api_server.get_web3_formal_status())
    assert "available" in status
    assert "solidity_invariant_translation_available" in status
    assert "exploit_witness_available" in status
    assert "audit_exploit_verification_available" in status
    assert "web3_formal_tools" in status
    assert "formal_capabilities" in status
    assert "tool_inventory" in status


def test_z3_api_web3_status_infers_tools_when_inventory_tools_missing(monkeypatch):
    z3_mcp_tools = pytest.importorskip("z3_mcp_tools")
    monkeypatch.setattr(
        z3_mcp_tools,
        "get_web3_formal_tool_inventory",
        lambda: {
            "available": True,
            "tools": [],
            "formal_capabilities": {
                "solidity_invariant_translation": True,
                "symbolic_exploit_witness": True,
                "composite_exploit_verification": True,
            },
        },
    )
    status = asyncio.run(z3_api_server.get_web3_formal_status())
    assert {
        "z3_translate_solidity_invariant",
        "z3_solve_smart_contract_exploit_witness",
        "z3_web3_audit_exploit_verification",
    }.issubset(set(status["web3_formal_tools"]))
    assert status["formal_capabilities"]["composite_exploit_verification"] is True


def test_z3_api_web3_status_infers_available_when_inventory_flag_false(monkeypatch):
    z3_mcp_tools = pytest.importorskip("z3_mcp_tools")
    monkeypatch.setattr(
        z3_mcp_tools,
        "get_web3_formal_tool_inventory",
        lambda: {
            "available": False,
            "tools": [],
            "formal_capabilities": {
                "solidity_invariant_translation": True,
                "symbolic_exploit_witness": True,
                "composite_exploit_verification": True,
            },
        },
    )
    status = asyncio.run(z3_api_server.get_web3_formal_status())
    assert status["available"] is True
    assert status["audit_exploit_verification_available"] is True


def test_z3_service_bubble_status_exposes_web3_formal_inventory(monkeypatch):
    monkeypatch.setattr(
        z3_api_server,
        "_normalize_web3_formal_inventory",
        lambda *_args, **_kwargs: {
            "available": True,
            "tools": [
                "z3_translate_solidity_invariant",
                "z3_solve_smart_contract_exploit_witness",
                "z3_web3_audit_exploit_verification",
            ],
            "formal_capabilities": {
                "solidity_invariant_translation": True,
                "symbolic_exploit_witness": True,
                "composite_exploit_verification": True,
            },
        },
    )
    bubble = z3_api_server.Z3ServiceBubble()
    status = bubble.get_status()
    assert status["web3_formal_available"] is True
    assert {
        "z3_translate_solidity_invariant",
        "z3_solve_smart_contract_exploit_witness",
        "z3_web3_audit_exploit_verification",
    }.issubset(set(status["web3_formal_tools"]))
    assert status["formal_capabilities"]["composite_exploit_verification"] is True
    assert status["audit_exploit_verification_available"] is True


def test_z3_api_web3_audit_exploit_verification_orchestration(monkeypatch):
    monkeypatch.setattr(
        z3_api_server,
        "translate_solidity_assignment_to_z3",
        lambda **kwargs: {
            "constraints": ["new_balance == old_balance - amount"],
            "invariants": ["new_balance >= 0"],
        },
    )
    monkeypatch.setattr(
        z3_api_server,
        "verify_solidity_invariant_translation",
        lambda **kwargs: {"proven": True, "reason": "constraints imply invariants"},
    )
    monkeypatch.setattr(
        z3_api_server,
        "solve_smart_contract_exploit_witness",
        lambda **kwargs: {"satisfiable": True, "model": {"amount": 1}},
    )
    request = z3_api_server.Web3AuditExploitVerificationRequest(
        statement="balance[msg.sender] -= amount;"
    )
    result = asyncio.run(z3_api_server.web3_audit_exploit_verification(request))
    assert result["success"] is True
    assert result["verified_exploit"] is True
    assert result["exploit_witness"]["satisfiable"] is True
