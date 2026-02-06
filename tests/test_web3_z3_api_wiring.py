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
