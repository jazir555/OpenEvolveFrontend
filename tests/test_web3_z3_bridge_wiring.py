import asyncio
import pytest


try:
    import z3_leanaide_bridge as z3_bridge
except Exception as exc:  # pragma: no cover - environment-dependent import fallback
    z3_bridge = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _require_bridge():
    if z3_bridge is None:
        pytest.skip(f"z3_leanaide_bridge unavailable: {_IMPORT_ERROR}")


def test_bridge_capabilities_include_web3_flags():
    _require_bridge()
    bridge = z3_bridge.Z3LeanAideBridge(config={"use_cav_nlp": False})
    capabilities = bridge.get_capabilities()
    assert "web3_formal_available" in capabilities
    assert "audit_exploit_verification_available" in capabilities
    assert "solidity_invariant_translation" in capabilities
    assert "invariant_translation_verification" in capabilities
    assert "smart_contract_exploit_witness" in capabilities
    assert "web3_audit_exploit_verification" in capabilities
    assert "web3_formal_tools" in capabilities
    assert "formal_capabilities" in capabilities


def test_bridge_translate_solidity_invariant(monkeypatch):
    _require_bridge()
    monkeypatch.setattr(
        z3_bridge,
        "translate_solidity_assignment_to_z3",
        lambda **kwargs: {"constraints": ["new_balance == old_balance - amount"]},
    )
    monkeypatch.setattr(
        z3_bridge,
        "verify_solidity_invariant_translation",
        lambda **kwargs: {"proven": True},
    )

    bridge = z3_bridge.Z3LeanAideBridge(config={"use_cav_nlp": False})
    result = asyncio.run(
        bridge.translate_solidity_invariant(
            statement="balance[msg.sender] -= amount;",
            verify_translation=True,
        )
    )
    assert result["success"] is True
    assert "translation" in result
    assert "verification" in result


def test_bridge_solve_web3_exploit_witness(monkeypatch):
    _require_bridge()
    monkeypatch.setattr(
        z3_bridge,
        "solve_smart_contract_exploit_witness",
        lambda **kwargs: {"status": "sat", "satisfiable": True},
    )
    bridge = z3_bridge.Z3LeanAideBridge(config={"use_cav_nlp": False})
    result = asyncio.run(bridge.solve_web3_exploit_witness(timeout=1.5))
    assert result["success"] is True
    assert result["result"]["status"] == "sat"


def test_bridge_web3_audit_exploit_verification(monkeypatch):
    _require_bridge()
    monkeypatch.setattr(
        z3_bridge,
        "translate_solidity_assignment_to_z3",
        lambda **kwargs: {"constraints": ["new_balance == old_balance - amount"]},
    )
    monkeypatch.setattr(
        z3_bridge,
        "verify_solidity_invariant_translation",
        lambda **kwargs: {"proven": True},
    )
    monkeypatch.setattr(
        z3_bridge,
        "solve_smart_contract_exploit_witness",
        lambda **kwargs: {"status": "sat", "satisfiable": True},
    )
    bridge = z3_bridge.Z3LeanAideBridge(config={"use_cav_nlp": False})
    result = asyncio.run(
        bridge.web3_audit_exploit_verification(statement="balance[msg.sender] -= amount;")
    )
    assert result["success"] is True
    assert result["verified_exploit"] is True
    assert "lean_proof_verification" in result
    assert "formal_evidence" in result
    assert result["formal_evidence"]["lean_proof_verification"] == result["lean_proof_verification"]


def test_bridge_quick_web3_helpers(monkeypatch):
    _require_bridge()

    class _StubBridge:
        async def translate_solidity_invariant(self, **kwargs):
            return {"success": True, "translation": {"statement": kwargs.get("statement")}}

        async def solve_web3_exploit_witness(self, **kwargs):
            return {"success": True, "result": {"status": "sat"}}

        async def web3_audit_exploit_verification(self, **kwargs):
            return {"success": True, "verified_exploit": True}

    monkeypatch.setattr(z3_bridge, "create_z3_lean_bridge", lambda *args, **kwargs: _StubBridge())

    translated = asyncio.run(
        z3_bridge.quick_translate_solidity_invariant("balance[msg.sender] -= amount;")
    )
    witness = asyncio.run(z3_bridge.quick_solve_web3_exploit_witness())
    composite = asyncio.run(z3_bridge.quick_web3_audit_exploit_verification())

    assert translated["success"] is True
    assert witness["success"] is True
    assert composite["verified_exploit"] is True
