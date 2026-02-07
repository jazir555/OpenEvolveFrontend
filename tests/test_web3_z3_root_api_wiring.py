import z3_api


def test_root_z3_api_exposes_web3_methods():
    api = z3_api.create_api()
    assert hasattr(api, "get_web3_status")
    assert hasattr(api, "translate_solidity_invariant")
    assert hasattr(api, "solve_web3_exploit_witness")
    assert hasattr(api, "web3_audit_exploit_verification")
    status = api.get_web3_status()
    assert "audit_exploit_verification_available" in status
    assert "web3_formal_available" in status
    assert "web3_formal_verification_available" in status
    assert "web3_formal_tools" in status
    assert "formal_capabilities" in status
    assert "tool_inventory" in status


def test_root_z3_api_status_infers_formal_tools_from_capabilities(monkeypatch):
    api = z3_api.create_api()
    monkeypatch.setattr(
        z3_api,
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
    status = api.get_web3_status()
    assert {
        "z3_translate_solidity_invariant",
        "z3_solve_smart_contract_exploit_witness",
        "z3_web3_audit_exploit_verification",
    }.issubset(set(status["web3_formal_tools"]))
    assert status["formal_capabilities"]["composite_exploit_verification"] is True
    assert status["available"] is True
    assert status["web3_formal_available"] is True
    assert status["web3_formal_verification_available"] is True
    assert status["tool_inventory"]["audit_exploit_verification_available"] is True


def test_root_z3_api_composite_orchestration(monkeypatch):
    api = z3_api.create_api()
    monkeypatch.setattr(
        z3_api,
        "translate_solidity_assignment_to_z3",
        lambda **kwargs: {"constraints": ["x == y"], "invariants": ["x >= 0"]},
    )
    monkeypatch.setattr(
        z3_api,
        "verify_solidity_invariant_translation",
        lambda **kwargs: {"proven": True},
    )
    monkeypatch.setattr(
        z3_api,
        "solve_smart_contract_exploit_witness",
        lambda **kwargs: {"satisfiable": True, "model": {"amount": 1}},
    )
    result = api.web3_audit_exploit_verification(statement="balance[msg.sender] -= amount;")
    assert result["success"] is True
    assert result["verified_exploit"] is True
    assert "lean_proof_verification" in result
    assert "formal_evidence" in result
