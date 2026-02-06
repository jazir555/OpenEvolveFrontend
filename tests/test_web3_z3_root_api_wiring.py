import z3_api


def test_root_z3_api_exposes_web3_methods():
    api = z3_api.create_api()
    assert hasattr(api, "get_web3_status")
    assert hasattr(api, "translate_solidity_invariant")
    assert hasattr(api, "solve_web3_exploit_witness")
    assert hasattr(api, "web3_audit_exploit_verification")


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
