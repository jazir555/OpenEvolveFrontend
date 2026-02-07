import asyncio

from web3_formal_evidence import (
    build_web3_formal_evidence,
    verify_web3_lean_proof_async,
)


def test_verify_web3_lean_proof_async_handles_missing_spec():
    result = asyncio.run(verify_web3_lean_proof_async({"constraints": ["x == y"]}))
    assert result["attempted"] is False
    assert result["status"] == "missing_lean_spec"
    assert result["method"] == "none"


def test_build_web3_formal_evidence_schema():
    lean_result = {"verified": False, "status": "missing_lean_spec"}
    evidence = build_web3_formal_evidence(
        verification={"proven": True},
        witness={"satisfiable": True, "status": "sat", "model": {"amount": 1}},
        lean_proof_verification=lean_result,
    )
    assert evidence["z3_invariant_verification"]["proven"] is True
    assert evidence["lean_proof_verification"] == lean_result
    assert evidence["symbolic_exploit_witness"]["satisfiable"] is True
    assert evidence["symbolic_exploit_witness"]["model_available"] is True
