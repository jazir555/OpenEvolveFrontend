import pytest


z3_integration = pytest.importorskip("z3prover_integration")


def test_z3_solver_engine_status_exposes_web3_formal_schema():
    engine = z3_integration.Z3SolverEngine()
    status = engine.get_status()
    assert "web3_formal_available" in status
    assert "web3_formal_tools" in status
    assert "formal_capabilities" in status
    assert "audit_exploit_verification_available" in status


def test_z3_solver_engine_status_includes_composite_capability():
    engine = z3_integration.Z3SolverEngine()
    status = engine.get_status()
    formal_capabilities = status["formal_capabilities"]
    assert "composite_exploit_verification" in formal_capabilities
