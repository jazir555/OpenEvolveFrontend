"""Wiring tests for root-level ``leanaide_integration`` compatibility."""

from leanaide_integration import (
    LeanAIDEIntegration,
    LeanAIDEVerifier,
    create_integration,
)


def test_root_module_exports_verifier_and_factory():
    integration = create_integration()
    assert isinstance(integration, LeanAIDEIntegration)
    assert hasattr(LeanAIDEVerifier, "verify_theorem")


def test_leanaide_verifier_returns_expected_contract():
    verifier = LeanAIDEVerifier(timeout=5.0)
    result = verifier.verify_theorem(
        code="theorem t : True := by trivial",
        context="theorem t : True := by trivial",
    )
    assert isinstance(result, dict)
    assert "proved" in result
    assert "tactics" in result
    assert "errors" in result


def test_leanaide_verifier_handles_empty_statement():
    verifier = LeanAIDEVerifier(timeout=1.0)
    result = verifier.verify_theorem(code="", context="")
    assert result["proved"] is False
    assert result["errors"]
