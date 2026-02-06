import pytest


root_leanaide = pytest.importorskip("leanaide_integration")
bubblelabs_leanaide = pytest.importorskip("bubblelabs_leanaide_integration")


def test_root_leanaide_web3_formal_status_schema():
    integration = root_leanaide.LeanAIDEIntegration()
    status = integration.get_web3_formal_status()
    assert "web3_formal_available" in status
    assert "web3_formal_verification_available" in status
    assert "web3_formal_tools" in status
    assert "formal_capabilities" in status
    assert "audit_exploit_verification_available" in status


def test_bubblelabs_leanaide_bridge_status_exposes_web3_formal_schema():
    bridge = bubblelabs_leanaide.get_leanaide_bridge()
    status = bridge.get_status()
    assert "web3_formal_available" in status
    assert "web3_formal_verification_available" in status
    assert "web3_formal_tools" in status
    assert "formal_capabilities" in status
    assert "audit_exploit_verification_available" in status
