import asyncio

import pytest


REQUIRED_WEB3_STATUS_KEYS = [
    "web3_formal_available",
    "web3_formal_verification_available",
    "web3_formal_tools",
    "formal_capabilities",
    "audit_exploit_verification_available",
]


def _as_dict(value):
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return value


def _assert_web3_status_schema(payload):
    for key in REQUIRED_WEB3_STATUS_KEYS:
        assert key in payload


def test_mcp_status_exposes_web3_formal_schema():
    mcp = pytest.importorskip("leanaide_mcp_tools")
    status = mcp.get_leanaide_status()
    _assert_web3_status_schema(status)


def test_lean_mdap_status_exposes_web3_formal_schema():
    mdap = pytest.importorskip("leanaide_mdap")
    status = mdap.get_lean_mdap_status()
    _assert_web3_status_schema(status)


def test_lean_mdap_workflow_progress_exposes_web3_formal_schema():
    workflow = pytest.importorskip("leanaide_mdap_workflow")
    monitor = workflow.LeanMDAPMonitor(mdap_integrator=None, maker_integrator=None)
    status = monitor.get_progress()
    _assert_web3_status_schema(status)


def test_leanaide_maker_capabilities_expose_web3_formal_schema():
    maker = pytest.importorskip("leanaide_hybrid_maker_enhanced")
    status = maker.get_leanaide_maker_capabilities()
    _assert_web3_status_schema(status)


def test_bubblelabs_lean_node_status_exposes_web3_formal_schema():
    node_mod = pytest.importorskip("bubblelabs_nodes.lean_autoformalization_node")
    node = node_mod.LeanAutoformalizationNode(config={})
    status = node.get_lean_status()
    _assert_web3_status_schema(status)


def test_leanaide_cav_nlp_bridge_capabilities_expose_web3_formal_schema():
    bridge_mod = pytest.importorskip("openevolve.leanaide_cav_nlp_bridge")
    bridge = bridge_mod.LeanAideCAVNLPBridge(use_cav_nlp=False, use_unified_service=False)
    status = bridge.get_capabilities()
    _assert_web3_status_schema(status)


def test_knowledge_engine_leanaide_status_exposes_web3_formal_schema():
    integration_mod = pytest.importorskip("knowledge_engine.integrations.leanaide_integration")
    integration = integration_mod.LeanAideIntegration()
    status = integration.get_leanaide_status()
    _assert_web3_status_schema(status)


def test_leanaide_api_status_exposes_web3_formal_schema():
    api_routes = pytest.importorskip("leanaide_api_routes")
    response = asyncio.run(api_routes.leanaide_status())
    payload = _as_dict(response)
    _assert_web3_status_schema(payload)


def test_lean4_system_health_exposes_web3_formal_schema():
    lean4_api = pytest.importorskip("lean4_system.lean4_api")
    api = lean4_api.MathematicalVerificationAPI()
    status = api.health_check()
    _assert_web3_status_schema(status)
