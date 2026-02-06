import asyncio
import pytest


z3_api_server = pytest.importorskip("z3_api_server")


def test_z3_api_registers_web3_formal_routes():
    paths = {route.path for route in z3_api_server.app.routes}
    expected = {
        "/web3/status",
        "/web3/invariants/translate",
        "/web3/exploits/symbolic-witness",
    }
    assert expected.issubset(paths)


def test_z3_api_web3_status_shape():
    status = asyncio.run(z3_api_server.get_web3_formal_status())
    assert "available" in status
    assert "solidity_invariant_translation_available" in status
    assert "exploit_witness_available" in status
    assert "tool_inventory" in status
