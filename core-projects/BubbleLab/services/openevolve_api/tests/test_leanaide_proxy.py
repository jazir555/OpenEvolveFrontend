"""
Tests for the BubbleLabs LeanAide proxy router.

These cover both the graceful-degrade path (LeanAide down -> non-200 with
``leanaide_available: false`` rather than 500) and the happy path (proxy
forwards method/status/body to the upstream). The upstream ``_forward`` helper
is monkeypatched so the tests are deterministic and do not require a live
LeanAide service.

Run with:
    python -m pytest tests/test_leanaide_proxy.py -q -p no:pytest_ethereum
"""

import types
import urllib.error
from pathlib import Path

import pytest
import tempfile

SERVICE_ROOT = Path(__file__).resolve().parents[1]

import os  # noqa: E402

os.environ.setdefault(
    "WORKFLOW_DB_PATH",
    str(Path(tempfile.gettempdir()) / "openevolve_api_leanaide_proxy.db"),
)

if "openevolve_api" not in __import__("sys").modules:
    import sys

    package = types.ModuleType("openevolve_api")
    package.__path__ = [str(SERVICE_ROOT)]  # type: ignore[attr-defined]
    sys.modules["openevolve_api"] = package

from fastapi.testclient import TestClient  # noqa: E402

from openevolve_api.main import app  # noqa: E402
from openevolve_api.api import leanaide as leanaide_module  # noqa: E402


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


# ----------------------------- graceful degrade (upstream down) ----------------------------- #
async def _unreachable(*_args, **_kwargs):
    return (
        502,
        {
            "error": "LeanAide upstream unreachable: connection refused",
            "upstream": "http://localhost:7654/status",
            "leanaide_available": False,
        },
    )


def test_leanaide_health_down_graceful(client, monkeypatch):
    monkeypatch.setattr(leanaide_module, "_forward", _unreachable)
    resp = client.get("/api/bubblelabs/leanaide/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["leanaide_available"] is False


def test_leanaide_status_down_returns_502(client, monkeypatch):
    monkeypatch.setattr(leanaide_module, "_forward", _unreachable)
    resp = client.get("/api/bubblelabs/leanaide/status")
    assert resp.status_code == 502
    body = resp.json()
    assert body["leanaide_available"] is False
    assert "error" in body and "upstream" in body


def test_leanaide_prove_down_returns_502(client, monkeypatch):
    monkeypatch.setattr(leanaide_module, "_forward", _unreachable)
    resp = client.post("/api/bubblelabs/leanaide/prove", json={"theorem": "x = x"})
    assert resp.status_code == 502
    body = resp.json()
    assert body["leanaide_available"] is False


def test_leanaide_trees_and_proofs_down_502(client, monkeypatch):
    monkeypatch.setattr(leanaide_module, "_forward", _unreachable)
    assert client.get("/api/bubblelabs/leanaide/trees").status_code == 502
    assert client.get("/api/bubblelabs/leanaide/trees/t1").status_code == 502
    assert client.get("/api/bubblelabs/leanaide/proofs").status_code == 502
    assert client.get("/api/bubblelabs/leanaide/proofs/p1").status_code == 502


# ----------------------------- happy path (proxy forwards) ----------------------------- #
async def _fake_forward(upstream_path, method, request=None, *, json_body=None):
    # Echo what was forwarded so we can assert the proxy passes through.
    return 200, {
        "leanaide_available": True,
        "server": "fake",
        "path": upstream_path,
        "method": method,
        "echo_body": json_body,
    }


def test_leanaide_status_forwards(client, monkeypatch):
    monkeypatch.setattr(leanaide_module, "_forward", _fake_forward)
    resp = client.get("/api/bubblelabs/leanaide/status")
    assert resp.status_code == 200
    assert resp.json()["path"] == "status"


def test_leanaide_execute_forwards_body(client, monkeypatch):
    monkeypatch.setattr(leanaide_module, "_forward", _fake_forward)
    resp = client.post(
        "/api/bubblelabs/leanaide/execute",
        json={"task_type": "prove", "payload": {"x": 1}},
    )
    body = resp.json()
    assert resp.status_code == 200
    assert body["path"] == "execute"
    assert body["method"] == "POST"
    assert body["echo_body"] == {"task_type": "prove", "payload": {"x": 1}}


def test_leanaide_prove_forwards_body_and_path(client, monkeypatch):
    monkeypatch.setattr(leanaide_module, "_forward", _fake_forward)
    resp = client.post("/api/bubblelabs/leanaide/prove", json={"theorem": "x=x"})
    body = resp.json()
    assert resp.status_code == 200
    assert body["path"] == "prove"
    assert body["echo_body"] == {"theorem": "x=x"}


def test_leanaide_tree_and_proof_paths_forward(client, monkeypatch):
    monkeypatch.setattr(leanaide_module, "_forward", _fake_forward)
    assert client.get("/api/bubblelabs/leanaide/trees").json()["path"] == "trees"
    assert (
        client.get("/api/bubblelabs/leanaide/trees/t1").json()["path"] == "trees/t1"
    )
    assert client.get("/api/bubblelabs/leanaide/proofs").json()["path"] == "proofs"
    assert (
        client.get("/api/bubblelabs/leanaide/proofs/p1").json()["path"]
        == "proofs/p1"
    )


# ----------------------------- route registration ----------------------------- #
def test_leanaide_routes_registered(client):
    paths = {route.path for route in app.routes}
    expected = {
        "/api/bubblelabs/leanaide/health",
        "/api/bubblelabs/leanaide/status",
        "/api/bubblelabs/leanaide/execute",
        "/api/bubblelabs/leanaide/trees",
        "/api/bubblelabs/leanaide/trees/{tree_id}",
        "/api/bubblelabs/leanaide/proofs",
        "/api/bubblelabs/leanaide/proofs/{proof_id}",
        "/api/bubblelabs/leanaide/prove",
    }
    assert expected.issubset(paths)
