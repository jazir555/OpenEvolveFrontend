"""
Tests for the BubbleLabs security proxy router.

The proxy forwards ``/security/*`` to the ``:8001`` engine, preserving the
upstream status code, headers and JSON body, and returns 502 when the engine is
unreachable. The forwarding uses ``httpx.AsyncClient.request``; we monkeypatch
``security_proxy.httpx_client`` so the tests are deterministic and do not
require a live ``:8001`` service.

Run with:
    python -m pytest tests/test_security_proxy.py -q -p no:pytest_ethereum
"""

import types
from pathlib import Path

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]

import os  # noqa: E402

os.environ.setdefault(
    "WORKFLOW_DB_PATH",
    str(Path(__file__).resolve().parent / "security_proxy_test.db"),
)

if "openevolve_api" not in __import__("sys").modules:
    import sys

    package = types.ModuleType("openevolve_api")
    package.__path__ = [str(SERVICE_ROOT)]  # type: ignore[attr-defined]
    sys.modules["openevolve_api"] = package

import httpx  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from openevolve_api.main import app  # noqa: E402
from openevolve_api.api import security_proxy as security_proxy_module  # noqa: E402


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


class _FakeAsyncClient:
    """Mimics httpx.AsyncClient for the forwarding path."""

    def __init__(self, response, raise_exc=None):
        self._response = response
        self._raise_exc = raise_exc

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def request(self, method, url, **kwargs):
        if self._raise_exc is not None:
            raise self._raise_exc
        self.last_request = (method, url, kwargs)
        return self._response


# ----------------------------- happy path (forwards status + body) ----------------------------- #
def test_security_get_forwards_status_and_body(client, monkeypatch):
    resp_obj = httpx.Response(
        200,
        json={"keys": [{"id": "k1"}]},
        headers={"content-type": "application/json"},
    )
    monkeypatch.setattr(
        security_proxy_module,
        "httpx_client",
        lambda: _FakeAsyncClient(resp_obj),
    )

    resp = client.get("/security/api-keys", headers={"X-API-Key": "admin-key"})
    assert resp.status_code == 200
    assert resp.json() == {"keys": [{"id": "k1"}]}


def test_security_post_forwards_body_and_method(client, monkeypatch):
    resp_obj = httpx.Response(201, json={"id": "k2", "created": True})
    fake = _FakeAsyncClient(resp_obj)
    monkeypatch.setattr(
        security_proxy_module, "httpx_client", lambda: fake
    )

    resp = client.post(
        "/security/api-keys",
        json={"name": "ci", "role": "admin"},
        headers={"X-API-Key": "admin-key"},
    )
    assert resp.status_code == 201
    assert resp.json() == {"id": "k2", "created": True}

    method, url, kwargs = fake.last_request
    assert method == "POST"
    assert url.endswith("/security/api-keys")
    assert b'"name":"ci"' in kwargs["content"]


def test_security_delete_forwards_path(client, monkeypatch):
    resp_obj = httpx.Response(204)
    fake = _FakeAsyncClient(resp_obj)
    monkeypatch.setattr(
        security_proxy_module, "httpx_client", lambda: fake
    )

    resp = client.delete(
        "/security/api-keys/k2", headers={"X-API-Key": "admin-key"}
    )
    assert resp.status_code == 204
    method, url, _ = fake.last_request
    assert method == "DELETE"
    assert url.endswith("/security/api-keys/k2")


def test_security_forwards_api_key_header(client, monkeypatch):
    captured = {}

    class _CaptureClient(_FakeAsyncClient):
        async def request(self, method, url, **kwargs):
            captured.update(kwargs)
            return httpx.Response(200, json={})

    monkeypatch.setattr(
        security_proxy_module,
        "httpx_client",
        lambda: _CaptureClient(httpx.Response(200, json={})),
    )

    client.get("/security/roles", headers={"X-API-Key": "abc"})
    assert captured["headers"].get("X-API-Key") == "abc"


def test_security_forwards_authorization_header(client, monkeypatch):
    captured = {}

    class _CaptureClient(_FakeAsyncClient):
        async def request(self, method, url, **kwargs):
            captured.update(kwargs)
            return httpx.Response(200, json={})

    monkeypatch.setattr(
        security_proxy_module,
        "httpx_client",
        lambda: _CaptureClient(httpx.Response(200, json={})),
    )

    client.get(
        "/security/audit-logs",
        headers={"Authorization": "Bearer tok", "X-API-Key": "abc"},
    )
    assert captured["headers"].get("Authorization") == "Bearer tok"


# ----------------------------- unreachable engine ----------------------------- #
def test_security_unreachable_returns_502(client, monkeypatch):
    monkeypatch.setattr(
        security_proxy_module,
        "httpx_client",
        lambda: _FakeAsyncClient(None, raise_exc=httpx.ConnectError("refused")),
    )

    resp = client.get("/security/api-keys", headers={"X-API-Key": "admin-key"})
    assert resp.status_code == 502
    assert resp.json()["detail"] == "OpenEvolve engine (security) unreachable"


def test_security_other_http_error_returns_502(client, monkeypatch):
    monkeypatch.setattr(
        security_proxy_module,
        "httpx_client",
        lambda: _FakeAsyncClient(None, raise_exc=httpx.ReadTimeout("boom")),
    )

    resp = client.post("/security/roles", json={"name": "r"})
    assert resp.status_code == 502
    assert "unreachable" in resp.json()["detail"]


# ----------------------------- route registration ----------------------------- #
def test_security_routes_registered(client):
    paths = {route.path for route in app.routes}
    # Catch-all mounted under /security; FastAPI exposes the mounted prefix path.
    assert any(p.startswith("/security") for p in paths)
