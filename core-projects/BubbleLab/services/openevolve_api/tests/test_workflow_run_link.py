"""
Tests for the durable :8000 -> :8001 engine link.

Covers:
  * POST /api/workflows/{id}/run now durably records the engine workflow id
    (``_workflow_executions`` + ``parameters.last_engine_workflow_id``) and flips
    the :8000 workflow status to RUNNING.
  * The /engine/* reverse-proxy routes forward to :8001 by the stored engine id,
    preserving the X-API-Key header, and 404 when the link/workflow is missing.

The :8001 engine is monkeypatched (no live server), so the test is hermetic.
Run with:
    python -m pytest tests/test_workflow_run_link.py -q -p no:pytest_ethereum
"""

import os
import sys
import types
from pathlib import Path

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault(
    "WORKFLOW_DB_PATH",
    str(Path(__file__).resolve().parent / "workflow_run_link_test.db"),
)
os.environ.setdefault("OPENEVOLVE_BRIDGE_ENABLED", "1")

if "openevolve_api" not in sys.modules:
    pkg = types.ModuleType("openevolve_api")
    pkg.__path__ = [str(SERVICE_ROOT)]
    sys.modules["openevolve_api"] = pkg

import httpx  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from openevolve_api.main import app  # noqa: E402
from openevolve_api.api import engine_proxy as engine_proxy  # noqa: E402
from openevolve_api.api import workflows as workflows_module  # noqa: E402


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def workflow_id(client):
    resp = client.post(
        "/api/workflows",
        json={
            "name": "run-link-test",
            "description": "x",
            "workflow_type": "sovereign",
            "problem_statement": "Optimise packet routing for low latency.",
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


# ----------------------------- run link ----------------------------- #
def test_run_records_engine_link_and_status(monkeypatch, client, workflow_id):
    async def fake_run(problem_statement, config, api_key=None):
        return {"workflow_id": "eng-123", "status": "running", "tenant_id": "t"}

    monkeypatch.setattr(engine_proxy, "run_workflow_on_engine", fake_run)

    resp = client.post(
        f"/api/workflows/{workflow_id}/run",
        json={"config": {}},
        headers={"X-API-Key": "secret-key"},
    )
    assert resp.status_code == 200, resp.text

    # last_engine_workflow_id is stored on the workflow.
    detail = client.get(f"/api/workflows/{workflow_id}").json()
    assert detail["parameters"]["last_engine_workflow_id"] == "eng-123"

    # _workflow_executions is durably linked.
    assert workflows_module._workflow_executions.get(workflow_id) == "eng-123"

    # The :8000 workflow reflects the running status.
    assert detail["status"] == "running"


# ----------------------------- proxy routes ----------------------------- #
class _FakeAsyncClient:
    """Mimics httpx.AsyncClient for the forwarding path."""

    def __init__(self, response, raise_exc=None):
        self._response = response
        self._raise_exc = raise_exc
        self.last_request = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def request(self, method, url, **kwargs):
        if self._raise_exc is not None:
            raise self._raise_exc
        self.last_request = (method, url, kwargs)
        return self._response


def test_engine_results_forwards_to_8001(monkeypatch, client, workflow_id):
    resp_obj = httpx.Response(200, json={"score": 0.9})
    fake = _FakeAsyncClient(resp_obj)
    monkeypatch.setattr(workflows_module, "httpx_client", lambda: fake)

    # First run to link the workflow to an engine id.
    async def fake_run(problem_statement, config, api_key=None):
        return {"workflow_id": "eng-123", "status": "running", "tenant_id": "t"}

    monkeypatch.setattr(engine_proxy, "run_workflow_on_engine", fake_run)
    run_resp = client.post(
        f"/api/workflows/{workflow_id}/run",
        json={},
        headers={"X-API-Key": "abc"},
    )
    assert run_resp.status_code == 200, run_resp.text

    resp = client.get(
        f"/api/workflows/{workflow_id}/engine/results",
        headers={"X-API-Key": "abc"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"score": 0.9}

    method, url, kwargs = fake.last_request
    assert method == "GET"
    assert url == "http://localhost:8001/workflows/eng-123/results"
    assert kwargs["headers"].get("X-API-Key") == "abc"


def test_engine_telemetry_and_resource_usage_routes(monkeypatch, client, workflow_id):
    fake = _FakeAsyncClient(httpx.Response(200, json={"ok": True}))
    monkeypatch.setattr(workflows_module, "httpx_client", lambda: fake)

    async def fake_run(problem_statement, config, api_key=None):
        return {"workflow_id": "eng-123", "status": "running", "tenant_id": "t"}

    monkeypatch.setattr(engine_proxy, "run_workflow_on_engine", fake_run)
    client.post(f"/api/workflows/{workflow_id}/run", json={})

    for suffix in ("telemetry", "resource-usage"):
        resp = client.get(f"/api/workflows/{workflow_id}/engine/{suffix}")
        assert resp.status_code == 200, resp.text
        method, url, _ = fake.last_request
        assert method == "GET"
        assert url == f"http://localhost:8001/workflows/eng-123/{suffix}"


def test_engine_truth_package_forwards_post(monkeypatch, client, workflow_id):
    fake = _FakeAsyncClient(httpx.Response(200, json={"packaged": True}))
    monkeypatch.setattr(workflows_module, "httpx_client", lambda: fake)

    async def fake_run(problem_statement, config, api_key=None):
        return {"workflow_id": "eng-123", "status": "running", "tenant_id": "t"}

    monkeypatch.setattr(engine_proxy, "run_workflow_on_engine", fake_run)
    client.post(f"/api/workflows/{workflow_id}/run", json={})

    resp = client.post(
        f"/api/workflows/{workflow_id}/engine/truth-package",
        json={"format": "json"},
        headers={"X-API-Key": "abc"},
    )
    assert resp.status_code == 200, resp.text
    method, url, kwargs = fake.last_request
    assert method == "POST"
    assert url == "http://localhost:8001/workflows/eng-123/truth-package"
    assert kwargs["headers"].get("X-API-Key") == "abc"


def test_engine_proxy_404_when_no_link(client, workflow_id):
    resp = client.get(f"/api/workflows/{workflow_id}/engine/results")
    assert resp.status_code == 404
    assert "no linked engine run" in resp.json()["detail"]


def test_engine_proxy_404_when_workflow_missing(client):
    resp = client.get("/api/workflows/does-not-exist/engine/results")
    assert resp.status_code == 404


def test_engine_proxy_502_when_unreachable(monkeypatch, client, workflow_id):
    fake = _FakeAsyncClient(None, raise_exc=httpx.ConnectError("refused"))
    monkeypatch.setattr(workflows_module, "httpx_client", lambda: fake)

    async def fake_run(problem_statement, config, api_key=None):
        return {"workflow_id": "eng-123", "status": "running", "tenant_id": "t"}

    monkeypatch.setattr(engine_proxy, "run_workflow_on_engine", fake_run)
    client.post(f"/api/workflows/{workflow_id}/run", json={})

    resp = client.get(f"/api/workflows/{workflow_id}/engine/results")
    assert resp.status_code == 502
    assert "unreachable" in resp.json()["detail"]
