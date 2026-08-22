"""
Tests for the unified :8000 workflow settings / plan / run surface.

Exercises GET/PUT /api/workflows/{id}/settings, PUT /api/workflows/{id}/decomposition-plan
and POST /api/workflows/{id}/run against the :8000 service. The :8001 engine call is
monkeypatched (no live server) so the test is hermetic.
"""

import os
import sys
import types
import tempfile
from pathlib import Path

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault(
    "WORKFLOW_DB_PATH",
    str(Path(tempfile.gettempdir()) / "openevolve_api_settings_run_workflows.db"),
)
os.environ.setdefault("OPENEVOLVE_BRIDGE_ENABLED", "1")

if "openevolve_api" not in sys.modules:
    pkg = types.ModuleType("openevolve_api")
    pkg.__path__ = [str(SERVICE_ROOT)]
    sys.modules["openevolve_api"] = pkg

from fastapi.testclient import TestClient  # noqa: E402

from openevolve_api.main import app  # noqa: E402
from openevolve_api.api import engine_proxy as engine_proxy  # noqa: E402


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def workflow_id(client):
    resp = client.post(
        "/api/workflows",
        json={
            "name": "run-settings-test",
            "description": "x",
            "workflow_type": "sovereign",
            "problem_statement": "Solve the travelling salesman problem efficiently.",
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def test_settings_get_defaults(client, workflow_id):
    resp = client.get(f"/api/workflows/{workflow_id}/settings")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["mdap_enabled"] is False
    assert data["max_refinement_loops"] == 5
    assert isinstance(data["resource_limits"], dict)
    assert isinstance(data["web3"], dict)


def test_settings_put_then_get(client, workflow_id):
    resp = client.put(
        f"/api/workflows/{workflow_id}/settings",
        json={"mdap_enabled": True, "max_refinement_loops": 7},
    )
    assert resp.status_code == 200, resp.text
    updated = resp.json()
    assert updated["mdap_enabled"] is True
    assert updated["max_refinement_loops"] == 7
    # Other fields preserved from defaults via merge.
    assert updated["maker_enabled"] is False

    resp2 = client.get(f"/api/workflows/{workflow_id}/settings")
    assert resp2.status_code == 200, resp2.text
    assert resp2.json()["mdap_enabled"] is True
    assert resp2.json()["max_refinement_loops"] == 7


def test_plan_put_returns_execution_order(client, workflow_id):
    resp = client.put(
        f"/api/workflows/{workflow_id}/decomposition-plan",
        json={
            "sub_problems": [
                {"id": "sp-1", "description": "a", "dependencies": []},
                {"id": "sp-2", "description": "b", "dependencies": ["sp-1"]},
                {"id": "sp-3", "description": "c", "dependencies": ["sp-2"]},
            ],
            "max_refinement_loops": 3,
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["message"]
    assert data["execution_order"] == ["sp-1", "sp-2", "sp-3"]

    # GET now returns the stored plan.
    resp2 = client.get(f"/api/workflows/{workflow_id}/decomposition-plan")
    assert resp2.status_code == 200, resp2.text
    plan = resp2.json()
    assert plan["plan"]["sub_problems"][0]["id"] == "sp-1"
    assert plan["dependency_graph"]["execution_order"] == ["sp-1", "sp-2", "sp-3"]


def test_run_forwards_to_engine_and_returns_its_response(monkeypatch, client, workflow_id):
    captured = {}

    async def fake_run(problem_statement, config, api_key=None):
        captured["problem_statement"] = problem_statement
        captured["config"] = config
        captured["api_key"] = api_key
        return {"workflow_id": "engine-wf-123", "status": "running", "tenant_id": "default"}

    monkeypatch.setattr(engine_proxy, "run_workflow_on_engine", fake_run)

    resp = client.post(
        f"/api/workflows/{workflow_id}/run",
        json={"config": {"maker_enabled": True}},
        headers={"X-API-Key": "secret-key"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["workflow_id"] == "engine-wf-123"
    # Problem statement came from the :8000 store.
    assert "travelling salesman" in captured["problem_statement"]
    # Stored settings + caller config are merged into the engine config.
    assert captured["config"]["maker_enabled"] is True
    assert captured["api_key"] == "secret-key"


def test_run_404_for_unknown_workflow(client):
    resp = client.post("/api/workflows/nope/run", json={"config": {}})
    assert resp.status_code == 404
