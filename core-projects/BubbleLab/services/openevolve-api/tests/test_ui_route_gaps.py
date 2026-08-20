"""
Tests for the UI route gaps that previously 404'd:
  - GET  /api/executions
  - GET  /api/workflows/{workflow_id}/decomposition-plan
  - /bubblelabs/control/{catalog,discover,execute}
  - /bubblelabs/workflow-definitions (+ /{id})
  - /bubblelabs/workflow-instances (+ /{id} and lifecycle sub-actions)

Run with:
    python -m pytest tests/test_ui_route_gaps.py -q -p no:pytest_ethereum
"""

import os
import sys
import tempfile
import types
from pathlib import Path

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault(
    "WORKFLOW_DB_PATH",
    str(Path(tempfile.gettempdir()) / "openevolve_api_ui_route_gaps_workflows.db"),
)

if "openevolve_api" not in sys.modules:
    package = types.ModuleType("openevolve_api")
    package.__path__ = [str(SERVICE_ROOT)]  # type: ignore[attr-defined]
    sys.modules["openevolve_api"] = package

from fastapi.testclient import TestClient  # noqa: E402

from openevolve_api.main import app  # noqa: E402


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


# ----------------------------- Executions list ----------------------------- #

def test_list_executions(client):
    resp = client.get("/api/executions")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "executions" in body
    assert "total" in body
    assert isinstance(body["executions"], list)


# ----------------------------- Decomposition plan ----------------------------- #

def test_workflow_decomposition_plan(client):
    workflow_id = "wf_route_gap_test"
    resp = client.get(f"/api/workflows/{workflow_id}/decomposition-plan")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["workflow_id"] == workflow_id
    assert "plan" in body
    assert "dependency_graph" in body
    assert "sub_problems" in body["plan"]
    assert "execution_order" in body["dependency_graph"]


# ----------------------------- Control plane ----------------------------- #

def test_control_catalog(client):
    resp = client.get("/bubblelabs/control/catalog")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True
    assert "components" in body


def test_control_discover(client):
    resp = client.post("/bubblelabs/control/discover", json={"force": True})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True
    assert "discovered_components" in body


def test_control_execute(client):
    resp = client.post(
        "/bubblelabs/control/execute",
        json={"component": "evolution", "action": "start", "payload": {"x": 1}},
    )
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["success"] is True
    assert "handle_id" in body


def test_control_execute_requires_component_and_action(client):
    resp = client.post("/bubblelabs/control/execute", json={})
    assert resp.status_code == 400, resp.text


# ----------------------------- Workflow definitions ----------------------------- #

def test_workflow_definitions_crud(client):
    # List (empty initially)
    resp = client.get("/bubblelabs/workflow-definitions")
    assert resp.status_code == 200, resp.text
    assert "definitions" in resp.json()

    # Create
    create = client.post(
        "/bubblelabs/workflow-definitions",
        json={
            "name": "Gap Test Definition",
            "description": "created by route gap test",
            "workflow_type": "sovereign",
            "parameters": {"max_iterations": 10},
        },
    )
    assert create.status_code == 201, create.text
    definition_id = create.json()["definition_id"]

    # Get detail
    get = client.get(f"/bubblelabs/workflow-definitions/{definition_id}")
    assert get.status_code == 200, get.text
    body = get.json()
    assert body["id"] == definition_id
    assert "parameters" in body
    assert "nodes" in body
    assert "edges" in body

    # Unknown id -> 404
    missing = client.get("/bubblelabs/workflow-definitions/does_not_exist")
    assert missing.status_code == 404


# ----------------------------- Workflow instances ----------------------------- #

def test_workflow_instances_lifecycle(client):
    # Create a definition to back the instance.
    definition = client.post(
        "/bubblelabs/workflow-definitions",
        json={"name": "Inst Def", "workflow_type": "sovereign", "parameters": {}},
    )
    definition_id = definition.json()["definition_id"]

    # List (empty initially)
    resp = client.get("/bubblelabs/workflow-instances")
    assert resp.status_code == 200, resp.text
    assert "instances" in resp.json()

    # Create instance
    create = client.post(
        "/bubblelabs/workflow-instances",
        json={
            "definition_id": definition_id,
            "instance_name": "gap-instance",
            "inputs": {"problem_statement": "solve the gap"},
            "parameters": {"temperature": 0.5},
        },
    )
    assert create.status_code == 201, create.text
    instance_id = create.json()["instance_id"]

    # Get detail
    get = client.get(f"/bubblelabs/workflow-instances/{instance_id}")
    assert get.status_code == 200, get.text
    detail = get.json()
    assert "status" in detail
    assert "parameters" in detail
    assert detail["parameters"]["temperature"] == 0.5

    # Start
    start = client.post(f"/bubblelabs/workflow-instances/{instance_id}/start")
    assert start.status_code == 200, start.text
    assert start.json()["status"] in ("running", "queued")

    # Pause / resume / stop / restart
    assert client.post(f"/bubblelabs/workflow-instances/{instance_id}/pause").status_code == 200
    assert client.post(f"/bubblelabs/workflow-instances/{instance_id}/resume").status_code == 200
    assert client.post(f"/bubblelabs/workflow-instances/{instance_id}/stop").status_code == 200
    assert client.post(f"/bubblelabs/workflow-instances/{instance_id}/restart").status_code == 200
    assert client.post(f"/bubblelabs/workflow-instances/{instance_id}/cancel").status_code == 200

    # Parameters GET returns the instance's parameters
    params = client.get(f"/bubblelabs/workflow-instances/{instance_id}/parameters")
    assert params.status_code == 200, params.text
    assert "parameters" in params.json()

    # Parameters POST syncs parameters
    sync = client.post(
        f"/bubblelabs/workflow-instances/{instance_id}/parameters",
        json={"parameters": {"a": 1, "b": 2}},
    )
    assert sync.status_code == 200, sync.text
    assert sync.json()["updated_count"] == 2

    # Unknown id -> 404
    assert client.get("/bubblelabs/workflow-instances/missing").status_code == 404
    assert client.post("/bubblelabs/workflow-instances/missing/start").status_code == 404
    assert client.delete("/bubblelabs/workflow-instances/missing").status_code == 404

    # Delete
    delete = client.delete(f"/bubblelabs/workflow-instances/{instance_id}")
    assert delete.status_code == 200, delete.text
