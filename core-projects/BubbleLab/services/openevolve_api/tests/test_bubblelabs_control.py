"""
Contract test for the BubbleLabs control plane workflow lifecycle.

Exercises the real mounted router via ``fastapi.testclient.TestClient`` with NO
live server and NO network. It creates a workflow definition, creates an
instance, starts it, and asserts the status progresses (200/valid, never 500).

The real engine dispatch (``execution_manager``) is mocked so the test runs
fully offline and deterministically.
"""

import os
import sys
import types
import uuid
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]

# Isolate the persistent stores before service modules import (eager openers).
os.environ.setdefault(
    "BUBBLELABS_DB_PATH",
    str(Path(tempfile.gettempdir()) / "openevolve_api_bubblelabs_test.json"),
)
os.environ.setdefault(
    "WORKFLOW_DB_PATH",
    str(Path(tempfile.gettempdir()) / "openevolve_api_bubblelabs_wf.db"),
)
os.environ.setdefault("OPENEVOLVE_BRIDGE_ENABLED", "1")

if "openevolve_api" not in sys.modules:
    package = types.ModuleType("openevolve_api")
    package.__path__ = [str(SERVICE_ROOT)]  # type: ignore[attr-defined]
    sys.modules["openevolve_api"] = package

from fastapi.testclient import TestClient  # noqa: E402

from openevolve_api.main import app  # noqa: E402
from openevolve_api.services.execution_service import execution_manager  # noqa: E402


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def _mock_engine_dispatch():
    """Offline, deterministic engine dispatch: queued -> completed on first poll."""
    real_start = execution_manager.start_execution
    real_status = execution_manager.get_execution_status
    runs: dict = {}

    async def fake_start(workflow_id, problem_statement, context=None):
        exec_id = f"exec_{uuid.uuid4().hex[:12]}"
        runs[exec_id] = {"status": "queued", "progress": 0.0}
        return {
            "execution_id": exec_id,
            "workflow_id": workflow_id,
            "status": "queued",
            "progress": 0.0,
            "started_at": datetime.now(timezone.utc),
            "completed_at": None,
            "result": None,
            "error": None,
            "workflow_type": "sovereign",
            "parameters": {},
        }

    async def fake_status(exec_id):
        rec = runs.get(exec_id)
        if rec is None:
            return None
        # First reconciliation poll marks the run completed.
        rec["status"] = "completed"
        rec["progress"] = 1.0
        return {
            "execution_id": exec_id,
            "workflow_id": "x",
            "status": rec["status"],
            "progress": rec["progress"],
            "started_at": datetime.now(timezone.utc),
            "completed_at": datetime.now(timezone.utc),
            "result": {"ok": True, "engine": "mock"},
            "error": None,
            "workflow_type": "sovereign",
            "parameters": {},
        }

    execution_manager.start_execution = fake_start
    execution_manager.get_execution_status = fake_status
    try:
        yield
    finally:
        execution_manager.start_execution = real_start
        execution_manager.get_execution_status = real_status


def test_bubblelabs_control_lifecycle(client):
    # Catalog is reachable and valid.
    cat = client.get("/bubblelabs/control/catalog")
    assert cat.status_code == 200
    assert cat.json().get("success") is True

    # Create a workflow definition.
    d = client.post(
        "/bubblelabs/workflow-definitions",
        json={"name": "bl-wf", "workflow_type": "sovereign", "description": "contract test"},
    )
    assert d.status_code == 201
    definition_id = d.json()["definition_id"]

    # Create an instance from the definition.
    i = client.post(
        "/bubblelabs/workflow-instances",
        json={"definition_id": definition_id, "inputs": {"problem_statement": "solve x"}},
    )
    assert i.status_code == 201
    instance_id = i.json()["instance_id"]

    # Start the instance -> dispatches a real run, status must progress (no 500).
    s = client.post(f"/bubblelabs/workflow-instances/{instance_id}/start")
    assert s.status_code == 200
    body = s.json()
    assert body["instance_id"] == instance_id
    assert body["action"] == "start"
    assert body["status"] in ("queued", "running")

    # GET reconciles with the real execution -> terminal/running state, never 500.
    g = client.get(f"/bubblelabs/workflow-instances/{instance_id}")
    assert g.status_code == 200
    detail = g.json()["status"]
    assert detail["status"] in ("queued", "running", "completed")
    assert detail["current_stage"] in ("queued", "running", "completed")

    # Control execute for an engine-backed component must not 500.
    ce = client.post(
        "/bubblelabs/control/execute",
        json={"component": "sovereign", "action": "start",
              "payload": {"problem_statement": "verify y"}},
    )
    assert ce.status_code in (200, 202)
    assert ce.json().get("success") is True


def test_bubblelabs_instance_crud_shapes(client):
    # Unknown instance -> 404 (valid, not 500).
    missing = client.get("/bubblelabs/workflow-instances/does_not_exist")
    assert missing.status_code == 404

    # Definition required to create an instance.
    bad = client.post("/bubblelabs/workflow-instances", json={})
    assert bad.status_code == 400

    # List endpoints return valid JSON. 200.
    assert client.get("/bubblelabs/workflow-definitions").status_code == 200
    assert client.get("/bubblelabs/workflow-instances").status_code == 200
