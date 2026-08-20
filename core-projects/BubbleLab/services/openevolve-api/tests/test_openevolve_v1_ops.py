"""
Tests for the workflow sub-operations added to ``/api/v1/*``: stop / pause /
resume / configure / batch / chain. These resolve the bubble's
WorkflowOrchestrator operations to real handlers (no 404).

Run with:
    python -m pytest tests/test_openevolve_v1_ops.py -q
"""

import os
import sys
import tempfile
import time
import types
import uuid
from pathlib import Path

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]

# Isolate the workflow DB before the service modules import (they open it eagerly).
os.environ.setdefault(
    "WORKFLOW_DB_PATH",
    str(Path(tempfile.gettempdir()) / "openevolve_api_v1_ops_workflows.db"),
)
os.environ.setdefault("OPENEVOLVE_BRIDGE_ENABLED", "1")

if "openevolve_api" not in sys.modules:
    package = types.ModuleType("openevolve_api")
    package.__path__ = [str(SERVICE_ROOT)]  # type: ignore[attr-defined]
    sys.modules["openevolve_api"] = package

pytest.importorskip(
    "openevolve", reason="the real openevolve library must be installed (pip install -e ../../../openevolve)"
)

from fastapi.testclient import TestClient  # noqa: E402

from openevolve_api.api import openevolve_v1 as v1  # noqa: E402
from openevolve_api.main import app  # noqa: E402


@pytest.fixture()
def client():
    # Reset module-level config overrides between tests for isolation.
    with v1._RUNS_LOCK:
        v1.DEFAULT_RUN_CONFIG.clear()
    with TestClient(app) as test_client:
        yield test_client


def _inject_run(status: str = "running") -> str:
    run_id = uuid.uuid4().hex
    with v1._RUNS_LOCK:
        v1.RUNS[run_id] = {
            "run_id": run_id,
            "status": status,
            "result": None,
            "error": None,
        }
    return run_id


# --------------------------------------------------------------------------- #
# stop / pause / resume
# --------------------------------------------------------------------------- #
def test_stop_workflow_updates_status(client):
    run_id = _inject_run("running")
    resp = client.post(f"/api/v1/workflows/{run_id}/stop")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["workflowId"] == run_id
    assert body["status"] == "stopped"
    assert v1._get_run(run_id)["status"] == "stopped"


def test_pause_then_resume_workflow(client):
    run_id = _inject_run("running")
    resp = client.post(f"/api/v1/workflows/{run_id}/pause")
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "paused"
    assert v1._get_run(run_id)["status"] == "paused"

    resp = client.post(f"/api/v1/workflows/{run_id}/resume")
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "running"
    assert v1._get_run(run_id)["status"] == "running"


def test_stop_preserves_terminal_status(client):
    run_id = _inject_run("completed")
    resp = client.post(f"/api/v1/workflows/{run_id}/stop")
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "completed"
    assert v1._get_run(run_id)["status"] == "completed"


def test_stop_unknown_run_is_404(client):
    resp = client.post(f"/api/v1/workflows/{uuid.uuid4().hex}/stop")
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# configure
# --------------------------------------------------------------------------- #
def test_configure_stores_overrides(client):
    resp = client.post(
        "/api/v1/workflows/configure",
        json={
            "workflowName": "demo",
            "definition": {"iterations": 7, "populationSize": 12},
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "configured"
    assert body["config"]["max_iterations"] == 7
    assert body["config"]["population_size"] == 12
    # Applied to a subsequent orchestrate call.
    assert v1.DEFAULT_RUN_CONFIG["max_iterations"] == 7


def test_configure_requires_definition(client):
    resp = client.post("/api/v1/workflows/configure", json={"workflowName": "x"})
    assert resp.status_code == 400


def test_configure_applies_to_future_orchestrate(client):
    client.post(
        "/api/v1/workflows/configure",
        json={"definition": {"iterations": 2, "populationSize": 3}},
    )
    resp = client.post(
        "/api/v1/workflows/orchestrate",
        json={"system": "evolutionary", "problemStatement": "add two numbers"},
    )
    assert resp.status_code == 202, resp.text


# --------------------------------------------------------------------------- #
# batch_execute
# --------------------------------------------------------------------------- #
def test_batch_execute_starts_multiple_runs(client):
    resp = client.post(
        "/api/v1/workflows/batch",
        json={
            "workflows": [
                {"system": "evolutionary", "problemStatement": "task one"},
                {"system": "evolutionary", "problemStatement": "task two"},
            ]
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["count"] == 2
    assert len(body["run_ids"]) == 2
    assert "batchId" in body
    for rid in body["run_ids"]:
        assert v1._get_run(rid) is not None


def test_batch_execute_requires_workflows(client):
    resp = client.post("/api/v1/workflows/batch", json={})
    assert resp.status_code == 400


def test_batch_execute_rejects_missing_problem_statement(client):
    resp = client.post(
        "/api/v1/workflows/batch",
        json={"workflows": [{"system": "evolutionary"}]},
    )
    assert resp.status_code == 400


# --------------------------------------------------------------------------- #
# chain_workflows
# --------------------------------------------------------------------------- #
def test_chain_workflows_starts_ordered_steps(client):
    resp = client.post(
        "/api/v1/workflows/chain",
        json={
            "chain": [
                {"system": "evolutionary", "problemStatement": "step one"},
                {"system": "evolutionary", "problemStatement": "step two"},
            ]
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["count"] == 2
    assert len(body["run_ids"]) == 2
    assert "chainId" in body
    for rid in body["run_ids"]:
        run = v1._get_run(rid)
        assert run is not None
        assert run["status"] in ("pending", "running", "completed", "failed")


def test_chain_workflows_requires_chain(client):
    resp = client.post("/api/v1/workflows/chain", json={})
    assert resp.status_code == 400


def test_chain_steps_complete_over_real_engine(client):
    resp = client.post(
        "/api/v1/workflows/chain",
        json={
            "chain": [
                {"system": "evolutionary", "problemStatement": "evolve adder", "generations": 2, "populationSize": 3},
            ]
        },
    )
    assert resp.status_code == 200, resp.text
    run_id = resp.json()["run_ids"][0]

    deadline = time.time() + 30
    while time.time() < deadline:
        run = client.get(f"/api/v1/runs/{run_id}")
        assert run.status_code == 200, run.text
        if run.json()["status"] in ("completed", "failed"):
            break
        time.sleep(0.5)

    assert run.json()["status"] == "completed", run.json().get("error")
    assert isinstance(run.json()["result"]["best_code"], str)
