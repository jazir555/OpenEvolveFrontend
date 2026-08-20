"""
End-to-end test: this primary service speaks the OpenEvolve ``/api/v1/*`` dialect
and runs a REAL evolution over HTTP (via the openevolve bridge, offline mock LLM).

This proves the BubbleLab integration bubbles can target THIS service rather
than the separate stdlib server.

Run with:
    python -m pytest tests/test_openevolve_v1_e2e.py -q
"""

import os
import sys
import tempfile
import time
import types
from pathlib import Path

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]

# Isolate the workflow DB before the service modules import (they open it eagerly).
os.environ.setdefault(
    "WORKFLOW_DB_PATH",
    str(Path(tempfile.gettempdir()) / "openevolve_api_v1_e2e_workflows.db"),
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

from openevolve_api.main import app  # noqa: E402


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_api_v1_health(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "healthy"
    assert "version" in body


def test_api_v1_orchestrate_runs_real_evolution_over_http(client):
    # a. Start a workflow-style evolution via the /api/v1 dialect.
    resp = client.post(
        "/api/v1/workflows/orchestrate",
        json={
            "system": "evolutionary",
            "problemStatement": "evolve a function that adds two numbers",
            "generations": 2,
            "populationSize": 4,
        },
    )
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["status"] == "running"
    workflow_id = body["workflowId"]
    assert isinstance(workflow_id, str) and workflow_id

    # b. Poll until the REAL engine completes (timeout ~30s).
    run = None
    deadline = time.time() + 30
    while time.time() < deadline:
        run = client.get(f"/api/v1/runs/{workflow_id}")
        assert run.status_code == 200, run.text
        if run.json()["status"] in ("completed", "failed"):
            break
        time.sleep(0.5)

    assert run is not None
    run_body = run.json()
    assert run_body["status"] == "completed", run_body.get("error")
    assert run_body["run_id"] == workflow_id

    # c. Prove the real openevolve engine produced a non-empty program.
    result = run_body["result"]
    assert result is not None
    assert isinstance(result["best_code"], str)
    assert result["best_code"].strip(), "best_code must be non-empty"
    assert result["engine"] == "openevolve"
    assert result["llm_mode"] == "mock"


def test_api_v1_unknown_run_is_404(client):
    resp = client.get("/api/v1/runs/does-not-exist")
    assert resp.status_code == 404
