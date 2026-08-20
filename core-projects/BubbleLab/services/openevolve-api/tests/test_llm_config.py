"""
Tests that the ``/api/v1/workflows/orchestrate`` endpoint accepts an optional
``llm`` field and forwards it to the openevolve bridge.

* A request WITHOUT an ``llm`` field (or without a real key) still completes in
  the offline ``mock`` mode (default).
* A request WITH an ``llm`` object is accepted (202 + workflowId) and the
  ``llm`` config is passed through to the bridge (proven by a stubbed bridge so
  no real network/key is required in tests).

Run with:
    python -m pytest tests/test_llm_config.py -q -p no:pytest_ethereum
"""

import os
import sys
import tempfile
import time
import types
from pathlib import Path

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault(
    "WORKFLOW_DB_PATH",
    str(Path(tempfile.gettempdir()) / "openevolve_api_llm_config_workflows.db"),
)
os.environ.setdefault("OPENEVOLVE_BRIDGE_ENABLED", "1")

if "openevolve_api" not in sys.modules:
    package = types.ModuleType("openevolve_api")
    package.__path__ = [str(SERVICE_ROOT)]  # type: ignore[attr-defined]
    sys.modules["openevolve_api"] = package

pytest.importorskip(
    "openevolve", reason="the real openevolve library must be installed"
)

from fastapi.testclient import TestClient  # noqa: E402

import openevolve_api.api.openevolve_v1 as v1  # noqa: E402
from openevolve_api.api.openevolve_v1 import (  # noqa: E402
    _orchestrate_request_to_bridge,
)
from openevolve_api.main import app  # noqa: E402


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


def _stub_bridge(monkeypatch, captured):
    """Replace the real bridge with a stub that records the request it got."""

    def fake_run(request):
        captured["request"] = dict(request)
        return {
            "best_code": "x = 1",
            "best_score": 1.0,
            "metrics": {},
            "generations": 1,
            "engine": "openevolve",
            "engine_entrypoint": "openevolve.api.run_evolution",
            "llm_mode": "live" if request.get("llm") else "mock",
            "iterations": 1,
            "population_size": 1,
            "mock_llm_calls": 0,
            "duration_seconds": 0.01,
            "started_at": "",
            "completed_at": "",
        }

    monkeypatch.setattr(v1, "run_openevolve_workflow", fake_run)


def test_orchestrate_builder_forwards_llm_config():
    body = {
        "system": "evolutionary",
        "problemStatement": "evolve x",
        "generations": 2,
        "populationSize": 4,
        "llm": {
            "name": "gpt-4o",
            "api_key": "sk-test-123",
            "api_base": "https://api.example.com/v1",
            "provider": "openai",
        },
    }
    bridge = _orchestrate_request_to_bridge(body)
    assert bridge["llm"] == body["llm"]
    assert bridge["problem_statement"] == "evolve x"
    assert bridge["parameters"]["max_iterations"] == 2
    assert bridge["parameters"]["population_size"] == 4


def test_orchestrate_builder_defaults_llm_to_empty_dict():
    bridge = _orchestrate_request_to_bridge({"problemStatement": "evolve x"})
    assert bridge["llm"] == {}


def test_orchestrate_accepts_llm_body_and_reaches_bridge(client, monkeypatch):
    captured: dict = {}
    _stub_bridge(monkeypatch, captured)

    llm = {
        "name": "gpt-4o",
        "api_key": "sk-fake-not-real",
        "api_base": "https://api.openai.com/v1",
        "provider": "openai",
    }
    resp = client.post(
        "/api/v1/workflows/orchestrate",
        json={
            "system": "evolutionary",
            "problemStatement": "evolve a function that multiplies by two",
            "generations": 2,
            "populationSize": 4,
            "llm": llm,
        },
    )
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["status"] == "running"
    assert isinstance(body["workflowId"], str) and body["workflowId"]

    # Wait for the (stubbed) run to finish so we can inspect what was forwarded.
    run = None
    deadline = time.time() + 15
    while time.time() < deadline:
        run = client.get(f"/api/v1/runs/{body['workflowId']}")
        assert run.status_code == 200, run.text
        if run.json()["status"] in ("completed", "failed"):
            break
        time.sleep(0.2)

    assert run is not None
    assert run.json()["status"] == "completed", run.json().get("error")
    assert captured.get("request", {}).get("llm") == llm


def test_orchestrate_no_llm_still_completes_in_mock_mode(client):
    resp = client.post(
        "/api/v1/workflows/orchestrate",
        json={
            "system": "evolutionary",
            "problemStatement": "evolve a trivial function",
            "generations": 2,
            "populationSize": 4,
        },
    )
    assert resp.status_code == 202, resp.text
    workflow_id = resp.json()["workflowId"]

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
    assert run_body["result"]["llm_mode"] == "mock"
