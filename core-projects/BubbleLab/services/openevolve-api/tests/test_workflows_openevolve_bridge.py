"""
Tests that the evolutionary workflow path is driven by the REAL openevolve library.

These tests run fully OFFLINE: the bridge configures the library with a
deterministic mock LLM (``name="mock", provider="mock"``), so no API keys or
network access are required.

Run with:
    python -m pytest tests/ -k workflow -q
"""

import os
import sys
import tempfile
import types
from pathlib import Path

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]

# Isolate the workflow DB before the service modules import (they open it eagerly).
os.environ.setdefault(
    "WORKFLOW_DB_PATH",
    str(Path(tempfile.gettempdir()) / "openevolve_api_bridge_test_workflows.db"),
)
os.environ.setdefault("OPENEVOLVE_BRIDGE_ENABLED", "1")

if "openevolve_api" not in sys.modules:
    package = types.ModuleType("openevolve_api")
    package.__path__ = [str(SERVICE_ROOT)]  # type: ignore[attr-defined]
    sys.modules["openevolve_api"] = package

openevolve = pytest.importorskip(
    "openevolve", reason="the real openevolve library must be installed (pip install -e ../../../openevolve)"
)

from fastapi.testclient import TestClient  # noqa: E402

from openevolve_api.api import workflows as workflows_api  # noqa: E402
from openevolve_api.core.openevolve_bridge import (  # noqa: E402
    OpenEvolveBridgeError,
    run_openevolve_workflow,
)
from openevolve_api.main import app  # noqa: E402


# Keep offline runs tiny so tests stay fast.
FAST_PARAMS = {"max_iterations": 2, "population_size": 4, "seed": 42}

MINIMAL_EVOLUTIONARY_WORKFLOW = {
    "name": "Bridge Evolution Workflow",
    "description": "Minimal evolutionary workflow driving the real engine",
    "problem_statement": "Evolve a function so that solve(21) returns 42.",
    "content_type": "code",
    "teams": [],
    "gauntlets": [],
    "workflow_type": "evolution",
    "parameters": FAST_PARAMS,
}


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


def _assert_real_evolution_result(result: dict) -> None:
    """Assert the payload came from openevolve.api.run_evolution, not a stub."""
    assert result, "no openevolve result attached"

    # Bridge contract.
    assert "best_score" in result
    assert isinstance(result["best_score"], (int, float))
    assert isinstance(result["best_code"], str)
    assert result["best_code"].strip(), "best_code must be non-empty"
    assert isinstance(result["metrics"], dict)
    assert result["metrics"], "metrics must be populated by the evaluator"
    assert isinstance(result["generations"], int)

    # Provenance: the real library entry point produced this.
    assert result["engine"] == "openevolve"
    assert result["engine_entrypoint"] == "openevolve.api.run_evolution"

    # The deterministic offline backend was actually exercised inside the
    # library, which is only possible if the real engine ran.
    assert result["llm_mode"] == "mock"
    assert result["mock_llm_calls"] is not None
    assert result["mock_llm_calls"] >= 1, "the library's MockLLM was never called"

    # The evaluator we handed the library ran against the candidate program.
    assert "combined_score" in result["metrics"]


# ==================== Bridge-level tests ====================

def test_openevolve_bridge_workflow_runs_offline():
    """The bridge drives the real engine with no API keys or network."""
    result = run_openevolve_workflow({"system": "evolutionary"})
    _assert_real_evolution_result(result)


def test_openevolve_bridge_workflow_honors_parameters():
    """Request parameters reach the library config."""
    result = run_openevolve_workflow(
        {
            "system": "evolutionary",
            "problem_statement": "Make solve(21) equal 42.",
            "parameters": FAST_PARAMS,
        }
    )
    _assert_real_evolution_result(result)
    assert result["iterations"] == FAST_PARAMS["max_iterations"]
    assert result["population_size"] == FAST_PARAMS["population_size"]


def test_openevolve_bridge_workflow_rejects_bad_evaluator():
    """Failures raise instead of silently returning fake data."""
    with pytest.raises(OpenEvolveBridgeError):
        run_openevolve_workflow(
            {"system": "evolutionary", "evaluator": "def not_an_evaluator(): pass"}
        )


def test_openevolve_bridge_workflow_accepts_custom_program():
    """A caller-supplied EVOLVE-BLOCK program is used verbatim as the seed."""
    program = (
        "# EVOLVE-BLOCK-START\n"
        "TARGET = 42\n"
        "def solve(x):\n"
        "    return x + 21\n"
        "# EVOLVE-BLOCK-END\n"
        "\n"
        "def run():\n"
        "    return solve(21)\n"
    )
    result = run_openevolve_workflow(
        {"system": "evolutionary", "initial_program": program, "parameters": FAST_PARAMS}
    )
    _assert_real_evolution_result(result)
    assert result["metrics"].get("value") is not None


# ==================== API-level tests ====================

def test_evolutionary_workflow_start_returns_real_engine_result(client):
    """POST a minimal evolutionary workflow and start it -> real evolution result."""
    create = client.post("/api/workflows", json=MINIMAL_EVOLUTIONARY_WORKFLOW)
    assert create.status_code == 201, create.text
    workflow_id = create.json()["id"]

    start = client.post(
        f"/api/workflows/{workflow_id}/start",
        json={"problem_statement": "Evolve a function so that solve(21) returns 42."},
    )
    assert start.status_code == 200, start.text

    body = start.json()
    # Existing response shape is preserved.
    assert body["id"] == workflow_id
    assert body["workflow_type"] == "evolution"
    assert body["status"] == "running"

    _assert_real_evolution_result(body["parameters"]["openevolve"])


def test_evolutionary_workflow_results_include_real_engine_result(client):
    """GET /{id}/results surfaces the evolved program from the real engine."""
    create = client.post("/api/workflows", json=MINIMAL_EVOLUTIONARY_WORKFLOW)
    assert create.status_code == 201, create.text
    workflow_id = create.json()["id"]

    start = client.post(f"/api/workflows/{workflow_id}/start", json={})
    assert start.status_code == 200, start.text

    results = client.get(f"/api/workflows/{workflow_id}/results")
    assert results.status_code == 200, results.text

    payload = results.json()
    assert payload["workflow_id"] == workflow_id

    import json

    final_solution = json.loads(payload["final_solution"])
    _assert_real_evolution_result(final_solution["openevolve"])


def test_non_evolution_workflow_does_not_invoke_bridge(client, monkeypatch):
    """Sovereign/adversarial workflows keep their existing behavior untouched."""
    called = {"count": 0}

    def _boom(_request):
        called["count"] += 1
        raise AssertionError("bridge must not run for non-evolution workflows")

    monkeypatch.setattr(workflows_api, "run_openevolve_workflow", _boom)

    create = client.post(
        "/api/workflows",
        json={
            **MINIMAL_EVOLUTIONARY_WORKFLOW,
            "name": "Sovereign Workflow",
            "workflow_type": "sovereign",
        },
    )
    assert create.status_code == 201, create.text
    workflow_id = create.json()["id"]

    start = client.post(f"/api/workflows/{workflow_id}/start", json={})
    assert start.status_code == 200, start.text
    assert called["count"] == 0
    assert "openevolve" not in (start.json().get("parameters") or {})


def test_workflow_bridge_is_optional(client, monkeypatch):
    """With the bridge unavailable the start path still succeeds (legacy fallback)."""
    monkeypatch.setattr(workflows_api, "OPENEVOLVE_BRIDGE_AVAILABLE", False)

    create = client.post("/api/workflows", json=MINIMAL_EVOLUTIONARY_WORKFLOW)
    assert create.status_code == 201, create.text
    workflow_id = create.json()["id"]

    start = client.post(f"/api/workflows/{workflow_id}/start", json={})
    assert start.status_code == 200, start.text
    assert "openevolve" not in (start.json().get("parameters") or {})
