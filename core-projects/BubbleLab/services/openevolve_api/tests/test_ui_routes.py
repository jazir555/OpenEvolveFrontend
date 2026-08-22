"""
Tests for the new UI-facing route groups (parameters, monitoring, validation,
analytics, statistics). These assert each route returns 200 and the expected
top-level keys, proving the BubbleLab client stops 404ing on them.

Run with:
    python -m pytest tests/test_ui_routes.py -q -p no:pytest_ethereum
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
    str(Path(tempfile.gettempdir()) / "openevolve_api_ui_routes_workflows.db"),
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


# ----------------------------- Parameters ----------------------------- #
def test_parameters_schema(client):
    resp = client.get("/api/parameters/schema")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "parameters" in body
    assert isinstance(body["parameters"], list)
    assert any(p["name"] == "max_iterations" for p in body["parameters"])
    assert all("type" in p and "default" in p and "category" in p for p in body["parameters"])


def test_parameters_defaults(client):
    resp = client.get("/api/parameters/defaults")
    assert resp.status_code == 200, resp.text
    assert "population_size" in resp.json()


def test_parameters_categories(client):
    resp = client.get("/api/parameters/categories")
    assert resp.status_code == 200, resp.text
    assert "categories" in resp.json()


def test_parameters_validate(client):
    resp = client.post("/api/parameters/validate", json={"temperature": 0.7})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["valid"] is True
    assert "errors" in body and "warnings" in body

    resp = client.post("/api/parameters/validate", json={"temperature": 99.0})
    assert resp.status_code == 200, resp.text
    assert resp.json()["valid"] is False


# ----------------------------- Monitoring ----------------------------- #
def test_monitoring_dashboard(client):
    resp = client.get("/api/monitoring/dashboard")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    for key in ("timestamp", "system", "health", "workflow", "recent_metrics"):
        assert key in body
    assert "uptime_seconds" in body["health"]


def test_monitoring_alerts(client):
    resp = client.get("/api/monitoring/alerts")
    assert resp.status_code == 200, resp.text
    assert "alerts" in resp.json()


def test_monitoring_services(client):
    resp = client.get("/api/monitoring/services")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "services" in body
    assert any(s["name"] == "openevolve-api" for s in body["services"])


def test_monitoring_logs(client):
    resp = client.get("/api/monitoring/logs")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "entries" in body and "total" in body


def test_monitoring_metrics(client):
    resp = client.get("/api/monitoring/metrics")
    assert resp.status_code == 200, resp.text
    assert "metrics" in resp.json()


def test_monitoring_health(client):
    resp = client.get("/api/monitoring/health")
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "healthy"


# ----------------------------- Validation ----------------------------- #
def test_validation_rules(client):
    resp = client.get("/api/validation/rules")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "rules" in body and "rule_names" in body


def test_validation_run(client):
    resp = client.post(
        "/api/validation/run",
        json={"content": "def add(a, b):\n    return a + b\n", "rule_names": ["min_documentation"]},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "overall_result" in body
    assert "validations" in body


def test_validation_compliance(client):
    resp = client.post(
        "/api/validation/compliance",
        json={"content": "def solve():\n    return 1\n", "framework": "default"},
    )
    assert resp.status_code == 200, resp.text
    assert "valid" in resp.json()


# ----------------------------- Analytics ----------------------------- #
def test_analytics_performance_metrics(client):
    resp = client.get("/api/analytics/performance-metrics")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "metrics" in body and "total" in body


def test_analytics_knowledge_stats(client):
    resp = client.get("/api/analytics/knowledge-stats")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "total_artifacts" in body


def test_analytics_workflow_metrics(client):
    resp = client.get("/api/analytics/workflow-metrics")
    assert resp.status_code == 200, resp.text
    assert "metrics" in resp.json()


def test_statistics(client):
    resp = client.get("/api/statistics")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    for key in ("total_workflows", "completed", "failed", "running", "total_teams", "total_gauntlets"):
        assert key in body


# ----------------------------- CrewAI ----------------------------- #
def test_crewai_workflows(client):
    resp = client.get("/api/crewai/workflows")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "workflows" in body and "total" in body
    assert isinstance(body["workflows"], list)
    if body["workflows"]:
        wf = body["workflows"][0]
        for key in ("workflow_id", "status", "execution_method"):
            assert key in wf


def test_crewai_workflow_detail(client):
    resp = client.get("/api/crewai/workflows/crewai-problem-decomp")
    assert resp.status_code == 200, resp.text
    assert resp.json()["workflow_id"] == "crewai-problem-decomp"


def test_crewai_workflow_tickets(client):
    resp = client.get("/api/crewai/workflows/crewai-problem-decomp/tickets")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "tickets" in body and "total" in body


# ----------------------------- Version Control ----------------------------- #
def test_version_control_versions(client):
    resp = client.get("/api/version-control/versions")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "versions" in body
    assert isinstance(body["versions"], list)
    assert any("id" in v and "protocol_text" in v for v in body["versions"])


def test_version_control_current(client):
    resp = client.get("/api/version-control/current")
    assert resp.status_code == 200, resp.text
    assert "current" in resp.json()


def test_version_control_create_compare_branch_delete(client):
    create = client.post(
        "/api/version-control/versions",
        json={"protocol_text": "branch-protocol-v2", "version_name": "v-ui-test"},
    )
    assert create.status_code == 200, create.text
    assert create.json()["version_id"] == "v-ui-test"

    versions = client.get("/api/version-control/versions").json()
    assert any(v["id"] == "v-ui-test" for v in versions["versions"])

    branch = client.post(
        "/api/version-control/versions/v-ui-test/branch",
        json={"new_version_name": "v-ui-test-branch"},
    )
    assert branch.status_code == 200, branch.text

    compare = client.post(
        "/api/version-control/compare",
        json={"version_id_1": "v1-initial", "version_id_2": "v-ui-test-branch"},
    )
    assert compare.status_code == 200, compare.text
    assert "chars_added" in compare.json()

    delete = client.delete("/api/version-control/versions/v-ui-test-branch")
    assert delete.status_code == 200, delete.text
    assert delete.json()["deleted"] is True


# ----------------------------- Evaluators ----------------------------- #
def test_evaluators_list(client):
    resp = client.get("/api/evaluators")
    assert resp.status_code == 200, resp.text
    assert "evaluators" in resp.json()


def test_evaluators_upload_delete(client):
    upload = client.post("/api/evaluators", json={"code": "def evaluate(x): return 1.0"})
    assert upload.status_code == 200, upload.text
    evaluator_id = upload.json()["evaluator_id"]
    assert evaluator_id

    delete = client.delete(f"/api/evaluators/{evaluator_id}")
    assert delete.status_code == 200, delete.text
    assert delete.json()["success"] is True


# ----------------------------- Integrated ----------------------------- #
def test_integrated_run(client):
    resp = client.post(
        "/api/integrated/run",
        json={"content_type": "protocol", "red_team_models": ["a"], "blue_team_models": ["b"]},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    for key in ("status", "timestamp", "monitoring", "parameters", "crewai"):
        assert key in body


# ----------------------------- BubbleLabs LeanAide (proxy) ----------------------------- #
def test_leanaide_health(client):
    resp = client.get("/api/bubblelabs/leanaide/health")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "leanaide_available" in body
    assert "server" in body
