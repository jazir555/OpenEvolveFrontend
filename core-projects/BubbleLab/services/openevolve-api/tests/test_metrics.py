"""
Additive tests for the dependency-free request metrics collector.

Verifies that the MetricsMiddleware records requests and that the
/api/monitoring/metrics endpoint (in the existing monitoring router) exposes a
nonzero request count.

Run with:
    python -m pytest tests/test_metrics.py -q -p no:pytest_ethereum
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
    str(Path(tempfile.gettempdir()) / "openevolve_api_metrics_workflows.db"),
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


def test_metrics_endpoint_reports_request_count(client):
    # Hit a known endpoint so the middleware records a request.
    health = client.get("/health")
    assert health.status_code == 200, health.text

    # The metrics endpoint must surface the recorded request count.
    resp = client.get("/api/monitoring/metrics")
    assert resp.status_code == 200, resp.text

    body = resp.json()
    assert "metrics" in body  # existing contract preserved
    assert "requests" in body
    assert body["request_count"] > 0
    assert body["requests"]["total_requests"] > 0


def test_metrics_by_route_tracks_known_endpoint(client):
    client.get("/health")
    client.get("/api/monitoring/metrics")

    body = client.get("/api/monitoring/metrics").json()
    by_route = body["requests"]["by_route"]
    assert "GET /health" in by_route
    assert by_route["GET /health"]["count"] > 0
