"""
Contract test for the PES Enhanced route group.

Asserts that ``/api/pes-enhanced/*`` is mounted by the OpenEvolve FastAPI
service and reachable (returns a non-404: either 200/2xx when the PES module
and its heavy deps are present, or 501/503 when they are not). This guards
against the group being silently unreachable (404) through the backend or the
Hono proxy (which forwards ``/*`` to this service verbatim).

Run with:
    python -m pytest tests/test_pes_routes.py -q -p no:pytest_ethereum
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
    str(Path(tempfile.gettempdir()) / "openevolve_api_pes_routes_workflows.db"),
)

if "openevolve_api" not in sys.modules:
    package = types.ModuleType("openevolve_api")
    package.__path__ = [str(SERVICE_ROOT)]  # type: ignore[attr-defined]
    sys.modules["openevolve_api"] = package

from fastapi.testclient import TestClient  # noqa: E402

from openevolve_api.main import app  # noqa: E402


def _mounted_pes_paths():
    paths = set()
    for route in app.routes:
        path = getattr(route, "path", None)
        if path and path.startswith("/api/pes-enhanced"):
            paths.add(path)
    return paths


def test_pes_group_is_mounted():
    mounted = _mounted_pes_paths()
    assert mounted, "no /api/pes-enhanced route is mounted"


def test_pes_health_reachable_not_404():
    client = TestClient(app)
    resp = client.get("/api/pes-enhanced/health")
    assert resp.status_code != 404, (
        "/api/pes-enhanced/health returned 404: PES group not reachable"
    )


def test_pes_runs_endpoint_reachable_not_404():
    client = TestClient(app)
    resp = client.get("/api/pes-enhanced/runs")
    assert resp.status_code != 404, (
        "/api/pes-enhanced/runs returned 404: PES group not reachable"
    )


def test_pes_post_endpoint_reachable_not_404():
    client = TestClient(app)
    resp = client.post(
        "/api/pes-enhanced/cost-estimate",
        json={"iterations": 10, "population_size": 20, "avg_tokens_per_eval": 500},
    )
    assert resp.status_code != 404, (
        "/api/pes-enhanced/cost-estimate returned 404: PES group not reachable"
    )
    assert resp.status_code in (200, 201, 202, 501, 503), (
        f"unexpected PES status code: {resp.status_code}"
    )
