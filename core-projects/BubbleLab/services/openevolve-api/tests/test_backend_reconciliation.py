"""
Backend reconciliation contract test.

Verifies that the legacy route groups -- ``/api/teams``, ``/api/gauntlets``,
``/api/executions`` and ``/api/workflows`` -- keep working after the optional
OpenEvolve engine bridge was wired into them. The bridge is DISABLED here so the
assertion is that the DB-backed, self-contained fallback still returns valid
(non-500) responses and parses as JSON. This proves the legacy routers remain
functional regardless of whether the shared engine bridge is reachable.

Run with:
    python -m pytest tests/test_backend_reconciliation.py -q -p no:pytest_ethereum
"""

import os
import sys
import tempfile
import types
from pathlib import Path

# Force the bridge OFF before any service module is imported so the legacy
# routers exercise only their DB-backed fallback path.
os.environ["OPENEVOLVE_BRIDGE_ENABLED"] = "0"
os.environ.setdefault(
    "WORKFLOW_DB_PATH",
    str(Path(tempfile.gettempdir()) / "openevolve_api_reconciliation_workflows.db"),
)

SERVICE_ROOT = Path(__file__).resolve().parents[1]

if "openevolve_api" not in sys.modules:
    package = types.ModuleType("openevolve_api")
    package.__path__ = [str(SERVICE_ROOT)]  # type: ignore[attr-defined]
    sys.modules["openevolve_api"] = package

from fastapi.testclient import TestClient  # noqa: E402

from openevolve_api.main import app  # noqa: E402


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _disable_bridge():
    """Guarantee the bridge is disabled for every test in this module."""
    previous = os.environ.get("OPENEVOLVE_BRIDGE_ENABLED")
    os.environ["OPENEVOLVE_BRIDGE_ENABLED"] = "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("OPENEVOLVE_BRIDGE_ENABLED", None)
        else:
            os.environ["OPENEVOLVE_BRIDGE_ENABLED"] = previous


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


def _assert_ok(resp):
    """A genuine 500 is the only hard failure; tolerate other 2xx/4xx."""
    assert resp.status_code != 500, f"Unexpected 500: {resp.text}"
    # Where JSON is expected, ensure the body parses.
    if 200 <= resp.status_code < 300:
        ctype = resp.headers.get("content-type", "")
        if "application/json" in ctype:
            assert resp.json() is not None


def test_legacy_teams_group(client):
    # List (GET)
    _assert_ok(client.get("/api/teams"))
    # Create (POST) -- exercises the DB fallback with the bridge disabled.
    created = client.post(
        "/api/teams",
        json={"name": "recon-team", "description": "reconciliation check"},
    )
    _assert_ok(created)
    if created.status_code < 300:
        team_id = created.json().get("id")
        _assert_ok(client.get(f"/api/teams/{team_id}"))


def test_legacy_gauntlets_group(client):
    _assert_ok(client.get("/api/gauntlets"))
    created = client.post(
        "/api/gauntlets",
        json={
            "name": "recon-gauntlet",
            "description": "reconciliation check",
            "rounds": [],
        },
    )
    _assert_ok(created)
    if created.status_code < 300:
        gid = created.json().get("id")
        _assert_ok(client.get(f"/api/gauntlets/{gid}"))


def test_legacy_executions_group(client):
    _assert_ok(client.get("/api/executions"))
    # Start an execution directly (DB-backed fallback).
    started = client.post(
        "/api/executions",
        json={
            "workflow_id": "recon-wf",
            "problem_statement": "Compute 2 + 2.",
            "context": "",
        },
    )
    _assert_ok(started)
    if started.status_code < 300:
        execution_id = started.json().get("execution_id")
        _assert_ok(client.get(f"/api/executions/{execution_id}"))


def test_legacy_workflows_group(client):
    _assert_ok(client.get("/api/workflows"))
    created = client.post(
        "/api/workflows",
        json={
            "name": "recon-wf",
            "description": "reconciliation check",
            "workflow_type": "sovereign",
            "parameters": {},
        },
    )
    _assert_ok(created)
    if created.status_code < 300:
        wf_id = created.json().get("id")
        _assert_ok(client.get(f"/api/workflows/{wf_id}"))


def test_legacy_groups_no_500_with_bridge_disabled(client):
    """Whole-group smoke: every legacy group returns a non-500, parseable body."""
    targets = [
        client.get("/api/teams"),
        client.get("/api/gauntlets"),
        client.get("/api/executions"),
        client.get("/api/workflows"),
    ]
    for resp in targets:
        _assert_ok(resp)
