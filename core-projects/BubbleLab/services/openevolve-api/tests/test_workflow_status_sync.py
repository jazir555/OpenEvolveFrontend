"""
Tests for the background engine-status sync (api/workflow_status_sync.py).

Hermetic: the real engine is never contacted. The sync module's fetch helpers
are monkeypatched to return canned engine payloads (or raise), and the
workflows module's DB writer is replaced with a Mock so we can assert on saves.

Run with:
    python -m pytest tests/test_workflow_status_sync.py -q -p no:pytest_ethereum
"""

import os
import sys
import types
from pathlib import Path

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault(
    "WORKFLOW_DB_PATH",
    str(Path(__file__).resolve().parent / "workflow_status_sync_test.db"),
)
os.environ.setdefault("OPENEVOLVE_API_KEY", "test-admin-key")
os.environ.setdefault("OPENEVOLVE_BRIDGE_ENABLED", "1")

if "openevolve_api" not in sys.modules:
    pkg = types.ModuleType("openevolve_api")
    pkg.__path__ = [str(SERVICE_ROOT)]
    sys.modules["openevolve_api"] = pkg

import httpx  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402
from openevolve_api.main import app  # noqa: E402
from openevolve_api.api import workflows as workflows_module  # noqa: E402
from openevolve_api.api import workflow_status_sync  # noqa: E402
from openevolve_api.models import WorkflowStatus  # noqa: E402


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def make_running_workflow(client):
    """Create a :8000 workflow linked to a fake engine id, set to RUNNING."""

    def _make(engine_id="eng-1", status=WorkflowStatus.RUNNING):
        resp = client.post(
            "/api/workflows",
            json={
                "name": "status-sync-test",
                "description": "x",
                "workflow_type": "sovereign",
                "problem_statement": "Solve the thing.",
            },
        )
        assert resp.status_code == 201, resp.text
        wid = resp.json()["id"]
        wf = workflows_module._workflows[wid]
        wf.parameters = {**wf.parameters, "last_engine_workflow_id": engine_id}
        wf.status = status
        return wid

    return _make


def test_sync_advances_to_completed(monkeypatch, client, make_running_workflow):
    wid = make_running_workflow("eng-1", WorkflowStatus.RUNNING)

    monkeypatch.setattr(
        workflow_status_sync, "_fetch_engine_status",
        lambda engine_id, api_key: {"status": "completed"},
    )
    saved = []
    monkeypatch.setattr(
        workflows_module, "_save_workflow_to_db",
        lambda wf: saved.append(wf),
    )

    updated = workflow_status_sync.sync_engine_statuses()

    assert updated == 1
    assert workflows_module._workflows[wid].status == WorkflowStatus.COMPLETED
    assert saved, "expected _save_workflow_to_db to be called"
    assert saved[0].id == wid


def test_sync_leaves_running_unchanged(monkeypatch, client, make_running_workflow):
    wid = make_running_workflow("eng-1", WorkflowStatus.RUNNING)

    monkeypatch.setattr(
        workflow_status_sync, "_fetch_engine_status",
        lambda engine_id, api_key: {"status": "running"},
    )
    saved = []
    monkeypatch.setattr(
        workflows_module, "_save_workflow_to_db",
        lambda wf: saved.append(wf),
    )

    updated = workflow_status_sync.sync_engine_statuses()

    assert updated == 0
    assert workflows_module._workflows[wid].status == WorkflowStatus.RUNNING
    assert saved == []


def test_sync_handles_404_and_network_errors(monkeypatch, client, make_running_workflow):
    wid = make_running_workflow("eng-1", WorkflowStatus.RUNNING)

    def _boom(engine_id, api_key):
        raise httpx.HTTPStatusError(
            "nf", request=None, response=httpx.Response(404)
        )

    monkeypatch.setattr(workflow_status_sync, "_fetch_engine_status", _boom)
    saved = []
    monkeypatch.setattr(
        workflows_module, "_save_workflow_to_db",
        lambda wf: saved.append(wf),
    )

    # Must not raise.
    updated = workflow_status_sync.sync_engine_statuses()

    assert updated == 0
    assert workflows_module._workflows[wid].status == WorkflowStatus.RUNNING
    assert saved == []


def test_sync_skipped_without_api_key(monkeypatch, client, make_running_workflow):
    wid = make_running_workflow("eng-1", WorkflowStatus.RUNNING)
    monkeypatch.setattr(workflow_status_sync, "_engine_api_key", lambda: None)
    saved = []
    monkeypatch.setattr(
        workflows_module, "_save_workflow_to_db",
        lambda wf: saved.append(wf),
    )

    assert workflow_status_sync.sync_engine_statuses() == 0
    assert saved == []
