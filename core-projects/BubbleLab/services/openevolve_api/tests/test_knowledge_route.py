"""
Tests for the ``/api/knowledge`` Knowledge Engine router.

Asserts the core knowledge routes return 200 with the expected top-level keys,
proving the BubbleLab client no longer 404s on them and that the endpoints
degrade gracefully (structured-empty) when no vector/text backend is
configured.

Run with:
    python -m pytest tests/test_knowledge_route.py -q -p no:pytest_ethereum
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
    str(Path(tempfile.gettempdir()) / "openevolve_api_knowledge_routes.db"),
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


# ----------------------------- Knowledge ----------------------------- #
def test_knowledge_documents(client):
    resp = client.get("/api/knowledge/documents")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "documents" in body
    assert isinstance(body["documents"], list)
    assert "backend" in body
    assert "total" in body


def test_knowledge_artifacts(client):
    resp = client.get("/api/knowledge/artifacts")
    assert resp.status_code == 200, resp.text
    assert "artifacts" in resp.json()


def test_knowledge_search(client):
    resp = client.post("/api/knowledge/search", json={"query": "optimization", "limit": 5})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "results" in body
    assert isinstance(body["results"], list)
    assert "backend" in body
    assert "query" in body
    assert body["query"] == "optimization"


def test_knowledge_stats(client):
    resp = client.get("/api/knowledge/stats")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    for key in ("total_artifacts", "total_usage", "average_effectiveness", "by_type"):
        assert key in body
    assert isinstance(body["by_type"], dict)


def test_knowledge_graph(client):
    resp = client.get("/api/knowledge/graph")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "nodes" in body and "edges" in body


def test_knowledge_recommendations(client):
    resp = client.post("/api/knowledge/recommendations", json={"query": "x"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    for key in (
        "recommended_approaches",
        "similar_problems",
        "team_recommendations",
        "gauntlet_recommendations",
    ):
        assert key in body


def test_knowledge_embed(client):
    resp = client.post("/api/knowledge/embed", json={"texts": ["hello world"]})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "dimension" in body
    assert "model" in body


def test_knowledge_sync(client):
    resp = client.post("/api/knowledge/sync", json={"source": "none"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "ok"
    assert "synced" in body


def test_knowledge_export_import(client):
    resp = client.get("/api/knowledge/export")
    assert resp.status_code == 200, resp.text
    assert "artifacts" in resp.json()

    resp = client.post("/api/knowledge/import", json={"artifacts": []})
    assert resp.status_code == 200, resp.text
    assert resp.json()["success"] is True
