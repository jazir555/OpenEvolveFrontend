"""
Additive contract test for the OpenEvolve team surface.

Covers the full team lifecycle via ``fastapi.testclient.TestClient`` with no live
server and no network access:

  - list teams     GET    /api/teams
  - create         POST   /api/teams
  - get by id      GET    /api/teams/{id}
  - update by id   PUT    /api/teams/{id}
  - delete by id   DELETE /api/teams/{id}

Every endpoint must return a 2xx and the documented JSON keys. A 500 is treated
as a real crash bug and fails the contract.

NOTE: the backend keys teams/gauntlets by their generated `id` (a `team_*` UUID),
NOT by name. The frontend client and this test therefore use the `id` returned
from the create response for all update/delete/get-by-id calls.

Run with:
    python -m pytest tests/test_teams.py -q -p no:pytest_ethereum
"""

import os
import sys
import types
import tempfile
from pathlib import Path

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]

# Inject the package stub so the service's relative imports resolve when the
# test file is collected (mirrors tests/conftest.py + test_route_contract.py).
if "openevolve_api" not in sys.modules:
    package = types.ModuleType("openevolve_api")
    package.__path__ = [str(SERVICE_ROOT)]  # type: ignore[attr-defined]
    sys.modules["openevolve_api"] = package

import openevolve_api.database as _db  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from openevolve_api.main import app  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_db():
    """Use a throwaway SQLite file so contract runs never touch the real DB."""
    original = _db.DB_PATH
    tmp = Path(tempfile.gettempdir()) / f"openevolve_api_teams_{os.getpid()}.db"
    _db.DB_PATH = tmp
    _db.init_db()
    try:
        yield
    finally:
        _db.DB_PATH = original


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


SAMPLE_TEAM = {
    "name": "contract-team",
    "description": "additive contract test team",
    "members": [
        {
            "name": "solver-1",
            "role": "solver",
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 4096,
        },
    ],
}


def _assert_not_500(resp, label):
    assert resp.status_code != 500, f"{label} returned 500: {resp.text}"


def test_team_lifecycle(client):
    # ---- list (baseline) ----
    r = client.get("/api/teams")
    _assert_not_500(r, "list")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "teams" in body and "total" in body
    assert isinstance(body["teams"], list)

    # ---- create ----
    r = client.post("/api/teams", json=SAMPLE_TEAM)
    _assert_not_500(r, "create")
    assert r.status_code == 201, r.text
    created = r.json()
    assert "id" in created and "name" in created
    assert created["name"] == SAMPLE_TEAM["name"]
    tid = created["id"]

    # ---- get by id ----
    r = client.get(f"/api/teams/{tid}")
    _assert_not_500(r, "get-by-id")
    assert r.status_code == 200, r.text
    got = r.json()
    assert got["id"] == tid
    assert got["name"] == SAMPLE_TEAM["name"]

    # ---- update by id ----
    r = client.put(
        f"/api/teams/{tid}",
        json={"description": "updated contract team", "members": SAMPLE_TEAM["members"]},
    )
    _assert_not_500(r, "update")
    assert r.status_code == 200, r.text
    updated = r.json()
    assert updated["id"] == tid
    assert updated["description"] == "updated contract team"

    # ---- get by id reflects the update ----
    r = client.get(f"/api/teams/{tid}")
    _assert_not_500(r, "get-by-id-after-update")
    assert r.status_code == 200, r.text
    assert r.json()["description"] == "updated contract team"

    # ---- delete by id ----
    r = client.delete(f"/api/teams/{tid}")
    _assert_not_500(r, "delete")
    assert r.status_code == 200, r.text
    assert "message" in r.json()

    # ---- get after delete proves deletion (404, not 500) ----
    r = client.get(f"/api/teams/{tid}")
    _assert_not_500(r, "get-after-delete")
    assert r.status_code == 404, r.text
