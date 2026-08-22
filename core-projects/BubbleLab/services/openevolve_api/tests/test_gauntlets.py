"""
Additive contract test for the OpenEvolve gauntlet surface.

Covers the full gauntlet lifecycle via ``fastapi.testclient.TestClient`` with
no live server and no network access:

  - list gauntlets          GET  /api/gauntlets
  - create                  POST /api/gauntlets
  - get by id               GET  /api/gauntlets/{id}
  - execute                 POST /api/gauntlets/{name}/execute  -> 202
  - get execution status    GET  /api/gauntlets/executions/{id}/status
  - list executions         GET  /api/gauntlets/executions
  - delete                  DELETE /api/gauntlets/{id}

Every endpoint must return a 2xx and the documented JSON keys. A 500 is
treated as a real crash bug and fails the contract.

Run with:
    python -m pytest tests/test_gauntlets.py -q -p no:pytest_ethereum
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
    """Use a throwaway SQLite file so contract runs never touch the real DB.

    ``database.py`` reads the module-level ``DB_PATH`` inside ``get_db()``, so
    swapping it here (and restoring it after) keeps other test modules using the
    default location.
    """
    original = _db.DB_PATH
    tmp = Path(tempfile.gettempdir()) / f"openevolve_api_gauntlets_{os.getpid()}.db"
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


SAMPLE_GAUNTLET = {
    "name": "contract-gauntlet",
    "description": "additive contract test gauntlet",
    "rounds": [
        {
            "id": "r1",
            "name": "round-one",
            "description": "first round",
            "quorum_threshold": 0.5,
            "confidence_threshold": 0.5,
            "evaluation_type": "standard",
        },
    ],
}


def _assert_not_500(resp, label):
    assert resp.status_code != 500, f"{label} returned 500: {resp.text}"


def test_gauntlet_lifecycle(client):
    # ---- list (baseline) ----
    r = client.get("/api/gauntlets")
    _assert_not_500(r, "list")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "gauntlets" in body and "total" in body
    assert isinstance(body["gauntlets"], list)

    # ---- create ----
    r = client.post("/api/gauntlets", json=SAMPLE_GAUNTLET)
    _assert_not_500(r, "create")
    assert r.status_code == 201, r.text
    created = r.json()
    assert "id" in created and "name" in created
    assert created["name"] == SAMPLE_GAUNTLET["name"]
    gid = created["id"]

    # ---- get by id ----
    r = client.get(f"/api/gauntlets/{gid}")
    _assert_not_500(r, "get-by-id")
    assert r.status_code == 200, r.text
    got = r.json()
    assert got["id"] == gid
    assert got["name"] == SAMPLE_GAUNTLET["name"]

    # ---- update by id ----
    r = client.put(
        f"/api/gauntlets/{gid}",
        json={"description": "updated contract gauntlet", "rounds": SAMPLE_GAUNTLET["rounds"]},
    )
    _assert_not_500(r, "update")
    assert r.status_code == 200, r.text
    updated = r.json()
    assert updated["id"] == gid
    assert updated["description"] == "updated contract gauntlet"

    # ---- get by id reflects the update ----
    r = client.get(f"/api/gauntlets/{gid}")
    _assert_not_500(r, "get-by-id-after-update")
    assert r.status_code == 200, r.text
    assert r.json()["description"] == "updated contract gauntlet"

    # ---- execute (expect 202) ----
    r = client.post(
        f"/api/gauntlets/{SAMPLE_GAUNTLET['name']}/execute",
        json={"content": "print('hi')", "content_type": "text_general"},
    )
    _assert_not_500(r, "execute")
    assert r.status_code == 202, r.text
    ex_body = r.json()
    assert "execution_id" in ex_body and ex_body["execution_id"]
    exec_id = ex_body["execution_id"]

    # ---- get execution status ----
    r = client.get(f"/api/gauntlets/executions/{exec_id}/status")
    _assert_not_500(r, "status")
    assert r.status_code == 200, r.text
    st = r.json()
    assert st["execution_id"] == exec_id
    assert "status" in st
    assert st["gauntlet_name"] == SAMPLE_GAUNTLET["name"]

    # ---- list executions ----
    r = client.get("/api/gauntlets/executions")
    _assert_not_500(r, "list-executions")
    assert r.status_code == 200, r.text
    ex_list = r.json()
    assert "executions" in ex_list and "total" in ex_list
    ids = [e["execution_id"] for e in ex_list["executions"]]
    assert exec_id in ids

    # ---- delete ----
    r = client.delete(f"/api/gauntlets/{gid}")
    _assert_not_500(r, "delete")
    assert r.status_code == 200, r.text
    assert "message" in r.json()

    # ---- get after delete proves deletion (404, not 500) ----
    r = client.get(f"/api/gauntlets/{gid}")
    _assert_not_500(r, "get-after-delete")
    assert r.status_code == 404, r.text


def test_gauntlet_endpoints_never_500(client):
    """Explicit guard: none of the gauntlet endpoints may emit a 500."""
    created = client.post("/api/gauntlets", json=SAMPLE_GAUNTLET)
    gid = created.json().get("id")
    name = SAMPLE_GAUNTLET["name"]

    executed = client.post(
        f"/api/gauntlets/{name}/execute",
        json={"content": "x"},
    )
    exec_id = executed.json().get("execution_id")

    checks = [
        ("GET /api/gauntlets", client.get("/api/gauntlets")),
        ("GET /api/gauntlets/{id}", client.get(f"/api/gauntlets/{gid}")),
        ("POST /api/gauntlets/{name}/execute", executed),
        ("GET /api/gauntlets/executions/{id}/status",
         client.get(f"/api/gauntlets/executions/{exec_id}/status")),
        ("GET /api/gauntlets/executions", client.get("/api/gauntlets/executions")),
        ("DELETE /api/gauntlets/{id}", client.delete(f"/api/gauntlets/{gid}")),
    ]
    for label, resp in checks:
        assert resp.status_code != 500, f"{label} returned 500: {resp.text}"
        assert resp.status_code < 500, f"{label} returned >=500: {resp.status_code}"
