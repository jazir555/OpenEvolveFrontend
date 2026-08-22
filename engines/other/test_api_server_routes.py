"""
Offline tests for the external-integration REST endpoints in engines/other/api_server.py.

These exercise the Sovereign-Grade Decomposition Workflow external API surface
(§6.3): team/gauntlet CRUD plus POST /workflows/run + GET /workflows/{id} polling.

Run:
    python -m pytest engines/other/test_api_server_routes.py -q -p no:pytest_ethereum
"""

import json
import os
import shutil
import sys

# Flat-style import path setup (must happen BEFORE importing the server module so
# that the env-driven API key table is populated and flat sibling modules resolve).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_SCRIPTS_DIR = os.path.join(_REPO_ROOT, "scripts")
for _p in (_THIS_DIR, _SCRIPTS_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Provide a usable API key for the auth-gated routes (read at import time).
# Format expected by api_server._load_api_keys: API_KEY_<name>=<secret>:<role>
os.environ["API_KEY_testkey"] = "testkey:admin"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import api_server  # noqa: E402
from api_server import app  # noqa: E402


API_KEY = "testkey"
HEADERS = {"X-API-Key": API_KEY}


@pytest.fixture(autouse=True)
def _hermetic_state():
    """Start every test from a clean on-disk state and work around an
    other-agent-owned serialization bug in gauntlet_manager.

    gauntlet_manager._save_gauntlets cannot persist GauntletDefinition because it
    carries a non-JSON-serializable VerificationMethod field; the resulting
    truncated file corrupts subsequent loads. We patch the saver with a safe
    JSON encoder (analogous to the permitted run_sovereign_workflow monkeypatch)
    so cross-request CRUD exercises real manager wiring without corruption.
    """
    for _p in ("data/tenants", "teams.json", "gauntlets.json"):
        if os.path.isdir(_p):
            shutil.rmtree(_p)
        elif os.path.exists(_p):
            os.remove(_p)

    def _safe_save(self):
        data = {}
        for _name, _g in self.gauntlets.items():
            _d = _g.__dict__.copy()
            _d["rounds"] = [r.__dict__ for r in _g.rounds]
            data[_name] = _d
        with open(self.gauntlets_file, "w", encoding="utf-8") as _fh:
            json.dump(
                data,
                _fh,
                indent=4,
                default=lambda o: o.__dict__ if hasattr(o, "__dict__") else str(o),
            )

    _orig_save = api_server.GauntletManager._save_gauntlets
    api_server.GauntletManager._save_gauntlets = _safe_save

    # Disable the security framework rate limiter so the offline CRUD suite is not
    # throttled (it makes several sequential requests against the same key).
    _orig_rl = getattr(api_server, "SecurityConfig", None)
    if _orig_rl is not None:
        _orig_rate_limit_enabled = _orig_rl.RATE_LIMIT_ENABLED
        _orig_rl.RATE_LIMIT_ENABLED = False
    yield
    api_server.GauntletManager._save_gauntlets = _orig_save
    if _orig_rl is not None:
        _orig_rl.RATE_LIMIT_ENABLED = _orig_rate_limit_enabled


def _route_paths():
    return {getattr(r, "path", None) for r in app.routes}


def _route_methods(path):
    methods = set()
    for r in app.routes:
        if getattr(r, "path", None) == path:
            methods |= set(getattr(r, "methods", set()) or set())
    return methods


# --------------------------------------------------------------------------- #
# 1. Module imports and routes are registered
# --------------------------------------------------------------------------- #
def test_module_imports():
    assert app is not None


def test_required_routes_registered():
    paths = _route_paths()
    assert "/teams" in paths, "GET/POST/PUT/DELETE /teams must be registered"
    assert "/gauntlets" in paths, "GET/POST/PUT/DELETE /gauntlets must be registered"
    assert "/workflows/run" in paths, "POST /workflows/run must be registered"


def test_team_crud_methods_registered():
    # Team CRUD is split across /teams (GET, POST) and /teams/{team_name} (PUT, DELETE).
    methods = {m.upper() for m in (_route_methods("/teams") | _route_methods("/teams/{team_name}"))}
    assert {"GET", "POST", "PUT", "DELETE"} <= methods


def test_gauntlet_crud_methods_registered():
    methods = {m.upper() for m in (_route_methods("/gauntlets") | _route_methods("/gauntlets/{gauntlet_name}"))}
    assert {"GET", "POST", "PUT", "DELETE"} <= methods


# --------------------------------------------------------------------------- #
# 2. Team CRUD via TestClient
# --------------------------------------------------------------------------- #
def test_team_crud():
    client = TestClient(app)
    name = "ext_team_blue"
    payload = {
        "name": name,
        "role": "Blue",
        "description": "external integration test team",
        "members": [{"model_id": "gpt-4o", "temperature": 0.5, "max_tokens": 1024}],
    }

    # Create
    r = client.post("/teams", json=payload, headers=HEADERS)
    assert r.status_code == 200, r.text
    assert r.json()["team_name"] == name

    # Read (list + single)
    r = client.get("/teams", headers=HEADERS)
    assert r.status_code == 200
    assert any(t["name"] == name for t in r.json()["teams"])

    r = client.get(f"/teams/{name}", headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["name"] == name

    # Update
    updated = dict(payload)
    updated["description"] = "updated description"
    r = client.put(f"/teams/{name}", json=updated, headers=HEADERS)
    assert r.status_code == 200, r.text
    r = client.get(f"/teams/{name}", headers=HEADERS)
    assert r.json()["description"] == "updated description"

    # 404 on missing
    r = client.get("/teams/does_not_exist_ext", headers=HEADERS)
    assert r.status_code == 404

    # Delete
    r = client.delete(f"/teams/{name}", headers=HEADERS)
    assert r.status_code == 200, r.text
    r = client.get(f"/teams/{name}", headers=HEADERS)
    assert r.status_code == 404


# --------------------------------------------------------------------------- #
# 3. Gauntlet CRUD via TestClient
# --------------------------------------------------------------------------- #
def test_gauntlet_crud():
    client = TestClient(app)
    name = "ext_gauntlet_red"
    payload = {
        "name": name,
        "team_name": "blue_team",
        "description": "external integration test gauntlet",
        "rounds": [
            {
                "round_number": 1,
                "quorum_required_approvals": 1,
                "quorum_from_panel_size": 1,
            }
        ],
    }

    r = client.post("/gauntlets", json=payload, headers=HEADERS)
    assert r.status_code == 200, r.text
    assert r.json()["gauntlet_name"] == name

    r = client.get(f"/gauntlets/{name}", headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["name"] == name

    r = client.get("/gauntlets/does_not_exist_ext", headers=HEADERS)
    assert r.status_code == 404

    updated = dict(payload)
    updated["description"] = "updated gauntlet"
    r = client.put(f"/gauntlets/{name}", json=updated, headers=HEADERS)
    assert r.status_code == 200, r.text

    r = client.delete(f"/gauntlets/{name}", headers=HEADERS)
    assert r.status_code == 200, r.text
    r = client.get(f"/gauntlets/{name}", headers=HEADERS)
    assert r.status_code == 404


# --------------------------------------------------------------------------- #
# 4. /workflows/run returns id + GET /workflows/{id} polls (offline stub)
# --------------------------------------------------------------------------- #
@pytest.fixture
def patched_runner():
    async def _stub(workflow_state, *args, **kwargs):
        workflow_state.status = "completed"
        workflow_state.current_stage = "DONE"
        return workflow_state

    original = api_server.run_sovereign_workflow
    api_server.run_sovereign_workflow = _stub
    yield
    api_server.run_sovereign_workflow = original


def test_workflow_run_and_poll(patched_runner):
    client = TestClient(app)

    r = client.post(
        "/workflows/run",
        json={"problem_statement": "Prove the buddy theorem for finite groups."},
        headers=HEADERS,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "workflow_id" in body
    assert body["status"] in {"running", "created", "completed"}
    workflow_id = body["workflow_id"]

    # Poll until the background thread finishes (offline stub is fast).
    final_status = None
    for _ in range(50):
        r = client.get(f"/workflows/{workflow_id}", headers=HEADERS)
        assert r.status_code == 200, r.text
        final_status = r.json().get("status")
        if final_status in {"completed", "failed"}:
            break

    assert final_status == "completed"
