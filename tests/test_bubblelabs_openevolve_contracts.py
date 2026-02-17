"""
Regression tests for BubbleLabs/OpenEvolve API contracts.

These tests cover:
- Legacy BubbleLab contract endpoints (/health, /evolutions, /adversarial-runs)
- Unified BubbleLabs control endpoints (/bubblelabs/control/*)
"""

from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

import api_server


TEST_API_KEY = "test-admin-key"


class _StubIntegration:
    def get_control_catalog(self) -> Dict[str, Any]:
        return {
            "success": True,
            "components": {"ace": ["create_skillbook"]},
            "auto_discovery": {"enabled": True},
        }

    def refresh_auto_discovery(self, force: bool = False) -> Dict[str, Any]:
        return {
            "success": True,
            "force": bool(force),
            "components": 1,
            "actions": 1,
        }

    def execute_control_action(
        self,
        component: str,
        action: str,
        payload: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        if component == "unknown":
            return {"success": False, "error": "Unknown component 'unknown'"}
        if component == "ace" and action == "fail":
            return {"success": False, "error": "Control action failed"}
        return {
            "success": True,
            "component": component,
            "action": action,
            "result": payload or {},
        }


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # Force deterministic auth path for tests.
    monkeypatch.setattr(api_server, "SECURITY_FRAMEWORK_AVAILABLE", False)
    api_server.API_KEYS[TEST_API_KEY] = {
        "role": api_server.UserRole.ADMIN,
        "name": "test-admin",
    }

    # Stabilize BubbleLabs integration behavior via stubbed bridge.
    stub = _StubIntegration()
    monkeypatch.setattr(api_server, "BUBBLELABS_AVAILABLE", True)
    monkeypatch.setattr(api_server, "get_extended_integration", lambda: stub)

    return TestClient(api_server.app)


def _auth_headers() -> Dict[str, str]:
    return {"x-api-key": TEST_API_KEY, "Content-Type": "application/json"}


def test_health_contract_fields(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert "version" in payload
    assert "timestamp" in payload


def test_legacy_evolutions_contract_crud(client: TestClient) -> None:
    create_response = client.post(
        "/evolutions",
        json={
            "name": "Contract Test Evolution",
            "base_prompt": "Solve this test problem",
            "adversarial_prompt": "Try to break it",
            "parameters": {"temperature": 0.1},
        },
    )
    assert create_response.status_code == 200
    created = create_response.json()
    assert created["id"].startswith("evo-")
    assert created["name"] == "Contract Test Evolution"
    assert "created_at" in created
    assert "updated_at" in created

    list_response = client.get("/evolutions?limit=10&offset=0")
    assert list_response.status_code == 200
    listed = list_response.json()
    assert isinstance(listed["evolutions"], list)
    assert listed["total"] >= 1
    assert len(listed["evolutions"]) <= 10

    by_id_response = client.get(f"/evolutions/{created['id']}")
    assert by_id_response.status_code == 200
    by_id = by_id_response.json()
    assert by_id["id"] == created["id"]
    assert by_id["base_prompt"] == "Solve this test problem"


def test_legacy_evolutions_contract_error_handling(client: TestClient) -> None:
    invalid_json_response = client.post(
        "/evolutions",
        data="invalid json{{{",
        headers={"Content-Type": "application/json"},
    )
    assert invalid_json_response.status_code == 400
    assert "error" in invalid_json_response.json()

    missing_required_response = client.post(
        "/evolutions",
        json={"name": "missing-base-prompt"},
    )
    assert missing_required_response.status_code == 400
    assert "error" in missing_required_response.json()

    missing_id_response = client.get("/evolutions/invalid-id-12345")
    assert missing_id_response.status_code == 404
    assert "error" in missing_id_response.json()


def test_legacy_adversarial_runs_contract(client: TestClient) -> None:
    response = client.get("/adversarial-runs")

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["runs"], list)
    assert "total" in payload


def test_control_catalog_requires_auth(client: TestClient) -> None:
    response = client.get("/bubblelabs/control/catalog")
    assert response.status_code == 422


def test_control_catalog_and_discover(client: TestClient) -> None:
    catalog_response = client.get("/bubblelabs/control/catalog", headers=_auth_headers())
    assert catalog_response.status_code == 200
    catalog_payload = catalog_response.json()
    assert catalog_payload["success"] is True
    assert "components" in catalog_payload
    assert "ace" in catalog_payload["components"]

    discover_response = client.post(
        "/bubblelabs/control/discover",
        headers=_auth_headers(),
        json={"force": True},
    )
    assert discover_response.status_code == 200
    discover_payload = discover_response.json()
    assert discover_payload["success"] is True
    assert discover_payload["force"] is True


def test_control_execute_success_and_error_mapping(client: TestClient) -> None:
    success_response = client.post(
        "/bubblelabs/control/execute",
        headers=_auth_headers(),
        json={
            "component": "ace",
            "action": "create_skillbook",
            "payload": {"name": "Smoke", "skills": [{"id": "s1"}]},
        },
    )
    assert success_response.status_code == 200
    success_payload = success_response.json()
    assert success_payload["success"] is True

    unknown_component_response = client.post(
        "/bubblelabs/control/execute",
        headers=_auth_headers(),
        json={"component": "unknown", "action": "anything", "payload": {}},
    )
    assert unknown_component_response.status_code == 404
    assert "Unknown" in unknown_component_response.json()["detail"]

    known_component_failure_response = client.post(
        "/bubblelabs/control/execute",
        headers=_auth_headers(),
        json={"component": "ace", "action": "fail", "payload": {}},
    )
    assert known_component_failure_response.status_code == 400
    assert "Control action failed" in known_component_failure_response.json()["detail"]
