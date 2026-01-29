"""
Evolution endpoint tests
"""
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def get_auth_token():
    """Helper to get auth token"""
    # Register user
    client.post(
        "/api/v1/auth/register",
        json={
            "email": "evolution@example.com",
            "password": "EvolPass123",
            "username": "evoluser",
        },
    )

    # Login
    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "evolution@example.com",
            "password": "EvolPass123",
        },
    )

    return response.json()["access_token"]


def test_start_evolution():
    """Test starting an evolution"""
    token = get_auth_token()

    response = client.post(
        "/api/v1/evolution/start",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "content": "def hello():\n    print('Hello World')",
            "mode": "standard",
            "parameters": {
                "max_iterations": 10,
                "population_size": 20,
                "temperature": 0.7,
                "top_p": 0.9,
            },
            "models": [
                {
                    "provider": "openai",
                    "model": "gpt-4",
                    "api_key": "sk-test-key",
                }
            ],
        },
    )

    assert response.status_code == 202
    data = response.json()
    assert "evolution_id" in data
    assert data["status"] == "running"
    assert "websocket_url" in data


def test_start_evolution_unauthorized():
    """Test starting evolution without authentication"""
    response = client.post(
        "/api/v1/evolution/start",
        json={
            "content": "test content",
            "mode": "standard",
            "parameters": {
                "max_iterations": 10,
                "population_size": 20,
            },
            "models": [
                {
                    "provider": "openai",
                    "model": "gpt-4",
                    "api_key": "sk-test",
                }
            ],
        },
    )

    assert response.status_code == 401


def test_get_evolution_status():
    """Test getting evolution status"""
    token = get_auth_token()

    # Start evolution
    start_response = client.post(
        "/api/v1/evolution/start",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "content": "test content",
            "mode": "standard",
            "parameters": {"max_iterations": 10, "population_size": 20},
            "models": [{"provider": "openai", "model": "gpt-4", "api_key": "sk-test"}],
        },
    )
    evolution_id = start_response.json()["evolution_id"]

    # Get status
    response = client.get(
        f"/api/v1/evolution/{evolution_id}",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["evolution_id"] == evolution_id
    assert "status" in data
    assert "progress" in data


def test_get_evolution_not_found():
    """Test getting non-existent evolution"""
    token = get_auth_token()

    response = client.get(
        "/api/v1/evolution/nonexistent-id",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 404


def test_pause_evolution():
    """Test pausing an evolution"""
    token = get_auth_token()

    # Start evolution
    start_response = client.post(
        "/api/v1/evolution/start",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "content": "test content",
            "mode": "standard",
            "parameters": {"max_iterations": 10, "population_size": 20},
            "models": [{"provider": "openai", "model": "gpt-4", "api_key": "sk-test"}],
        },
    )
    evolution_id = start_response.json()["evolution_id"]

    # Pause evolution
    response = client.post(
        f"/api/v1/evolution/{evolution_id}/pause",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "paused"


def test_list_evolutions():
    """Test listing evolutions"""
    token = get_auth_token()

    # Start a few evolutions
    for i in range(3):
        client.post(
            "/api/v1/evolution/start",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "content": f"test content {i}",
                "mode": "standard",
                "parameters": {"max_iterations": 5, "population_size": 10},
                "models": [{"provider": "openai", "model": "gpt-4", "api_key": "sk-test"}],
            },
        )

    # List evolutions
    response = client.get(
        "/api/v1/evolution?limit=10",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "evolutions" in data
    assert "total" in data
    assert len(data["evolutions"]) <= 10


def test_delete_evolution():
    """Test deleting an evolution"""
    token = get_auth_token()

    # Start evolution
    start_response = client.post(
        "/api/v1/evolution/start",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "content": "test content",
            "mode": "standard",
            "parameters": {"max_iterations": 5, "population_size": 10},
            "models": [{"provider": "openai", "model": "gpt-4", "api_key": "sk-test"}],
        },
    )
    evolution_id = start_response.json()["evolution_id"]

    # Delete evolution
    response = client.delete(
        f"/api/v1/evolution/{evolution_id}",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 204

    # Verify it's deleted
    get_response = client.get(
        f"/api/v1/evolution/{evolution_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert get_response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
