"""
WebSocket endpoint tests
"""
import pytest
import asyncio
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def get_auth_token():
    """Helper to get auth token"""
    client.post(
        "/api/v1/auth/register",
        json={
            "email": "websocket@example.com",
            "password": "SocketPass123",
            "username": "websocketuser",
        },
    )

    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "websocket@example.com",
            "password": "SocketPass123",
        },
    )

    return response.json()["access_token"]


def test_websocket_evolution_connection():
    """Test WebSocket connection for evolution updates"""
    with client.websocket_connect(
        "/ws/evolution/test-evolution-1?user_id=test-user"
    ) as websocket:
        # Send a message
        websocket.send_json({"type": "subscribe", "data": {}})

        # Receive response
        data = websocket.receive_json()
        assert data["type"] == "echo"


def test_websocket_adversarial_connection():
    """Test WebSocket connection for adversarial testing"""
    with client.websocket_connect(
        "/ws/adversarial/test-adversarial-1?user_id=test-user"
    ) as websocket:
        # Send a message
        websocket.send_json({"type": "subscribe", "data": {}})

        # Receive response
        data = websocket.receive_json()
        assert data["type"] == "echo"


def test_websocket_workflow_connection():
    """Test WebSocket connection for workflow updates"""
    with client.websocket_connect(
        "/ws/workflow/test-workflow-1?user_id=test-user"
    ) as websocket:
        # Send a message
        websocket.send_json({"type": "subscribe", "data": {}})

        # Receive response
        data = websocket.receive_json()
        assert data["type"] == "echo"


def test_websocket_collaboration_connection():
    """Test WebSocket connection for collaboration"""
    with client.websocket_connect(
        "/ws/collaboration/test-room-1?user_id=test-user&username=TestUser"
    ) as websocket:
        # Send content update
        websocket.send_json({
            "type": "content_update",
            "content": "Test content update"
        })

        # Receive response
        data = websocket.receive_json()
        assert data["type"] == "echo"


def test_websocket_monitoring_connection():
    """Test WebSocket connection for monitoring"""
    with client.websocket_connect("/ws/monitoring") as websocket:
        # Should receive periodic monitoring updates
        data = websocket.receive_json()
        assert "type" in data
        assert data["type"] == "resource_update"
        assert "data" in data
        assert "cpu_percent" in data["data"]
        assert "memory_percent" in data["data"]


def test_websocket_send_and_receive():
    """Test sending and receiving multiple WebSocket messages"""
    with client.websocket_connect(
        "/ws/evolution/test-evolution-2?user_id=test-user"
    ) as websocket:
        # Send multiple messages
        messages = [
            {"type": "subscribe", "data": {}},
            {"type": "status_update", "data": {"progress": 50}},
            {"type": "complete", "data": {"result": "success"}},
        ]

        for msg in messages:
            websocket.send_json(msg)

        # Receive responses
        for msg in messages:
            response = websocket.receive_json()
            assert response["type"] == "echo"


def test_websocket_disconnect():
    """Test WebSocket disconnection"""
    with client.websocket_connect(
        "/ws/evolution/test-evolution-3?user_id=test-user"
    ) as websocket:
        # Send message
        websocket.send_json({"type": "subscribe", "data": {}})

        # Receive response
        websocket.receive_json()

    # WebSocket should be disconnected after context exit
    # No assertion needed, if no exception is raised, test passes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
