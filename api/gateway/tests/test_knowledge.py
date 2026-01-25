"""
Knowledge base endpoint tests
"""
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def get_auth_token():
    """Helper to get auth token"""
    client.post(
        "/api/v1/auth/register",
        json={
            "email": "knowledge@example.com",
            "password": "KnowPass123",
            "username": "knowledgeuser",
        },
    )

    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "knowledge@example.com",
            "password": "KnowPass123",
        },
    )

    return response.json()["access_token"]


def test_create_artifact():
    """Test creating a knowledge artifact"""
    token = get_auth_token()

    response = client.post(
        "/api/v1/knowledge/artifacts",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "title": "Test Artifact",
            "content": "This is test content",
            "type": "note",
            "tags": ["test", "sample"],
            "language": "python",
        },
    )

    assert response.status_code == 201
    data = response.json()
    assert "artifact_id" in data
    assert data["title"] == "Test Artifact"
    assert data["content"] == "This is test content"


def test_create_artifact_unauthorized():
    """Test creating artifact without authentication"""
    response = client.post(
        "/api/v1/knowledge/artifacts",
        json={
            "title": "Test Artifact",
            "content": "This is test content",
            "type": "note",
        },
    )

    assert response.status_code == 401


def test_get_artifact():
    """Test getting a knowledge artifact"""
    token = get_auth_token()

    # Create artifact first
    create_response = client.post(
        "/api/v1/knowledge/artifacts",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "title": "Get Test Artifact",
            "content": "Content to retrieve",
            "type": "note",
        },
    )
    artifact_id = create_response.json()["artifact_id"]

    # Get artifact
    response = client.get(
        f"/api/v1/knowledge/artifacts/{artifact_id}",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["artifact_id"] == artifact_id
    assert data["title"] == "Get Test Artifact"


def test_get_artifact_not_found():
    """Test getting non-existent artifact"""
    token = get_auth_token()

    response = client.get(
        "/api/v1/knowledge/artifacts/nonexistent-id",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 404


def test_search_artifacts():
    """Test searching knowledge artifacts"""
    token = get_auth_token()

    # Create some artifacts
    client.post(
        "/api/v1/knowledge/artifacts",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "title": "Python Tutorial",
            "content": "Learn Python programming",
            "type": "tutorial",
            "tags": ["python", "programming"],
        },
    )

    client.post(
        "/api/v1/knowledge/artifacts",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "title": "JavaScript Guide",
            "content": "Learn JavaScript",
            "type": "tutorial",
            "tags": ["javascript", "programming"],
        },
    )

    # Search for Python
    response = client.get(
        "/api/v1/knowledge/artifacts",
        headers={"Authorization": f"Bearer {token}"},
        params={"search": "Python"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "artifacts" in data
    assert len(data["artifacts"]) > 0
    # Should find Python tutorial
    assert any("Python" in artifact["title"] for artifact in data["artifacts"])


def test_update_artifact():
    """Test updating a knowledge artifact"""
    token = get_auth_token()

    # Create artifact
    create_response = client.post(
        "/api/v1/knowledge/artifacts",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "title": "Original Title",
            "content": "Original content",
            "type": "note",
        },
    )
    artifact_id = create_response.json()["artifact_id"]

    # Update artifact
    response = client.put(
        f"/api/v1/knowledge/artifacts/{artifact_id}",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "title": "Updated Title",
            "content": "Updated content",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "Updated Title"
    assert data["content"] == "Updated content"


def test_delete_artifact():
    """Test deleting a knowledge artifact"""
    token = get_auth_token()

    # Create artifact
    create_response = client.post(
        "/api/v1/knowledge/artifacts",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "title": "To Delete",
            "content": "Will be deleted",
            "type": "note",
        },
    )
    artifact_id = create_response.json()["artifact_id"]

    # Delete artifact
    response = client.delete(
        f"/api/v1/knowledge/artifacts/{artifact_id}",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 204

    # Verify deletion
    get_response = client.get(
        f"/api/v1/knowledge/artifacts/{artifact_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert get_response.status_code == 404


def test_list_artifacts_by_tag():
    """Test listing artifacts filtered by tag"""
    token = get_auth_token()

    # Create artifacts with different tags
    client.post(
        "/api/v1/knowledge/artifacts",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "title": "Machine Learning",
            "content": "ML content",
            "type": "note",
            "tags": ["ml", "ai"],
        },
    )

    client.post(
        "/api/v1/knowledge/artifacts",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "title": "Web Development",
            "content": "Web dev content",
            "type": "note",
            "tags": ["web", "frontend"],
        },
    )

    # Filter by ML tag
    response = client.get(
        "/api/v1/knowledge/artifacts",
        headers={"Authorization": f"Bearer {token}"},
        params={"tag": "ml"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "artifacts" in data
    # All results should have ml tag
    for artifact in data["artifacts"]:
        assert "ml" in artifact.get("tags", [])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
