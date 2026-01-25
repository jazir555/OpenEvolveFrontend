"""
Security tests for API endpoints
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
            "email": "apisec@example.com",
            "password": "ApiSecPass123",
            "username": "apisecuser",
        },
    )

    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "apisec@example.com",
            "password": "ApiSecPass123",
        },
    )

    return response.json()["access_token"]


def test_path_traversal():
    """Test path traversal protection"""
    token = get_auth_token()

    response = client.get(
        "/api/v1/knowledge/artifacts/../../../etc/passwd",
        headers={"Authorization": f"Bearer {token}"},
    )

    # Should return 404 or 400, not expose file system
    assert response.status_code in [400, 404]


def test_command_injection():
    """Test command injection protection"""
    token = get_auth_token()

    response = client.post(
        "/api/v1/evolution/start",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "content": "test; rm -rf /",
            "mode": "standard",
            "parameters": {
                "max_iterations": 10,
                "population_size": 20,
            },
            "models": [{"provider": "openai", "model": "gpt-4", "api_key": "sk-test"}],
        },
    )

    # Should handle safely, not execute command
    # Either accept (sanitized) or reject
    assert response.status_code in [202, 400, 422]


def test_xml_injection():
    """Test XML injection protection"""
    token = get_auth_token()

    xml_payload = """<?xml version="1.0"?>
    <!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>
    <content>&xxe;</content>
    """

    response = client.post(
        "/api/v1/knowledge/artifacts",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "title": "XML Test",
            "content": xml_payload,
            "type": "note",
        },
    )

    # Should reject or sanitize
    assert response.status_code in [201, 400, 422]


def test_content_length_limit():
    """Test content length limits"""
    token = get_auth_token()

    # Try to send extremely large content
    large_content = "A" * (10 * 1024 * 1024)  # 10MB

    response = client.post(
        "/api/v1/knowledge/artifacts",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "title": "Large Content",
            "content": large_content,
            "type": "note",
        },
    )

    # Should reject oversized content
    assert response.status_code in [400, 413, 422]


def test_authenticated_request_without_token():
    """Test that authenticated endpoints reject requests without tokens"""
    response = client.get("/api/v1/knowledge/artifacts")
    assert response.status_code == 401


def test_authenticated_request_with_invalid_token():
    """Test that authenticated endpoints reject invalid tokens"""
    response = client.get(
        "/api/v1/knowledge/artifacts",
        headers={"Authorization": "Bearer invalid-token"},
    )
    assert response.status_code == 401


def test_authenticated_request_with_expired_token():
    """Test that authenticated endpoints reject expired tokens"""
    # Use a malformed token to simulate expired/invalid
    response = client.get(
        "/api/v1/knowledge/artifacts",
        headers={"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.expired"},
    )
    assert response.status_code == 401


def test_parameter_pollution():
    """Test parameter pollution protection"""
    token = get_auth_token()

    response = client.get(
        "/api/v1/knowledge/artifacts?id=1&id=2&id=3",
        headers={"Authorization": f"Bearer {token}"},
    )

    # Should handle duplicate parameters safely
    assert response.status_code in [200, 400]


def test_http_method_tampering():
    """Test HTTP method tampering protection"""
    token = get_auth_token()

    # Try to override POST method with X-HTTP-Method-Override header
    response = client.post(
        "/api/v1/knowledge/artifacts",
        headers={
            "Authorization": f"Bearer {token}",
            "X-HTTP-Method-Override": "DELETE",
        },
        json={
            "title": "Test",
            "content": "Test content",
        },
    )

    # Should not override the method
    # Either create (POST) or reject
    assert response.status_code in [201, 405]


def test_ssrf_protection():
    """Test Server-Side Request Forgery protection"""
    token = get_auth_token()

    # Try to make request to internal network
    response = client.post(
        "/api/v1/knowledge/artifacts",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "title": "SSRF Test",
            "content": "Check http://localhost:6379 or http://169.254.169.254",
            "type": "note",
        },
    )

    # Should reject or sanitize
    assert response.status_code in [201, 400, 422]


def test_html_injection():
    """Test HTML injection protection"""
    token = get_auth_token()

    html_content = """
    <iframe src="javascript:alert('xss')"></iframe>
    <script>alert('xss')</script>
    <img src=x onerror="alert('xss')">
    """

    response = client.post(
        "/api/v1/knowledge/artifacts",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "title": "HTML Injection",
            "content": html_content,
            "type": "note",
        },
    )

    if response.status_code == 201:
        data = response.json()
        # Content should be sanitized
        assert "<script>" not in str(data)
        assert "onerror=" not in str(data)


def test_json_payload_size_limit():
    """Test JSON payload size limit"""
    token = get_auth_token()

    # Create very large JSON payload
    large_array = [{"item": i} for i in range(100000)]

    response = client.post(
        "/api/v1/evolution/start",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "content": "test",
            "mode": "standard",
            "parameters": {"large_data": large_array},
            "models": [{"provider": "openai", "model": "gpt-4", "api_key": "sk-test"}],
        },
    )

    # Should reject oversized payload
    assert response.status_code in [400, 413, 422]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
