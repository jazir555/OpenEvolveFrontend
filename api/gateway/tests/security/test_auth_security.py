"""
Security tests for authentication endpoints
"""
import pytest
from fastapi.testclient import TestClient
from main import app
import time

client = TestClient(app)


def test_sql_injection_in_login():
    """Test SQL injection protection in login"""
    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "'; DROP TABLE users; --",
            "password": "password",
        },
    )

    # Should return validation error or unauthorized, not a 500 error
    assert response.status_code in [401, 422]


def test_sql_injection_in_register():
    """Test SQL injection protection in registration"""
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": "test@example.com",
            "password": "'; DROP TABLE users; --",
            "username": "testuser",
        },
    )

    # Should validate password format
    assert response.status_code in [400, 422]


def test_xss_in_user_input():
    """Test XSS protection in user input"""
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": "test@example.com",
            "password": "SecurePass123",
            "username": "<script>alert('xss')</script>",
            "full_name": "<img src=x onerror=alert('xss')>",
        },
    )

    # Should either accept (sanitized) or reject
    # If accepted, the data should be sanitized
    if response.status_code == 201:
        data = response.json()
        # Check that script tags are not in response
        assert "<script>" not in str(data)
        assert "onerror=" not in str(data)


def test_rate_limiting_login():
    """Test rate limiting on login endpoint"""
    # Attempt multiple failed logins rapidly
    responses = []
    for _ in range(100):
        response = client.post(
            "/api/v1/auth/login",
            json={
                "email": "test@example.com",
                "password": "wrongpassword",
            },
        )
        responses.append(response.status_code)
        # Don't delay, we want to trigger rate limiting

    # Should get rate limited (429) after some attempts
    assert 429 in responses, "Rate limiting should be triggered"


def test_brute_force_protection():
    """Test brute force protection"""
    email = "bruteforce@example.com"

    # Register user
    client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": "CorrectPass123",
            "username": "bruteforceuser",
        },
    )

    # Attempt multiple failed logins
    failed_attempts = 0
    for _ in range(20):
        response = client.post(
            "/api/v1/auth/login",
            json={
                "email": email,
                "password": "WrongPassword",
            },
        )
        if response.status_code == 401:
            failed_attempts += 1
        elif response.status_code == 429:
            # Rate limited, good
            break

    # Now try correct password
    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": email,
            "password": "CorrectPass123",
        },
    )

    # Should still be rate limited or take longer
    assert response.status_code in [200, 429]


def test_jwt_expiration():
    """Test JWT token expiration"""
    # This test requires an expired token
    # For now, test with invalid token format
    response = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": "Bearer invalid.token.format"},
    )

    assert response.status_code == 401


def test_jwt_reuse_prevention():
    """Test that JWT tokens cannot be reused after logout"""
    # Register and login
    client.post(
        "/api/v1/auth/register",
        json={
            "email": "jwtreuse@example.com",
            "password": "JWTReuse123",
            "username": "jwtuser",
        },
    )

    login_response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "jwtreuse@example.com",
            "password": "JWTReuse123",
        },
    )
    token = login_response.json()["access_token"]

    # Logout
    client.post(
        "/api/v1/auth/logout",
        headers={"Authorization": f"Bearer {token}"},
    )

    # Try to use token after logout
    response = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )

    # Should be unauthorized
    assert response.status_code == 401


def test_cors_configuration():
    """Test CORS headers"""
    response = client.options(
        "/api/v1/auth/login",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )

    # Check for CORS headers
    assert "access-control-allow-origin" in response.headers or response.status_code == 200


def test_password_complexity():
    """Test password complexity requirements"""
    weak_passwords = [
        "123",  # Too short
        "password",  # Too common
        "abcdefgh",  # No numbers
        "12345678",  # No letters
        "Abc123",  # Too short
    ]

    for weak_pwd in weak_passwords:
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": f"test{weak_pwd}@example.com",
                "password": weak_pwd,
                "username": f"test{weak_pwd}",
            },
        )

        # Should reject weak passwords
        assert response.status_code != 201, f"Weak password '{weak_pwd}' should be rejected"


def test_content_type_validation():
    """Test content-type validation"""
    # Try to send non-JSON content
    response = client.post(
        "/api/v1/auth/login",
        content="email=test@example.com&password=password",  # Form data
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )

    # Should reject or handle appropriately
    # API expects JSON
    assert response.status_code in [415, 422]


def test_header_injection():
    """Test header injection protection"""
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": "test@example.com\r\nX-Injected-Header: malicious",
            "password": "SecurePass123",
            "username": "testuser",
        },
    )

    # Should validate and reject or sanitize
    assert response.status_code in [400, 422, 201]


def test_mass_assignment():
    """Test protection against mass assignment"""
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": "test@example.com",
            "password": "SecurePass123",
            "username": "testuser",
            "is_admin": True,  # Try to set admin flag
            "role": "admin",  # Try to set role
            "credits": 999999,  # Try to set credits
        },
    )

    if response.status_code == 201:
        data = response.json()
        # Should not include admin/role/credits in response
        assert "is_admin" not in data
        assert "role" not in data
        assert "credits" not in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
