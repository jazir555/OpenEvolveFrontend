"""
Authentication endpoint tests
"""
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "OpenEvolve API Gateway"
    assert data["status"] == "operational"


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "services" in data


def test_register_user():
    """Test user registration"""
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": "test@example.com",
            "password": "SecurePass123",
            "username": "testuser",
            "full_name": "Test User",
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert "user_id" in data
    assert data["email"] == "test@example.com"
    assert data["username"] == "testuser"


def test_register_duplicate_user():
    """Test registering duplicate user"""
    # Register user first time
    client.post(
        "/api/v1/auth/register",
        json={
            "email": "duplicate@example.com",
            "password": "SecurePass123",
            "username": "duplicate",
        },
    )

    # Try to register again
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": "duplicate@example.com",
            "password": "SecurePass123",
            "username": "duplicate2",
        },
    )

    assert response.status_code == 409


def test_login_success():
    """Test successful login"""
    # Register user first
    client.post(
        "/api/v1/auth/register",
        json={
            "email": "login@example.com",
            "password": "LoginPass123",
            "username": "loginuser",
        },
    )

    # Login
    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "login@example.com",
            "password": "LoginPass123",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"


def test_login_invalid_credentials():
    """Test login with invalid credentials"""
    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "nonexistent@example.com",
            "password": "WrongPassword",
        },
    )

    assert response.status_code == 422


def test_get_current_user():
    """Test getting current user profile"""
    # Register and login
    client.post(
        "/api/v1/auth/register",
        json={
            "email": "profile@example.com",
            "password": "ProfilePass123",
            "username": "profileuser",
        },
    )

    login_response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "profile@example.com",
            "password": "ProfilePass123",
        },
    )
    token = login_response.json()["access_token"]

    # Get profile
    response = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "profile@example.com"
    assert data["username"] == "profileuser"


def test_get_current_user_unauthorized():
    """Test getting profile without authentication"""
    response = client.get("/api/v1/auth/me")
    assert response.status_code == 401


def test_update_user_profile():
    """Test updating user profile"""
    # Register and login
    client.post(
        "/api/v1/auth/register",
        json={
            "email": "update@example.com",
            "password": "UpdatePass123",
            "username": "updateuser",
            "full_name": "Original Name",
        },
    )

    login_response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "update@example.com",
            "password": "UpdatePass123",
        },
    )
    token = login_response.json()["access_token"]

    # Update profile
    response = client.put(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "full_name": "Updated Name",
            "preferences": {"theme": "dark"},
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["full_name"] == "Updated Name"
    assert data["preferences"]["theme"] == "dark"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
