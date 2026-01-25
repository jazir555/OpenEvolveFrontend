"""
Analytics endpoint tests
"""
import pytest
from fastapi.testclient import TestClient
from main import app
from datetime import datetime, timedelta

client = TestClient(app)


def get_auth_token():
    """Helper to get auth token"""
    client.post(
        "/api/v1/auth/register",
        json={
            "email": "analytics@example.com",
            "password": "AnalPass123",
            "username": "analyticsuser",
        },
    )

    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "analytics@example.com",
            "password": "AnalPass123",
        },
    )

    return response.json()["access_token"]


def test_get_metrics():
    """Test getting analytics metrics"""
    token = get_auth_token()

    # Calculate date range
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=7)

    response = client.get(
        "/api/v1/analytics/metrics",
        headers={"Authorization": f"Bearer {token}"},
        params={
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "granularity": "day",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert "total_evolutions" in data["metrics"]
    assert "success_rate" in data["metrics"]
    assert "average_duration" in data["metrics"]


def test_get_metrics_unauthorized():
    """Test getting metrics without authentication"""
    response = client.get("/api/v1/analytics/metrics")
    assert response.status_code == 401


def test_get_performance_data():
    """Test getting performance analytics"""
    token = get_auth_token()

    response = client.get(
        "/api/v1/analytics/performance",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "performance" in data
    assert isinstance(data["performance"], list)


def test_get_metrics_invalid_date_range():
    """Test getting metrics with invalid date range"""
    token = get_auth_token()

    # End date before start date
    response = client.get(
        "/api/v1/analytics/metrics",
        headers={"Authorization": f"Bearer {token}"},
        params={
            "start_date": "2025-01-10T00:00:00Z",
            "end_date": "2025-01-01T00:00:00Z",
            "granularity": "day",
        },
    )

    assert response.status_code == 400


def test_get_metrics_invalid_granularity():
    """Test getting metrics with invalid granularity"""
    token = get_auth_token()

    response = client.get(
        "/api/v1/analytics/metrics",
        headers={"Authorization": f"Bearer {token}"},
        params={
            "start_date": "2025-01-01T00:00:00Z",
            "end_date": "2025-01-10T00:00:00Z",
            "granularity": "invalid",
        },
    )

    assert response.status_code == 422


def test_get_top_content():
    """Test getting top performing content"""
    token = get_auth_token()

    response = client.get(
        "/api/v1/analytics/top-content",
        headers={"Authorization": f"Bearer {token}"},
        params={"limit": 10},
    )

    assert response.status_code == 200
    data = response.json()
    assert "content" in data
    assert isinstance(data["content"], list)


def test_get_user_statistics():
    """Test getting user statistics"""
    token = get_auth_token()

    response = client.get(
        "/api/v1/analytics/user-stats",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "total_evolutions" in data
    assert "total_adversarial_tests" in data
    assert "success_rate" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
