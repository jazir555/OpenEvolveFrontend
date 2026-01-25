"""
Load testing configuration for OpenEvolve API Gateway
"""
from locust import HttpUser, task, between, events
from locust.runners import MasterRunner
import json
import random


class OpenEvolveUser(HttpUser):
    """
    Simulates a typical OpenEvolve user workflow
    """
    wait_time = between(1, 3)

    def on_start(self):
        """
        Login on start to get auth token
        """
        # Login
        response = self.client.post(
            "/api/v1/auth/login",
            json={
                "email": "test@example.com",
                "password": "SecurePass123",
            },
        )
        if response.status_code == 200:
            data = response.json()
            self.token = data.get("access_token")
            self.client.headers.update({
                "Authorization": f"Bearer {self.token}"
            })
        else:
            self.token = None

    @task(3)
    def get_workflows(self):
        """Fetch workflow list"""
        if self.token:
            self.client.get("/api/v1/evolution")

    @task(2)
    def get_analytics(self):
        """Fetch analytics metrics"""
        if self.token:
            self.client.get("/api/v1/analytics/metrics", params={
                "start_date": "2025-01-01T00:00:00Z",
                "end_date": "2025-01-06T00:00:00Z",
                "granularity": "day",
            })

    @task(1)
    def get_knowledge_artifacts(self):
        """Fetch knowledge base artifacts"""
        if self.token:
            self.client.get("/api/v1/knowledge/artifacts")

    @task(1)
    def create_artifact(self):
        """Create a new knowledge artifact"""
        if self.token:
            self.client.post(
                "/api/v1/knowledge/artifacts",
                json={
                    "title": f"Load Test Artifact {random.randint(1, 1000)}",
                    "content": "This is a load test artifact",
                    "type": "note",
                    "tags": ["load-test"],
                },
            )

    @task(1)
    def get_user_profile(self):
        """Get user profile"""
        if self.token:
            self.client.get("/api/v1/auth/me")


class AdminUser(HttpUser):
    """
    Simulates an admin user with different patterns
    """
    wait_time = between(2, 5)

    def on_start(self):
        """Login as admin"""
        response = self.client.post(
            "/api/v1/auth/login",
            json={
                "email": "admin@example.com",
                "password": "AdminPass123",
            },
        )
        if response.status_code == 200:
            data = response.json()
            self.token = data.get("access_token")
            self.client.headers.update({
                "Authorization": f"Bearer {self.token}"
            })
        else:
            self.token = None

    @task(2)
    def get_system_health(self):
        """Check system health"""
        self.client.get("/health")

    @task(1)
    def get_monitoring_data(self):
        """Fetch monitoring data"""
        if self.token:
            self.client.get("/api/v1/monitoring/health")

    @task(1)
    def view_logs(self):
        """View application logs"""
        if self.token:
            self.client.get("/api/v1/monitoring/logs", params={"limit": 50})


# Event handlers for reporting
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """
    Custom event handler for request logging
    """
    if exception:
        print(f"Request failed: {name} - {exception}")
    elif response_time > 1000:  # Log slow requests
        print(f"Slow request: {name} - {response_time}ms")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """
    Custom event handler for test stop
    """
    if isinstance(environment.runner, MasterRunner):
        print("Test completed on master node")
    else:
        print("Test completed on worker node")
