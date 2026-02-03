"""
OpenEvolve API Integration Tests

Comprehensive test suite for OpenEvolve FastAPI service
Tests all endpoints, error handling, and circuit breaker functionality

Run with: pytest tests/test_api_integration.py -v
"""

import pytest
import httpx
from typing import AsyncGenerator, Dict, Any
import asyncio
import json
import time


# ==================== Configuration ====================

BASE_URL = "http://localhost:8001"
TEST_TIMEOUT = 30.0


# ==================== Fixtures ====================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
async def client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """HTTP client for testing"""
    async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
        yield client


@pytest.fixture(scope="module")
async def verify_service_running(client: httpx.AsyncClient) -> bool:
    """Verify the service is running before tests"""
    try:
        response = await client.get(f"{BASE_URL}/health")
        assert response.status_code == 201
        return True
    except Exception as e:
        pytest.skip(f"Service not running: {e}")


# ==================== Test Data ====================

TEST_WORKFLOWS = [
    {
        "name": "Test Evolution Workflow",
        "description": "Evolution workflow for testing",
        "problem_statement": "Optimize a simple math function",
        "content_type": "code",
        "teams": [],
        "gauntlets": [],
        "metadata": {
            "mdap_enabled": False,
            "maker_enabled": False,
        },
        "workflow_type": "evolution",
    },
    {
        "name": "Test Adversarial Workflow",
        "description": "Adversarial workflow for testing",
        "problem_statement": "Harden API input validation",
        "content_type": "text",
        "teams": [],
        "gauntlets": [],
        "metadata": {},
        "workflow_type": "adversarial",
    },
    {
        "name": "Test Sovereign Workflow",
        "description": "Sovereign workflow for testing",
        "problem_statement": "Decompose a complex system design",
        "content_type": "document",
        "teams": [],
        "gauntlets": [],
        "metadata": {},
        "workflow_type": "sovereign",
    }
]

TEST_TEAM = {
    "name": "Test Team",
    "description": "Team for testing",
    "members": [
        {
            "name": "Test Model",
            "role": "coder",
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 2048
        }
    ]
}

TEST_GAUNTLET = {
    "name": "Test Gauntlet",
    "description": "Gauntlet for testing",
    "rounds": [
        {
            "name": "Round 1",
            "quorum_threshold": 0.7,
            "confidence_threshold": 0.8,
            "evaluation_type": "accuracy"
        }
    ]
}


# ==================== Health & Info Tests ====================

class TestHealthAndInfo:
    """Test health check and info endpoints"""

    @pytest.mark.asyncio
    async def test_health_check(self, client: httpx.AsyncClient, verify_service_running):
        """Test health check endpoint returns correct structure"""
        response = await client.get(f"{BASE_URL}/health")

        assert response.status_code == 201
        data = response.json()

        assert data["status"] == "healthy"
        assert data["service"] == "openevolve-api"
        assert "version" in data
        assert isinstance(data["features"], dict)
        assert data["features"]["evolution"] is True
        assert data["features"]["adversarial"] is True
        assert data["features"]["sovereign"] is True

    @pytest.mark.asyncio
    async def test_root_endpoint(self, client: httpx.AsyncClient, verify_service_running):
        """Test root endpoint returns API information"""
        response = await client.get(f"{BASE_URL}/")

        assert response.status_code == 201
        data = response.json()

        assert "service" in data
        assert "version" in data
        assert data["docs"] == "/docs"
        assert data["health"] == "/health"

    @pytest.mark.asyncio
    async def test_api_docs_accessible(self, client: httpx.AsyncClient, verify_service_running):
        """Test that API documentation is accessible"""
        response = await client.get(f"{BASE_URL}/docs")

        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")


# ==================== Workflow CRUD Tests ====================

class TestWorkflows:
    """Test workflow CRUD operations"""

    @pytest.mark.asyncio
    async def test_create_evolution_workflow(self, client: httpx.AsyncClient, verify_service_running):
        """Test creating an evolution workflow"""
        workflow = TEST_WORKFLOWS[0]

        response = await client.post(
            f"{BASE_URL}/api/workflows",
            json=workflow
        )

        assert response.status_code == 200
        data = response.json()

        assert "id" in data
        assert data["name"] == workflow["name"]
        assert data["description"] == workflow["description"]
        assert data["workflow_type"] == workflow["workflow_type"]
        assert data["status"] == "created"
        assert "created_at" in data
        assert "updated_at" in data

    @pytest.mark.asyncio
    async def test_create_adversarial_workflow(self, client: httpx.AsyncClient, verify_service_running):
        """Test creating an adversarial workflow"""
        workflow = TEST_WORKFLOWS[1]

        response = await client.post(
            f"{BASE_URL}/api/workflows",
            json=workflow
        )

        assert response.status_code == 201
        data = response.json()

        assert data["workflow_type"] == "adversarial"

    @pytest.mark.asyncio
    async def test_create_sovereign_workflow(self, client: httpx.AsyncClient, verify_service_running):
        """Test creating a sovereign workflow"""
        workflow = TEST_WORKFLOWS[2]

        response = await client.post(
            f"{BASE_URL}/api/workflows",
            json=workflow
        )

        assert response.status_code == 201
        data = response.json()

        assert data["workflow_type"] == "sovereign"

    @pytest.mark.asyncio
    async def test_list_workflows(self, client: httpx.AsyncClient, verify_service_running):
        """Test listing workflows"""
        # Create a workflow first
        await client.post(
            f"{BASE_URL}/api/workflows",
            json=TEST_WORKFLOWS[0]
        )

        # List workflows
        response = await client.get(f"{BASE_URL}/api/workflows")

        assert response.status_code == 200
        data = response.json()

        assert "workflows" in data
        assert "total" in data
        assert isinstance(data["workflows"], list)
        assert len(data["workflows"]) >= 1

    @pytest.mark.asyncio
    async def test_get_workflow(self, client: httpx.AsyncClient, verify_service_running):
        """Test getting a specific workflow"""
        # Create a workflow
        create_response = await client.post(
            f"{BASE_URL}/api/workflows",
            json=TEST_WORKFLOWS[0]
        )
        workflow_id = create_response.json()["id"]

        # Get the workflow
        response = await client.get(f"{BASE_URL}/api/workflows/{workflow_id}")

        assert response.status_code == 200
        data = response.json()

        assert data["id"] == workflow_id
        assert data["name"] == TEST_WORKFLOWS[0]["name"]

    @pytest.mark.asyncio
    async def test_update_workflow(self, client: httpx.AsyncClient, verify_service_running):
        """Test updating a workflow"""
        # Create a workflow
        create_response = await client.post(
            f"{BASE_URL}/api/workflows",
            json=TEST_WORKFLOWS[0]
        )
        workflow_id = create_response.json()["id"]

        # Update the workflow
        updates = {
            "name": "Updated Workflow Name",
            "description": "Updated description"
        }

        response = await client.put(
            f"{BASE_URL}/api/workflows/{workflow_id}",
            json=updates
        )

        assert response.status_code == 200
        data = response.json()

        assert data["name"] == updates["name"]
        assert data["description"] == updates["description"]

    @pytest.mark.asyncio
    async def test_delete_workflow(self, client: httpx.AsyncClient, verify_service_running):
        """Test deleting a workflow"""
        # Create a workflow
        create_response = await client.post(
            f"{BASE_URL}/api/workflows",
            json=TEST_WORKFLOWS[0]
        )
        workflow_id = create_response.json()["id"]

        # Delete the workflow
        response = await client.delete(f"{BASE_URL}/api/workflows/{workflow_id}")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data

        # Verify it's deleted
        get_response = await client.get(f"{BASE_URL}/api/workflows/{workflow_id}")
        assert get_response.status_code == 404

    @pytest.mark.asyncio
    async def test_create_workflow_validation(self, client: httpx.AsyncClient, verify_service_running):
        """Test workflow creation validation"""
        # Missing required fields
        invalid_workflow = {
            "name": "Test"
            # Missing problem_statement and content_type
        }

        response = await client.post(
            f"{BASE_URL}/api/workflows",
            json=invalid_workflow
        )

        assert response.status_code == 422  # Validation error


# ==================== Execution Tests ====================

class TestExecution:
    """Test workflow execution"""

    @pytest.mark.asyncio
    async def test_execute_workflow(self, client: httpx.AsyncClient, verify_service_running):
        """Test executing a workflow"""
        # Create a workflow
        create_response = await client.post(
            f"{BASE_URL}/api/workflows",
            json=TEST_WORKFLOWS[0]
        )
        workflow_id = create_response.json()["id"]

        # Execute the workflow
        execution_data = {
            "problem_statement": "Test problem: Create a function to add two numbers",
            "context": "Testing execution"
        }

        response = await client.post(
            f"{BASE_URL}/api/executions/workflows/{workflow_id}/execute",
            json=execution_data
        )

        assert response.status_code == 202
        data = response.json()

        assert "execution_id" in data
        assert data["workflow_id"] == workflow_id
        assert data["status"] in ["queued", "running"]
        assert 0.0 <= data["progress"] <= 1.0

    @pytest.mark.asyncio
    async def test_get_execution_status(self, client: httpx.AsyncClient, verify_service_running):
        """Test getting execution status"""
        # Create and execute workflow
        create_response = await client.post(
            f"{BASE_URL}/api/workflows",
            json=TEST_WORKFLOWS[0]
        )
        workflow_id = create_response.json()["id"]

        exec_response = await client.post(
            f"{BASE_URL}/api/executions/workflows/{workflow_id}/execute",
            json={
                "problem_statement": "Test problem"
            }
        )
        execution_id = exec_response.json()["execution_id"]

        # Get status
        response = await client.get(
            f"{BASE_URL}/api/executions/workflows/{workflow_id}/executions/{execution_id}"
        )

        assert response.status_code == 200
        data = response.json()

        assert data["execution_id"] == execution_id
        assert "status" in data
        assert "progress" in data

    @pytest.mark.asyncio
    async def test_pause_execution(self, client: httpx.AsyncClient, verify_service_running):
        """Test pausing an execution"""
        # Create and execute workflow
        create_response = await client.post(
            f"{BASE_URL}/api/workflows",
            json=TEST_WORKFLOWS[0]
        )
        workflow_id = create_response.json()["id"]

        exec_response = await client.post(
            f"{BASE_URL}/api/executions/workflows/{workflow_id}/execute",
            json={
                "problem_statement": "Test problem for pause"
            }
        )
        execution_id = exec_response.json()["execution_id"]

        # Pause execution
        response = await client.post(
            f"{BASE_URL}/api/executions/workflows/{workflow_id}/executions/{execution_id}/pause",
            json={}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["execution_id"] == execution_id

    @pytest.mark.asyncio
    async def test_resume_execution(self, client: httpx.AsyncClient, verify_service_running):
        """Test resuming a paused execution"""
        # Create and execute workflow
        create_response = await client.post(
            f"{BASE_URL}/api/workflows",
            json=TEST_WORKFLOWS[0]
        )
        workflow_id = create_response.json()["id"]

        exec_response = await client.post(
            f"{BASE_URL}/api/executions/workflows/{workflow_id}/execute",
            json={
                "problem_statement": "Test problem for resume"
            }
        )
        execution_id = exec_response.json()["execution_id"]

        # Pause first
        await client.post(
            f"{BASE_URL}/api/executions/workflows/{workflow_id}/executions/{execution_id}/pause",
            json={}
        )

        # Resume
        response = await client.post(
            f"{BASE_URL}/api/executions/workflows/{workflow_id}/executions/{execution_id}/resume",
            json={}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["execution_id"] == execution_id

    @pytest.mark.asyncio
    async def test_cancel_execution(self, client: httpx.AsyncClient, verify_service_running):
        """Test cancelling an execution"""
        # Create and execute workflow
        create_response = await client.post(
            f"{BASE_URL}/api/workflows",
            json=TEST_WORKFLOWS[0]
        )
        workflow_id = create_response.json()["id"]

        exec_response = await client.post(
            f"{BASE_URL}/api/executions/workflows/{workflow_id}/execute",
            json={
                "problem_statement": "Test problem for cancel"
            }
        )
        execution_id = exec_response.json()["execution_id"]

        # Cancel execution
        response = await client.post(
            f"{BASE_URL}/api/executions/workflows/{workflow_id}/executions/{execution_id}/cancel",
            json={}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["execution_id"] == execution_id
        assert data["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_get_execution_logs(self, client: httpx.AsyncClient, verify_service_running):
        """Test getting execution logs"""
        # Create and execute workflow
        create_response = await client.post(
            f"{BASE_URL}/api/workflows",
            json=TEST_WORKFLOWS[0]
        )
        workflow_id = create_response.json()["id"]

        exec_response = await client.post(
            f"{BASE_URL}/api/executions/workflows/{workflow_id}/execute",
            json={
                "problem_statement": "Test problem for logs"
            }
        )
        execution_id = exec_response.json()["execution_id"]

        # Get logs
        response = await client.get(
            f"{BASE_URL}/api/executions/workflows/{workflow_id}/executions/{execution_id}/logs"
        )

        assert response.status_code == 200
        data = response.json()

        assert "logs" in data
        assert "total" in data
        assert isinstance(data["logs"], list)


# ==================== Team Tests ====================

class TestTeams:
    """Test team management"""

    @pytest.mark.asyncio
    async def test_create_team(self, client: httpx.AsyncClient, verify_service_running):
        """Test creating a team"""
        response = await client.post(
            f"{BASE_URL}/api/teams",
            json=TEST_TEAM
        )

        assert response.status_code == 200
        data = response.json()

        assert "id" in data
        assert data["name"] == TEST_TEAM["name"]
        assert len(data["members"]) == len(TEST_TEAM["members"])
        assert "created_at" in data

    @pytest.mark.asyncio
    async def test_list_teams(self, client: httpx.AsyncClient, verify_service_running):
        """Test listing teams"""
        # Create a team first
        await client.post(
            f"{BASE_URL}/api/teams",
            json=TEST_TEAM
        )

        # List teams
        response = await client.get(f"{BASE_URL}/api/teams")

        assert response.status_code == 200
        data = response.json()

        assert "teams" in data
        assert "total" in data
        assert isinstance(data["teams"], list)

    @pytest.mark.asyncio
    async def test_get_team(self, client: httpx.AsyncClient, verify_service_running):
        """Test getting a specific team"""
        # Create a team
        create_response = await client.post(
            f"{BASE_URL}/api/teams",
            json=TEST_TEAM
        )
        team_id = create_response.json()["id"]

        # Get the team
        response = await client.get(f"{BASE_URL}/api/teams/{team_id}")

        assert response.status_code == 200
        data = response.json()

        assert data["id"] == team_id
        assert data["name"] == TEST_TEAM["name"]


# ==================== Gauntlet Tests ====================

class TestGauntlets:
    """Test gauntlet management"""

    @pytest.mark.asyncio
    async def test_create_gauntlet(self, client: httpx.AsyncClient, verify_service_running):
        """Test creating a gauntlet"""
        response = await client.post(
            f"{BASE_URL}/api/gauntlets",
            json=TEST_GAUNTLET
        )

        assert response.status_code == 200
        data = response.json()

        assert "id" in data
        assert data["name"] == TEST_GAUNTLET["name"]
        assert len(data["rounds"]) == len(TEST_GAUNTLET["rounds"])
        assert "created_at" in data

    @pytest.mark.asyncio
    async def test_list_gauntlets(self, client: httpx.AsyncClient, verify_service_running):
        """Test listing gauntlets"""
        # Create a gauntlet first
        await client.post(
            f"{BASE_URL}/api/gauntlets",
            json=TEST_GAUNTLET
        )

        # List gauntlets
        response = await client.get(f"{BASE_URL}/api/gauntlets")

        assert response.status_code == 200
        data = response.json()

        assert "gauntlets" in data
        assert "total" in data
        assert isinstance(data["gauntlets"], list)

    @pytest.mark.asyncio
    async def test_get_gauntlet(self, client: httpx.AsyncClient, verify_service_running):
        """Test getting a specific gauntlet"""
        # Create a gauntlet
        create_response = await client.post(
            f"{BASE_URL}/api/gauntlets",
            json=TEST_GAUNTLET
        )
        gauntlet_id = create_response.json()["id"]

        # Get the gauntlet
        response = await client.get(f"{BASE_URL}/api/gauntlets/{gauntlet_id}")

        assert response.status_code == 200
        data = response.json()

        assert data["id"] == gauntlet_id
        assert data["name"] == TEST_GAUNTLET["name"]


# ==================== Error Handling Tests ====================

class TestErrorHandling:
    """Test error handling"""

    @pytest.mark.asyncio
    async def test_404_not_found(self, client: httpx.AsyncClient, verify_service_running):
        """Test 404 error handling"""
        response = await client.get(f"{BASE_URL}/api/workflows/nonexistent-id")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_invalid_workflow_type(self, client: httpx.AsyncClient, verify_service_running):
        """Test invalid workflow type validation"""
        invalid_workflow = {
            "name": "Invalid Workflow",
            "description": "Test",
            "problem_statement": "Invalid workflow type should fail",
            "content_type": "text",
            "teams": [],
            "gauntlets": [],
            "workflow_type": "invalid_type"
        }

        response = await client.post(
            f"{BASE_URL}/api/workflows",
            json=invalid_workflow
        )

        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_missing_required_fields(self, client: httpx.AsyncClient, verify_service_running):
        """Test missing required fields validation"""
        incomplete_workflow = {
            "name": "Incomplete"
            # Missing problem_statement
        }

        response = await client.post(
            f"{BASE_URL}/api/workflows",
            json=incomplete_workflow
        )

        assert response.status_code == 422


# ==================== Performance Tests ====================

class TestPerformance:
    """Test API performance"""

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, client: httpx.AsyncClient, verify_service_running):
        """Test handling concurrent requests"""
        # Create multiple workflows concurrently
        tasks = [
            client.post(
                f"{BASE_URL}/api/workflows",
                json=TEST_WORKFLOWS[0]
            )
            for _ in range(10)
        ]

        responses = await asyncio.gather(*tasks)

        # All should succeed
        for response in responses:
            assert response.status_code == 201

    @pytest.mark.asyncio
    async def test_response_time(self, client: httpx.AsyncClient, verify_service_running):
        """Test API response time is acceptable"""
        start_time = time.time()

        response = await client.get(f"{BASE_URL}/health")

        end_time = time.time()
        response_time = end_time - start_time

        assert response.status_code == 200
        assert response_time < 1.0  # Should respond in under 1 second


# ==================== Run Tests ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
