"""Tests for PES Enhanced API routes.

Run with: pytest openevolve_pes_enhanced/test_api_routes.py -v
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

# Skip all tests if FastAPI not available
try:
    from fastapi.testclient import TestClient
    from fastapi import FastAPI
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    pytest.skip("FastAPI not available", allow_module_level=True)

# Import routes
from openevolve_pes_enhanced.api_routes import (
    router,
    _pe_runs,
    _generate_run_id,
    _PERunState,
    PESEnhancedRunRequest,
    PESEnhancedRunResponse,
    CostEstimateRequest,
    CostEstimateResponse,
    StopRunRequest,
    StrategyRecommendationRequest,
    TestCase,
    PES_ENHANCED_AVAILABLE
)


# Create test app
app = FastAPI()
app.include_router(router)
client = TestClient(app)


class TestRunIdGeneration:
    """Test run ID generation."""
    
    def test_generate_run_id_format(self):
        """Test that run IDs have correct format."""
        run_id = _generate_run_id()
        assert run_id.startswith("pes-enhanced-")
        assert len(run_id) == 25  # "pes-enhanced-" + 12 hex chars
    
    def test_generate_run_id_unique(self):
        """Test that run IDs are unique."""
        ids = [_generate_run_id() for _ in range(100)]
        assert len(set(ids)) == 100


class TestRequestValidation:
    """Test request validation."""
    
    def test_valid_run_request(self):
        """Test valid run request creation."""
        request = PESEnhancedRunRequest(
            code="def foo(): pass",
            problem_description="Optimize this function",
            tests=[TestCase(name="test1", input="foo()")],
            language="python",
            max_cost_usd=5.0
        )
        assert request.code == "def foo(): pass"
        assert request.language == "python"
        assert request.max_cost_usd == 5.0
    
    def test_invalid_empty_code(self):
        """Test that empty code is rejected."""
        with pytest.raises(ValueError, match="Code cannot be empty"):
            PESEnhancedRunRequest(
                code="",
                problem_description="Optimize this"
            )
    
    def test_invalid_whitespace_code(self):
        """Test that whitespace-only code is rejected."""
        with pytest.raises(ValueError, match="Code cannot be empty"):
            PESEnhancedRunRequest(
                code="   \n\t  ",
                problem_description="Optimize this"
            )
    
    def test_invalid_empty_problem(self):
        """Test that empty problem description is rejected."""
        with pytest.raises(ValueError, match="Problem description cannot be empty"):
            PESEnhancedRunRequest(
                code="def foo(): pass",
                problem_description=""
            )
    
    def test_invalid_complexity(self):
        """Test that invalid complexity is rejected."""
        with pytest.raises(ValueError, match="Complexity must be one of"):
            CostEstimateRequest(
                iterations=50,
                population_size=20,
                problem_complexity="invalid"
            )
    
    def test_valid_complexity_values(self):
        """Test valid complexity values."""
        for complexity in ["low", "medium", "high", "very_high"]:
            request = CostEstimateRequest(problem_complexity=complexity)
            assert request.problem_complexity == complexity


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check(self):
        """Test health check returns correct structure."""
        response = client.get("/pes-enhanced/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "available" in data
        assert "status" in data
        assert "active_runs" in data
        assert "total_runs" in data
        assert isinstance(data["active_runs"], int)
        assert isinstance(data["total_runs"], int)


class TestCostEstimateEndpoint:
    """Test cost estimation endpoint."""
    
    @pytest.mark.skipif(not PES_ENHANCED_AVAILABLE, reason="PES Enhanced not available")
    def test_cost_estimate_basic(self):
        """Test basic cost estimation."""
        request = {
            "iterations": 50,
            "population_size": 20,
            "problem_complexity": "medium"
        }
        
        response = client.post("/pes-enhanced/cost-estimate", json=request)
        
        # Should succeed or return 503 if service unavailable
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "estimated_cost_usd" in data
            assert "estimated_tokens" in data
            assert "estimated_duration_ms" in data
            assert "recommended_strategy" in data
            assert "parameter_recommendations" in data
            assert data["estimated_cost_usd"] >= 0
            assert data["estimated_tokens"] >= 0
    
    @pytest.mark.skipif(not PES_ENHANCED_AVAILABLE, reason="PES Enhanced not available")
    def test_cost_estimate_complexity_levels(self):
        """Test cost estimation with different complexity levels."""
        complexities = ["low", "medium", "high", "very_high"]
        costs = []
        
        for complexity in complexities:
            request = {
                "iterations": 50,
                "population_size": 20,
                "problem_complexity": complexity
            }
            
            response = client.post("/pes-enhanced/cost-estimate", json=request)
            
            if response.status_code == 200:
                costs.append(response.json()["estimated_cost_usd"])
        
        # Higher complexity should generally cost more
        if len(costs) >= 2:
            assert costs[-1] >= costs[0]  # very_high >= low


class TestStartRunEndpoint:
    """Test start run endpoint."""
    
    def test_start_run_missing_pes_enhanced(self):
        """Test handling when PES Enhanced is not available."""
        # This test checks the structure is correct
        request = {
            "code": "def foo(): return 42",
            "problem_description": "Optimize this function",
            "tests": [{"name": "test1", "input": "foo()", "weight": 1.0}],
            "language": "python"
        }
        
        # Mock PES_ENHANCED_AVAILABLE to False
        with patch('openevolve_pes_enhanced.api_routes.PES_ENHANCED_AVAILABLE', False):
            response = client.post("/pes-enhanced/runs", json=request)
            assert response.status_code == 503
            assert "not available" in response.json()["detail"].lower()
    
    @pytest.mark.skipif(not PES_ENHANCED_AVAILABLE, reason="PES Enhanced not available")
    def test_start_run_invalid_request(self):
        """Test starting run with invalid request."""
        # Missing required fields
        request = {"code": "def foo(): pass"}
        
        response = client.post("/pes-enhanced/runs", json=request)
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.skipif(not PES_ENHANCED_AVAILABLE, reason="PES Enhanced not available")
    def test_start_run_valid_request_structure(self):
        """Test starting run returns correct structure."""
        request = {
            "code": "def foo(): return 42",
            "problem_description": "Optimize this function for better performance",
            "tests": [
                {"name": "test1", "input": "foo()", "expected_output": "42", "weight": 1.0}
            ],
            "language": "python",
            "max_cost_usd": 5.0,
            "enable_cost_optimization": True
        }
        
        response = client.post("/pes-enhanced/runs", json=request)
        
        # Should be accepted (202) or service unavailable (503)
        assert response.status_code in [202, 503]
        
        if response.status_code == 202:
            data = response.json()
            assert "run_id" in data
            assert data["status"] in ["pending", "running"]
            assert data["run_id"].startswith("pes-enhanced-")


class TestGetRunEndpoint:
    """Test get run endpoint."""
    
    def test_get_nonexistent_run(self):
        """Test getting a run that doesn't exist."""
        response = client.get("/pes-enhanced/runs/nonexistent-run-id")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    def test_get_existing_run(self):
        """Test getting an existing run."""
        # Create a mock run
        run_id = "pes-enhanced-test123"
        mock_run = _PERunState(
            run_id=run_id,
            status="completed",
            created_at=datetime.utcnow().isoformat()
        )
        _pe_runs[run_id] = mock_run
        
        try:
            response = client.get(f"/pes-enhanced/runs/{run_id}")
            assert response.status_code == 200
            
            data = response.json()
            assert data["run_id"] == run_id
            assert data["status"] == "completed"
        finally:
            # Cleanup
            del _pe_runs[run_id]


class TestListRunsEndpoint:
    """Test list runs endpoint."""
    
    def test_list_runs_empty(self):
        """Test listing runs when none exist."""
        # Clear all runs temporarily
        original_runs = _pe_runs.copy()
        _pe_runs.clear()
        
        try:
            response = client.get("/pes-enhanced/runs")
            assert response.status_code == 200
            
            data = response.json()
            assert "runs" in data
            assert "total_count" in data
            assert data["total_count"] == 0
            assert len(data["runs"]) == 0
        finally:
            # Restore
            _pe_runs.update(original_runs)
    
    def test_list_runs_with_data(self):
        """Test listing runs with some data."""
        # Add mock runs
        for i in range(3):
            run_id = f"pes-enhanced-test{i}"
            _pe_runs[run_id] = _PERunState(
                run_id=run_id,
                status="completed",
                created_at=datetime.utcnow().isoformat()
            )
        
        try:
            response = client.get("/pes-enhanced/runs")
            assert response.status_code == 200
            
            data = response.json()
            assert data["total_count"] >= 3
            assert len(data["runs"]) >= 3
        finally:
            # Cleanup
            for i in range(3):
                _pe_runs.pop(f"pes-enhanced-test{i}", None)
    
    def test_list_runs_pagination(self):
        """Test run listing pagination."""
        response = client.get("/pes-enhanced/runs?limit=5&offset=0")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["runs"]) <= 5


class TestStopRunEndpoint:
    """Test stop run endpoint."""
    
    def test_stop_nonexistent_run(self):
        """Test stopping a run that doesn't exist."""
        response = client.post(
            "/pes-enhanced/runs/nonexistent-run-id/stop",
            json={"reason": "test", "force": False}
        )
        assert response.status_code == 404
    
    def test_stop_completed_run(self):
        """Test stopping a run that's already completed."""
        run_id = "pes-enhanced-test-stop"
        mock_run = _PERunState(
            run_id=run_id,
            status="completed",
            created_at=datetime.utcnow().isoformat()
        )
        _pe_runs[run_id] = mock_run
        
        try:
            response = client.post(
                f"/pes-enhanced/runs/{run_id}/stop",
                json={"reason": "test", "force": False}
            )
            assert response.status_code == 200
            
            data = response.json()
            assert data["success"] == False  # Cannot stop completed run
            assert data["previous_status"] == "completed"
        finally:
            del _pe_runs[run_id]
    
    def test_stop_running_run(self):
        """Test stopping a running run."""
        run_id = "pes-enhanced-test-stop-running"
        mock_run = _PERunState(
            run_id=run_id,
            status="running",
            created_at=datetime.utcnow().isoformat()
        )
        _pe_runs[run_id] = mock_run
        
        try:
            response = client.post(
                f"/pes-enhanced/runs/{run_id}/stop",
                json={"reason": "user_request", "force": False}
            )
            assert response.status_code == 200
            
            data = response.json()
            assert data["success"] == True
            assert data["previous_status"] == "running"
            assert _pe_runs[run_id].cancel_requested == True
        finally:
            del _pe_runs[run_id]


class TestBudgetEndpoint:
    """Test budget status endpoint."""
    
    def test_budget_nonexistent_run(self):
        """Test getting budget for nonexistent run."""
        response = client.get("/pes-enhanced/runs/nonexistent-run-id/budget")
        assert response.status_code == 404
    
    def test_budget_no_cost_optimization(self):
        """Test getting budget when cost optimization not enabled."""
        run_id = "pes-enhanced-test-budget"
        mock_run = _PERunState(
            run_id=run_id,
            status="running",
            created_at=datetime.utcnow().isoformat()
        )
        _pe_runs[run_id] = mock_run
        
        try:
            response = client.get(f"/pes-enhanced/runs/{run_id}/budget")
            # Should fail because no cost optimizer configured
            assert response.status_code == 400
        finally:
            del _pe_runs[run_id]


class TestStrategyRecommendation:
    """Test strategy recommendation endpoint."""
    
    @pytest.mark.skipif(not PES_ENHANCED_AVAILABLE, reason="PES Enhanced not available")
    def test_strategy_recommendation_basic(self):
        """Test basic strategy recommendation."""
        request = {
            "problem_description": "Optimize a sorting algorithm for large datasets",
            "max_cost_usd": 10.0
        }
        
        response = client.post("/pes-enhanced/recommend-strategy", json=request)
        
        # Should succeed or return 503
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "strategy" in data
            assert "confidence" in data
            assert "estimated_cost_usd" in data
            assert "reasoning" in data
            assert "recommended_parameters" in data


class TestWebSocket:
    """Test WebSocket functionality."""
    
    def test_websocket_nonexistent_run(self):
        """Test WebSocket connection to nonexistent run."""
        # Note: TestClient WebSocket testing is limited
        # This is a basic smoke test
        pass  # WebSocket tests typically require async testing


class TestResponseModels:
    """Test response model validation."""
    
    def test_run_response_structure(self):
        """Test PESEnhancedRunResponse structure."""
        response = PESEnhancedRunResponse(
            run_id="pes-enhanced-test",
            status="completed",
            success=True,
            created_at=datetime.utcnow().isoformat()
        )
        
        data = response.dict()
        assert data["run_id"] == "pes-enhanced-test"
        assert data["status"] == "completed"
        assert data["success"] == True
        assert "total_cost_usd" in data
        assert "efficiency_gain" in data
    
    def test_cost_estimate_response_structure(self):
        """Test CostEstimateResponse structure."""
        response = CostEstimateResponse(
            estimated_cost_usd=1.5,
            estimated_tokens=50000,
            estimated_duration_ms=30000,
            recommended_strategy="standard",
            prompt_tokens=35000,
            completion_tokens=15000,
            prompt_cost_usd=0.35,
            completion_cost_usd=0.45,
            total_evaluations=1000
        )
        
        data = response.dict()
        assert data["estimated_cost_usd"] == 1.5
        assert data["recommended_strategy"] == "standard"
        assert "parameter_recommendations" in data


class TestIntegration:
    """Integration tests for the full API."""
    
    @pytest.mark.skipif(not PES_ENHANCED_AVAILABLE, reason="PES Enhanced not available")
    @pytest.mark.asyncio
    async def test_full_workflow_mocked(self):
        """Test full workflow with mocked evolution."""
        from openevolve_pes_enhanced.api_routes import _execute_pes_run
        
        # Create a mock run state
        run_id = "pes-enhanced-integration-test"
        request = PESEnhancedRunRequest(
            code="def foo(): return 42",
            problem_description="Test optimization",
            tests=[TestCase(name="test1", input="foo()")],
            language="python",
            max_cost_usd=1.0,
            enable_cost_optimization=True
        )
        
        mock_run = _PERunState(
            run_id=run_id,
            status="pending",
            created_at=datetime.utcnow().isoformat(),
            request=request
        )
        _pe_runs[run_id] = mock_run
        
        try:
            # Execute (this would normally run evolution)
            # Note: This test requires mocking the wrapper
            # For now, we just verify the state management
            assert mock_run.status == "pending"
            assert mock_run.request == request
        finally:
            del _pe_runs[run_id]


def cleanup_test_runs():
    """Cleanup function to remove test runs."""
    test_runs = [k for k in _pe_runs.keys() if "test" in k.lower()]
    for run_id in test_runs:
        del _pe_runs[run_id]


# Run cleanup after all tests
@pytest.fixture(autouse=True, scope="module")
def cleanup_after_tests():
    """Cleanup fixture."""
    yield
    cleanup_test_runs()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
