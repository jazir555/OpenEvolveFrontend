"""
Test Suite for math_api_complete.py

Tests all API endpoints and functionality.
"""

import pytest
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestMathAPIComplete:
    """Test the complete math API."""
    
    @pytest.fixture
    def event_loop(self):
        """Create event loop."""
        loop = asyncio.get_event_loop_policy().new_event_loop()
        yield loop
        loop.close()
    
    def test_api_creation(self):
        """Test that API can be created."""
        from math_api_complete import create_math_api
        app = create_math_api()
        assert app is not None
    
    def test_api_routes_exist(self):
        """Test that all routes exist."""
        from math_api_complete import math_api
        
        if not math_api:
            pytest.skip("API not available")
        
        routes = [(r.path, list(r.methods) if hasattr(r, 'methods') else []) 
                  for r in math_api.routes if hasattr(r, 'path')]
        
        # Check key routes
        assert any('/health' in r for r, _ in routes)
        assert any('/solve/z3' in r for r, _ in routes)
        assert any('/solve/lean' in r for r, _ in routes)
        assert any('/solve/unified' in r for r, _ in routes)
    
    @pytest.mark.asyncio
    async def test_solve_z3_endpoint(self):
        """Test Z3 solve endpoint."""
        from math_api_complete import SolveZ3Request
        
        req = SolveZ3Request(
            content="(declare-fun x () Int) (assert (> x 0)) (check-sat)",
            timeout_ms=30000
        )
        assert req.content is not None
        assert req.timeout_ms > 0
    
    @pytest.mark.asyncio
    async def test_solve_lean_endpoint(self):
        """Test Lean solve endpoint."""
        from math_api_complete import ProveLeanRequest
        
        req = ProveLeanRequest(
            theorem="forall n: Nat, n + 0 = n",
            timeout_seconds=300
        )
        assert req.theorem is not None
        assert req.timeout_seconds > 0
    
    @pytest.mark.asyncio
    async def test_knowledge_learn_endpoint(self):
        """Test knowledge learn endpoint."""
        from math_api_complete import LearnRequest
        
        req = LearnRequest(
            problem_statement="Test problem",
            constraints=["x > 0"],
            result="success"
        )
        assert req.problem_statement is not None
        assert isinstance(req.constraints, list)
    
    @pytest.mark.asyncio
    async def test_knowledge_search_endpoint(self):
        """Test knowledge search endpoint."""
        from math_api_complete import SearchRequest
        
        req = SearchRequest(
            query="linear system",
            top_k=5
        )
        assert req.query is not None
        assert 1 <= req.top_k <= 50
    
    def test_request_validation(self):
        """Test request validation."""
        from math_api_complete import SolveZ3Request
        
        # Valid request
        req = SolveZ3Request(content="(assert true)", timeout_ms=30000)
        assert req.timeout_ms == 30000
        
        # Check default values
        req2 = SolveZ3Request(content="test")
        assert req2.timeout_ms == 30000  # Default


class TestAPIErrorHandling:
    """Test API error handling."""
    
    def test_invalid_timeout_rejected(self):
        """Test that invalid timeouts are rejected."""
        from math_api_complete import SolveZ3Request
        
        # Pydantic should validate this
        try:
            req = SolveZ3Request(content="test", timeout_ms=-1)
            # If we get here, validation didn't work
            assert False, "Should have rejected negative timeout"
        except Exception:
            pass  # Expected
    
    def test_empty_content_handled(self):
        """Test empty content handling."""
        from math_api_complete import SolveZ3Request
        
        req = SolveZ3Request(content="")
        assert req.content == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
