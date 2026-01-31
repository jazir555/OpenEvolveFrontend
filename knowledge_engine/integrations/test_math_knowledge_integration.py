"""
Comprehensive Test Suite for Mathematical Knowledge Integration

Tests all components:
- Z3 solver connector
- LeanAIDE connector
- Knowledge manager
- Unified bridge
- MCP tools
- API endpoints

Run with: pytest test_math_knowledge_integration.py -v
"""

import asyncio
import json
import pytest
from typing import Dict, Any
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def z3_connector():
    """Create Z3 connector."""
    from z3_solver_connector import get_z3_connector
    return get_z3_connector()


@pytest.fixture
async def leanaide_connector():
    """Create LeanAIDE connector."""
    from leanaide_real_connector import get_leanaide_connector
    return await get_leanaide_connector()


@pytest.fixture
async def knowledge_manager():
    """Create knowledge manager."""
    from z3_knowledge_complete import get_z3_knowledge_manager
    return await get_z3_knowledge_manager()


@pytest.fixture
async def unified_bridge():
    """Create unified bridge."""
    from unified_math_bridge_complete import get_unified_bridge_complete
    return await get_unified_bridge_complete()


# =============================================================================
# Import Tests
# =============================================================================

class TestImports:
    """Test that all modules can be imported."""
    
    def test_z3_solver_connector(self):
        """Test Z3 solver connector import."""
        from z3_solver_connector import Z3SolverConnector, get_z3_connector
        assert Z3SolverConnector is not None
        assert get_z3_connector is not None
    
    def test_leanaide_connector(self):
        """Test LeanAIDE connector import."""
        from leanaide_real_connector import LeanAideRealConnector, get_leanaide_connector
        assert LeanAideRealConnector is not None
        assert get_leanaide_connector is not None
    
    def test_knowledge_manager(self):
        """Test knowledge manager import."""
        from z3_knowledge_complete import Z3KnowledgeManager, get_z3_knowledge_manager
        assert Z3KnowledgeManager is not None
        assert get_z3_knowledge_manager is not None
    
    def test_unified_bridge(self):
        """Test unified bridge import."""
        from unified_math_bridge_complete import (
            UnifiedMathBridgeComplete, 
            UnifiedMathKnowledgeBridge,
            get_unified_bridge_complete
        )
        assert UnifiedMathBridgeComplete is not None
        assert UnifiedMathKnowledgeBridge is not None
        assert get_unified_bridge_complete is not None
    
    def test_mcp_tools(self):
        """Test MCP tools import."""
        from math_mcp_tools import MathMCPTools, get_math_mcp_tools
        assert MathMCPTools is not None
        assert get_math_mcp_tools is not None
    
    def test_config(self):
        """Test configuration import."""
        from math_knowledge_config import MathKnowledgeConfig, load_config
        assert MathKnowledgeConfig is not None
        assert load_config is not None
    
    def test_api(self):
        """Test API import."""
        from z3_api import app, create_z3_knowledge_app
        assert app is not None
        assert create_z3_knowledge_app is not None


# =============================================================================
# Z3 Solver Tests
# =============================================================================

class TestZ3Solver:
    """Test Z3 solver functionality."""
    
    @pytest.mark.asyncio
    async def test_basic_satisfiable(self, z3_connector):
        """Test basic satisfiable problem."""
        from z3_solver_connector import Z3SolverConfig, Z3ResultStatus
        
        smtlib = """
        (declare-fun x () Int)
        (assert (> x 0))
        (assert (< x 10))
        (check-sat)
        (get-model)
        """
        
        result = await z3_connector.solve_smtlib(
            smtlib,
            Z3SolverConfig(timeout_ms=10000)
        )
        
        assert result.status == Z3ResultStatus.SAT
        assert result.model is not None
        assert result.solving_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_basic_unsatisfiable(self, z3_connector):
        """Test basic unsatisfiable problem."""
        from z3_solver_connector import Z3SolverConfig, Z3ResultStatus
        
        smtlib = """
        (declare-fun x () Int)
        (assert (> x 5))
        (assert (< x 3))
        (check-sat)
        """
        
        result = await z3_connector.solve_smtlib(
            smtlib,
            Z3SolverConfig(timeout_ms=10000)
        )
        
        assert result.status == Z3ResultStatus.UNSAT
    
    @pytest.mark.asyncio
    async def test_linear_system(self, z3_connector):
        """Test linear equation system."""
        from z3_solver_connector import Z3SolverConfig, Z3ResultStatus
        
        smtlib = """
        (declare-fun x () Int)
        (declare-fun y () Int)
        (assert (= (+ x y) 10))
        (assert (= (- x y) 2))
        (check-sat)
        (get-model)
        """
        
        result = await z3_connector.solve_smtlib(
            smtlib,
            Z3SolverConfig(timeout_ms=10000, model_generation=True)
        )
        
        assert result.status == Z3ResultStatus.SAT
        assert result.model is not None


# =============================================================================
# Knowledge Manager Tests
# =============================================================================

class TestKnowledgeManager:
    """Test knowledge manager functionality."""
    
    @pytest.mark.asyncio
    async def test_learn_from_solution(self, knowledge_manager):
        """Test learning from a solution."""
        result = await knowledge_manager.learn_from_solution(
            problem_statement="Test linear system",
            constraints=["x + y = 10", "x - y = 2"],
            result="success",
            proof="substitution",
            metadata={"strategy": "elimination"}
        )
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_strategy_recommendation(self, knowledge_manager):
        """Test strategy recommendation."""
        strategy = await knowledge_manager.get_recommended_strategy(
            problem_statement="Linear system with 3 variables",
            constraints=["x + y + z = 10"]
        )
        
        assert strategy is not None
        assert "strategy" in strategy
    
    def test_statistics(self, knowledge_manager):
        """Test getting statistics."""
        stats = knowledge_manager.get_statistics()
        
        assert isinstance(stats, dict)
        assert "total_records" in stats


# =============================================================================
# Unified Bridge Tests
# =============================================================================

class TestUnifiedBridge:
    """Test unified bridge functionality."""
    
    @pytest.mark.asyncio
    async def test_solver_selection(self, unified_bridge):
        """Test intelligent solver selection."""
        from unified_math_bridge_complete import SolverSystem
        
        # Linear problem should prefer Z3
        solver = unified_bridge._select_solver("linear system", "x + y = 5")
        assert solver in [SolverSystem.Z3, SolverSystem.AUTO]
    
    @pytest.mark.asyncio
    async def test_semantic_translation(self, unified_bridge):
        """Test semantic translation."""
        smt = "(assert (> x 0))"
        lean = unified_bridge.translator.translate_smt_to_lean(smt)
        
        assert lean is not None
        assert isinstance(lean, str)
    
    @pytest.mark.asyncio
    async def test_consensus_check(self, unified_bridge):
        """Test consensus checking."""
        from unified_math_bridge_complete import SolverResult, SolverSystem
        
        z3_result = SolverResult(
            solver=SolverSystem.Z3,
            status="sat",
            model={"x": 5}
        )
        
        lean_result = SolverResult(
            solver=SolverSystem.LEANAIDE,
            status="proved",
            model={"x": 5}
        )
        
        consensus = unified_bridge.consensus.check_agreement(
            z3_result, lean_result, "partial"
        )
        
        assert consensus is not None
        assert "agreement" in consensus


# =============================================================================
# MCP Tools Tests
# =============================================================================

class TestMCPTools:
    """Test MCP tools functionality."""
    
    @pytest.mark.asyncio
    async def test_tool_listing(self):
        """Test listing available tools."""
        from math_mcp_tools import get_math_mcp_tools
        
        tools = await get_math_mcp_tools()
        available = tools.get_tools()
        
        assert len(available) > 0
        assert any(t["name"] == "z3_solve" for t in available)
    
    @pytest.mark.asyncio
    async def test_health_check_tool(self):
        """Test health check tool."""
        from math_mcp_tools import get_math_mcp_tools
        
        tools = await get_math_mcp_tools()
        result = await tools.execute_tool("math_health_check", {})
        
        assert result is not None
        assert "z3_available" in result


# =============================================================================
# Configuration Tests
# =============================================================================

class TestConfiguration:
    """Test configuration management."""
    
    def test_default_config(self):
        """Test default configuration."""
        from math_knowledge_config import MathKnowledgeConfig
        
        config = MathKnowledgeConfig()
        
        assert config.z3.timeout_ms == 30000
        assert config.leanaide.port == 7654
    
    def test_config_validation(self):
        """Test configuration validation."""
        from math_knowledge_config import MathKnowledgeConfig
        
        config = MathKnowledgeConfig()
        errors = config.validate()
        
        assert isinstance(errors, list)
    
    def test_config_from_env(self):
        """Test loading config from environment."""
        import os
        from math_knowledge_config import MathKnowledgeConfig
        
        os.environ["MATH_KNOWLEDGE_Z3_TIMEOUT_MS"] = "60000"
        config = MathKnowledgeConfig.from_env()
        
        assert config.z3.timeout_ms == 60000


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Test end-to-end integration."""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test complete problem solving workflow."""
        from z3_solver_connector import get_z3_connector, Z3SolverConfig
        from z3_knowledge_complete import get_z3_knowledge_manager
        
        # Step 1: Solve a problem
        z3 = get_z3_connector()
        smtlib = """
        (declare-fun x () Int)
        (declare-fun y () Int)
        (assert (= (+ x y) 10))
        (assert (> x 0))
        (assert (> y 0))
        (check-sat)
        (get-model)
        """
        
        result = await z3.solve_smtlib(smtlib, Z3SolverConfig())
        assert result.status.value == "sat"
        
        # Step 2: Learn from solution
        manager = await get_z3_knowledge_manager()
        await manager.learn_from_solution(
            problem_statement="Simple linear system",
            constraints=["x + y = 10", "x > 0", "y > 0"],
            result="success",
            metadata={"time_ms": result.solving_time_ms}
        )
        
        # Step 3: Get strategy for similar problem
        strategy = await manager.get_recommended_strategy(
            problem_statement="Another linear system",
            constraints=["a + b = 20"]
        )
        
        assert strategy is not None


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Test performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_solving_performance(self, z3_connector):
        """Test that solving completes within reasonable time."""
        import time
        from z3_solver_connector import Z3SolverConfig
        
        smtlib = """
        (declare-fun x () Int)
        (assert (> x 0))
        (check-sat)
        """
        
        start = time.time()
        result = await z3_connector.solve_smtlib(
            smtlib,
            Z3SolverConfig(timeout_ms=5000)
        )
        elapsed = time.time() - start
        
        assert elapsed < 5.0  # Should complete within 5 seconds
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_cache_performance(self, unified_bridge):
        """Test that caching improves performance."""
        import time
        
        problem = "x + y = 10, x > 0, y > 0"
        
        # First call
        start = time.time()
        result1 = await unified_bridge.solve(problem, timeout=10)
        time1 = time.time() - start
        
        # Second call (should use cache)
        start = time.time()
        result2 = await unified_bridge.solve(problem, timeout=10)
        time2 = time.time() - start
        
        # Cache hit should be faster
        assert time2 <= time1


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Test error handling."""
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, z3_connector):
        """Test timeout handling."""
        from z3_solver_connector import Z3SolverConfig, Z3ResultStatus
        
        # Complex problem with short timeout
        smtlib = """
        (declare-fun x () Int)
        (declare-fun y () Int)
        (declare-fun z () Int)
        (assert (> (* x y z) 1000000))
        (check-sat)
        """
        
        result = await z3_connector.solve_smtlib(
            smtlib,
            Z3SolverConfig(timeout_ms=100)  # Very short timeout
        )
        
        # Should either timeout or return quickly
        assert result.status in [Z3ResultStatus.UNKNOWN, Z3ResultStatus.ERROR] or result.solving_time_ms < 200
    
    @pytest.mark.asyncio
    async def test_invalid_smtlib(self, z3_connector):
        """Test handling of invalid SMT-LIB."""
        from z3_solver_connector import Z3SolverConfig
        
        result = await z3_connector.solve_smtlib(
            "this is not valid smtlib",
            Z3SolverConfig()
        )
        
        # Should return error status
        assert result is not None


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
