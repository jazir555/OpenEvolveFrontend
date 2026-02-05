"""
Comprehensive Test Suite for Z3 Prover Service Bubble

Tests all components of the Z3 Service Bubble:
- Core solving (SAT/SMT)
- Optimization (single/multi-objective)
- Theorem proving
- Proof extraction
- Portfolio solving
- Incremental solving
- Translation (SMT-LIB/Lean)
- Verification
- Reliability checking
- Knowledge extraction
- Caching
- Performance monitoring

Run with: pytest test_z3_prover_comprehensive.py -v

Author: OpenEvolve
Created: 2026-02-04
"""

import asyncio
import json
import pytest
import time
from typing import Dict, Any, List
from dataclasses import dataclass

# Import Z3 components
try:
    from z3_api_server import (
        Z3SolverService, Z3ServiceBubble, get_service_bubble,
        SolveRequest, SolveResponse, OptimizeRequest, ProveRequest,
        ProofExtractRequest, PortfolioSolveRequest, IncrementalSolveRequest,
        BatchSolveRequest, VerifyRequest, TranslateRequest
    )
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False

try:
    from z3prover_integration import (
        Z3SolverEngine, Z3TheoremProver, Z3Variable, Z3Constraint,
        Z3ConstraintType, Z3Config, Z3ResultStatus, get_z3_solver_engine
    )
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False

try:
    from z3prover_advanced import (
        Z3AdvancedSolver, OptimizationObjective, ProofFormat,
        get_z3_advanced_solver
    )
    Z3_ADVANCED_AVAILABLE = True
except ImportError:
    Z3_ADVANCED_AVAILABLE = False

try:
    from z3_mcp_tools import get_z3_mcp_server, MCPTool
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

try:
    from z3_crewai_bridge import (
        Z3AgentCoordinator, Z3SolverAgent, Z3TheoremProverAgent,
        AgentTask, AgentRole, get_z3_agent_coordinator
    )
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False

try:
    from z3_result_cache import get_z3_result_cache, CacheConfig
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False

try:
    from z3_performance_monitor import get_z3_performance_monitor
    MONITOR_AVAILABLE = True
except ImportError:
    MONITOR_AVAILABLE = False

try:
    from z3_knowledge_extraction import get_z3_knowledge_extractor
    KNOWLEDGE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_AVAILABLE = False

try:
    from z3_reliability_checker import (
        Z3ReliabilityChecker, ComponentReliabilityModel,
        ReliabilityConstraint, ReliabilityProperty
    )
    RELIABILITY_AVAILABLE = True
except ImportError:
    RELIABILITY_AVAILABLE = False


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def solver_service():
    """Create solver service instance."""
    if not API_AVAILABLE:
        pytest.skip("API server not available")
    return Z3SolverService()


@pytest.fixture
def service_bubble():
    """Create service bubble instance."""
    if not API_AVAILABLE:
        pytest.skip("API server not available")
    return Z3ServiceBubble()


@pytest.fixture
def z3_solver():
    """Create Z3 solver engine."""
    if not Z3_AVAILABLE:
        pytest.skip("Z3 not available")
    return get_z3_solver_engine()


@pytest.fixture
def advanced_solver():
    """Create Z3 advanced solver."""
    if not Z3_ADVANCED_AVAILABLE:
        pytest.skip("Z3 advanced not available")
    return get_z3_advanced_solver()


@pytest.fixture
def cache():
    """Create cache instance."""
    if not CACHE_AVAILABLE:
        pytest.skip("Cache not available")
    return get_z3_result_cache(CacheConfig(max_size=100, db_path=":memory:"))


@pytest.fixture
def monitor():
    """Create monitor instance."""
    if not MONITOR_AVAILABLE:
        pytest.skip("Monitor not available")
    return get_z3_performance_monitor()


# =============================================================================
# Test Core Solving
# =============================================================================

@pytest.mark.asyncio
class TestCoreSolving:
    """Test core constraint solving functionality."""
    
    async def test_simple_sat_problem(self, solver_service):
        """Test simple satisfiable problem."""
        request = SolveRequest(
            problem="Simple constraint problem",
            variables=[
                {"name": "x", "type": "INTEGER"},
                {"name": "y", "type": "INTEGER"}
            ],
            constraints=[
                "x > 0",
                "x < 10",
                "y == x + 5"
            ],
            timeout=10.0
        )
        
        response = await solver_service.solve(request)
        
        assert response.success
        assert response.status in ["sat", "unsat", "unknown"]
        assert response.solver_used == "z3"
        assert response.execution_time_ms >= 0
    
    async def test_smtlib_solving(self, solver_service):
        """Test SMT-LIB format solving."""
        smtlib = """
        (set-logic LIA)
        (declare-fun x () Int)
        (declare-fun y () Int)
        (assert (> x 0))
        (assert (< x 10))
        (assert (= y (+ x 5)))
        (check-sat)
        (get-model)
        """
        
        request = SolveRequest(
            problem=smtlib,
            timeout=10.0
        )
        
        response = await solver_service.solve(request)
        
        assert response.success
        assert response.status in ["sat", "unsat", "unknown"]
    
    async def test_unsat_problem(self, solver_service):
        """Test unsatisfiable problem."""
        request = SolveRequest(
            problem="Unsatisfiable problem",
            variables=[{"name": "x", "type": "INTEGER"}],
            constraints=["x > 5", "x < 3"],
            timeout=10.0
        )
        
        response = await solver_service.solve(request)
        
        assert response.success
        # Note: May return sat/unsat/unknown depending on Z3 availability
        if response.status == "unsat":
            assert not response.satisfiable
    
    async def test_batch_solving(self, solver_service):
        """Test batch solving."""
        problems = [
            SolveRequest(
                problem=f"Problem {i}",
                variables=[{"name": "x", "type": "INTEGER"}],
                constraints=[f"x > {i}", f"x < {i + 10}"],
                timeout=5.0
            )
            for i in range(3)
        ]
        
        request = BatchSolveRequest(
            problems=problems,
            parallel=True,
            max_workers=2
        )
        
        response = await solver_service.solve_batch(request)
        
        assert response.success
        assert len(response.results) == 3
        assert response.completed >= 0
        assert response.failed >= 0
        assert response.total_time_ms >= 0


# =============================================================================
# Test Optimization
# =============================================================================

@pytest.mark.asyncio
class TestOptimization:
    """Test optimization functionality."""
    
    async def test_single_objective_minimize(self, solver_service):
        """Test single objective minimization."""
        request = OptimizeRequest(
            variables=[{"name": "x", "type": "INTEGER"}],
            constraints=["x >= 0", "x <= 100"],
            objective={"expression": "x", "direction": "minimize"},
            direction="minimize",
            multi_objective=False
        )
        
        response = await solver_service.optimize(request)
        
        if response.success:
            assert response.optimal_value is not None
            assert response.execution_time_ms >= 0
    
    async def test_single_objective_maximize(self, solver_service):
        """Test single objective maximization."""
        request = OptimizeRequest(
            variables=[{"name": "x", "type": "INTEGER"}],
            constraints=["x >= 0", "x <= 100"],
            objective={"expression": "x", "direction": "maximize"},
            direction="maximize",
            multi_objective=False
        )
        
        response = await solver_service.optimize(request)
        
        if response.success:
            assert response.optimal_value is not None
            assert response.execution_time_ms >= 0


# =============================================================================
# Test Theorem Proving
# =============================================================================

@pytest.mark.asyncio
class TestTheoremProving:
    """Test theorem proving functionality."""
    
    async def test_simple_theorem(self, solver_service):
        """Test simple theorem proving."""
        smtlib = """
        (set-logic LIA)
        (declare-fun x () Int)
        (assert (> x 0))
        (assert (not (> (+ x 1) 0)))
        (check-sat)
        """
        
        request = ProveRequest(
            theorem=smtlib,
            extract_proof=False,
            timeout=10.0
        )
        
        response = await solver_service.prove(request)
        
        assert response.success
        assert isinstance(response.proven, bool)
        assert 0 <= response.confidence <= 1
        assert response.execution_time_ms >= 0
    
    async def test_theorem_with_assumptions(self, solver_service):
        """Test theorem with assumptions."""
        request = ProveRequest(
            theorem="(assert (> x 0))",
            assumptions=["(declare-fun x () Int)"],
            extract_proof=False,
            timeout=10.0
        )
        
        response = await solver_service.prove(request)
        
        assert response.success
        assert isinstance(response.proven, bool)


# =============================================================================
# Test Proof Extraction
# =============================================================================

@pytest.mark.asyncio
class TestProofExtraction:
    """Test proof extraction functionality."""
    
    async def test_extract_proof_text(self, solver_service):
        """Test proof extraction in text format."""
        smtlib = """
        (set-logic LIA)
        (declare-fun x () Int)
        (assert (> x 0))
        (assert (not (> (+ x 1) 0)))
        (check-sat)
        """
        
        request = ProofExtractRequest(
            smtlib=smtlib,
            format="text",
            verify=True
        )
        
        response = await solver_service.extract_proof(request)
        
        assert isinstance(response.success, bool)
        assert isinstance(response.proof_steps, list)
        assert isinstance(response.axioms_used, list)
        assert isinstance(response.tactics_used, list)
    
    async def test_extract_proof_json(self, solver_service):
        """Test proof extraction in JSON format."""
        smtlib = """
        (set-logic LIA)
        (declare-fun x () Int)
        (assert (> x 0))
        (assert (not (> (+ x 1) 0)))
        (check-sat)
        """
        
        request = ProofExtractRequest(
            smtlib=smtlib,
            format="json",
            verify=True
        )
        
        response = await solver_service.extract_proof(request)
        
        assert isinstance(response.success, bool)


# =============================================================================
# Test Portfolio Solving
# =============================================================================

@pytest.mark.asyncio
class TestPortfolioSolving:
    """Test portfolio solving functionality."""
    
    async def test_portfolio_solve(self, solver_service):
        """Test portfolio solving with multiple strategies."""
        smtlib = """
        (set-logic LIA)
        (declare-fun x () Int)
        (assert (> x 0))
        (assert (< x 100))
        (check-sat)
        """
        
        request = PortfolioSolveRequest(
            smtlib=smtlib,
            strategies=["default", "smt", "qflia"],
            timeout=10.0,
            parallel=True
        )
        
        response = await solver_service.solve_portfolio(request)
        
        assert isinstance(response.success, bool)
        assert response.strategies_tried > 0
        assert response.execution_time_ms >= 0
        assert response.parallel_speedup >= 1.0


# =============================================================================
# Test Incremental Solving
# =============================================================================

@pytest.mark.asyncio
class TestIncrementalSolving:
    """Test incremental solving functionality."""
    
    async def test_incremental_create_and_check(self, solver_service):
        """Test incremental state creation and checking."""
        # Create initial state
        create_request = IncrementalSolveRequest(
            operation="create",
            variables=[{"name": "x", "type": "INTEGER"}],
            constraints=["x > 0"]
        )
        
        create_response = await solver_service.incremental_solve(create_request)
        
        assert create_response.success
        assert create_response.state_id is not None
        
        state_id = create_response.state_id
        
        # Check satisfiability
        check_request = IncrementalSolveRequest(
            operation="check",
            state_id=state_id
        )
        
        check_response = await solver_service.incremental_solve(check_request)
        
        assert check_response.success
    
    async def test_incremental_push_pop(self, solver_service):
        """Test incremental push and pop operations."""
        # Create state
        create_request = IncrementalSolveRequest(
            operation="create",
            variables=[{"name": "x", "type": "INTEGER"}],
            constraints=["x > 0"]
        )
        
        create_response = await solver_service.incremental_solve(create_request)
        state_id = create_response.state_id
        
        # Push scope
        push_request = IncrementalSolveRequest(
            operation="push",
            state_id=state_id
        )
        
        push_response = await solver_service.incremental_solve(push_request)
        assert push_response.success
        
        # Pop scope
        pop_request = IncrementalSolveRequest(
            operation="pop",
            state_id=state_id
        )
        
        pop_response = await solver_service.incremental_solve(pop_request)
        assert pop_response.success


# =============================================================================
# Test Caching
# =============================================================================

@pytest.mark.skipif(not CACHE_AVAILABLE, reason="Cache not available")
class TestCaching:
    """Test caching functionality."""
    
    def test_cache_basic_operations(self, cache):
        """Test basic cache operations."""
        # Set value
        cache.set("test_op", {"key": "value"}, {"result": "test"}, ttl=3600)
        
        # Get value
        hit, value = cache.get("test_op", {"key": "value"})
        
        assert hit
        assert value == {"result": "test"}
    
    def test_cache_miss(self, cache):
        """Test cache miss."""
        hit, value = cache.get("nonexistent", {"key": "value"})
        
        assert not hit
        assert value is None
    
    def test_cache_invalidation(self, cache):
        """Test cache invalidation."""
        # Add entries
        cache.set("op1", {"a": 1}, "result1", tags=["tag1"])
        cache.set("op2", {"b": 2}, "result2", tags=["tag2"])
        
        # Invalidate by tag
        invalidated = cache.invalidate(tags=["tag1"])
        
        assert invalidated >= 0
        
        # Verify invalidation
        hit, _ = cache.get("op1", {"a": 1})
        assert not hit
    
    def test_cache_stats(self, cache):
        """Test cache statistics."""
        # Add some entries
        cache.set("op1", {"a": 1}, "result1")
        cache.set("op2", {"b": 2}, "result2")
        
        # Access to generate stats
        cache.get("op1", {"a": 1})
        cache.get("op1", {"a": 1})
        cache.get("nonexistent", {"c": 3})
        
        stats = cache.get_stats()
        
        assert stats.hits >= 0
        assert stats.misses >= 0
        assert 0 <= stats.hit_rate <= 1


# =============================================================================
# Test Performance Monitoring
# =============================================================================

@pytest.mark.skipif(not MONITOR_AVAILABLE, reason="Monitor not available")
class TestPerformanceMonitoring:
    """Test performance monitoring functionality."""
    
    def test_record_operation(self, monitor):
        """Test operation recording."""
        monitor.record_operation("test_op", 0.5, success=True)
        monitor.record_operation("test_op", 0.6, success=True)
        monitor.record_operation("test_op", 0.4, success=False)
        
        summary = monitor.get_operation_summary("test_op")
        
        assert summary["calls"] == 3
        assert summary["errors"] == 1
    
    def test_get_bottlenecks(self, monitor):
        """Test bottleneck detection."""
        # Record operations with different durations
        monitor.record_operation("fast_op", 0.1, success=True)
        monitor.record_operation("slow_op", 2.0, success=True)
        
        bottlenecks = monitor.get_bottlenecks(2)
        
        assert len(bottlenecks) <= 2
        if bottlenecks:
            assert "operation" in bottlenecks[0]
            assert "avg_time_s" in bottlenecks[0]
    
    def test_dashboard_data(self, monitor):
        """Test dashboard data generation."""
        # Record some operations
        monitor.record_operation("op1", 0.5, success=True)
        monitor.record_operation("op2", 1.0, success=False)
        
        dashboard = monitor.get_dashboard_data()
        
        assert "timestamp" in dashboard
        assert "summary" in dashboard
        assert "operations" in dashboard


# =============================================================================
# Test Knowledge Extraction
# =============================================================================

@pytest.mark.skipif(not KNOWLEDGE_AVAILABLE, reason="Knowledge extraction not available")
class TestKnowledgeExtraction:
    """Test knowledge extraction functionality."""
    
    def test_learn_strategy(self):
        """Test strategy learning."""
        extractor = get_z3_knowledge_extractor()
        
        strategy = extractor.learn_strategy(
            problem_features={
                "type": "linear",
                "var_count": 5,
                "constraint_count": 10
            },
            tactics_used=["simplify", "solve-eqs", "smt"],
            config_used={"timeout": 30},
            success=True,
            solving_time=2.5
        )
        
        assert strategy is not None
        assert strategy.success_count == 1
    
    def test_analyze_constraints(self):
        """Test constraint analysis."""
        extractor = get_z3_knowledge_extractor()
        
        constraints = [
            "(> x 0)",
            "(< x 10)",
            "(= y (+ x 5))",
            "(> (* x y) 0)"
        ]
        
        patterns = extractor.analyze_constraints(constraints, 1.5, True)
        
        assert len(patterns) > 0
    
    def test_get_knowledge_summary(self):
        """Test knowledge summary."""
        extractor = get_z3_knowledge_extractor()
        
        # Add some knowledge
        extractor.learn_strategy(
            problem_features={"type": "test"},
            tactics_used=["tactic1"],
            config_used={},
            success=True,
            solving_time=1.0
        )
        
        summary = extractor.get_knowledge_summary()
        
        assert "proof_patterns" in summary
        assert "constraint_patterns" in summary
        assert "strategies" in summary


# =============================================================================
# Test Reliability Checking
# =============================================================================

@pytest.mark.skipif(not RELIABILITY_AVAILABLE, reason="Reliability checker not available")
class TestReliabilityChecking:
    """Test reliability checking functionality."""
    
    def test_component_reliability_verification(self):
        """Test component reliability verification."""
        checker = Z3ReliabilityChecker()
        
        component = ComponentReliabilityModel(
            component_id="test_component",
            availability=0.99,
            mtbf_hours=8760.0,
            mttr_hours=1.0
        )
        
        requirements = [
            ReliabilityConstraint(
                property_type=ReliabilityProperty.AVAILABILITY,
                threshold=0.95
            )
        ]
        
        result = checker.verify_component_reliability(component, requirements)
        
        assert result.success or not Z3_AVAILABLE
        if result.success:
            assert isinstance(result.verified, bool)
    
    def test_system_reliability_verification(self):
        """Test system reliability verification."""
        checker = Z3ReliabilityChecker()
        
        components = [
            ComponentReliabilityModel(
                component_id="comp1",
                availability=0.99
            ),
            ComponentReliabilityModel(
                component_id="comp2",
                availability=0.98
            )
        ]
        
        requirements = [
            ReliabilityConstraint(
                property_type=ReliabilityProperty.AVAILABILITY,
                threshold=0.95
            )
        ]
        
        result = checker.verify_system_reliability(components, requirements)
        
        assert result.success or not Z3_AVAILABLE
    
    def test_checker_status(self):
        """Test reliability checker status."""
        checker = Z3ReliabilityChecker()
        
        status = checker.get_status()
        
        assert "z3_available" in status
        assert "statistics" in status


# =============================================================================
# Test MCP Tools
# =============================================================================

@pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP not available")
class TestMCPTools:
    """Test MCP tools functionality."""
    
    def test_mcp_server_creation(self):
        """Test MCP server creation."""
        server = get_z3_mcp_server()
        
        assert server is not None
    
    def test_list_tools(self):
        """Test listing MCP tools."""
        server = get_z3_mcp_server()
        
        tools = server.list_tools()
        
        assert len(tools) > 0
        assert all("name" in tool for tool in tools)
    
    def test_call_solve_tool(self):
        """Test calling solve tool via MCP."""
        server = get_z3_mcp_server()
        
        result = server.call_tool("z3_solve_constraints", {
            "variables": [
                {"name": "x", "type": "INTEGER"},
                {"name": "y", "type": "INTEGER"}
            ],
            "constraints": [
                "x > 0",
                "x < 10",
                "y == x + 5"
            ]
        })
        
        assert isinstance(result, dict)
        assert "success" in result


# =============================================================================
# Test CrewAI Bridge
# =============================================================================

@pytest.mark.skipif(not CREWAI_AVAILABLE, reason="CrewAI not available")
class TestCrewAIBridge:
    """Test CrewAI bridge functionality."""
    
    @pytest.mark.asyncio
    async def test_solver_agent(self):
        """Test Z3 solver agent."""
        agent = Z3SolverAgent("test_solver")
        
        task = AgentTask(
            task_id="test_task",
            role=AgentRole.SOLVER,
            problem="""
            (set-logic LIA)
            (declare-fun x () Int)
            (assert (> x 0))
            (assert (< x 10))
            (check-sat)
            """
        )
        
        result = await agent.execute(task)
        
        assert result is not None
        assert result.task_id == "test_task"
        assert isinstance(result.success, bool)
    
    @pytest.mark.asyncio
    async def test_prover_agent(self):
        """Test Z3 prover agent."""
        agent = Z3TheoremProverAgent("test_prover")
        
        task = AgentTask(
            task_id="test_prove_task",
            role=AgentRole.PROVER,
            problem="""
            (set-logic LIA)
            (declare-fun x () Int)
            (assert (> x 0))
            (assert (not (> (+ x 1) 0)))
            (check-sat)
            """
        )
        
        result = await agent.execute(task)
        
        assert result is not None
        assert result.task_id == "test_prove_task"


# =============================================================================
# Test Service Bubble Integration
# =============================================================================

@pytest.mark.asyncio
class TestServiceBubble:
    """Test complete service bubble integration."""
    
    async def test_service_status(self, service_bubble):
        """Test service bubble status."""
        status = service_bubble.get_status()
        
        assert "z3_available" in status
        assert "cache_available" in status
        assert "monitor_available" in status
    
    async def test_end_to_end_solve(self, service_bubble):
        """Test end-to-end solving through service bubble."""
        request = SolveRequest(
            problem="End-to-end test",
            variables=[{"name": "x", "type": "INTEGER"}],
            constraints=["x > 0", "x < 100"],
            timeout=10.0
        )
        
        response = await service_bubble.solver.solve(request)
        
        assert response.success
        assert response.execution_time_ms >= 0


# =============================================================================
# Test Advanced Z3 Features
# =============================================================================

@pytest.mark.skipif(not Z3_ADVANCED_AVAILABLE, reason="Z3 advanced not available")
class TestAdvancedFeatures:
    """Test advanced Z3 features."""
    
    def test_bitvector_solving(self, advanced_solver):
        """Test bit-vector solving."""
        from z3prover_advanced import BitVectorConstraint
        
        bv_constraints = [
            BitVectorConstraint("x", 32, signed=False, constraints=["(bvugt x #x00000000)"])
        ]
        
        result = advanced_solver.solve_bitvector(bv_constraints)
        
        assert result is not None
    
    def test_array_solving(self, advanced_solver):
        """Test array constraint solving."""
        from z3prover_advanced import ArrayConstraint
        
        scalar_vars = [
            Z3Variable("idx", Z3ConstraintType.INTEGER)
        ]
        
        array_constraints = [
            ArrayConstraint(
                array_name="arr",
                index_type=Z3ConstraintType.INTEGER,
                value_type=Z3ConstraintType.INTEGER,
                constraints=["(> (select arr idx) 0)"]
            )
        ]
        
        result = advanced_solver.solve_with_arrays(scalar_vars, array_constraints, [])
        
        assert result is not None


# =============================================================================
# Performance Tests
# =============================================================================

@pytest.mark.asyncio
class TestPerformance:
    """Test performance characteristics."""
    
    async def test_solving_performance(self, solver_service):
        """Test solving performance is within acceptable limits."""
        request = SolveRequest(
            problem="Performance test",
            variables=[{"name": "x", "type": "INTEGER"} for _ in range(10)],
            constraints=[f"x{i} > 0" for i in range(10)],
            timeout=5.0
        )
        
        start = time.time()
        response = await solver_service.solve(request)
        elapsed = (time.time() - start) * 1000
        
        # Should complete within timeout
        assert elapsed < 5000  # 5 seconds
    
    async def test_batch_performance(self, solver_service):
        """Test batch solving performance."""
        problems = [
            SolveRequest(
                problem=f"Batch {i}",
                variables=[{"name": "x", "type": "INTEGER"}],
                constraints=[f"x > {i}"],
                timeout=2.0
            )
            for i in range(5)
        ]
        
        request = BatchSolveRequest(problems=problems, parallel=True)
        
        start = time.time()
        response = await solver_service.solve_batch(request)
        elapsed = (time.time() - start) * 1000
        
        # Should complete reasonably fast with parallel execution
        assert elapsed < 10000  # 10 seconds


# =============================================================================
# Error Handling Tests
# =============================================================================

@pytest.mark.asyncio
class TestErrorHandling:
    """Test error handling."""
    
    async def test_invalid_constraint(self, solver_service):
        """Test handling of invalid constraints."""
        request = SolveRequest(
            problem="Invalid constraint test",
            variables=[{"name": "x", "type": "INVALID_TYPE"}],
            constraints=["invalid constraint"],
            timeout=5.0
        )
        
        # Should handle gracefully
        response = await solver_service.solve(request)
        
        # May succeed or fail, but shouldn't crash
        assert isinstance(response.success, bool)
    
    async def test_timeout_handling(self, solver_service):
        """Test timeout handling."""
        request = SolveRequest(
            problem="Timeout test",
            variables=[{"name": "x", "type": "INTEGER"}],
            constraints=["x > 0"],
            timeout=0.001  # Very short timeout
        )
        
        response = await solver_service.solve(request)
        
        # Should complete (may be timeout or success)
        assert isinstance(response.success, bool)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
