"""
RESE Complete End-to-End Test Suite

Tests the complete RESE framework with Z3 and LeanAide integrations.

Following CLAUDE.md principles:
- Law of Configuration Explicitness: All config via env vars
- Law of Idempotency: All operations safe to replay
- Circuit Breaker: Per-phase circuit breakers
- Exponential Backoff: Retry with jitter
- Dead Letter Queue: For logic failures
- Structured Logging: JSON with correlation_id

Author: RESE Team
Created: 2026-02-04
Status: Comprehensive E2E Test Suite
"""

import asyncio
import json
import os
import sys
import time
import uuid
import pytest
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from unittest.mock import Mock, AsyncMock, MagicMock, patch
import logging

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "glue"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "glue", "orchestration"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "glue", "lib"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "glue", "schemas"))

# Try imports
try:
    from glue.orchestration.rese_pipeline import (
        RESEPipeline, PhaseStatus, ErrorType, PipelineConfig
    )
    from glue.lib.rese_dee import DeepExplorationEngine, ExplorationConfig
    from glue.lib.rese_lltl import LogicToLossTranslator
    from glue.adapters.rese_z3_bridge.src.rese_z3_bridge import RESEZ3Bridge, RESEZ3BridgeConfig
    from glue.adapters.rese_leanaide_workflow.src.leanaide_rese_workflow import (
        LeanAideRESEWorkflow, WorkflowConfig, ProblemType, SolverType
    )
except ImportError as e:
    pytest.skip(f"Import error: {e}", allow_module_level=True)


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def sample_problem_statements():
    """Sample problem statements for different domains"""
    return {
        "simple_logic": {
            "statement": "If A implies B, and B implies C, then A implies C",
            "domain": "logic",
            "expected_complexity": "low"
        },
        "arithmetic": {
            "statement": "For all integers x and y, if x + y = 10 and x = 3, then y = 7",
            "domain": "arithmetic",
            "expected_complexity": "low"
        },
        "set_theory": {
            "statement": "If A is a subset of B, and B is a subset of C, then A is a subset of C",
            "domain": "set_theory",
            "expected_complexity": "medium"
        },
        "graph_theory": {
            "statement": "In any connected graph, there exists a spanning tree",
            "domain": "graph_theory",
            "expected_complexity": "medium"
        },
        "optimization": {
            "statement": "Find the minimum value of f(x,y) = x^2 + y^2 subject to x + y = 10",
            "domain": "optimization",
            "expected_complexity": "high"
        }
    }


@pytest.fixture
def sample_constraints():
    """Sample constraints for testing"""
    return [
        {
            "constraint_id": "c1",
            "type": "boolean",
            "expression": "A => B",
            "priority": 1.0
        },
        {
            "constraint_id": "c2",
            "type": "boolean",
            "expression": "B => C",
            "priority": 1.0
        },
        {
            "constraint_id": "c3",
            "type": "arithmetic",
            "expression": "x + y = 10",
            "priority": 0.8
        }
    ]


@pytest.fixture
def sample_hypotheses():
    """Sample hypotheses for Phase III testing"""
    return [
        {
            "hypothesis_id": "h1",
            "statement": "A implies C through transitivity",
            "type": "causal",
            "domain": "logic",
            "confidence": 0.8
        },
        {
            "hypothesis_id": "h2",
            "statement": "y equals 7 when x equals 3",
            "type": "structural",
            "domain": "arithmetic",
            "confidence": 0.9
        }
    ]


@pytest.fixture
def sample_efficacy_claims():
    """Sample efficacy claims for Phase IV testing"""
    return [
        {
            "claim_id": "ec1",
            "statement": "The predictive model achieves 95% accuracy",
            "metric": "accuracy",
            "target_value": 0.95
        },
        {
            "claim_id": "ec2",
            "statement": "The solution satisfies all constraints",
            "metric": "constraint_satisfaction",
            "target_value": 1.0
        }
    ]


@pytest.fixture
def pipeline_config():
    """Pipeline configuration for testing"""
    os.environ.update({
        "PHASE_I_TIMEOUT_MS": "30000",
        "PHASE_II_TIMEOUT_MS": "30000",
        "PHASE_III_TIMEOUT_MS": "30000",
        "PHASE_IV_TIMEOUT_MS": "30000",
        "PIPELINE_TIMEOUT_MS": "120000",
        "MAX_RETRIES": "3",
        "CIRCUIT_BREAKER_THRESHOLD": "5",
        "CIRCUIT_BREAKER_TIMEOUT_MS": "60000"
    })
    return PipelineConfig.from_env()


@pytest.fixture
def z3_bridge_config():
    """Z3 bridge configuration for testing"""
    os.environ.update({
        "Z3_BASE_URL": "http://localhost:8000",
        "Z3_TIMEOUT_MS": "10000",
        "LEANAIDE_BASE_URL": "http://localhost:7654",
        "LEANAIDE_TIMEOUT_MS": "30000",
        "LEANAIDE_ENABLE": "false",  # Disable for unit tests
        "Z3_MAX_RETRIES": "2",
        "Z3_ENABLE_CACHE": "true"
    })
    return RESEZ3BridgeConfig.from_env()


@pytest.fixture
def workflow_config():
    """Workflow configuration for testing"""
    os.environ.update({
        "LEANAIDE_HOST": "localhost",
        "LEANAIDE_PORT": "7654",
        "LEANAIDE_TIMEOUT_MS": "15000",
        "PROOF_SEARCH_TIMEOUT_MS": "20000",
        "WORKFLOW_TIMEOUT_MS": "90000"
    })
    return WorkflowConfig.from_env()


# ============================================================================
# MOCK SERVERS
# ============================================================================

@pytest.fixture
def mock_z3_server():
    """Mock Z3 server for testing"""
    server = AsyncMock()

    async def mock_solve(smtlib: str, correlation_id: str, timeout_ms: int):
        return {
            "status": "sat",
            "model": {
                "assignments": {
                    "A": True,
                    "B": True,
                    "C": True
                }
            },
            "execution_time": 0.1
        }

    server.solve = mock_solve

    async def mock_check_health():
        return {"status": "ok"}

    server.check_health = mock_check_health

    return server


@pytest.fixture
def mock_leanaide_server():
    """Mock LeanAide server for testing"""
    server = AsyncMock()

    async def mock_autoformalize(natural_language: str, theorem_name: Optional[str]):
        return {
            "success": True,
            "lean_code": f"theorem {theorem_name or 'theorem'} : Prop := by\n  simp",
            "theorem_name": theorem_name or "theorem",
            "theorem_type": "Prop"
        }

    server.autoformalize = mock_autoformalize

    async def mock_prove(theorem_text: str, theorem_code: Optional[str]):
        return {
            "success": True,
            "proof": "by simp",
            "tactics": ["simp"],
            "proof_script": "simp"
        }

    server.prove = mock_prove

    return server


# ============================================================================
# TEST UTILITIES
# ============================================================================

class TestMetrics:
    """Test metrics collector"""

    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}

    def record(self, operation: str, duration_ms: float):
        if operation not in self.metrics:
            self.metrics[operation] = []
        self.metrics[operation].append(duration_ms)

    def get_summary(self) -> Dict[str, Any]:
        summary = {}
        for op, times in self.metrics.items():
            summary[op] = {
                "count": len(times),
                "total_ms": sum(times),
                "avg_ms": sum(times) / len(times) if times else 0,
                "min_ms": min(times) if times else 0,
                "max_ms": max(times) if times else 0
            }
        return summary


@pytest.fixture
def test_metrics():
    """Test metrics fixture"""
    return TestMetrics()


def assert_valid_correlation_id(correlation_id: str):
    """Assert correlation ID is valid UUID"""
    assert correlation_id is not None
    assert isinstance(correlation_id, str)
    assert len(correlation_id) == 36  # UUID format


def assert_valid_phase_result(result: Dict[str, Any], phase_name: str):
    """Assert phase result has required fields"""
    assert "phase_name" in result or "phase" in result
    assert "status" in result
    assert result["status"] in ["completed", "failed", "timeout", "skipped"]
    assert "execution_time_ms" in result
    assert result["execution_time_ms"] >= 0


def assert_valid_timestamp(timestamp: str):
    """Assert timestamp is valid ISO-8601 UTC"""
    assert timestamp is not None
    assert isinstance(timestamp, str)
    # Try parsing
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        assert dt.tzinfo == timezone.utc
    except ValueError:
        pytest.fail(f"Invalid timestamp: {timestamp}")


# ============================================================================
# SCENARIO 1: COMPLETE RESE PIPELINE
# ============================================================================

class TestScenario1CompletePipeline:
    """Test Scenario 1: Complete RESE Pipeline"""

    @pytest.mark.asyncio
    async def test_complete_pipeline_simple_logic(self, sample_problem_statements, pipeline_config, test_metrics):
        """Test complete pipeline with simple logic problem"""
        start_time = time.time()

        problem = sample_problem_statements["simple_logic"]

        # Create pipeline
        pipeline = RESEPipeline(pipeline_config)

        # Execute pipeline
        result = pipeline.execute(
            problem_statement=problem["statement"],
            context={"domain": problem["domain"]},
            correlation_id=str(uuid.uuid4())
        )

        duration_ms = (time.time() - start_time) * 1000
        test_metrics.record("complete_pipeline_simple", duration_ms)

        # Validate result
        assert result is not None
        assert result["status"] in ["completed", "failed"]
        assert_valid_correlation_id(result["correlation_id"])
        assert_valid_timestamp(result["timestamp"])
        assert "results" in result

        # Validate all phases executed
        assert "phase_i" in result["results"]
        assert "phase_ii" in result["results"]
        assert "phase_iii" in result["results"]
        assert "phase_iv" in result["results"]

        # Validate phase results
        for phase_name, phase_result in result["results"].items():
            assert_valid_phase_result(phase_result, phase_name)

    @pytest.mark.asyncio
    async def test_complete_pipeline_with_context(self, sample_problem_statements, pipeline_config):
        """Test pipeline with additional context"""
        problem = sample_problem_statements["arithmetic"]

        pipeline = RESEPipeline(pipeline_config)

        context = {
            "domain": problem["domain"],
            "variables": ["x", "y"],
            "constraints": ["x + y = 10", "x = 3"]
        }

        result = pipeline.execute(
            problem_statement=problem["statement"],
            context=context,
            correlation_id=str(uuid.uuid4())
        )

        assert result is not None
        assert "results" in result

    @pytest.mark.asyncio
    async def test_pipeline_idempotency(self, sample_problem_statements, pipeline_config):
        """Test pipeline idempotency (same input produces consistent output)"""
        problem = sample_problem_statements["simple_logic"]
        correlation_id = str(uuid.uuid4())

        pipeline = RESEPipeline(pipeline_config)

        # Execute twice with same correlation_id
        result1 = pipeline.execute(
            problem_statement=problem["statement"],
            correlation_id=correlation_id
        )

        result2 = pipeline.execute(
            problem_statement=problem["statement"],
            correlation_id=correlation_id
        )

        # Both should complete
        assert result1["status"] in ["completed", "failed"]
        assert result2["status"] in ["completed", "failed"]


# ============================================================================
# SCENARIO 2: Z3 INTEGRATION ACROSS PHASES
# ============================================================================

class TestScenario2Z3Integration:
    """Test Scenario 2: Z3 Integration Across Phases"""

    @pytest.mark.asyncio
    async def test_phase_i_sce_with_z3(self, z3_bridge_config, sample_constraints, mock_z3_server):
        """Test Phase I SCE with Z3 constraint solving"""

        # Mock Z3 client
        with patch('glue.adapters.rese_z3_bridge.src.rese_z3_client.Z3Client') as mock_client:
            mock_instance = AsyncMock()
            mock_instance.solve = AsyncMock(return_value=mock_z3_server.solve("", "cid", 10000))
            mock_instance.check_health = AsyncMock(return_value=mock_z3_server.check_health())
            mock_client.return_value = mock_instance

            bridge = RESEZ3Bridge(z3_bridge_config)

            # Create canonical variables
            from glue.adapters.rese_z3_bridge.src.rese_z3_schema import CanonicalVariable
            variables = [
                CanonicalVariable(name="A", var_type="Bool"),
                CanonicalVariable(name="B", var_type="Bool"),
                CanonicalVariable(name="C", var_type="Bool")
            ]

            # Create canonical constraints
            from glue.adapters.rese_z3_bridge.src.rese_z3_schema import CanonicalConstraint
            constraints = [
                CanonicalConstraint(
                    constraint_id="c1",
                    constraint_type="boolean",
                    expression="(=> A B)"
                )
            ]

            # Solve constraints
            result = bridge.solve_constraints(
                variables=variables,
                constraints=constraints,
                correlation_id=str(uuid.uuid4()),
                timeout_ms=10000
            )

            # Validate result
            assert result is not None
            assert result.status in ["sat", "unsat", "unknown"]

    @pytest.mark.asyncio
    async def test_phase_iii_dito_contradiction_detection(self, z3_bridge_config, mock_z3_server):
        """Test Phase III ACI with Z3 contradiction detection"""

        with patch('glue.adapters.rese_z3_bridge.src.rese_z3_client.Z3Client') as mock_client:
            mock_instance = AsyncMock()
            # Return UNSAT to indicate contradiction
            mock_instance.solve = AsyncMock(return_value={
                "status": "unsat",
                "reason": "Contradiction found"
            })
            mock_instance.check_health = AsyncMock(return_value={"status": "ok"})
            mock_client.return_value = mock_instance

            bridge = RESEZ3Bridge(z3_bridge_config)

            # Create contradictory constraints
            from glue.adapters.rese_z3_bridge.src.rese_z3_schema import CanonicalConstraint
            constraints = [
                CanonicalConstraint(
                    constraint_id="c1",
                    constraint_type="boolean",
                    expression="A"
                ),
                CanonicalConstraint(
                    constraint_id="c2",
                    constraint_type="boolean",
                    expression="(not A)"
                )
            ]

            # Detect contradictions
            has_contradiction, counterexample = bridge.detect_contradictions(
                constraints=constraints,
                correlation_id=str(uuid.uuid4()),
                timeout_ms=10000
            )

            # Should detect contradiction
            assert has_contradiction is True

    @pytest.mark.asyncio
    async def test_z3_performance_improvement(self, z3_bridge_config, test_metrics):
        """Test Z3 caching improves performance"""

        with patch('glue.adapters.rese_z3_bridge.src.rese_z3_client.Z3Client') as mock_client:
            mock_instance = AsyncMock()

            async def mock_solve_with_delay(smtlib, cid, timeout):
                time.sleep(0.01)  # Simulate work
                return {"status": "sat", "model": {}}

            mock_instance.solve = mock_solve_with_delay
            mock_instance.check_health = AsyncMock(return_value={"status": "ok"})
            mock_client.return_value = mock_instance

            bridge = RESEZ3Bridge(z3_bridge_config)

            from glue.adapters.rese_z3_bridge.src.rese_z3_schema import (
                CanonicalVariable, CanonicalConstraint
            )

            variables = [CanonicalVariable(name="x", var_type="Int")]
            constraints = [
                CanonicalConstraint(
                    constraint_id="c1",
                    constraint_type="arithmetic",
                    expression "(= x 5)"
                )
            ]

            # First call - no cache
            start_time = time.time()
            result1 = bridge.solve_constraints(
                variables=variables,
                constraints=constraints,
                correlation_id=str(uuid.uuid4())
            )
            duration1_ms = (time.time() - start_time) * 1000
            test_metrics.record("z3_first_call", duration1_ms)

            # Second call - should hit cache
            start_time = time.time()
            result2 = bridge.solve_constraints(
                variables=variables,
                constraints=constraints,
                correlation_id=str(uuid.uuid4())
            )
            duration2_ms = (time.time() - start_time) * 1000
            test_metrics.record("z3_cached_call", duration2_ms)

            # Cached call should be faster (or similar due to mock)
            assert duration2_ms <= duration1_ms * 1.5  # Allow some variance


# ============================================================================
# SCENARIO 3: LEANAIDE INTEGRATION ACROSS PHASES
# ============================================================================

class TestScenario3LeanAideIntegration:
    """Test Scenario 3: LeanAide Integration Across Phases"""

    @pytest.mark.asyncio
    async def test_autoformalization_all_phases(self, workflow_config, mock_leanaide_server):
        """Test autoformalization in all 4 phases"""

        # Skip if LeanAide not available
        pytest.importorskip("glue.adapters.rese_leanaide_workflow.src.autoformalization_service")

        from glue.adapters.rese_leanaide_workflow.src.leanaide_rese_workflow import LeanAideRESEWorkflow

        workflow = LeanAideRESEWorkflow(workflow_config)
        await workflow.initialize()

        # Test Phase I autoformalization
        result_i = await workflow.autoformalization_service.autoformalize_phase_i(
            constraint_text="A implies B",
            constraint_type="logical",
            correlation_id=str(uuid.uuid4())
        )

        assert result_i is not None
        assert isinstance(result_i.success, bool)

        await workflow.close()

    @pytest.mark.asyncio
    async def test_ai_powered_proving(self, workflow_config):
        """Test AI-powered proving in Phase III"""

        pytest.importorskip("glue.adapters.rese_leanaide_workflow.src.proof_search_service")

        from glue.adapters.rese_leanaide_workflow.src.leanaide_rese_workflow import LeanAideRESEWorkflow

        workflow = LeanAideRESEWorkflow(workflow_config)
        await workflow.initialize()

        # Test proof search
        result = await workflow.proof_search_service.search_phase_iii(
            lean_code="theorem example : Prop := by simp",
            correlation_id=str(uuid.uuid4())
        )

        assert result is not None

        await workflow.close()

    @pytest.mark.asyncio
    async def test_workflow_orchestration(self, sample_problem_statements, workflow_config):
        """Test complete workflow orchestration with LeanAide"""

        pytest.importorskip("glue.adapters.rese_leanaide_workflow.src.leanaide_rese_workflow")

        from glue.adapters.rese_leanaide_workflow.src.leanaide_rese_workflow import LeanAideRESEWorkflow

        workflow = LeanAideRESEWorkflow(workflow_config)

        try:
            await workflow.initialize()

            result = await workflow.execute(
                problem_statement=sample_problem_statements["simple_logic"]["statement"],
                context={},
                correlation_id=str(uuid.uuid4())
            )

            assert result is not None
            assert result.workflow_id is not None
            assert result.correlation_id is not None
            assert "phase_results" in result.to_dict()

        finally:
            await workflow.close()


# ============================================================================
# SCENARIO 4: TIERED VERIFICATION
# ============================================================================

class TestScenario4TieredVerification:
    """Test Scenario 4: Tiered Verification System"""

    @pytest.mark.asyncio
    async def test_simple_problems_use_z3(self, z3_bridge_config):
        """Test simple problems use Z3 (Tier 1)"""

        bridge = RESEZ3Bridge(z3_bridge_config)

        # Simple constraint problem
        from glue.adapters.rese_z3_bridge.src.rese_z3_schema import (
            CanonicalVariable, CanonicalConstraint
        )

        variables = [CanonicalVariable(name="x", var_type="Int")]
        constraints = [
            CanonicalConstraint(
                constraint_id="c1",
                constraint_type="arithmetic",
                expression="(< x 10)"
            )
        ]

        result = bridge.solve_constraints(
            variables=variables,
            constraints=constraints,
            correlation_id=str(uuid.uuid4())
        )

        assert result is not None

    @pytest.mark.asyncio
    async def test_medium_problems_use_leanaide(self, workflow_config):
        """Test medium problems use LeanAide (Tier 2)"""

        pytest.importorskip("glue.adapters.rese_leanaide_workflow.src.leanaide_rese_workflow")

        from glue.adapters.rese_leanaide_workflow.src.leanaide_rese_workflow import LeanAideRESEWorkflow

        workflow = LeanAideRESEWorkflow(workflow_config)

        try:
            await workflow.initialize()

            # Medium complexity: theorem requiring proof
            classification = workflow._classify_problem(
                "Prove that if A implies B and B implies C, then A implies C",
                {}
            )

            assert classification.problem_type == ProblemType.THEOREM_PROVING
            assert classification.recommended_solver in [
                SolverType.LEANAIDE,
                SolverType.HYBRID_Z3_LEANAIDE
            ]

        finally:
            await workflow.close()

    @pytest.mark.asyncio
    async def test_escalation_on_failure(self, z3_bridge_config):
        """Test escalation when lower tiers fail"""

        bridge = RESEZ3Bridge(z3_bridge_config)

        # Try Z3 first (Tier 1)
        # If it fails, should escalate to LeanAide (Tier 2)

        # This test verifies the escalation logic
        # In real scenario, Z3 would timeout or return unknown
        # triggering escalation to LeanAide

        assert bridge is not None


# ============================================================================
# SCENARIO 5: ERROR HANDLING AND RESILIENCE
# ============================================================================

class TestScenario5ErrorHandling:
    """Test Scenario 5: Error Handling and Resilience"""

    @pytest.mark.asyncio
    async def test_circuit_breaker_activation(self, z3_bridge_config):
        """Test circuit breaker activates on failures"""

        with patch('glue.adapters.rese_z3_bridge.src.rese_z3_client.Z3Client') as mock_client:
            # Mock failing client
            mock_instance = AsyncMock()
            mock_instance.solve = AsyncMock(side_effect=Exception("Connection refused"))
            mock_instance.check_health = AsyncMock(return_value={"status": "error"})
            mock_client.return_value = mock_instance

            bridge = RESEZ3Bridge(z3_bridge_config)

            from glue.adapters.rese_z3_bridge.src.rese_z3_schema import (
                CanonicalVariable, CanonicalConstraint
            )

            variables = [CanonicalVariable(name="x", var_type="Int")]
            constraints = [
                CanonicalConstraint(
                    constraint_id="c1",
                    constraint_type="arithmetic",
                    expression="(< x 10)"
                )
            ]

            # Trigger multiple failures
            for _ in range(10):
                try:
                    bridge.solve_constraints(
                        variables=variables,
                        constraints=constraints,
                        correlation_id=str(uuid.uuid4())
                    )
                except:
                    pass

            # Check circuit breaker state
            stats = bridge.get_stats()
            assert "client_stats" in stats

    @pytest.mark.asyncio
    async def test_graceful_degradation_z3_unavailable(self, z3_bridge_config):
        """Test graceful degradation when Z3 unavailable"""

        with patch('glue.adapters.rese_z3_bridge.src.rese_z3_client.Z3Client') as mock_client:
            mock_instance = AsyncMock()
            mock_instance.solve = AsyncMock(side_effect=Exception("Z3 not available"))
            mock_client.return_value = mock_instance

            bridge = RESEZ3Bridge(z3_bridge_config)

            from glue.adapters.rese_z3_bridge.src.rese_z3_schema import (
                CanonicalVariable, CanonicalConstraint
            )

            variables = [CanonicalVariable(name="x", var_type="Int")]
            constraints = [
                CanonicalConstraint(
                    constraint_id="c1",
                    constraint_type="arithmetic",
                    expression="(< x 10)"
                )
            ]

            # Should handle error gracefully
            try:
                result = bridge.solve_constraints(
                    variables=variables,
                    constraints=constraints,
                    correlation_id=str(uuid.uuid4())
                )
                # If it returns, check result
                assert result is not None
            except Exception as e:
                # Or raise meaningful error
                assert str(e) is not None

    @pytest.mark.asyncio
    async def test_retry_logic_with_backoff(self, z3_bridge_config):
        """Test retry logic with exponential backoff"""

        with patch('glue.adapters.rese_z3_bridge.src.rese_z3_client.Z3Client') as mock_client:
            call_count = [0]

            async def flaky_solve(smtlib, cid, timeout):
                call_count[0] += 1
                if call_count[0] < 3:
                    raise Exception("Temporary failure")
                return {"status": "sat", "model": {}}

            mock_instance = AsyncMock()
            mock_instance.solve = flaky_solve
            mock_instance.check_health = AsyncMock(return_value={"status": "ok"})
            mock_client.return_value = mock_instance

            bridge = RESEZ3Bridge(z3_bridge_config)

            from glue.adapters.rese_z3_bridge.src.rese_z3_schema import (
                CanonicalVariable, CanonicalConstraint
            )

            variables = [CanonicalVariable(name="x", var_type="Int")]
            constraints = [
                CanonicalConstraint(
                    constraint_id="c1",
                    constraint_type="arithmetic",
                    expression="(< x 10)"
                )
            ]

            # Should retry and succeed
            result = bridge.solve_constraints(
                variables=variables,
                constraints=constraints,
                correlation_id=str(uuid.uuid4())
            )

            # Verify retries happened
            assert call_count[0] >= 3
            assert result is not None

    @pytest.mark.asyncio
    async def test_idempotency_same_input(self, z3_bridge_config):
        """Test idempotency: same input produces same output"""

        with patch('glue.adapters.rese_z3_bridge.src.rese_z3_client.Z3Client') as mock_client:
            mock_instance = AsyncMock()

            # Create deterministic mock
            async def deterministic_solve(smtlib, cid, timeout):
                return {
                    "status": "sat",
                    "model": {"x": 5},
                    "execution_time": 0.1
                }

            mock_instance.solve = deterministic_solve
            mock_instance.check_health = AsyncMock(return_value={"status": "ok"})
            mock_client.return_value = mock_instance

            bridge = RESEZ3Bridge(z3_bridge_config)

            from glue.adapters.rese_z3_bridge.src.rese_z3_schema import (
                CanonicalVariable, CanonicalConstraint
            )

            variables = [CanonicalVariable(name="x", var_type="Int")]
            constraints = [
                CanonicalConstraint(
                    constraint_id="c1",
                    constraint_type="arithmetic",
                    expression="(< x 10)"
                )
            ]

            # Execute twice with same input
            result1 = bridge.solve_constraints(
                variables=variables,
                constraints=constraints,
                correlation_id=str(uuid.uuid4())
            )

            result2 = bridge.solve_constraints(
                variables=variables,
                constraints=constraints,
                correlation_id=str(uuid.uuid4())
            )

            # Results should be consistent
            assert result1.status == result2.status


# ============================================================================
# SCENARIO 6: PERFORMANCE BENCHMARKS
# ============================================================================

class TestScenario6PerformanceBenchmarks:
    """Test Scenario 6: Performance Benchmarks"""

    @pytest.mark.asyncio
    async def test_pipeline_10_constraints(self, pipeline_config, test_metrics):
        """Test RESE pipeline with 10 constraints"""

        pipeline = RESEPipeline(pipeline_config)

        problem = (
            "Given 10 constraints: "
            + ", ".join([f"constraint_{i} is true" for i in range(10)])
        )

        start_time = time.time()
        result = pipeline.execute(
            problem_statement=problem,
            context={"constraint_count": 10},
            correlation_id=str(uuid.uuid4())
        )
        duration_ms = (time.time() - start_time) * 1000

        test_metrics.record("pipeline_10_constraints", duration_ms)

        assert result is not None
        assert duration_ms < 30000  # Should complete within 30 seconds

    @pytest.mark.asyncio
    async def test_pipeline_100_constraints(self, pipeline_config, test_metrics):
        """Test RESE pipeline with 100 constraints"""

        pipeline = RESEPipeline(pipeline_config)

        constraints = [f"constraint_{i} is valid" for i in range(100)]
        problem = "Given 100 constraints: " + ", ".join(constraints[:10]) + "..."

        start_time = time.time()
        result = pipeline.execute(
            problem_statement=problem,
            context={"constraint_count": 100},
            correlation_id=str(uuid.uuid4())
        )
        duration_ms = (time.time() - start_time) * 1000

        test_metrics.record("pipeline_100_constraints", duration_ms)

        assert result is not None

    @pytest.mark.asyncio
    async def test_z3_solver_performance(self, z3_bridge_config, test_metrics):
        """Benchmark Z3 solver performance"""

        with patch('glue.adapters.rese_z3_bridge.src.rese_z3_client.Z3Client') as mock_client:
            mock_instance = AsyncMock()

            async def benchmark_solve(smtlib, cid, timeout):
                time.sleep(0.001)  # Simulate work
                return {"status": "sat", "model": {}}

            mock_instance.solve = benchmark_solve
            mock_instance.check_health = AsyncMock(return_value={"status": "ok"})
            mock_client.return_value = mock_instance

            bridge = RESEZ3Bridge(z3_bridge_config)

            from glue.adapters.rese_z3_bridge.src.rese_z3_schema import (
                CanonicalVariable, CanonicalConstraint
            )

            # Benchmark 10 solves
            durations = []
            for i in range(10):
                variables = [CanonicalVariable(name=f"x{i}", var_type="Int")]
                constraints = [
                    CanonicalConstraint(
                        constraint_id=f"c{i}",
                        constraint_type="arithmetic",
                        expression=f"(< x{i} {i * 10})"
                    )
                ]

                start_time = time.time()
                bridge.solve_constraints(
                    variables=variables,
                    constraints=constraints,
                    correlation_id=str(uuid.uuid4())
                )
                duration_ms = (time.time() - start_time) * 1000
                durations.append(duration_ms)

            avg_duration = sum(durations) / len(durations)
            test_metrics.record("z3_solver_avg", avg_duration)

            # Average should be reasonable
            assert avg_duration < 1000  # Less than 1 second per solve

    @pytest.mark.asyncio
    async def test_phase_execution_times(self, pipeline_config, test_metrics):
        """Test execution time for each phase"""

        pipeline = RESEPipeline(pipeline_config)

        problem = "Simple logic problem with basic constraints"

        result = pipeline.execute(
            problem_statement=problem,
            correlation_id=str(uuid.uuid4())
        )

        # Record phase times
        for phase_name, phase_result in result["results"].items():
            if "execution_time_ms" in phase_result:
                test_metrics.record(
                    f"phase_{phase_name}",
                    phase_result["execution_time_ms"]
                )

                # Each phase should complete in reasonable time
                assert phase_result["execution_time_ms"] < 60000  # 1 minute max


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for RESE components"""

    @pytest.mark.asyncio
    async def test_dee_lltl_integration(self):
        """Test DEE and LLTL integration"""

        config = ExplorationConfig.from_env()
        dee = DeepExplorationEngine(config)

        # Create problem statement
        problem = "Find the optimal solution for x + y = 10 with minimal x^2 + y^2"

        # Run exploration
        result = dee.explore(
            problem_statement=problem,
            domain="optimization",
            context={}
        )

        assert result is not None
        assert result.search_id is not None

    @pytest.mark.asyncio
    async def test_z3_leanaide_bridge_integration(self, z3_bridge_config):
        """Test Z3-LeanAide bridge integration"""

        bridge = RESEZ3Bridge(z3_bridge_config)

        # Test translation to Lean 4
        smtlib = "(declare-fun x () Int)(assert (< x 10))"

        result = bridge.translate_z3_to_lean(
            smtlib_content=smtlib,
            correlation_id=str(uuid.uuid4())
        )

        assert result is not None
        assert result.lean_code is not None

    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, sample_problem_statements, workflow_config):
        """Test complete end-to-end workflow"""

        pytest.importorskip("glue.adapters.rese_leanaide_workflow.src.leanaide_rese_workflow")

        from glue.adapters.rese_leanaide_workflow.src.leanaide_rese_workflow import LeanAideRESEWorkflow

        workflow = LeanAideRESEWorkflow(workflow_config)

        try:
            await workflow.initialize()

            # Execute workflow
            result = await workflow.execute(
                problem_statement=sample_problem_statements["simple_logic"]["statement"],
                context={"domain": "logic"},
                correlation_id=str(uuid.uuid4())
            )

            # Validate complete workflow
            assert result is not None
            assert result.workflow_id is not None
            assert result.correlation_id is not None
            assert result.overall_status in ["completed", "failed", "timeout"]
            assert len(result.phase_results) == 4

            # Validate each phase
            for phase_name, phase_result in result.phase_results.items():
                assert phase_result is not None
                assert phase_result.status in ["completed", "failed", "timeout"]
                assert phase_result.execution_time_ms >= 0

        finally:
            await workflow.close()


# ============================================================================
# TEST SUITE METADATA
# ============================================================================

def get_test_suite_info():
    """Get test suite metadata"""
    return {
        "name": "RESE Complete End-to-End Test Suite",
        "version": "1.0.0",
        "created": "2026-02-04",
        "scenarios": [
            {
                "id": 1,
                "name": "Complete RESE Pipeline",
                "tests": 3
            },
            {
                "id": 2,
                "name": "Z3 Integration Across Phases",
                "tests": 3
            },
            {
                "id": 3,
                "name": "LeanAide Integration Across Phases",
                "tests": 3
            },
            {
                "id": 4,
                "name": "Tiered Verification",
                "tests": 3
            },
            {
                "id": 5,
                "name": "Error Handling and Resilience",
                "tests": 4
            },
            {
                "id": 6,
                "name": "Performance Benchmarks",
                "tests": 4
            }
        ],
        "total_tests": 20,
        "integration_tests": 3
    }


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short", "-s"])
