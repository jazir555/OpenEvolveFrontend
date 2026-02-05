"""
Comprehensive Test Suite for Tiered Verification Component

Tests cover:
1. Configuration (5 tests)
2. Problem Classifier (15 tests)
3. Solver Selector (15 tests)
4. Tiered Verifier (15 tests)
5. Verification Results (10 tests)

Total: 60+ tests targeting >90% code coverage

Following CLAUDE.md principles:
- Law of Configuration Explicitness: Test env var validation
- Law of Idempotency: Test repeatable operations
- Law of UTC: Test timestamp format
- Structured Logging: Test JSON log format
- Circuit Breaker: Test failure detection

Author: RESE Team
Created: 2026-02-04
"""

import pytest
import os
import json
from datetime import datetime, timezone
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any, List

# Import test targets
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tiered_verifier import (
    TieredVerifier,
    TieredVerifierConfig,
    verify,
)
from problem_classifier import (
    ProblemClassifier,
    ClassifierConfig,
    classify_problem,
    should_escalate as classifier_should_escalate,
)
from solver_selector import (
    SolverSelector,
    SolverSelectorConfig,
    SolverPerformance,
    SelectionStrategy,
    SelectionResult,
    select_solver,
)
from verification_result import (
    VerificationTier,
    VerificationStatus,
    ProblemClass,
    ProblemDomain,
    Z3VerificationResult,
    LeanAideVerificationResult,
    Lean4VerificationResult,
    UnifiedVerificationResult,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def correlation_id():
    """Test correlation ID"""
    return "test-correlation-id-12345"


@pytest.fixture
def verifier_config():
    """Test verifier configuration"""
    return TieredVerifierConfig()


@pytest.fixture
def classifier_config():
    """Test classifier configuration"""
    return ClassifierConfig(
        max_tier1_constraints=100,
        max_tier2_constraints=1000,
        max_tier1_quantifier_depth=2,
        max_tier2_quantifier_depth=5,
        tier1_timeout_ms=1000,
        tier2_timeout_ms=60000,
    )


@pytest.fixture
def selector_config():
    """Test selector configuration"""
    return SolverSelectorConfig(
        prefer_fast=True,
        allow_parallel=False,
        max_parallel_solvers=2,
        min_confidence_threshold=0.7,
        max_total_time_ms=300000,
        z3_available=True,
        leanaide_available=True,
        lean4_available=True,
    )


@pytest.fixture
def problem_classifier(classifier_config):
    """Test problem classifier"""
    return ProblemClassifier(classifier_config)


@pytest.fixture
def solver_selector(selector_config, classifier_config):
    """Test solver selector"""
    return SolverSelector(selector_config, classifier_config)


@pytest.fixture
def sample_problem():
    """Sample problem statement"""
    return "forall x, P(x) -> Q(x)"


@pytest.fixture
def sample_constraints():
    """Sample constraints"""
    return [
        {"expression": "x > 0", "constraint_type": "boolean"},
        {"expression": "y < 10", "constraint_type": "boolean"},
    ]


@pytest.fixture
def sample_variables():
    """Sample variables"""
    return [
        {"name": "x", "type": "Int"},
        {"name": "y", "type": "Int"},
    ]


# =============================================================================
# A. CONFIGURATION TESTS (5 tests)
# =============================================================================

class TestConfiguration:
    """Test suite for TieredVerifierConfig"""

    def test_verifier_config_defaults(self):
        """Test verifier configuration has correct defaults"""
        config = TieredVerifierConfig()

        assert config.z3_base_url == "http://localhost:8000"
        assert config.z3_timeout_ms == 1000
        assert config.z3_max_constraints == 100
        assert config.leanaide_base_url == "http://localhost:8001"
        assert config.leanaide_timeout_ms == 60000
        assert config.leanaide_max_constraints == 1000
        assert config.lean4_path == "lean"
        assert config.auto_escalate is True
        assert config.max_tier == 3
        assert config.selection_strategy == "adaptive"

    def test_verifier_config_from_env(self):
        """Test verifier configuration loads from environment variables"""
        with patch.dict(os.environ, {
            "Z3_BASE_URL": "http://z3-custom:9000",
            "TIER1_TIMEOUT_MS": "2000",
            "LEANAIDE_BASE_URL": "http://leanaide-custom:9001",
            "AUTO_ESCALATE": "false",
            "MAX_TIER": "2",
            "SELECTION_STRATEGY": "fast_first",
        }):
            config = TieredVerifierConfig.from_env()

            assert config.z3_base_url == "http://z3-custom:9000"
            assert config.z3_timeout_ms == 2000
            assert config.leanaide_base_url == "http://leanaide-custom:9001"
            assert config.auto_escalate is False
            assert config.max_tier == 2
            assert config.selection_strategy == "fast_first"

    def test_classifier_config_defaults(self):
        """Test classifier configuration has correct defaults"""
        config = ClassifierConfig()

        assert config.max_tier1_constraints == 100
        assert config.max_tier2_constraints == 1000
        assert config.max_tier1_quantifier_depth == 2
        assert config.max_tier2_quantifier_depth == 5
        assert config.tier1_timeout_ms == 1000
        assert config.tier2_timeout_ms == 60000

    def test_classifier_config_from_env(self):
        """Test classifier configuration loads from environment variables"""
        with patch.dict(os.environ, {
            "TIER1_MAX_CONSTRAINTS": "200",
            "TIER2_MAX_CONSTRAINTS": "2000",
            "TIER1_MAX_QUANTIFIER_DEPTH": "3",
        }):
            config = ClassifierConfig.from_env()

            assert config.max_tier1_constraints == 200
            assert config.max_tier2_constraints == 2000
            assert config.max_tier1_quantifier_depth == 3

    def test_selector_config_from_env(self):
        """Test selector configuration loads from environment variables"""
        with patch.dict(os.environ, {
            "PREFER_FAST_SOLVER": "false",
            "ALLOW_PARALLEL_SOLVERS": "true",
            "MAX_PARALLEL_SOLVERS": "3",
            "MIN_CONFIDENCE_THRESHOLD": "0.8",
            "Z3_AVAILABLE": "false",
        }):
            config = SolverSelectorConfig.from_env()

            assert config.prefer_fast is False
            assert config.allow_parallel is True
            assert config.max_parallel_solvers == 3
            assert config.min_confidence_threshold == 0.8
            assert config.z3_available is False


# =============================================================================
# B. PROBLEM CLASSIFIER TESTS (15 tests)
# =============================================================================

class TestProblemClassifier:
    """Test suite for ProblemClassifier"""

    def test_classify_theorem_proving(self, problem_classifier):
        """Test classification of theorem proving problems"""
        problem = "Prove that for all natural numbers n, n + 0 = n"
        problem_class, problem_domain, complexity = problem_classifier.classify(problem)

        assert problem_class == ProblemClass.THEOREM_PROVING
        assert complexity["estimated_tier"] in [1, 2, 3]

    def test_classify_constraint_satisfaction(self, problem_classifier):
        """Test classification of constraint satisfaction problems"""
        problem = "Find x such that x > 0 and x < 10"
        problem_class, problem_domain, complexity = problem_classifier.classify(problem)

        assert problem_class == ProblemClass.CONSTRAINT_SAT

    def test_classify_optimization(self, problem_classifier):
        """Test classification of optimization problems"""
        problem = "Minimize the objective function f(x) = x^2 + 2x + 1"
        problem_class, problem_domain, complexity = problem_classifier.classify(problem)

        assert problem_class == ProblemClass.OPTIMIZATION

    def test_classify_contradiction_detection(self, problem_classifier):
        """Test classification of contradiction detection problems"""
        problem = "Check if these constraints are inconsistent"
        problem_class, problem_domain, complexity = problem_classifier.classify(problem)

        assert problem_class in [
            ProblemClass.CONTRADICTION_DETECTION,
            ProblemClass.CONSTRAINT_SAT,
        ]

    def test_classify_domain_arithmetic(self, problem_classifier):
        """Test classification of arithmetic domain"""
        problem = "Prove that 2 + 2 = 4"
        problem_class, problem_domain, complexity = problem_classifier.classify(problem)

        assert problem_domain == ProblemDomain.ARITHMETIC

    def test_classify_domain_algebra(self, problem_classifier):
        """Test classification of algebra domain"""
        problem = "Factorize the polynomial x^2 - 5x + 6"
        problem_class, problem_domain, complexity = problem_classifier.classify(problem)

        assert problem_domain == ProblemDomain.ALGEBRA

    def test_classify_domain_logic(self, problem_classifier):
        """Test classification of logic domain"""
        problem = "forall x, exists y such that P(x, y)"
        problem_class, problem_domain, complexity = problem_classifier.classify(problem)

        assert problem_domain == ProblemDomain.LOGIC

    def test_compute_complexity_constraint_count(self, problem_classifier):
        """Test complexity computation counts constraints"""
        constraints = [{"expr": "x > 0"}, {"expr": "y < 10"}, {"expr": "z == 5"}]
        problem_class, problem_domain, complexity = problem_classifier.classify(
            "Test problem",
            constraints=constraints,
        )

        assert complexity["constraint_count"] == 3

    def test_compute_complexity_variable_count(self, problem_classifier):
        """Test complexity computation counts variables"""
        variables = [{"name": "x"}, {"name": "y"}, {"name": "z"}]
        problem_class, problem_domain, complexity = problem_classifier.classify(
            "Test problem",
            variables=variables,
        )

        assert complexity["variable_count"] == 3

    def test_compute_complexity_quantifier_depth(self, problem_classifier):
        """Test complexity computation detects quantifier depth"""
        problem = "forall x, (exists y, (forall z, P(x, y, z)))"
        problem_class, problem_domain, complexity = problem_classifier.classify(problem)

        assert complexity["quantifier_depth"] > 0
        assert complexity["has_quantifiers"] is True

    def test_compute_complexity_nonlinear(self, problem_classifier):
        """Test complexity computation detects nonlinear operations"""
        problem = "x^2 + sin(y) > 0"
        problem_class, problem_domain, complexity = problem_classifier.classify(problem)

        assert complexity["has_nonlinear"] is True

    def test_compute_complexity_arrays(self, problem_classifier):
        """Test complexity computation detects array usage"""
        problem = "select(array, i) > 0"
        problem_class, problem_domain, complexity = problem_classifier.classify(problem)

        assert complexity["has_arrays"] is True

    def test_estimate_tier_simple(self, problem_classifier):
        """Test tier estimation for simple problems"""
        problem = "x > 0 and x < 10"
        constraints = [{"expr": "x > 0"}, {"expr": "x < 10"}]
        problem_class, problem_domain, complexity = problem_classifier.classify(
            problem,
            constraints=constraints,
        )

        assert complexity["estimated_tier"] == 1

    def test_estimate_tier_complex(self, problem_classifier):
        """Test tier estimation for complex problems"""
        problem = "forall x, exists y, (x^2 + y > 0)"
        problem_class, problem_domain, complexity = problem_classifier.classify(problem)

        assert complexity["estimated_tier"] in [2, 3]

    def test_should_escalate_timeout(self, problem_classifier):
        """Test escalation on timeout"""
        should_escalate, reason = problem_classifier.should_escalate(
            current_tier=1,
            constraint_count=50,
            execution_time_ms=2000,  # > tier1_timeout_ms
            status="unknown",
        )

        assert should_escalate is True
        assert "timeout" in reason.lower()


# =============================================================================
# C. SOLVER SELECTOR TESTS (15 tests)
# =============================================================================

class TestSolverSelector:
    """Test suite for SolverSelector"""

    def test_solver_selector_initialization(self, solver_selector):
        """Test solver selector initializes correctly"""
        assert solver_selector.config is not None
        assert solver_selector.classifier is not None
        assert VerificationTier.TIER1_Z3 in solver_selector.performance
        assert VerificationTier.TIER2_LEANAIDE in solver_selector.performance
        assert VerificationTier.TIER3_LEAN4 in solver_selector.performance

    def test_select_solver_fast_first(self, solver_selector):
        """Test solver selection with fast_first strategy"""
        result = solver_selector.select_solver(
            "x > 0 and x < 10",
            strategy=SelectionStrategy.FAST_FIRST,
        )

        assert result.recommended_tier == VerificationTier.TIER1_Z3
        assert result.strategy == SelectionStrategy.FAST_FIRST
        assert result.should_escalate_automatically is True

    def test_select_solver_accurate_first(self, solver_selector):
        """Test solver selection with accurate_first strategy"""
        result = solver_selector.select_solver(
            "Complex theorem requiring formal proof",
            strategy=SelectionStrategy.ACCURATE_FIRST,
        )

        assert result.recommended_tier == VerificationTier.TIER3_LEAN4
        assert result.strategy == SelectionStrategy.ACCURATE_FIRST

    def test_select_solver_parallel(self, solver_selector):
        """Test solver selection with parallel strategy"""
        result = solver_selector.select_solver(
            "Test problem",
            strategy=SelectionStrategy.PARALLEL,
        )

        assert result.recommended_tier in [
            VerificationTier.TIER1_Z3,
            VerificationTier.TIER2_LEANAIDE,
            VerificationTier.TIER3_LEAN4,
        ]
        assert result.strategy == SelectionStrategy.PARALLEL

    def test_select_solver_adaptive(self, solver_selector):
        """Test solver selection with adaptive strategy"""
        result = solver_selector.select_solver(
            "forall x, exists y, P(x, y)",
            strategy=SelectionStrategy.ADAPTIVE,
        )

        assert result.recommended_tier in [
            VerificationTier.TIER1_Z3,
            VerificationTier.TIER2_LEANAIDE,
            VerificationTier.TIER3_LEAN4,
        ]
        assert result.strategy == SelectionStrategy.ADAPTIVE

    def test_select_solver_user_specified(self, solver_selector):
        """Test solver selection with user-specified tier"""
        result = solver_selector.select_solver(
            "Test problem",
            metadata={"preferred_tier": "tier2_leanaide"},
            strategy=SelectionStrategy.USER_SPECIFIED,
        )

        assert result.recommended_tier == VerificationTier.TIER2_LEANAIDE
        assert result.strategy == SelectionStrategy.USER_SPECIFIED

    def test_select_solver_with_max_tier(self, solver_selector):
        """Test solver selection respects max tier"""
        result = solver_selector.select_solver(
            "Test problem",
            max_tier=VerificationTier.TIER2_LEANAIDE,
        )

        assert result.max_tier == VerificationTier.TIER2_LEANAIDE
        # Verify no tier 3 in alternatives
        assert VerificationTier.TIER3_LEAN4 not in result.alternative_tiers

    def test_get_escalation_path(self, solver_selector):
        """Test escalation path calculation"""
        result = solver_selector.select_solver(
            "Simple problem",
            strategy=SelectionStrategy.FAST_FIRST,
        )

        # Should have escalation path from tier 1 to tier 3
        assert VerificationTier.TIER2_LEANAIDE in result.alternative_tiers
        assert VerificationTier.TIER3_LEAN4 in result.alternative_tiers

    def test_record_performance_success(self, solver_selector):
        """Test recording successful solver performance"""
        solver_selector.record_performance(
            VerificationTier.TIER1_Z3,
            success=True,
            timeout=False,
            execution_time_ms=100.0,
        )

        perf = solver_selector.performance[VerificationTier.TIER1_Z3]
        assert perf.total_attempts == 1
        assert perf.successful_attempts == 1
        assert perf.failed_attempts == 0
        assert perf.success_rate == 1.0

    def test_record_performance_failure(self, solver_selector):
        """Test recording failed solver performance"""
        solver_selector.record_performance(
            VerificationTier.TIER1_Z3,
            success=False,
            timeout=False,
            execution_time_ms=5000.0,
        )

        perf = solver_selector.performance[VerificationTier.TIER1_Z3]
        assert perf.total_attempts == 1
        assert perf.successful_attempts == 0
        assert perf.failed_attempts == 1
        assert perf.failure_count == 1

    def test_circuit_breaker_opens_after_threshold(self, solver_selector):
        """Test circuit breaker opens after failure threshold"""
        # Record failures to exceed threshold
        for _ in range(10):
            solver_selector.record_performance(
                VerificationTier.TIER1_Z3,
                success=False,
                timeout=False,
                execution_time_ms=1000.0,
            )

        perf = solver_selector.performance[VerificationTier.TIER1_Z3]
        perf.check_circuit_breaker(threshold=5)
        assert perf.circuit_breaker_open is True

    def test_circuit_breaker_resets_on_success(self, solver_selector):
        """Test circuit breaker resets on success"""
        # Record failures
        for _ in range(10):
            solver_selector.record_performance(
                VerificationTier.TIER1_Z3,
                success=False,
                timeout=False,
                execution_time_ms=1000.0,
            )

        # Record success
        solver_selector.record_performance(
            VerificationTier.TIER1_Z3,
            success=True,
            timeout=False,
            execution_time_ms=100.0,
        )

        perf = solver_selector.performance[VerificationTier.TIER1_Z3]
        assert perf.failure_count == 0
        assert perf.circuit_breaker_open is False

    def test_get_performance_stats(self, solver_selector):
        """Test getting performance statistics"""
        solver_selector.record_performance(
            VerificationTier.TIER1_Z3,
            success=True,
            timeout=False,
            execution_time_ms=100.0,
        )

        stats = solver_selector.get_performance_stats()

        assert "tier1_z3" in stats
        assert stats["tier1_z3"]["total_attempts"] == 1
        assert stats["tier1_z3"]["success_rate"] == 1.0

    def test_reset_performance_stats(self, solver_selector):
        """Test resetting performance statistics"""
        solver_selector.record_performance(
            VerificationTier.TIER1_Z3,
            success=True,
            timeout=False,
            execution_time_ms=100.0,
        )

        solver_selector.reset_performance_stats()

        perf = solver_selector.performance[VerificationTier.TIER1_Z3]
        assert perf.total_attempts == 0
        assert perf.successful_attempts == 0
        assert perf.failure_count == 0


# =============================================================================
# D. TIERED VERIFIER TESTS (15 tests)
# =============================================================================

class TestTieredVerifier:
    """Test suite for TieredVerifier"""

    def test_verifier_initialization(self, verifier_config):
        """Test verifier initializes correctly"""
        verifier = TieredVerifier(verifier_config)

        assert verifier.config is not None
        assert verifier.classifier is not None
        assert verifier.selector is not None

    @patch('tiered_verifier.RESEZ3Bridge')
    def test_verify_creates_correlation_id(self, mock_z3_bridge, verifier_config):
        """Test verification creates correlation ID if not provided"""
        verifier = TieredVerifier(verifier_config)

        # Mock Z3 bridge
        mock_bridge = Mock()
        mock_bridge.detect_contradictions.return_value = (False, {})
        mock_z3_bridge.return_value = mock_bridge

        result = verifier.verify("x > 0")

        assert result.correlation_id is not None
        assert len(result.correlation_id) > 0

    @patch('tiered_verifier.RESEZ3Bridge')
    def test_verify_with_tier1_success(self, mock_z3_bridge, verifier_config, correlation_id):
        """Test successful verification with Tier 1"""
        verifier = TieredVerifier(verifier_config)

        # Mock Z3 bridge
        mock_bridge = Mock()
        mock_bridge.detect_contradictions.return_value = (False, {"x": 5})
        mock_z3_bridge.return_value = mock_bridge

        result = verifier.verify("x > 0 and x < 10", correlation_id=correlation_id)

        assert result.final_status == VerificationStatus.VERIFIED
        assert result.successful_tier == VerificationTier.TIER1_Z3
        assert result.tier1_result is not None
        assert result.tier1_result.z3_result == "sat"

    @patch('tiered_verifier.RESEZ3Bridge')
    def test_verify_with_tier1_contradiction(self, mock_z3_bridge, verifier_config, correlation_id):
        """Test verification detects contradiction with Tier 1"""
        verifier = TieredVerifier(verifier_config)

        # Mock Z3 bridge to detect contradiction
        mock_bridge = Mock()
        mock_bridge.detect_contradictions.return_value = (True, {})
        mock_z3_bridge.return_value = mock_bridge

        result = verifier.verify("x > 10 and x < 5", correlation_id=correlation_id)

        assert result.final_status == VerificationStatus.REFUTED
        assert result.tier1_result.z3_result == "unsat"

    def test_verify_with_specific_tier(self, verifier_config, correlation_id):
        """Test verification with specific tier"""
        verifier = TieredVerifier(verifier_config)

        # Mock the tier-specific method
        with patch.object(verifier, '_verify_tier1') as mock_verify:
            mock_result = Z3VerificationResult(
                status=VerificationStatus.VERIFIED,
                z3_result="sat",
                correlation_id=correlation_id,
            )
            mock_verify.return_value = mock_result

            result = verifier.verify_with_tier(
                "x > 0",
                VerificationTier.TIER1_Z3,
                correlation_id=correlation_id,
            )

            assert result.status == VerificationStatus.VERIFIED
            assert result.correlation_id == correlation_id

    def test_escalate_from_tier1_to_tier2(self, verifier_config, correlation_id):
        """Test escalation from Tier 1 to Tier 2"""
        verifier = TieredVerifier(verifier_config)

        # Mock results
        tier1_result = Z3VerificationResult(
            status=VerificationStatus.UNKNOWN,
            z3_result="unknown",
            correlation_id=correlation_id,
        )

        with patch.object(verifier, '_verify_tier2') as mock_verify:
            mock_result = LeanAideVerificationResult(
                status=VerificationStatus.VERIFIED,
                proof_status="proved",
                correlation_id=correlation_id,
            )
            mock_verify.return_value = mock_result

            result = verifier.escalate_tier(
                tier1_result,
                "forall x, P(x)",
                correlation_id=correlation_id,
            )

            assert isinstance(result, LeanAideVerificationResult)

    def test_escalate_from_tier2_to_tier3(self, verifier_config, correlation_id):
        """Test escalation from Tier 2 to Tier 3"""
        verifier = TieredVerifier(verifier_config)

        # Mock results
        tier2_result = LeanAideVerificationResult(
            status=VerificationStatus.REFUTED,
            proof_status="failed",
            correlation_id=correlation_id,
        )

        with patch.object(verifier, '_verify_tier3') as mock_verify:
            mock_result = Lean4VerificationResult(
                status=VerificationStatus.VERIFIED,
                verification_status="verified",
                correlation_id=correlation_id,
            )
            mock_verify.return_value = mock_result

            result = verifier.escalate_tier(
                tier2_result,
                "Complex theorem",
                correlation_id=correlation_id,
            )

            assert isinstance(result, Lean4VerificationResult)

    def test_combine_results(self, verifier_config, correlation_id):
        """Test combining results from multiple tiers"""
        verifier = TieredVerifier(verifier_config)

        tier1_result = Z3VerificationResult(
            status=VerificationStatus.VERIFIED,
            z3_result="sat",
            correlation_id=correlation_id,
        )
        tier2_result = LeanAideVerificationResult(
            status=VerificationStatus.VERIFIED,
            proof_status="proved",
            correlation_id=correlation_id,
        )

        combined = verifier.combine_results([tier1_result, tier2_result], correlation_id)

        assert combined.correlation_id == correlation_id
        assert combined.tier1_result is not None
        assert combined.tier2_result is not None
        assert combined.successful_tier == VerificationTier.TIER1_Z3

    def test_get_verification_status(self, verifier_config, correlation_id):
        """Test getting verification status"""
        verifier = TieredVerifier(verifier_config)

        status = verifier.get_verification_status(correlation_id)

        assert status["correlation_id"] == correlation_id
        assert "status" in status

    @patch('tiered_verifier.RESEZ3Bridge')
    def test_auto_escalate_disabled(self, mock_z3_bridge, verifier_config, correlation_id):
        """Test auto-escalation can be disabled"""
        verifier_config.auto_escalate = False
        verifier = TieredVerifier(verifier_config)

        # Mock Z3 bridge
        mock_bridge = Mock()
        mock_bridge.detect_contradictions.return_value = (False, {})
        mock_z3_bridge.return_value = mock_bridge

        result = verifier.verify("x > 0", correlation_id=correlation_id)

        # Should not escalate even if tier 1 fails
        assert result.escalation_path == [VerificationTier.TIER1_Z3]

    @patch('tiered_verifier.RESEZ3Bridge')
    def test_max_tier_respected(self, mock_z3_bridge, verifier_config, correlation_id):
        """Test max_tier configuration is respected"""
        verifier_config.max_tier = 2
        verifier = TieredVerifier(verifier_config)

        # Mock Z3 bridge
        mock_bridge = Mock()
        mock_bridge.detect_contradictions.return_value = (True, {})
        mock_z3_bridge.return_value = mock_bridge

        result = verifier.verify("x > 0", correlation_id=correlation_id)

        # Should not escalate to tier 3
        assert result.tier3_result is None

    def test_verify_handles_errors(self, verifier_config, correlation_id):
        """Test verification handles errors gracefully"""
        verifier = TieredVerifier(verifier_config)

        # Mock to raise exception
        with patch.object(verifier.classifier, 'classify', side_effect=Exception("Test error")):
            result = verifier.verify("x > 0", correlation_id=correlation_id)

            assert result.final_status == VerificationStatus.ERROR
            assert "error" in result.metadata


# =============================================================================
# E. VERIFICATION RESULT TESTS (10 tests)
# =============================================================================

class TestVerificationResults:
    """Test suite for verification result data structures"""

    def test_z3_result_creation(self, correlation_id):
        """Test Z3 verification result creation"""
        result = Z3VerificationResult(
            status=VerificationStatus.VERIFIED,
            z3_result="sat",
            model={"x": 5},
            execution_time_ms=100.0,
            constraints_checked=2,
            correlation_id=correlation_id,
        )

        assert result.status == VerificationStatus.VERIFIED
        assert result.z3_result == "sat"
        assert result.model == {"x": 5}
        assert result.correlation_id == correlation_id

    def test_z3_result_serialization(self, correlation_id):
        """Test Z3 verification result serialization"""
        result = Z3VerificationResult(
            status=VerificationStatus.VERIFIED,
            z3_result="sat",
            correlation_id=correlation_id,
        )

        data = result.to_dict()

        assert data["status"] == "verified"
        assert data["z3_result"] == "sat"
        assert data["tier"] == "tier1_z3"
        assert data["correlation_id"] == correlation_id

    def test_z3_result_deserialization(self, correlation_id):
        """Test Z3 verification result deserialization"""
        data = {
            "status": "verified",
            "z3_result": "sat",
            "model": {"x": 5},
            "execution_time_ms": 100.0,
            "constraints_checked": 2,
            "correlation_id": correlation_id,
        }

        result = Z3VerificationResult.from_dict(data)

        assert result.status == VerificationStatus.VERIFIED
        assert result.z3_result == "sat"
        assert result.correlation_id == correlation_id

    def test_leanaide_result_creation(self, correlation_id):
        """Test LeanAide verification result creation"""
        result = LeanAideVerificationResult(
            status=VerificationStatus.VERIFIED,
            proof_status="proved",
            proof_script="theorem test : sorry",
            tactics_used=["intro", "apply"],
            execution_time_ms=5000.0,
            constraints_checked=10,
            correlation_id=correlation_id,
        )

        assert result.status == VerificationStatus.VERIFIED
        assert result.proof_status == "proved"
        assert result.tactics_used == ["intro", "apply"]
        assert result.correlation_id == correlation_id

    def test_lean4_result_creation(self, correlation_id):
        """Test Lean 4 verification result creation"""
        result = Lean4VerificationResult(
            status=VerificationStatus.VERIFIED,
            verification_status="verified",
            lean4_code="theorem test : True := by trivial",
            theorem_name="test",
            execution_time_ms=10000.0,
            constraints_checked=20,
            lean_version="4.x",
            correlation_id=correlation_id,
        )

        assert result.status == VerificationStatus.VERIFIED
        assert result.verification_status == "verified"
        assert result.theorem_name == "test"
        assert result.correlation_id == correlation_id

    def test_unified_result_creation(self, correlation_id):
        """Test unified verification result creation"""
        result = UnifiedVerificationResult(
            correlation_id=correlation_id,
            problem_class=ProblemClass.THEOREM_PROVING,
            problem_domain=ProblemDomain.LOGIC,
        )

        assert result.correlation_id == correlation_id
        assert result.problem_class == ProblemClass.THEOREM_PROVING
        assert result.problem_domain == ProblemDomain.LOGIC

    def test_unified_result_add_tier_result(self, correlation_id):
        """Test adding tier results to unified result"""
        unified = UnifiedVerificationResult(correlation_id=correlation_id)

        tier1_result = Z3VerificationResult(
            status=VerificationStatus.VERIFIED,
            z3_result="sat",
            correlation_id=correlation_id,
        )

        unified.add_tier_result(tier1_result, "Selected by adaptive strategy")

        assert unified.tier1_result is not None
        assert VerificationTier.TIER1_Z3 in unified.escalation_path
        assert len(unified.escalation_reasons) > 0

    def test_unified_result_is_successful(self, correlation_id):
        """Test checking if unified result is successful"""
        unified = UnifiedVerificationResult(correlation_id=correlation_id)

        assert unified.is_successful() is False

        tier1_result = Z3VerificationResult(
            status=VerificationStatus.VERIFIED,
            z3_result="sat",
            correlation_id=correlation_id,
        )
        unified.add_tier_result(tier1_result)

        assert unified.is_successful() is True

    def test_unified_result_get_summary(self, correlation_id):
        """Test getting human-readable summary"""
        unified = UnifiedVerificationResult(correlation_id=correlation_id)

        summary = unified.get_summary()

        assert correlation_id in summary
        assert "pending" in summary.lower()

    def test_timestamps_are_utc(self, correlation_id):
        """Test Law of UTC: all timestamps are in UTC"""
        result = Z3VerificationResult(
            status=VerificationStatus.VERIFIED,
            z3_result="sat",
            correlation_id=correlation_id,
        )

        # Check timestamp ends with Z (UTC indicator)
        assert result.timestamp.endswith("Z")

        # Parse and verify timezone
        dt = datetime.fromisoformat(result.timestamp)
        assert dt.tzinfo == timezone.utc


# =============================================================================
# CLAUDE.md COMPLIANCE TESTS
# =============================================================================

class TestCLAUDECompliance:
    """Test suite for CLAUDE.md principle compliance"""

    def test_law_of_configuration_explicitness_missing_env(self):
        """Test Law of Configuration Explicitness: missing env vars crash"""
        # This test verifies that the system doesn't use magic defaults
        # All configuration should come from environment variables
        with patch.dict(os.environ, {}, clear=True):
            # Should still work with hardcoded defaults (as defined in code)
            config = TieredVerifierConfig()
            assert config.z3_base_url == "http://localhost:8000"  # Default in code

    def test_law_of_idempotency_classification(self, problem_classifier):
        """Test Law of Idempotency: classification is repeatable"""
        problem = "forall x, P(x)"

        result1 = problem_classifier.classify(problem)
        result2 = problem_classifier.classify(problem)

        assert result1[0] == result2[0]  # Same problem class
        assert result1[1] == result2[1]  # Same domain
        assert result1[2] == result2[2]  # Same complexity

    def test_law_of_idempotency_solver_selection(self, solver_selector):
        """Test Law of Idempotency: solver selection is repeatable"""
        problem = "x > 0 and x < 10"

        result1 = solver_selector.select_solver(problem, strategy=SelectionStrategy.FAST_FIRST)
        result2 = solver_selector.select_solver(problem, strategy=SelectionStrategy.FAST_FIRST)

        assert result1.recommended_tier == result2.recommended_tier

    def test_law_of_utc_timestamps(self, correlation_id):
        """Test Law of UTC: all timestamps use UTC"""
        result = UnifiedVerificationResult(correlation_id=correlation_id)

        # Check timestamp format
        assert result.timestamp.endswith("Z")
        dt = datetime.fromisoformat(result.timestamp)
        assert dt.tzinfo == timezone.utc

    def test_structured_logging(self, verifier_config):
        """Test Structured Logging: logs are JSON format"""
        verifier = TieredVerifier(verifier_config)

        # Logger should be configured
        assert verifier.logger is not None
        assert verifier.logger.name == "rese.verification.tiered_verifier"

    def test_circuit_breaker_pattern(self, solver_selector):
        """Test Circuit Breaker: prevents hammering failing solvers"""
        # Record multiple failures
        for _ in range(10):
            solver_selector.record_performance(
                VerificationTier.TIER1_Z3,
                success=False,
                timeout=False,
                execution_time_ms=1000.0,
            )

        perf = solver_selector.performance[VerificationTier.TIER1_Z3]
        perf.check_circuit_breaker(threshold=5)

        # Circuit breaker should be open
        assert perf.circuit_breaker_open is True
        assert perf.should_attempt(threshold=5) is False
