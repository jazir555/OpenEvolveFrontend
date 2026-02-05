"""
Comprehensive Test Suite for Tiered Verification System

Tests all 3 tiers independently and together:
- Tier 1: Z3 Fast Verification
- Tier 2: LeanAide AI-Assisted Proving
- Tier 3: Lean 4 Formal Verification

Coverage:
- Problem classification
- Solver selection
- Tier escalation
- Result combination
- Unified API
- Performance monitoring

Following CLAUDE.md principles:
- Law of Runtime Truth: Test against real systems
- Law of Idempotency: Tests safe to run 100x
- Structured Logging: JSON output

Author: RESE Team
Created: 2026-02-04
"""

import os
import sys
import unittest
import json
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Try relative imports first, then absolute
try:
    from src.verification_result import (
        VerificationTier,
        VerificationStatus,
        ProblemClass,
        ProblemDomain,
        Z3VerificationResult,
        LeanAideVerificationResult,
        Lean4VerificationResult,
        UnifiedVerificationResult,
    )

    from src.problem_classifier import (
        ClassifierConfig,
        ProblemClassifier,
        classify_problem,
        should_escalate,
    )

    from src.solver_selector import (
        SolverSelectorConfig,
        SolverPerformance,
        SelectionStrategy,
        SelectionResult,
        SolverSelector,
        select_solver,
    )

    from src.tiered_verifier import (
        TieredVerifierConfig,
        TieredVerifier,
        verify,
    )
except ImportError:
    # Fallback to direct imports
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

    from problem_classifier import (
        ClassifierConfig,
        ProblemClassifier,
        classify_problem,
        should_escalate,
    )

    from solver_selector import (
        SolverSelectorConfig,
        SolverPerformance,
        SelectionStrategy,
        SelectionResult,
        SolverSelector,
        select_solver,
    )

    from tiered_verifier import (
        TieredVerifierConfig,
        TieredVerifier,
        verify,
    )


# =============================================================================
# TEST VERIFICATION RESULT DATA STRUCTURES
# =============================================================================

class TestVerificationResults(unittest.TestCase):
    """Test verification result data structures"""

    def test_z3_result_creation(self):
        """Test Z3 verification result creation"""
        result = Z3VerificationResult(
            status=VerificationStatus.VERIFIED,
            z3_result="sat",
            model={"x": "1"},
            execution_time_ms=100.0,
            constraints_checked=5,
            correlation_id="test-123",
        )

        self.assertEqual(result.status, VerificationStatus.VERIFIED)
        self.assertEqual(result.z3_result, "sat")
        self.assertEqual(result.execution_time_ms, 100.0)
        self.assertTrue(result.is_successful())
        self.assertFalse(result.should_escalate())

    def test_z3_result_serialization(self):
        """Test Z3 result serialization"""
        result = Z3VerificationResult(
            status=VerificationStatus.VERIFIED,
            z3_result="sat",
            correlation_id="test-123",
        )

        data = result.to_dict()
        self.assertEqual(data["tier"], "tier1_z3")
        self.assertEqual(data["status"], "verified")
        self.assertEqual(data["z3_result"], "sat")

        # Test deserialization
        result2 = Z3VerificationResult.from_dict(data)
        self.assertEqual(result2.status, result.status)
        self.assertEqual(result2.z3_result, result.z3_result)

    def test_z3_result_escalation(self):
        """Test Z3 result escalation conditions"""
        # Too many constraints
        result1 = Z3VerificationResult(
            status=VerificationStatus.UNKNOWN,
            z3_result="unknown",
            constraints_checked=150,
        )
        self.assertTrue(result1.should_escalate())

        # Timeout
        result2 = Z3VerificationResult(
            status=VerificationStatus.TIMEOUT,
            z3_result="unknown",
            execution_time_ms=6000,
        )
        self.assertTrue(result2.should_escalate())

        # Successful - no escalation
        result3 = Z3VerificationResult(
            status=VerificationStatus.VERIFIED,
            z3_result="sat",
            constraints_checked=10,
        )
        self.assertFalse(result3.should_escalate())

    def test_leanaide_result_creation(self):
        """Test LeanAide verification result creation"""
        result = LeanAideVerificationResult(
            status=VerificationStatus.VERIFIED,
            proof_status="proved",
            proof_script="theorem test : True := by trivial",
            tactics_used=["trivial"],
            autoformalization_confidence=0.95,
            execution_time_ms=5000.0,
            constraints_checked=50,
            correlation_id="test-123",
        )

        self.assertEqual(result.status, VerificationStatus.VERIFIED)
        self.assertEqual(result.proof_status, "proved")
        self.assertIn("trivial", result.tactics_used)
        self.assertTrue(result.is_successful())
        self.assertFalse(result.should_escalate())

    def test_leanaide_result_escalation(self):
        """Test LeanAide result escalation conditions"""
        # Failed proof
        result1 = LeanAideVerificationResult(
            status=VerificationStatus.REFUTED,
            proof_status="failed",
            constraints_checked=50,
        )
        self.assertTrue(result1.should_escalate())

        # Too many constraints
        result2 = LeanAideVerificationResult(
            status=VerificationStatus.VERIFIED,
            proof_status="proved",
            constraints_checked=1500,
        )
        self.assertTrue(result2.should_escalate())

    def test_lean4_result_creation(self):
        """Test Lean 4 verification result creation"""
        result = Lean4VerificationResult(
            status=VerificationStatus.VERIFIED,
            verification_status="verified",
            lean4_code="theorem test : True := by trivial",
            theorem_name="test_theorem",
            execution_time_ms=10000.0,
            constraints_checked=500,
            lean_version="4.0.0",
            correlation_id="test-123",
        )

        self.assertEqual(result.status, VerificationStatus.VERIFIED)
        self.assertEqual(result.verification_status, "verified")
        self.assertTrue(result.is_successful())
        # Lean 4 never escalates
        self.assertFalse(result.should_escalate())

    def test_unified_result_creation(self):
        """Test unified verification result creation"""
        unified = UnifiedVerificationResult(
            correlation_id="test-123",
            problem_class=ProblemClass.THEOREM_PROVING,
            problem_domain=ProblemDomain.LOGIC,
        )

        self.assertEqual(unified.correlation_id, "test-123")
        self.assertEqual(unified.problem_class, ProblemClass.THEOREM_PROVING)
        self.assertEqual(unified.final_status, VerificationStatus.PENDING)

    def test_unified_result_add_tier1(self):
        """Test adding Tier 1 result to unified result"""
        unified = UnifiedVerificationResult(correlation_id="test-123")

        z3_result = Z3VerificationResult(
            status=VerificationStatus.VERIFIED,
            z3_result="sat",
            execution_time_ms=100.0,
            constraints_checked=5,
        )

        unified.add_tier_result(z3_result, "Initial verification")

        self.assertEqual(unified.tier1_result, z3_result)
        self.assertEqual(unified.final_status, VerificationStatus.VERIFIED)
        self.assertEqual(unified.successful_tier, VerificationTier.TIER1_Z3)
        self.assertEqual(unified.confidence, 0.7)  # Z3 confidence
        self.assertEqual(unified.total_execution_time_ms, 100.0)
        self.assertEqual(unified.total_constraints_checked, 5)

    def test_unified_result_add_tier2(self):
        """Test adding Tier 2 result to unified result"""
        unified = UnifiedVerificationResult(correlation_id="test-123")

        leanaide_result = LeanAideVerificationResult(
            status=VerificationStatus.VERIFIED,
            proof_status="proved",
            execution_time_ms=5000.0,
            constraints_checked=50,
        )

        unified.add_tier_result(leanaide_result, "Escalated from Tier 1")

        self.assertEqual(unified.tier2_result, leanaide_result)
        self.assertEqual(unified.final_status, VerificationStatus.VERIFIED)
        self.assertEqual(unified.successful_tier, VerificationTier.TIER2_LEANAIDE)
        self.assertEqual(unified.confidence, 0.85)  # LeanAide confidence

    def test_unified_result_add_tier3(self):
        """Test adding Tier 3 result to unified result"""
        unified = UnifiedVerificationResult(correlation_id="test-123")

        lean4_result = Lean4VerificationResult(
            status=VerificationStatus.VERIFIED,
            verification_status="verified",
            execution_time_ms=10000.0,
            constraints_checked=500,
        )

        unified.add_tier_result(lean4_result, "Escalated from Tier 2")

        self.assertEqual(unified.tier3_result, lean4_result)
        self.assertEqual(unified.final_status, VerificationStatus.VERIFIED)
        self.assertEqual(unified.successful_tier, VerificationTier.TIER3_LEAN4)
        self.assertEqual(unified.confidence, 1.0)  # Lean 4 confidence

    def test_unified_result_get_summary(self):
        """Test unified result summary generation"""
        unified = UnifiedVerificationResult(correlation_id="test-123")

        # Successful verification
        z3_result = Z3VerificationResult(
            status=VerificationStatus.VERIFIED,
            z3_result="sat",
            execution_time_ms=100.0,
            constraints_checked=5,
        )
        unified.add_tier_result(z3_result)

        summary = unified.get_summary()
        self.assertIn("✓", summary)
        self.assertIn("Tier1_Z3", summary)
        self.assertIn("70.0%", summary)

        # Failed verification
        unified2 = UnifiedVerificationResult(correlation_id="test-456")
        unified2.final_status = VerificationStatus.REFUTED
        unified2.escalation_path = [VerificationTier.TIER1_Z3, VerificationTier.TIER2_LEANAIDE]
        unified2.total_execution_time_ms = 6000.0

        summary2 = unified2.get_summary()
        self.assertIn("✗", summary2)
        self.assertIn("failed", summary2)


# =============================================================================
# TEST PROBLEM CLASSIFIER
# =============================================================================

class TestProblemClassifier(unittest.TestCase):
    """Test problem classifier"""

    def setUp(self):
        """Set up test fixtures"""
        self.classifier = ProblemClassifier()

    def test_classify_simple_constraint(self):
        """Test classification of simple constraint problem"""
        problem = "Find x such that x > 0 and x < 10"
        problem_class, problem_domain, complexity = self.classifier.classify(problem)

        self.assertEqual(problem_class, ProblemClass.CONSTRAINT_SAT)
        self.assertEqual(complexity["constraint_count"], 0)  # No constraints provided
        self.assertEqual(complexity["has_quantifiers"], False)
        self.assertEqual(complexity["estimated_tier"], 1)  # Simple enough for Tier 1

    def test_classify_theorem_proving(self):
        """Test classification of theorem proving problem"""
        problem = "Prove that for all x, if P(x) then Q(x)"
        problem_class, problem_domain, complexity = self.classifier.classify(problem)

        self.assertEqual(problem_class, ProblemClass.THEOREM_PROVING)
        self.assertEqual(complexity["has_quantifiers"], True)
        self.assertGreater(complexity["quantifier_depth"], 0)
        self.assertGreaterEqual(complexity["estimated_tier"], 2)  # Needs at least Tier 2

    def test_classify_optimization(self):
        """Test classification of optimization problem"""
        problem = "Minimize f(x) = x^2 subject to x > 0"
        problem_class, problem_domain, complexity = self.classifier.classify(problem)

        self.assertEqual(problem_class, ProblemClass.OPTIMIZATION)

    def test_classify_arithmetic_domain(self):
        """Test domain classification for arithmetic"""
        problem = "Prove that 1 + 1 = 2"
        problem_class, problem_domain, complexity = self.classifier.classify(problem)

        self.assertEqual(problem_domain, ProblemDomain.ARITHMETIC)

    def test_classify_algebra_domain(self):
        """Test domain classification for algebra"""
        problem = "Prove that (a + b)^2 = a^2 + 2ab + b^2"
        problem_class, problem_domain, complexity = self.classifier.classify(problem)

        self.assertEqual(problem_domain, ProblemDomain.ALGEBRA)

    def test_classify_logic_domain(self):
        """Test domain classification for logic"""
        problem = "For all propositions P and Q, P and Q implies P"
        problem_class, problem_domain, complexity = self.classifier.classify(problem)

        self.assertEqual(problem_domain, ProblemDomain.LOGIC)

    def test_count_quantifier_depth(self):
        """Test quantifier depth counting"""
        # No quantifiers
        depth1 = self.classifier._count_quantifier_depth("x > 0 and x < 10")
        self.assertEqual(depth1, 0)

        # Single quantifier
        depth2 = self.classifier._count_quantifier_depth("for all x, P(x)")
        self.assertGreater(depth2, 0)

        # Nested quantifiers
        depth3 = self.classifier._count_quantifier_depth("for all x, exists y, P(x, y)")
        self.assertGreater(depth3, depth2)

    def test_detect_nonlinear(self):
        """Test nonlinear operation detection"""
        # Linear
        nonlinear1 = self.classifier._has_nonlinear("x + y > 0")
        self.assertFalse(nonlinear1)

        # Nonlinear (power)
        nonlinear2 = self.classifier._has_nonlinear("x^2 + y > 0")
        self.assertTrue(nonlinear2)

        # Nonlinear (trigonometric)
        nonlinear3 = self.classifier._has_nonlinear("sin(x) > 0")
        self.assertTrue(nonlinear3)

    def test_should_escalate_from_tier1(self):
        """Test escalation logic from Tier 1"""
        # Too many constraints
        should1, reason1 = self.classifier.should_escalate(
            current_tier=1,
            constraint_count=150,
            execution_time_ms=500,
            status="sat"
        )
        self.assertTrue(should1)
        self.assertIn("Too many constraints", reason1)

        # Timeout
        should2, reason2 = self.classifier.should_escalate(
            current_tier=1,
            constraint_count=10,
            execution_time_ms=1500,
            status="unknown"
        )
        self.assertTrue(should2)
        self.assertIn("timeout", reason2.lower())

        # No escalation needed
        should3, reason3 = self.classifier.should_escalate(
            current_tier=1,
            constraint_count=10,
            execution_time_ms=100,
            status="sat"
        )
        self.assertFalse(should3)

    def test_should_not_escalate_from_tier3(self):
        """Test that Tier 3 never escalates"""
        should, reason = self.classifier.should_escalate(
            current_tier=3,
            constraint_count=10000,
            execution_time_ms=100000,
            status="unknown"
        )
        self.assertFalse(should)
        self.assertIn("final tier", reason)


# =============================================================================
# TEST SOLVER SELECTOR
# =============================================================================

class TestSolverSelector(unittest.TestCase):
    """Test solver selector"""

    def setUp(self):
        """Set up test fixtures"""
        self.selector = SolverSelector()

    def test_select_fast_first_simple(self):
        """Test fast-first selection for simple problem"""
        result = self.selector.select_solver("Find x such that x > 0")

        self.assertEqual(result.recommended_tier, VerificationTier.TIER1_Z3)
        self.assertEqual(result.strategy, SelectionStrategy.FAST_FIRST)
        self.assertIn(VerificationTier.TIER2_LEANAIDE, result.alternative_tiers)
        self.assertTrue(result.should_escalate_automatically)

    def test_select_accurate_first(self):
        """Test accurate-first selection"""
        result = self.selector.select_solver(
            "Prove theorem",
            strategy=SelectionStrategy.ACCURATE_FIRST
        )

        # Should select highest available tier
        self.assertIn(result.recommended_tier, [
            VerificationTier.TIER3_LEAN4,
            VerificationTier.TIER2_LEANAIDE,
            VerificationTier.TIER1_Z3
        ])
        self.assertEqual(result.strategy, SelectionStrategy.ACCURATE_FIRST)
        self.assertFalse(result.should_escalate_automatically)

    def test_select_parallel(self):
        """Test parallel selection"""
        result = self.selector.select_solver(
            "Prove theorem",
            strategy=SelectionStrategy.PARALLEL
        )

        self.assertEqual(result.strategy, SelectionStrategy.PARALLEL)
        self.assertGreater(len(result.alternative_tiers), 0)

    def test_select_adaptive(self):
        """Test adaptive selection"""
        # Simple problem
        result1 = self.selector.select_solver("Find x such that x > 0")

        self.assertEqual(result1.strategy, SelectionStrategy.FAST_FIRST if self.selector.config.prefer_fast else SelectionStrategy.ADAPTIVE)

        # Complex problem
        result2 = self.selector.select_solver(
            "For all x, exists y, such that P(x, y) implies Q(x, y)"
        )

        # Should select higher tier for complex problem
        self.assertIn(result2.recommended_tier, [
            VerificationTier.TIER2_LEANAIDE,
            VerificationTier.TIER3_LEAN4
        ])

    def test_record_performance(self):
        """Test performance recording"""
        # Record successful attempt
        self.selector.record_performance(
            VerificationTier.TIER1_Z3,
            success=True,
            timeout=False,
            execution_time_ms=100.0
        )

        stats = self.selector.get_performance_stats()
        z3_stats = stats["tier1_z3"]

        self.assertEqual(z3_stats["total_attempts"], 1)
        self.assertEqual(z3_stats["successful_attempts"], 1)
        self.assertEqual(z3_stats["failed_attempts"], 0)
        self.assertEqual(z3_stats["success_rate"], 1.0)
        self.assertEqual(z3_stats["average_time_ms"], 100.0)

        # Record failed attempt
        self.selector.record_performance(
            VerificationTier.TIER1_Z3,
            success=False,
            timeout=False,
            execution_time_ms=200.0
        )

        stats = self.selector.get_performance_stats()
        z3_stats = stats["tier1_z3"]

        self.assertEqual(z3_stats["total_attempts"], 2)
        self.assertEqual(z3_stats["successful_attempts"], 1)
        self.assertEqual(z3_stats["failed_attempts"], 1)
        self.assertEqual(z3_stats["success_rate"], 0.5)
        self.assertEqual(z3_stats["average_time_ms"], 150.0)

    def test_circuit_breaker(self):
        """Test circuit breaker functionality"""
        # Record multiple failures to open circuit breaker
        for _ in range(10):
            self.selector.record_performance(
                VerificationTier.TIER1_Z3,
                success=False,
                timeout=False,
                execution_time_ms=100.0
            )

        stats = self.selector.get_performance_stats()
        z3_stats = stats["tier1_z3"]

        # Circuit breaker should be open
        self.assertTrue(z3_stats["circuit_breaker_open"])
        self.assertEqual(z3_stats["failure_count"], 10)

        # Reset stats
        self.selector.reset_performance_stats()
        stats = self.selector.get_performance_stats()
        z3_stats = stats["tier1_z3"]

        self.assertEqual(z3_stats["total_attempts"], 0)
        self.assertFalse(z3_stats["circuit_breaker_open"])


# =============================================================================
# TEST TIERED VERIFIER
# =============================================================================

class TestTieredVerifier(unittest.TestCase):
    """Test tiered verifier"""

    def setUp(self):
        """Set up test fixtures"""
        self.verifier = TieredVerifier()

    def test_verifier_initialization(self):
        """Test verifier initialization"""
        self.assertIsNotNone(self.verifier.classifier)
        self.assertIsNotNone(self.verifier.selector)
        self.assertEqual(self.verifier.config.auto_escalate, True)

    def test_verify_with_tier1_mock(self):
        """Test Tier 1 verification with mocked Z3"""
        with patch('tiered_verifier.RESEZ3Bridge') as mock_bridge:
            # Mock Z3 bridge
            mock_instance = Mock()
            mock_instance.detect_contradictions.return_value = (False, {"x": "1"})
            mock_bridge.return_value = mock_instance

            result = self.verifier.verify_with_tier(
                "Find x such that x > 0",
                VerificationTier.TIER1_Z3
            )

            self.assertIsInstance(result, Z3VerificationResult)
            self.assertEqual(result.tier, VerificationTier.TIER1_Z3)

    def test_verify_with_tier2_mock(self):
        """Test Tier 2 verification with mocked LeanAide"""
        with patch('tiered_verifier.Z3LeanAideBridge') as mock_bridge:
            # Mock LeanAide bridge
            mock_instance = Mock()
            mock_proof_result = Mock()
            mock_proof_result.success = True
            mock_proof_result.lean_component = "theorem test : True := by trivial"
            mock_proof_result.tactics_used = ["trivial"]
            mock_instance.prove = Mock(return_value=mock_proof_result)
            mock_bridge.return_value = mock_instance

            result = self.verifier.verify_with_tier(
                "Prove theorem",
                VerificationTier.TIER2_LEANAIDE
            )

            self.assertIsInstance(result, LeanAideVerificationResult)
            self.assertEqual(result.tier, VerificationTier.TIER2_LEANAIDE)

    def test_verify_with_tier3_mock(self):
        """Test Tier 3 verification with mocked Lean 4"""
        with patch('tiered_verifier.Lean4Interface') as mock_interface:
            # Mock Lean 4 interface
            mock_instance = Mock()
            mock_instance.formalize_constraint.return_value = {
                "verification_status": "verified",
                "lean4_code": "theorem test : True := by trivial",
                "theorem_name": "test_theorem",
                "errors": [],
            }
            mock_interface.return_value = mock_instance

            result = self.verifier.verify_with_tier(
                "Prove theorem",
                VerificationTier.TIER3_LEAN4
            )

            self.assertIsInstance(result, Lean4VerificationResult)
            self.assertEqual(result.tier, VerificationTier.TIER3_LEAN4)

    @patch('tiered_verifier.RESEZ3Bridge')
    def test_verify_unified_mock(self, mock_bridge):
        """Test unified verification with mocked solvers"""
        # Mock Z3 bridge
        mock_instance = Mock()
        mock_instance.detect_contradictions.return_value = (False, {"x": "1"})
        mock_bridge.return_value = mock_instance

        result = self.verifier.verify(
            "Find x such that x > 0 and x < 10",
        )

        self.assertIsInstance(result, UnifiedVerificationResult)
        self.assertIsNotNone(result.correlation_id)
        self.assertEqual(result.final_status, VerificationStatus.VERIFIED)

    def test_get_verification_status(self):
        """Test getting verification status"""
        status = self.verifier.get_verification_status("test-123")

        self.assertIsInstance(status, dict)
        self.assertEqual(status["correlation_id"], "test-123")

    def test_combine_results(self):
        """Test combining results from multiple tiers"""
        z3_result = Z3VerificationResult(
            status=VerificationStatus.VERIFIED,
            z3_result="sat",
            execution_time_ms=100.0,
            constraints_checked=5,
        )

        leanaide_result = LeanAideVerificationResult(
            status=VerificationStatus.VERIFIED,
            proof_status="proved",
            execution_time_ms=5000.0,
            constraints_checked=50,
        )

        combined = self.verifier.combine_results(
            [z3_result, leanaide_result],
            correlation_id="test-123"
        )

        self.assertIsInstance(combined, UnifiedVerificationResult)
        self.assertEqual(combined.correlation_id, "test-123")
        self.assertEqual(combined.tier1_result, z3_result)
        self.assertEqual(combined.tier2_result, leanaide_result)


# =============================================================================
# TEST CONVENIENCE FUNCTIONS
# =============================================================================

class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions"""

    def test_classify_problem(self):
        """Test classify_problem convenience function"""
        problem_class, problem_domain, complexity = classify_problem(
            "Find x such that x > 0"
        )

        self.assertIsNotNone(problem_class)
        self.assertIsNotNone(problem_domain)
        self.assertIsInstance(complexity, dict)

    def test_should_escalate_convenience(self):
        """Test should_escalate convenience function"""
        should, reason = should_escalate(
            current_tier=1,
            constraint_count=10,
            execution_time_ms=100,
            status="sat"
        )

        self.assertFalse(should)

    def test_select_solver_convenience(self):
        """Test select_solver convenience function"""
        result = select_solver("Find x such that x > 0")

        self.assertIsInstance(result, SelectionResult)
        self.assertIsNotNone(result.recommended_tier)


# =============================================================================
# RUN TESTS
# =============================================================================

def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestVerificationResults))
    suite.addTests(loader.loadTestsFromTestCase(TestProblemClassifier))
    suite.addTests(loader.loadTestsFromTestCase(TestSolverSelector))
    suite.addTests(loader.loadTestsFromTestCase(TestTieredVerifier))
    suite.addTests(loader.loadTestsFromTestCase(TestConvenienceFunctions))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    print("=" * 70)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
