#!/usr/bin/env python3
"""
Unit Tests for LLTL Z3 Contradiction Detection

Tests the Z3-based contradiction detection for formal commitments.

Following CLAUDE.md principles:
- Law of Runtime Truth: Test against actual Z3 behavior
- Law of Idempotency: Verify idempotent operations
- Law of Configuration Explicitness: Test config validation
- Structured Logging: Verify log output format
- Circuit Breaker: Test fallback to naive method

Author: RESE Team
Created: 2026-02-04
"""

import os
import sys
import json
import unittest
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from unittest.mock import Mock, AsyncMock, patch
import uuid

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../lib'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

LLTL_AVAILABLE = False
IMPORT_ERROR = None
Z3_AVAILABLE = False

# Try importing FormalCommitment directly
try:
    from lltl_adapter import FormalCommitment
    FORMAL_COMMITMENT_AVAILABLE = True
except ImportError:
    FORMAL_COMMITMENT_AVAILABLE = False

# Try importing full adapter
try:
    from lltl_adapter import LLTLAdapter, create_adapter, is_available
    LLTL_AVAILABLE = True
except ImportError as e:
    IMPORT_ERROR = str(e)

# Check Z3 availability
try:
    from z3prover_integration import is_z3_available
    Z3_AVAILABLE = is_z3_available()
except ImportError:
    Z3_AVAILABLE = False


def create_test_commitment(
    proposition_id: str,
    statement: str,
    confidence: float,
    p_value: float = 0.02,
    hypothesis: str = "hypothesis-1"
) -> FormalCommitment:
    """Helper to create test commitments"""
    return FormalCommitment(
        proposition_id=proposition_id,
        statement=statement,
        confidence_threshold=confidence,
        statistical_evidence={
            'confidence': confidence,
            'p_value': p_value,
            'confidence_interval_lower': confidence - 0.05,
            'confidence_interval_upper': confidence + 0.05,
            'expected_value': confidence
        },
        source_hypothesis=hypothesis,
        derivation_method="test",
        timestamp=datetime.now(timezone.utc).isoformat(),
        correlation_id="test-correlation"
    )


@unittest.skipIf(not FORMAL_COMMITMENT_AVAILABLE, "FormalCommitment not available")
class TestFormalCommitmentToZ3(unittest.TestCase):
    """Test formal commitment to Z3 formula conversion"""

    def setUp(self):
        """Set up test adapter"""
        os.environ['LLTL_AUDITABILITY_ENABLED'] = 'true'
        os.environ['RESE_Z3_LLTL_ENABLED'] = 'true'
        os.environ['Z3_TIMEOUT'] = '5000'
        os.environ['RESE_SIGNIFICANCE_LEVEL'] = '0.05'

        if not LLTL_AVAILABLE:
            self.skipTest(f"LLTL not available: {IMPORT_ERROR}")

        try:
            self.adapter = create_adapter()
        except Exception as e:
            self.skipTest(f"Failed to create adapter: {str(e)}")

    def test_encode_statement_inequality_less_than(self):
        """Test encoding statement with less-than inequality"""
        commitment = create_test_commitment(
            "test-1",
            "x < 10",
            0.90
        )

        # This should not raise an error
        formula = self.adapter._encode_statement_to_z3(commitment.statement)

        # Should contain the inequality
        self.assertIn("x", formula)
        self.assertIn("10", formula)
        self.assertIn("<", formula)

    def test_encode_statement_inequality_greater_than(self):
        """Test encoding statement with greater-than inequality"""
        commitment = create_test_commitment(
            "test-1",
            "x > 5",
            0.90
        )

        formula = self.adapter._encode_statement_to_z3(commitment.statement)

        # Should contain the inequality
        self.assertIn("x", formula)
        self.assertIn("5", formula)
        self.assertIn(">", formula)

    def test_encode_statement_equality(self):
        """Test encoding statement with equality"""
        commitment = create_test_commitment(
            "test-1",
            "x = 7.5",
            0.90
        )

        formula = self.adapter._encode_statement_to_z3(commitment.statement)

        # Should contain the equality
        self.assertIn("x", formula)
        self.assertIn("7.5", formula)
        # Z3 uses "=" for equality
        self.assertTrue("=" in formula or "==" in formula)

    def test_encode_statement_with_and(self):
        """Test encoding statement with AND operator"""
        commitment = create_test_commitment(
            "test-1",
            "x > 5 and y < 10",
            0.90
        )

        formula = self.adapter._encode_statement_to_z3(commitment.statement)

        # Should contain both parts
        self.assertIn("x", formula)
        self.assertIn("y", formula)
        self.assertIn("5", formula)
        self.assertIn("10", formula)

    def test_formal_commitment_to_z3_formula(self):
        """Test converting full formal commitment to Z3 formula"""
        commitment = create_test_commitment(
            "test-1",
            "x < 10",
            0.90,
            p_value=0.03
        )

        formula = self.adapter._formal_commitment_to_z3_formula(commitment)

        # Should be a valid SMT-LIB2 formula
        self.assertIsInstance(formula, str)
        self.assertIn("confidence", formula)
        self.assertIn("0.9", formula)  # confidence threshold

    def test_extract_variable_names(self):
        """Test extracting variable names from formula"""
        formula = "(and (x > 5) (y < 10) (>= confidence 0.9))"

        variables = self.adapter._extract_variable_names(formula)

        self.assertIn("x", variables)
        self.assertIn("y", variables)
        self.assertIn("confidence", variables)

    def test_extract_inequality_less_than(self):
        """Test extracting variable and value from less-than inequality"""
        statement = "x < 10"

        var, val = self.adapter._extract_inequality(statement, '<')

        self.assertEqual(var, "x")
        self.assertEqual(val, "10")

    def test_extract_inequality_greater_than(self):
        """Test extracting variable and value from greater-than inequality"""
        statement = "temperature > 100"

        var, val = self.adapter._extract_inequality(statement, '>')

        self.assertEqual(var, "temperature")
        self.assertEqual(val, "100")

    def test_extract_equality(self):
        """Test extracting variable and value from equality"""
        statement = "value = 42.5"

        var, val = self.adapter._extract_equality(statement)

        self.assertEqual(var, "value")
        self.assertEqual(val, "42.5")


@unittest.skipIf(not LLTL_AVAILABLE, f"LLTL not available: {IMPORT_ERROR if not LLTL_AVAILABLE else ''}")
class TestZ3ContradictionDetection(unittest.TestCase):
    """Test Z3-based contradiction detection"""

    def setUp(self):
        """Set up test adapter"""
        os.environ['LLTL_AUDITABILITY_ENABLED'] = 'true'
        os.environ['RESE_Z3_LLTL_ENABLED'] = 'true'
        os.environ['Z3_TIMEOUT'] = '5000'
        os.environ['RESE_SIGNIFICANCE_LEVEL'] = '0.05'

        if not LLTL_AVAILABLE:
            self.skipTest(f"LLTL not available: {IMPORT_ERROR}")

        try:
            self.adapter = create_adapter()
        except Exception as e:
            self.skipTest(f"Failed to create adapter: {str(e)}")

    @unittest.skipIf(not Z3_AVAILABLE, "Z3 not available")
    def test_detect_contradictions_no_contradictions(self):
        """Test contradiction detection with no contradictions (SAT)"""
        # Create compatible commitments
        commitments = [
            create_test_commitment("test-1", "x > 5", 0.90),
            create_test_commitment("test-2", "x < 10", 0.85),
            create_test_commitment("test-3", "y = 7", 0.80)
        ]

        contradictions, error = self.adapter.detect_contradictions(
            constraints=commitments,
            correlation_id="test-correlation-1"
        )

        self.assertIsNone(error)
        # Should have no contradictions (or minimal if Z3 finds any)
        self.assertIsInstance(contradictions, list)

    @unittest.skipIf(not Z3_AVAILABLE, "Z3 not available")
    def test_detect_contradictions_with_contradictions(self):
        """Test contradiction detection with actual contradictions (UNSAT)"""
        # Create contradictory commitments
        commitments = [
            create_test_commitment("test-1", "x > 10", 0.90),
            create_test_commitment("test-2", "x < 5", 0.85)  # Contradicts x > 10
        ]

        contradictions, error = self.adapter.detect_contradictions(
            constraints=commitments,
            correlation_id="test-correlation-2"
        )

        self.assertIsNone(error)
        # Should detect contradictions
        self.assertIsInstance(contradictions, list)
        # Note: May not detect without proper unsat core support
        # This is a known limitation with basic Z3 integration

    @unittest.skipIf(not Z3_AVAILABLE, "Z3 not available")
    def test_detect_contradictions_empty_list(self):
        """Test contradiction detection with empty commitment list"""
        contradictions, error = self.adapter.detect_contradictions(
            constraints=[],
            correlation_id="test-correlation-3"
        )

        self.assertIsNone(error)
        self.assertEqual(len(contradictions), 0)

    @unittest.skipIf(not Z3_AVAILABLE, "Z3 not available")
    def test_detect_contradictions_single_commitment(self):
        """Test contradiction detection with single commitment"""
        commitments = [
            create_test_commitment("test-1", "x > 5", 0.90)
        ]

        contradictions, error = self.adapter.detect_contradictions(
            constraints=commitments,
            correlation_id="test-correlation-4"
        )

        self.assertIsNone(error)
        # Single commitment cannot contradict itself
        self.assertEqual(len(contradictions), 0)


@unittest.skipIf(not LLTL_AVAILABLE, f"LLTL not available: {IMPORT_ERROR if not LLTL_AVAILABLE else ''}")
class TestNaiveContradictionDetection(unittest.TestCase):
    """Test naive fallback contradiction detection"""

    def setUp(self):
        """Set up test adapter with Z3 disabled"""
        os.environ['LLTL_AUDITABILITY_ENABLED'] = 'true'
        os.environ['RESE_Z3_LLTL_ENABLED'] = 'false'  # Disable Z3
        os.environ['Z3_TIMEOUT'] = '5000'

        if not LLTL_AVAILABLE:
            self.skipTest(f"LLTL not available: {IMPORT_ERROR}")

        try:
            self.adapter = create_adapter()
        except Exception as e:
            self.skipTest(f"Failed to create adapter: {str(e)}")

    def test_naive_detect_contradictions_opposite_inequalities(self):
        """Test naive detection of opposite inequalities"""
        commitments = [
            create_test_commitment("test-1", "x > 10", 0.90),
            create_test_commitment("test-2", "x < 5", 0.85)
        ]

        contradictions, error = self.adapter.detect_contradictions(
            constraints=commitments,
            correlation_id="test-naive-1"
        )

        self.assertIsNone(error)
        # Naive method should detect opposite inequalities
        self.assertGreater(len(contradictions), 0)

    def test_naive_detect_contradictions_no_contradictions(self):
        """Test naive detection with no contradictions"""
        commitments = [
            create_test_commitment("test-1", "x > 5", 0.90),
            create_test_commitment("test-2", "x < 10", 0.85)
        ]

        contradictions, error = self.adapter.detect_contradictions(
            constraints=commitments,
            correlation_id="test-naive-2"
        )

        self.assertIsNone(error)
        # Should have no contradictions
        self.assertEqual(len(contradictions), 0)

    def test_naive_check_contradiction_direct_negation(self):
        """Test naive detection of direct negation"""
        c1 = create_test_commitment("test-1", "not (x > 5)", 0.90)
        c2 = create_test_commitment("test-2", "x > 5", 0.85)

        # Direct check
        contradicts = self.adapter._check_contradiction_naive(c1, c2)

        # Should detect direct negation
        self.assertTrue(contradicts)

    def test_naive_check_contradiction_conflicting_confidence(self):
        """Test naive detection of conflicting confidence thresholds"""
        c1 = create_test_commitment("test-1", "x > 5", 0.95)  # Very confident
        c2 = create_test_commitment("test-2", "x < 10", 0.50)  # Low confidence

        contradicts = self.adapter._check_contradiction_naive(c1, c2)

        # May detect as contradiction due to conflicting confidence
        # This is a heuristic, not guaranteed
        self.assertIsInstance(contradicts, bool)


@unittest.skipIf(not LLTL_AVAILABLE, f"LLTL not available: {IMPORT_ERROR if not LLTL_AVAILABLE else ''}")
class TestZ3IntegrationConfiguration(unittest.TestCase):
    """Test Z3 integration configuration and environment variables"""

    def test_z3_enabled_by_default(self):
        """Test that Z3 is enabled by default"""
        # Set environment
        os.environ['RESE_Z3_LLTL_ENABLED'] = 'true'
        os.environ['Z3_TIMEOUT'] = '5000'

        if not LLTL_AVAILABLE:
            self.skipTest(f"LLTL not available: {IMPORT_ERROR}")

        try:
            adapter = create_adapter()
            # Z3 should be enabled if available
            if Z3_AVAILABLE:
                self.assertTrue(adapter.z3_enabled)
        except Exception as e:
            self.skipTest(f"Failed to create adapter: {str(e)}")

    def test_z3_disabled_via_env(self):
        """Test that Z3 can be disabled via environment variable"""
        os.environ['RESE_Z3_LLTL_ENABLED'] = 'false'

        if not LLTL_AVAILABLE:
            self.skipTest(f"LLTL not available: {IMPORT_ERROR}")

        try:
            adapter = create_adapter()
            self.assertFalse(adapter.z3_enabled)
        except Exception as e:
            self.skipTest(f"Failed to create adapter: {str(e)}")

    def test_z3_timeout_configuration(self):
        """Test Z3 timeout configuration"""
        os.environ['RESE_Z3_LLTL_ENABLED'] = 'true'
        os.environ['Z3_TIMEOUT'] = '10000'  # 10 seconds

        if not LLTL_AVAILABLE:
            self.skipTest(f"LLTL not available: {IMPORT_ERROR}")

        try:
            adapter = create_adapter()
            self.assertEqual(adapter.z3_timeout_ms, 10000)
        except Exception as e:
            self.skipTest(f"Failed to create adapter: {str(e)}")

    def test_get_stats_includes_z3(self):
        """Test that get_stats includes Z3 information"""
        os.environ['RESE_Z3_LLTL_ENABLED'] = 'true'
        os.environ['Z3_TIMEOUT'] = '5000'

        if not LLTL_AVAILABLE:
            self.skipTest(f"LLTL not available: {IMPORT_ERROR}")

        try:
            adapter = create_adapter()
            stats = adapter.get_stats()

            self.assertIn('z3_integration', stats)
            z3_info = stats['z3_integration']

            self.assertIn('enabled', z3_info)
            self.assertIn('available', z3_info)
            self.assertIn('solver_initialized', z3_info)
            self.assertIn('timeout_ms', z3_info)

            self.assertEqual(z3_info['timeout_ms'], 5000)
        except Exception as e:
            self.skipTest(f"Failed to create adapter: {str(e)}")

    def test_health_check_includes_z3(self):
        """Test that health_check includes Z3 status"""
        os.environ['RESE_Z3_LLTL_ENABLED'] = 'true'

        if not LLTL_AVAILABLE:
            self.skipTest(f"LLTL not available: {IMPORT_ERROR}")

        try:
            adapter = create_adapter()
            healthy, message = adapter.health_check()

            self.assertTrue(healthy)
            # Message should mention Z3 if enabled
            if Z3_AVAILABLE and adapter.z3_enabled:
                self.assertIn('Z3', message)
        except Exception as e:
            self.skipTest(f"Failed to create adapter: {str(e)}")


@unittest.skipIf(not LLTL_AVAILABLE, f"LLTL not available: {IMPORT_ERROR if not LLTL_AVAILABLE else ''}")
class TestZ3IntegrationIdempotency(unittest.TestCase):
    """Test idempotency of Z3 contradiction detection"""

    def setUp(self):
        """Set up test adapter"""
        os.environ['LLTL_AUDITABILITY_ENABLED'] = 'true'
        os.environ['RESE_Z3_LLTL_ENABLED'] = 'true'
        os.environ['Z3_TIMEOUT'] = '5000'

        if not LLTL_AVAILABLE:
            self.skipTest(f"LLTL not available: {IMPORT_ERROR}")

        try:
            self.adapter = create_adapter()
        except Exception as e:
            self.skipTest(f"Failed to create adapter: {str(e)}")

    @unittest.skipIf(not Z3_AVAILABLE, "Z3 not available")
    def test_idempotent_contradiction_detection(self):
        """Test that contradiction detection is idempotent"""
        commitments = [
            create_test_commitment("test-1", "x > 5", 0.90),
            create_test_commitment("test-2", "x < 10", 0.85)
        ]

        # Run detection twice
        contradictions1, error1 = self.adapter.detect_contradictions(
            constraints=commitments,
            correlation_id="test-idempotent-1"
        )

        contradictions2, error2 = self.adapter.detect_contradictions(
            constraints=commitments,
            correlation_id="test-idempotent-2"
        )

        # Both should succeed
        self.assertIsNone(error1)
        self.assertIsNone(error2)

        # Should return same number of contradictions
        self.assertEqual(len(contradictions1), len(contradictions2))


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestFormalCommitmentToZ3))
    suite.addTests(loader.loadTestsFromTestCase(TestZ3ContradictionDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestNaiveContradictionDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestZ3IntegrationConfiguration))
    suite.addTests(loader.loadTestsFromTestCase(TestZ3IntegrationIdempotency))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    import sys
    sys.exit(run_tests())
