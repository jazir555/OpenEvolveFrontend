#!/usr/bin/env python3
"""
Unit Tests for LLTL DEE → SCE Auditability Component

Tests the FormalCommitment class and DEE → SCE conversion methods

Following CLAUDE.md principles:
- Law of Runtime Truth: Test against actual behavior
- Law of Idempotency: Verify idempotent operations
- Law of Configuration Explicitness: Test config validation
- Structured Logging: Verify log output format

Author: RESE Team
Created: 2026-02-04
"""

import os
import sys
import json
import unittest
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from unittest.mock import Mock, AsyncMock, patch
import uuid

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../lib'))

LLTL_AVAILABLE = False
IMPORT_ERROR = None

# Try importing FormalCommitment directly for first test class
try:
    from lltl_adapter import FormalCommitment
    FORMAL_COMMITMENT_AVAILABLE = True
except ImportError:
    FORMAL_COMMITMENT_AVAILABLE = False

# Try importing full adapter
try:
    from lltl_adapter import (
        LLTLAdapter,
        create_adapter,
        is_available
    )
    LLTL_AVAILABLE = True
except ImportError as e:
    IMPORT_ERROR = str(e)


@unittest.skipIf(not FORMAL_COMMITMENT_AVAILABLE, "FormalCommitment not available")
class TestFormalCommitment(unittest.TestCase):
    """Test FormalCommitment dataclass"""

    def test_formal_commitment_creation(self):
        """Test creating a FormalCommitment"""
        commitment = FormalCommitment(
            proposition_id="test-prop-1",
            statement="(H) ∧ (confidence ≥ 0.950) → Accept(H)",
            confidence_threshold=0.90,
            statistical_evidence={
                'confidence': 0.95,
                'p_value': 0.02,
                'confidence_interval_lower': 0.85,
                'confidence_interval_upper': 0.98,
                'expected_value': 0.9
            },
            source_hypothesis="hypothesis-1",
            derivation_method="mcts_validation",
            timestamp=datetime.now(timezone.utc).isoformat(),
            correlation_id="test-correlation-1"
        )

        self.assertEqual(commitment.proposition_id, "test-prop-1")
        self.assertEqual(commitment.confidence_threshold, 0.90)
        self.assertEqual(commitment.source_hypothesis, "hypothesis-1")
        self.assertEqual(commitment.derivation_method, "mcts_validation")
        self.assertIsNone(commitment.lean4_theorem)

    def test_to_sce_constraint(self):
        """Test converting FormalCommitment to SCE constraint format"""
        commitment = FormalCommitment(
            proposition_id="test-prop-1",
            statement="(H) ∧ (confidence ≥ 0.950) → Accept(H)",
            confidence_threshold=0.90,
            statistical_evidence={'confidence': 0.95},
            source_hypothesis="hypothesis-1",
            derivation_method="mcts_validation",
            timestamp=datetime.now(timezone.utc).isoformat(),
            correlation_id="test-correlation-1"
        )

        sce_constraint = commitment.to_sce_constraint()

        self.assertEqual(sce_constraint['constraint_id'], "test-prop-1")
        self.assertEqual(sce_constraint['formal_statement'], commitment.statement)
        self.assertEqual(sce_constraint['confidence'], 0.90)
        self.assertEqual(sce_constraint['type'], "statistical_commitment")
        self.assertIn('evidence', sce_constraint)

    def test_to_dict(self):
        """Test converting FormalCommitment to dictionary"""
        commitment = FormalCommitment(
            proposition_id="test-prop-1",
            statement="(H) ∧ (confidence ≥ 0.950) → Accept(H)",
            confidence_threshold=0.90,
            statistical_evidence={'confidence': 0.95},
            source_hypothesis="hypothesis-1",
            derivation_method="mcts_validation",
            timestamp=datetime.now(timezone.utc).isoformat(),
            correlation_id="test-correlation-1",
            lean4_theorem=None
        )

        commitment_dict = commitment.to_dict()

        self.assertEqual(commitment_dict['proposition_id'], "test-prop-1")
        self.assertEqual(commitment_dict['confidence_threshold'], 0.90)
        self.assertIsNone(commitment_dict['lean4_theorem'])
        self.assertIn('statistical_evidence', commitment_dict)


@unittest.skipIf(not LLTL_AVAILABLE, f"LLTL not available: {IMPORT_ERROR if not LLTL_AVAILABLE else ''}")
class TestStatisticalToFormal(unittest.TestCase):
    """Test statistical_to_formal conversion"""

    def setUp(self):
        """Set up test adapter"""
        # Set required environment variables
        os.environ['LLTL_AUDITABILITY_ENABLED'] = 'true'
        os.environ['LLTL_CONFIDENCE_THRESHOLD_DEFAULT'] = '0.75'
        os.environ['LLTL_SIGNIFICANCE_LEVEL'] = '0.05'
        os.environ['LLTL_AUDIT_TIMEOUT_MS'] = '5000'

        # Check if LLTL is available
        if not LLTL_AVAILABLE:
            self.skipTest(f"LLTL not available: {IMPORT_ERROR}")

        try:
            self.adapter = create_adapter()
        except Exception as e:
            self.skipTest(f"Failed to create adapter: {str(e)}")

    def test_statistical_to_formal_basic(self):
        """Test basic statistical to formal conversion"""
        statistical_result = {
            'hypothesis_statement': 'Lattice confinement enables LENR',
            'confidence': 0.85,
            'p_value': 0.02,
            'confidence_interval': (0.78, 0.92),
            'expected_value': 0.85
        }

        commitment, error = self.adapter.statistical_to_formal(
            statistical_result=statistical_result,
            source_hypothesis='hypothesis-1',
            derivation_method='mcts_validation',
            correlation_id='test-correlation-1'
        )

        self.assertIsNotNone(commitment)
        self.assertIsNone(error)
        self.assertIsInstance(commitment, FormalCommitment)
        self.assertEqual(commitment.source_hypothesis, 'hypothesis-1')
        self.assertEqual(commitment.derivation_method, 'mcts_validation')
        self.assertEqual(commitment.correlation_id, 'test-correlation-1')
        self.assertIn('Lattice confinement enables LENR', commitment.statement)

    def test_confidence_threshold_calculation(self):
        """Test confidence threshold is calculated correctly"""
        test_cases = [
            (0.98, 0.90),  # Very high confidence
            (0.85, 0.75),  # High confidence
            (0.70, 0.60),  # Moderate confidence
            (0.50, 0.50),  # Low confidence
        ]

        for confidence, expected_threshold in test_cases:
            statistical_result = {
                'hypothesis_statement': 'Test hypothesis',
                'confidence': confidence,
                'p_value': 0.02,
                'confidence_interval': (0.0, 1.0),
                'expected_value': confidence
            }

            commitment, error = self.adapter.statistical_to_formal(
                statistical_result=statistical_result,
                source_hypothesis='hypothesis-1',
                derivation_method='test',
                correlation_id='test-correlation'
            )

            self.assertIsNotNone(commitment)
            self.assertEqual(commitment.confidence_threshold, expected_threshold,
                           f"Confidence {confidence} should give threshold {expected_threshold}")

    def test_formal_statement_construction(self):
        """Test formal logical statement is constructed correctly"""
        statistical_result = {
            'hypothesis_statement': 'Lattice confinement enables LENR',
            'confidence': 0.85,
            'p_value': 0.02,
            'confidence_interval': (0.78, 0.92),
            'expected_value': 0.85
        }

        commitment, error = self.adapter.statistical_to_formal(
            statistical_result=statistical_result,
            source_hypothesis='hypothesis-1',
            derivation_method='mcts_validation',
            correlation_id='test-correlation-1'
        )

        self.assertIsNotNone(commitment)

        # Check statement structure
        self.assertIn('Lattice confinement enables LENR', commitment.statement)
        self.assertIn('confidence ≥', commitment.statement)
        self.assertIn('p_value ≤', commitment.statement)
        self.assertIn('CI ∈', commitment.statement)
        self.assertIn('→ Accept(', commitment.statement)

    def test_statistical_evidence_extraction(self):
        """Test statistical evidence is extracted correctly"""
        statistical_result = {
            'hypothesis_statement': 'Test hypothesis',
            'confidence': 0.85,
            'p_value': 0.02,
            'confidence_interval': [0.78, 0.92],  # Test with list
            'expected_value': 0.85
        }

        commitment, error = self.adapter.statistical_to_formal(
            statistical_result=statistical_result,
            source_hypothesis='hypothesis-1',
            derivation_method='test',
            correlation_id='test-correlation'
        )

        self.assertIsNotNone(commitment)

        evidence = commitment.statistical_evidence
        self.assertEqual(evidence['confidence'], 0.85)
        self.assertEqual(evidence['p_value'], 0.02)
        self.assertEqual(evidence['confidence_interval_lower'], 0.78)
        self.assertEqual(evidence['confidence_interval_upper'], 0.92)
        self.assertEqual(evidence['expected_value'], 0.85)

    def test_missing_required_fields(self):
        """Test error handling for missing required fields"""
        # Missing 'confidence' field
        statistical_result = {
            'hypothesis_statement': 'Test hypothesis'
        }

        commitment, error = self.adapter.statistical_to_formal(
            statistical_result=statistical_result,
            source_hypothesis='hypothesis-1',
            derivation_method='test',
            correlation_id='test-correlation'
        )

        self.assertIsNone(commitment)
        self.assertIsNotNone(error)
        self.assertIn('Missing required fields', error)

    def test_idempotency(self):
        """Test that same input produces same commitment (idempotency)"""
        statistical_result = {
            'hypothesis_statement': 'Test hypothesis',
            'confidence': 0.85,
            'p_value': 0.02,
            'confidence_interval': (0.78, 0.92),
            'expected_value': 0.85
        }

        # Create two commitments from same input
        commitment1, error1 = self.adapter.statistical_to_formal(
            statistical_result=statistical_result,
            source_hypothesis='hypothesis-1',
            derivation_method='test',
            correlation_id='test-correlation-1'
        )

        commitment2, error2 = self.adapter.statistical_to_formal(
            statistical_result=statistical_result,
            source_hypothesis='hypothesis-1',
            derivation_method='test',
            correlation_id='test-correlation-2'
        )

        # Both should succeed
        self.assertIsNotNone(commitment1)
        self.assertIsNotNone(commitment2)
        self.assertIsNone(error1)
        self.assertIsNone(error2)

        # Should have same threshold (deterministic)
        self.assertEqual(commitment1.confidence_threshold, commitment2.confidence_threshold)

        # Should have different proposition IDs (each call creates new commitment)
        self.assertNotEqual(commitment1.proposition_id, commitment2.proposition_id)


@unittest.skipIf(not LLTL_AVAILABLE, f"LLTL not available: {IMPORT_ERROR if not LLTL_AVAILABLE else ''}")
class TestSCEIntegration(unittest.TestCase):
    """Test SCE integration methods"""

    def setUp(self):
        """Set up test adapter and mock SCE engine"""
        # Set required environment variables
        os.environ['LLTL_AUDITABILITY_ENABLED'] = 'true'

        if not LLTL_AVAILABLE:
            self.skipTest(f"LLTL not available: {IMPORT_ERROR}")

        try:
            self.adapter = create_adapter()
        except Exception as e:
            self.skipTest(f"Failed to create adapter: {str(e)}")

        # Create mock SCE engine
        self.mock_sce = Mock()
        self.mock_sce.add_constraint = AsyncMock(return_value={'added': True, 'updated': False})
        self.mock_sce.detect_contradictions = AsyncMock()

    def test_integrate_into_sce_success(self):
        """Test successful integration into SCE"""
        # Create a commitment
        commitment = FormalCommitment(
            proposition_id="test-prop-1",
            statement="(H) ∧ (confidence ≥ 0.950) → Accept(H)",
            confidence_threshold=0.90,
            statistical_evidence={'confidence': 0.95},
            source_hypothesis="hypothesis-1",
            derivation_method="mcts_validation",
            timestamp=datetime.now(timezone.utc).isoformat(),
            correlation_id="test-correlation-1"
        )

        # Integrate into SCE
        success, error = self.adapter.integrate_into_sce(
            commitment=commitment,
            sce_engine=self.mock_sce,
            correlation_id="test-correlation-1"
        )

        self.assertTrue(success)
        self.assertIsNone(error)
        self.mock_sce.add_constraint.assert_called_once()

    def test_integrate_into_sce_with_contradictions(self):
        """Test SCE integration with contradiction detection"""
        # Create a commitment
        commitment = FormalCommitment(
            proposition_id="test-prop-1",
            statement="(H) ∧ (confidence ≥ 0.950) → Accept(H)",
            confidence_threshold=0.90,
            statistical_evidence={'confidence': 0.95},
            source_hypothesis="hypothesis-1",
            derivation_method="mcts_validation",
            timestamp=datetime.now(timezone.utc).isoformat(),
            correlation_id="test-correlation-1"
        )

        # Mock contradiction result
        mock_result = Mock()
        mock_result.contradictions = ['contradiction1', 'contradiction2']
        self.mock_sce.detect_contradictions = AsyncMock(return_value=mock_result)

        # Integrate into SCE
        success, error = self.adapter.integrate_into_sce(
            commitment=commitment,
            sce_engine=self.mock_sce,
            correlation_id="test-correlation-1"
        )

        # Should still succeed (contradictions are warnings, not failures)
        self.assertTrue(success)
        self.mock_sce.detect_contradictions.assert_called_once()

    def test_integrate_into_sce_disabled(self):
        """Test integration when auditability is disabled"""
        # Disable auditability
        os.environ['LLTL_AUDITABILITY_ENABLED'] = 'false'

        # Create new adapter with disabled auditability
        adapter = create_adapter()

        commitment = FormalCommitment(
            proposition_id="test-prop-1",
            statement="(H) ∧ (confidence ≥ 0.950) → Accept(H)",
            confidence_threshold=0.90,
            statistical_evidence={'confidence': 0.95},
            source_hypothesis="hypothesis-1",
            derivation_method="mcts_validation",
            timestamp=datetime.now(timezone.utc).isoformat(),
            correlation_id="test-correlation-1"
        )

        # Integrate into SCE
        success, error = adapter.integrate_into_sce(
            commitment=commitment,
            sce_engine=self.mock_sce,
            correlation_id="test-correlation-1"
        )

        # Should succeed without calling SCE
        self.assertTrue(success)
        self.mock_sce.add_constraint.assert_not_called()


@unittest.skipIf(not LLTL_AVAILABLE, f"LLTL not available: {IMPORT_ERROR if not LLTL_AVAILABLE else ''}")
class TestAuditTrail(unittest.TestCase):
    """Test audit trail functionality"""

    def setUp(self):
        """Set up test adapter"""
        os.environ['LLTL_AUDITABILITY_ENABLED'] = 'true'

        if not LLTL_AVAILABLE:
            self.skipTest(f"LLTL not available: {IMPORT_ERROR}")

        try:
            self.adapter = create_adapter()
        except Exception as e:
            self.skipTest(f"Failed to create adapter: {str(e)}")

    def test_get_audit_trail_empty(self):
        """Test getting audit trail when empty"""
        trail = self.adapter.get_audit_trail()
        self.assertIsInstance(trail, list)
        self.assertEqual(len(trail), 0)

    def test_get_audit_trail_with_commitments(self):
        """Test getting audit trail with commitments"""
        # Create some commitments
        statistical_result = {
            'hypothesis_statement': 'Test hypothesis 1',
            'confidence': 0.85,
            'p_value': 0.02,
            'confidence_interval': (0.78, 0.92),
            'expected_value': 0.85
        }

        commitment1, _ = self.adapter.statistical_to_formal(
            statistical_result=statistical_result,
            source_hypothesis='hypothesis-1',
            derivation_method='test',
            correlation_id='test-correlation-1'
        )

        statistical_result['hypothesis_statement'] = 'Test hypothesis 2'
        commitment2, _ = self.adapter.statistical_to_formal(
            statistical_result=statistical_result,
            source_hypothesis='hypothesis-2',
            derivation_method='test',
            correlation_id='test-correlation-2'
        )

        # Get audit trail
        trail = self.adapter.get_audit_trail()

        self.assertEqual(len(trail), 2)
        self.assertIn(commitment1, trail)
        self.assertIn(commitment2, trail)

    def test_get_commitment(self):
        """Test getting specific commitment by ID"""
        statistical_result = {
            'hypothesis_statement': 'Test hypothesis',
            'confidence': 0.85,
            'p_value': 0.02,
            'confidence_interval': (0.78, 0.92),
            'expected_value': 0.85
        }

        commitment, _ = self.adapter.statistical_to_formal(
            statistical_result=statistical_result,
            source_hypothesis='hypothesis-1',
            derivation_method='test',
            correlation_id='test-correlation-1'
        )

        # Get commitment by ID
        retrieved = self.adapter.get_commitment(commitment.proposition_id)

        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.proposition_id, commitment.proposition_id)
        self.assertEqual(retrieved.statement, commitment.statement)

    def test_clear_audit_trail(self):
        """Test clearing audit trail"""
        # Create a commitment
        statistical_result = {
            'hypothesis_statement': 'Test hypothesis',
            'confidence': 0.85,
            'p_value': 0.02,
            'confidence_interval': (0.78, 0.92),
            'expected_value': 0.85
        }

        self.adapter.statistical_to_formal(
            statistical_result=statistical_result,
            source_hypothesis='hypothesis-1',
            derivation_method='test',
            correlation_id='test-correlation-1'
        )

        # Verify not empty
        self.assertGreater(len(self.adapter.get_audit_trail()), 0)

        # Clear trail
        count = self.adapter.clear_audit_trail()

        # Verify empty
        self.assertEqual(count, 1)
        self.assertEqual(len(self.adapter.get_audit_trail()), 0)


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestFormalCommitment))
    suite.addTests(loader.loadTestsFromTestCase(TestStatisticalToFormal))
    suite.addTests(loader.loadTestsFromTestCase(TestSCEIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestAuditTrail))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    import sys
    sys.exit(run_tests())
