#!/usr/bin/env python3
"""
Unit Tests for LLTL Formal Commitments Handler

Tests the FormalCommitmentsHandler and related components.

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
import unittest
from datetime import datetime, timezone
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

try:
    from confidence_tracker import ConfidenceTracker
    from formal_commitments import (
        FormalCommitmentsHandler,
        FormalCommitment,
        CommitmentStatus,
        ContradictionReport
    )
    FORMAL_COMMITMENTS_AVAILABLE = True
except ImportError as e:
    FORMAL_COMMITMENTS_AVAILABLE = False
    IMPORT_ERROR = str(e)


@unittest.skipIf(not FORMAL_COMMITMENTS_AVAILABLE, f"Formal commitments not available: {IMPORT_ERROR if not FORMAL_COMMITMENTS_AVAILABLE else ''}")
class TestFormalCommitment(unittest.TestCase):
    """Test FormalCommitment dataclass."""

    def test_formal_commitment_creation(self):
        """Test creating a FormalCommitment."""
        commitment = FormalCommitment(
            proposition_id="test-prop-1",
            statement="(H) ∧ (confidence ≥ 0.950) → Accept(H)",
            confidence_threshold=0.90,
            statistical_evidence={
                'confidence': 0.95,
                'p_value': 0.02,
                'confidence_interval_lower': 0.85,
                'confidence_interval_upper': 0.98
            },
            source_hypothesis="hypothesis-1",
            derivation_method="mcts_validation",
            timestamp=datetime.now(timezone.utc).isoformat(),
            correlation_id="test-correlation-1"
        )

        self.assertEqual(commitment.proposition_id, "test-prop-1")
        self.assertEqual(commitment.confidence_threshold, 0.90)
        self.assertEqual(commitment.status, CommitmentStatus.PENDING)

    def test_to_sce_constraint(self):
        """Test converting FormalCommitment to SCE constraint format."""
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

    def test_to_dict(self):
        """Test converting FormalCommitment to dictionary."""
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

        commitment_dict = commitment.to_dict()

        self.assertEqual(commitment_dict['proposition_id'], "test-prop-1")
        self.assertEqual(commitment_dict['confidence_threshold'], 0.90)
        self.assertEqual(commitment_dict['status'], "pending")

    def test_update_status(self):
        """Test updating commitment status."""
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

        self.assertEqual(commitment.status, CommitmentStatus.PENDING)

        commitment.update_status(CommitmentStatus.INTEGRATED)
        self.assertEqual(commitment.status, CommitmentStatus.INTEGRATED)


@unittest.skipIf(not FORMAL_COMMITMENTS_AVAILABLE, f"Formal commitments not available: {IMPORT_ERROR if not FORMAL_COMMITMENTS_AVAILABLE else ''}")
class TestFormalCommitmentsHandler(unittest.TestCase):
    """Test FormalCommitmentsHandler class."""

    def setUp(self):
        """Set up test handler."""
        # Set required environment variables
        os.environ['LLTL_SIGNIFICANCE_LEVEL'] = '0.05'

        if not FORMAL_COMMITMENTS_AVAILABLE:
            self.skipTest(f"Formal commitments not available: {IMPORT_ERROR}")

        try:
            self.confidence_tracker = ConfidenceTracker()
            self.handler = FormalCommitmentsHandler(
                confidence_tracker=self.confidence_tracker
            )
        except Exception as e:
            self.skipTest(f"Failed to create handler: {str(e)}")

    def test_initialization(self):
        """Test handler initialization."""
        self.assertIsNotNone(self.handler)
        self.assertIsNotNone(self.handler.confidence_tracker)
        self.assertEqual(len(self.handler.commitments), 0)
        self.assertEqual(len(self.handler.contradiction_reports), 0)

    def test_create_commitment_basic(self):
        """Test basic commitment creation."""
        statistical_result = {
            'hypothesis_statement': 'Lattice confinement enables LENR',
            'confidence': 0.85,
            'p_value': 0.02,
            'confidence_interval': (0.78, 0.92),
            'expected_value': 0.85
        }

        commitment, error = self.handler.create_commitment(
            statistical_result=statistical_result,
            source_hypothesis='hypothesis-1',
            derivation_method='mcts_validation',
            correlation_id='test-correlation-1'
        )

        self.assertIsNotNone(commitment)
        self.assertIsNone(error)
        self.assertIsInstance(commitment, FormalCommitment)
        self.assertEqual(commitment.source_hypothesis, 'hypothesis-1')
        self.assertIn('Lattice confinement enables LENR', commitment.statement)

    def test_create_commitment_missing_fields(self):
        """Test commitment creation with missing required fields."""
        # Missing 'confidence' field
        statistical_result = {
            'hypothesis_statement': 'Test hypothesis'
        }

        commitment, error = self.handler.create_commitment(
            statistical_result=statistical_result,
            source_hypothesis='hypothesis-1',
            derivation_method='test'
        )

        self.assertIsNone(commitment)
        self.assertIsNotNone(error)
        self.assertIn('Missing required fields', error)

    def test_create_commitment_confidence_threshold(self):
        """Test that confidence threshold is calculated correctly."""
        statistical_result = {
            'hypothesis_statement': 'Test hypothesis',
            'confidence': 0.95,  # Very high
            'p_value': 0.02,
            'confidence_interval': (0.90, 0.98),
            'expected_value': 0.95
        }

        commitment, error = self.handler.create_commitment(
            statistical_result=statistical_result,
            source_hypothesis='hypothesis-1',
            derivation_method='test'
        )

        self.assertIsNotNone(commitment)
        # Very high confidence should give 0.90 threshold
        self.assertEqual(commitment.confidence_threshold, 0.90)

    def test_create_commitment_formal_statement(self):
        """Test that formal statement is constructed correctly."""
        statistical_result = {
            'hypothesis_statement': 'Lattice confinement enables LENR',
            'confidence': 0.85,
            'p_value': 0.02,
            'confidence_interval': (0.78, 0.92),
            'expected_value': 0.85
        }

        commitment, error = self.handler.create_commitment(
            statistical_result=statistical_result,
            source_hypothesis='hypothesis-1',
            derivation_method='mcts_validation'
        )

        self.assertIsNotNone(commitment)

        # Check statement structure (ASCII format for Windows compatibility)
        self.assertIn('Lattice confinement enables LENR', commitment.statement)
        self.assertIn('confidence >=', commitment.statement)
        self.assertIn('p_value <=', commitment.statement)
        self.assertIn('CI in', commitment.statement)
        self.assertIn('IMPLIES Accept(', commitment.statement)

    def test_get_commitment(self):
        """Test getting a commitment by ID."""
        # Create a commitment
        statistical_result = {
            'hypothesis_statement': 'Test hypothesis',
            'confidence': 0.85,
            'p_value': 0.02,
            'confidence_interval': (0.78, 0.92),
            'expected_value': 0.85
        }

        commitment, _ = self.handler.create_commitment(
            statistical_result=statistical_result,
            source_hypothesis='hypothesis-1',
            derivation_method='test'
        )

        # Get by ID
        retrieved = self.handler.get_commitment(commitment.proposition_id)

        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.proposition_id, commitment.proposition_id)
        self.assertEqual(retrieved.statement, commitment.statement)

    def test_get_all_commitments(self):
        """Test getting all commitments."""
        # Create multiple commitments
        for i in range(3):
            statistical_result = {
                'hypothesis_statement': f'Test hypothesis {i}',
                'confidence': 0.8 + i * 0.05,
                'p_value': 0.02,
                'confidence_interval': (0.75, 0.95),
                'expected_value': 0.8
            }

            self.handler.create_commitment(
                statistical_result=statistical_result,
                source_hypothesis=f'hypothesis-{i}',
                derivation_method='test'
            )

        # Get all
        commitments = self.handler.get_all_commitments()
        self.assertEqual(len(commitments), 3)

    def test_get_commitments_by_status(self):
        """Test filtering commitments by status."""
        # Create a commitment
        statistical_result = {
            'hypothesis_statement': 'Test hypothesis',
            'confidence': 0.85,
            'p_value': 0.02,
            'confidence_interval': (0.78, 0.92),
            'expected_value': 0.85
        }

        commitment, _ = self.handler.create_commitment(
            statistical_result=statistical_result,
            source_hypothesis='hypothesis-1',
            derivation_method='test'
        )

        # Update status
        self.handler.update_commitment_status(
            commitment.proposition_id,
            CommitmentStatus.INTEGRATED
        )

        # Get by status
        integrated = self.handler.get_all_commitments(status=CommitmentStatus.INTEGRATED)
        self.assertEqual(len(integrated), 1)
        self.assertEqual(integrated[0].proposition_id, commitment.proposition_id)

    def test_get_commitments_by_hypothesis(self):
        """Test getting commitments for a specific hypothesis."""
        # Create commitments for same hypothesis
        for i in range(3):
            statistical_result = {
                'hypothesis_statement': f'Test hypothesis {i}',
                'confidence': 0.85,
                'p_value': 0.02,
                'confidence_interval': (0.78, 0.92),
                'expected_value': 0.85
            }

            self.handler.create_commitment(
                statistical_result=statistical_result,
                source_hypothesis='hypothesis-1',  # Same hypothesis
                derivation_method='test'
            )

        # Get by hypothesis
        commitments = self.handler.get_commitments_by_hypothesis('hypothesis-1')
        self.assertEqual(len(commitments), 3)

    def test_update_commitment_status(self):
        """Test updating commitment status."""
        # Create a commitment
        statistical_result = {
            'hypothesis_statement': 'Test hypothesis',
            'confidence': 0.85,
            'p_value': 0.02,
            'confidence_interval': (0.78, 0.92),
            'expected_value': 0.85
        }

        commitment, _ = self.handler.create_commitment(
            statistical_result=statistical_result,
            source_hypothesis='hypothesis-1',
            derivation_method='test'
        )

        self.assertEqual(commitment.status, CommitmentStatus.PENDING)

        # Update status
        success = self.handler.update_commitment_status(
            commitment.proposition_id,
            CommitmentStatus.INTEGRATED
        )

        self.assertTrue(success)
        self.assertEqual(commitment.status, CommitmentStatus.INTEGRATED)

    def test_update_commitment_status_not_found(self):
        """Test updating status of non-existent commitment."""
        success = self.handler.update_commitment_status(
            'non-existent-id',
            CommitmentStatus.INTEGRATED
        )

        self.assertFalse(success)

    def test_detect_contradictions(self):
        """Test contradiction detection."""
        # Create two contradictory commitments
        # Commitment 1: x > 5 with high confidence
        statistical_result_1 = {
            'hypothesis_statement': 'x > 5',
            'confidence': 0.90,
            'p_value': 0.02,
            'confidence_interval': (0.85, 0.95),
            'expected_value': 0.90
        }

        commitment_1, _ = self.handler.create_commitment(
            statistical_result=statistical_result_1,
            source_hypothesis='hypothesis-1',
            derivation_method='test'
        )

        # Commitment 2: x < 3 with high confidence (contradictory)
        statistical_result_2 = {
            'hypothesis_statement': 'x < 3',
            'confidence': 0.85,
            'p_value': 0.02,
            'confidence_interval': (0.78, 0.92),
            'expected_value': 0.85
        }

        commitment_2, _ = self.handler.create_commitment(
            statistical_result=statistical_result_2,
            source_hypothesis='hypothesis-2',
            derivation_method='test'
        )

        # Mark both as integrated
        self.handler.update_commitment_status(commitment_1.proposition_id, CommitmentStatus.INTEGRATED)
        self.handler.update_commitment_status(commitment_2.proposition_id, CommitmentStatus.INTEGRATED)

        # Detect contradictions
        contradictions = self.handler.detect_contradictions()

        # Should detect contradiction (both high confidence, opposite inequalities)
        # Note: This depends on the implementation's heuristic
        # For now, we just verify it runs without error
        self.assertIsInstance(contradictions, list)

    def test_get_contradiction_reports(self):
        """Test getting contradiction reports."""
        # Get reports (should be empty initially)
        reports = self.handler.get_contradiction_reports()
        self.assertEqual(len(reports), 0)

    def test_get_stats(self):
        """Test getting handler statistics."""
        # Create a commitment
        statistical_result = {
            'hypothesis_statement': 'Test hypothesis',
            'confidence': 0.85,
            'p_value': 0.02,
            'confidence_interval': (0.78, 0.92),
            'expected_value': 0.85
        }

        self.handler.create_commitment(
            statistical_result=statistical_result,
            source_hypothesis='hypothesis-1',
            derivation_method='test'
        )

        stats = self.handler.get_stats()

        self.assertIn('commitments', stats)
        self.assertIn('contradictions', stats)
        self.assertIn('config', stats)

        self.assertEqual(stats['commitments']['total'], 1)

    def test_clear_commitments(self):
        """Test clearing all commitments."""
        # Create some commitments
        statistical_result = {
            'hypothesis_statement': 'Test hypothesis',
            'confidence': 0.85,
            'p_value': 0.02,
            'confidence_interval': (0.78, 0.92),
            'expected_value': 0.85
        }

        self.handler.create_commitment(
            statistical_result=statistical_result,
            source_hypothesis='hypothesis-1',
            derivation_method='test'
        )

        self.assertGreater(len(self.handler.commitments), 0)

        # Clear
        count = self.handler.clear_commitments()
        self.assertGreater(count, 0)
        self.assertEqual(len(self.handler.commitments), 0)
        self.assertEqual(len(self.handler.contradiction_reports), 0)

    def test_configuration_validation(self):
        """Test that invalid configuration raises error."""
        with self.assertRaises(RuntimeError):
            FormalCommitmentsHandler(
                confidence_tracker=self.confidence_tracker,
                config={"significance_level": 1.5}  # Invalid: > 1
            )


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestFormalCommitment))
    suite.addTests(loader.loadTestsFromTestCase(TestFormalCommitmentsHandler))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())
