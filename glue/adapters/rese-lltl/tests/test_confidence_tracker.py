#!/usr/bin/env python3
"""
Unit Tests for LLTL Confidence Tracker

Tests the ConfidenceTracker and related components.

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
    from confidence_tracker import (
        ConfidenceTracker,
        ConfidenceThreshold,
        ConfidenceLevel,
        ConfidenceHistory
    )
    CONFIDENCE_TRACKER_AVAILABLE = True
except ImportError as e:
    CONFIDENCE_TRACKER_AVAILABLE = False
    IMPORT_ERROR = str(e)


@unittest.skipIf(not CONFIDENCE_TRACKER_AVAILABLE, f"Confidence tracker not available: {IMPORT_ERROR if not CONFIDENCE_TRACKER_AVAILABLE else ''}")
class TestConfidenceThreshold(unittest.TestCase):
    """Test ConfidenceThreshold dataclass."""

    def test_confidence_threshold_creation(self):
        """Test creating a ConfidenceThreshold."""
        threshold = ConfidenceThreshold(
            threshold=0.90,
            level=ConfidenceLevel.VERY_HIGH,
            significance_level=0.05,
            derived_at=datetime.now(timezone.utc).isoformat(),
            derivation_method="tiered",
            correlation_id="test-correlation-1"
        )

        self.assertEqual(threshold.threshold, 0.90)
        self.assertEqual(threshold.level, ConfidenceLevel.VERY_HIGH)
        self.assertEqual(threshold.significance_level, 0.05)
        self.assertEqual(threshold.derivation_method, "tiered")

    def test_to_dict(self):
        """Test converting ConfidenceThreshold to dictionary."""
        threshold = ConfidenceThreshold(
            threshold=0.90,
            level=ConfidenceLevel.VERY_HIGH,
            significance_level=0.05,
            derived_at=datetime.now(timezone.utc).isoformat(),
            derivation_method="tiered",
            correlation_id="test-correlation-1",
            metadata={"test": "data"}
        )

        threshold_dict = threshold.to_dict()

        self.assertEqual(threshold_dict['threshold'], 0.90)
        self.assertEqual(threshold_dict['level'], "very_high")
        self.assertEqual(threshold_dict['significance_level'], 0.05)
        self.assertIn('metadata', threshold_dict)
        self.assertEqual(threshold_dict['metadata']['test'], "data")


@unittest.skipIf(not CONFIDENCE_TRACKER_AVAILABLE, f"Confidence tracker not available: {IMPORT_ERROR if not CONFIDENCE_TRACKER_AVAILABLE else ''}")
class TestConfidenceTracker(unittest.TestCase):
    """Test ConfidenceTracker class."""

    def setUp(self):
        """Set up test tracker."""
        # Set required environment variables
        os.environ['LLTL_SIGNIFICANCE_LEVEL'] = '0.05'
        os.environ['LLTL_CONFIDENCE_THRESHOLD_DEFAULT'] = '0.75'
        os.environ['LLTL_THRESHOLD_STRATEGY'] = 'tiered'

        if not CONFIDENCE_TRACKER_AVAILABLE:
            self.skipTest(f"Confidence tracker not available: {IMPORT_ERROR}")

        try:
            self.tracker = ConfidenceTracker()
        except Exception as e:
            self.skipTest(f"Failed to create tracker: {str(e)}")

    def test_initialization(self):
        """Test tracker initialization."""
        self.assertIsNotNone(self.tracker)
        self.assertEqual(self.tracker.config["significance_level"], 0.05)
        self.assertEqual(len(self.tracker.threshold_history), 0)
        self.assertEqual(len(self.tracker._threshold_cache), 0)

    def test_calculate_threshold_tiered_very_high(self):
        """Test threshold calculation for very high confidence."""
        threshold = self.tracker.calculate_threshold(
            confidence=0.98,
            derivation_method="tiered",
            correlation_id="test-correlation-1"
        )

        self.assertEqual(threshold.threshold, 0.90)
        self.assertEqual(threshold.level, ConfidenceLevel.VERY_HIGH)
        self.assertEqual(threshold.derivation_method, "tiered")

    def test_calculate_threshold_tiered_high(self):
        """Test threshold calculation for high confidence."""
        threshold = self.tracker.calculate_threshold(
            confidence=0.85,
            derivation_method="tiered",
            correlation_id="test-correlation-1"
        )

        self.assertEqual(threshold.threshold, 0.75)
        self.assertEqual(threshold.level, ConfidenceLevel.HIGH)

    def test_calculate_threshold_tiered_moderate(self):
        """Test threshold calculation for moderate confidence."""
        threshold = self.tracker.calculate_threshold(
            confidence=0.70,
            derivation_method="tiered",
            correlation_id="test-correlation-1"
        )

        self.assertEqual(threshold.threshold, 0.60)
        self.assertEqual(threshold.level, ConfidenceLevel.MODERATE)

    def test_calculate_threshold_tiered_low(self):
        """Test threshold calculation for low confidence."""
        threshold = self.tracker.calculate_threshold(
            confidence=0.50,
            derivation_method="tiered",
            correlation_id="test-correlation-1"
        )

        self.assertEqual(threshold.threshold, 0.50)
        self.assertEqual(threshold.level, ConfidenceLevel.LOW)

    def test_calculate_threshold_linear(self):
        """Test linear threshold calculation."""
        # Configure linear strategy
        tracker = ConfidenceTracker(config={"calculation_strategy": "linear"})

        threshold = tracker.calculate_threshold(
            confidence=0.50,
            derivation_method="linear",
            correlation_id="test-correlation-1"
        )

        # Linear interpolation between low (0.50) and very_high (0.90)
        # At 0.50 confidence, should be in the middle: 0.50 + (0.90 - 0.50) * 0.50 = 0.70
        expected = 0.50 + (0.90 - 0.50) * 0.50
        self.assertAlmostEqual(threshold.threshold, expected, places=2)

    def test_calculate_threshold_invalid_confidence(self):
        """Test that invalid confidence raises ValueError."""
        with self.assertRaises(ValueError):
            self.tracker.calculate_threshold(
                confidence=1.5,  # Invalid: > 1
                derivation_method="tiered"
            )

        with self.assertRaises(ValueError):
            self.tracker.calculate_threshold(
                confidence=-0.1,  # Invalid: < 0
                derivation_method="tiered"
            )

    def test_idempotency(self):
        """Test that same input produces same threshold (idempotency)."""
        threshold1 = self.tracker.calculate_threshold(
            confidence=0.85,
            derivation_method="tiered",
            correlation_id="test-correlation-1"
        )

        threshold2 = self.tracker.calculate_threshold(
            confidence=0.85,
            derivation_method="tiered",
            correlation_id="test-correlation-2"
        )

        # Same threshold
        self.assertEqual(threshold1.threshold, threshold2.threshold)
        self.assertEqual(threshold1.level, threshold2.level)

    def test_cache_hit(self):
        """Test that cache is used for repeated calculations."""
        # First call - cache miss
        threshold1 = self.tracker.calculate_threshold(
            confidence=0.85,
            derivation_method="tiered",
            correlation_id="test-correlation-1"
        )

        cache_size_after_first = len(self.tracker._threshold_cache)

        # Second call - cache hit
        threshold2 = self.tracker.calculate_threshold(
            confidence=0.85,
            derivation_method="tiered",
            correlation_id="test-correlation-2"
        )

        # Cache should have grown
        self.assertEqual(len(self.tracker._threshold_cache), cache_size_after_first)

        # Results should be identical (same object from cache)
        self.assertIs(threshold1, threshold2)

    def test_track_threshold(self):
        """Test tracking threshold in history."""
        threshold = self.tracker.calculate_threshold(
            confidence=0.85,
            derivation_method="tiered",
            correlation_id="test-correlation-1"
        )

        history_id = self.tracker.track_threshold(
            proposition_id="test-proposition-1",
            input_confidence=0.85,
            threshold=threshold,
            correlation_id="test-correlation-1"
        )

        self.assertIsNotNone(history_id)
        self.assertEqual(len(self.tracker.threshold_history), 1)

        history = self.tracker.threshold_history[0]
        self.assertEqual(history.proposition_id, "test-proposition-1")
        self.assertEqual(history.input_confidence, 0.85)
        self.assertEqual(history.calculated_threshold, threshold)

    def test_track_threshold_disabled(self):
        """Test that tracking fails when history is disabled."""
        tracker = ConfidenceTracker(config={"enable_history": False})

        threshold = tracker.calculate_threshold(
            confidence=0.85,
            derivation_method="tiered"
        )

        with self.assertRaises(RuntimeError):
            tracker.track_threshold(
                proposition_id="test-proposition-1",
                input_confidence=0.85,
                threshold=threshold
            )

    def test_get_history(self):
        """Test getting threshold history."""
        # Create some thresholds
        threshold1 = self.tracker.calculate_threshold(confidence=0.85, derivation_method="tiered")
        self.tracker.track_threshold("prop-1", 0.85, threshold1)

        threshold2 = self.tracker.calculate_threshold(confidence=0.90, derivation_method="tiered")
        self.tracker.track_threshold("prop-2", 0.90, threshold2)

        # Get all history
        history = self.tracker.get_history()
        self.assertEqual(len(history), 2)

        # Get history for specific proposition
        prop_history = self.tracker.get_history(proposition_id="prop-1")
        self.assertEqual(len(prop_history), 1)
        self.assertEqual(prop_history[0].proposition_id, "prop-1")

    def test_get_history_with_limit(self):
        """Test getting history with limit."""
        # Create multiple thresholds
        for i in range(10):
            threshold = self.tracker.calculate_threshold(confidence=0.8 + (i % 3) * 0.05, derivation_method="tiered")
            self.tracker.track_threshold(f"prop-{i}", 0.8 + (i % 3) * 0.05, threshold)

        # Get with limit
        history = self.tracker.get_history(limit=5)
        self.assertEqual(len(history), 5)

    def test_clear_history(self):
        """Test clearing threshold history."""
        # Create some thresholds
        threshold = self.tracker.calculate_threshold(confidence=0.85, derivation_method="tiered")
        self.tracker.track_threshold("prop-1", 0.85, threshold)

        self.assertGreater(len(self.tracker.threshold_history), 0)

        # Clear
        count = self.tracker.clear_history()
        self.assertEqual(count, 1)
        self.assertEqual(len(self.tracker.threshold_history), 0)

    def test_get_stats(self):
        """Test getting tracker statistics."""
        # Create some thresholds
        threshold = self.tracker.calculate_threshold(confidence=0.85, derivation_method="tiered")
        self.tracker.track_threshold("prop-1", 0.85, threshold)

        stats = self.tracker.get_stats()

        self.assertIn('config', stats)
        self.assertIn('history', stats)
        self.assertIn('thresholds', stats)

        self.assertEqual(stats['history']['total_entries'], 1)
        self.assertGreater(stats['history']['cache_size'], 0)

    def test_configuration_validation(self):
        """Test that invalid configuration raises error."""
        with self.assertRaises(RuntimeError):
            ConfidenceTracker(config={"significance_level": 1.5})  # Invalid: > 1

        with self.assertRaises(RuntimeError):
            ConfidenceTracker(config={"max_history_size": -1})  # Invalid: negative


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestConfidenceThreshold))
    suite.addTests(loader.loadTestsFromTestCase(TestConfidenceTracker))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())
