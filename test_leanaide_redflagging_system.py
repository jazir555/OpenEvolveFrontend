"""
Test Suite for LeanAide Red-Flagging System

This module provides comprehensive tests for the red-flagging system
that integrates with MDAP-MCTS-MAKER.
"""

import asyncio
import unittest
from unittest.mock import Mock, AsyncMock, patch
import json
import time

from leanaide_redflagging_system import (
    RedFlagConfig,
    RedFlag,
    RedFlagType,
    RedFlagAnalysis,
    RedFlaggingSystem,
    MDAPRedFlaggingSystem,
    MCTSRedFlaggingSystem,
    MAKERRedFlaggingSystem,
    IntegratedRedFlaggingSystem,
    create_integrated_red_flagging_system,
    flag_mdap_mcts_item
)


class TestRedFlagConfig(unittest.TestCase):
    """Test red-flagging configuration."""

    def test_config_creation(self):
        """Test creating red-flagging configuration."""
        config = RedFlagConfig(
            confidence_threshold=0.3,
            confidence_variance_threshold=0.1,
            blocked_patterns=["sorry", "admit"],
            suspicious_patterns=["error", "failed"],
            max_proof_length=1000,
            max_token_count=4000,
            min_proof_length=1,
            performance_threshold=0.1,
            vote_agreement_threshold=0.3,
            enable_adaptive_thresholds=True,
            threshold_adjustment_rate=0.05,
            enable_detailed_analysis=True,
            enable_performance_tracking=True,
            enable_pattern_learning=True,
            enable_flagging=True,
            enable_pruning=True,
            enable_fallback=True
        )

        self.assertEqual(config.confidence_threshold, 0.3)
        self.assertEqual(config.confidence_variance_threshold, 0.1)
        self.assertEqual(config.blocked_patterns, ["sorry", "admit"])
        self.assertEqual(config.suspicious_patterns, ["error", "failed"])
        self.assertEqual(config.max_proof_length, 1000)
        self.assertEqual(config.max_token_count, 4000)
        self.assertEqual(config.min_proof_length, 1)
        self.assertEqual(config.performance_threshold, 0.1)
        self.assertEqual(config.vote_agreement_threshold, 0.3)
        self.assertTrue(config.enable_adaptive_thresholds)
        self.assertEqual(config.threshold_adjustment_rate, 0.05)
        self.assertTrue(config.enable_detailed_analysis)
        self.assertTrue(config.enable_performance_tracking)
        self.assertTrue(config.enable_pattern_learning)
        self.assertTrue(config.enable_flagging)
        self.assertTrue(config.enable_pruning)
        self.assertTrue(config.enable_fallback)


class TestRedFlag(unittest.TestCase):
    """Test red flag data class."""

    def test_red_flag_creation(self):
        """Test creating a red flag."""
        flag = RedFlag(
            flag_type=RedFlagType.CONFIDENCE_LOW,
            reason="Confidence too low",
            severity=0.8,
            confidence=0.9
        )

        self.assertEqual(flag.flag_type, RedFlagType.CONFIDENCE_LOW)
        self.assertEqual(flag.reason, "Confidence too low")
        self.assertEqual(flag.severity, 0.8)
        self.assertEqual(flag.confidence, 0.9)

    def test_red_flag_to_dict(self):
        """Test converting red flag to dictionary."""
        flag = RedFlag(
            flag_type=RedFlagType.CONFIDENCE_LOW,
            reason="Confidence too low",
            severity=0.8,
            confidence=0.9
        )

        flag_dict = flag.to_dict()
        self.assertEqual(flag_dict["flag_type"], "confidence_low")
        self.assertEqual(flag_dict["reason"], "Confidence too low")
        self.assertEqual(flag_dict["severity"], 0.8)
        self.assertEqual(flag_dict["confidence"], 0.9)


class TestRedFlagAnalysis(unittest.TestCase):
    """Test red flag analysis."""

    def test_red_flag_analysis_creation(self):
        """Test creating red flag analysis."""
        analysis = RedFlagAnalysis(
            total_flags=5,
            flag_types={"confidence_low": 2, "pattern_blocked": 3},
            severity_distribution={"0.0-0.2": 1, "0.8-1.0": 4},
            flagged_items=["item1", "item2"]
        )

        self.assertEqual(analysis.total_flags, 5)
        self.assertEqual(analysis.flag_types["confidence_low"], 2)
        self.assertEqual(analysis.severity_distribution["0.8-1.0"], 4)
        self.assertEqual(analysis.flagged_items, ["item1", "item2"])

    def test_red_flag_analysis_to_dict(self):
        """Test converting red flag analysis to dictionary."""
        analysis = RedFlagAnalysis(
            total_flags=2,
            flag_types={"confidence_low": 1},
            flagged_items=["item1"]
        )

        analysis_dict = analysis.to_dict()
        self.assertEqual(analysis_dict["total_flags"], 2)
        self.assertEqual(analysis_dict["flag_types"]["confidence_low"], 1)
        self.assertEqual(analysis_dict["flagged_items"], ["item1"])


class TestRedFlaggingSystem(unittest.TestCase):
    """Test basic red-flagging system."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = RedFlagConfig(
            confidence_threshold=0.3,
            max_proof_length=100,
            blocked_patterns=["sorry", "admit"]
        )
        self.system = RedFlaggingSystem(self.config)

    def test_flag_item_with_low_confidence(self):
        """Test flagging an item with low confidence."""
        item = Mock()
        item.confidence = 0.2  # Below threshold
        
        is_flagged, flags = self.system.flag_item(item)
        
        self.assertTrue(is_flagged)
        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0].flag_type, RedFlagType.CONFIDENCE_LOW)
        self.assertIn("0.200", flags[0].reason)

    def test_flag_item_with_blocked_pattern(self):
        """Test flagging an item with blocked pattern."""
        item = "theorem test : True := by sorry"
        
        is_flagged, flags = self.system.flag_item(item)
        
        self.assertTrue(is_flagged)
        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0].flag_type, RedFlagType.PATTERN_BLOCKED)
        self.assertIn("sorry", flags[0].reason)

    def test_flag_item_with_high_confidence(self):
        """Test that high confidence items are not flagged."""
        item = Mock()
        item.confidence = 0.8  # Above threshold
        
        is_flagged, flags = self.system.flag_item(item)
        
        self.assertFalse(is_flagged)
        self.assertEqual(len(flags), 0)

    def test_flag_item_with_long_proof(self):
        """Test flagging an item with too many lines."""
        long_proof = "\n".join([f"  -- Line {i}" for i in range(150)])  # More than 100 lines
        
        is_flagged, flags = self.system.flag_item(long_proof)
        
        self.assertTrue(is_flagged)
        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0].flag_type, RedFlagType.LENGTH_TOO_LONG)

    def test_analyze_flags(self):
        """Test flag analysis."""
        flags = [
            RedFlag(RedFlagType.CONFIDENCE_LOW, "Low confidence", 0.9, 0.8),
            RedFlag(RedFlagType.PATTERN_BLOCKED, "Blocked pattern", 0.7, 0.9)
        ]
        
        analysis = self.system.analyze_flags(flags)
        
        self.assertEqual(analysis.total_flags, 2)
        self.assertEqual(analysis.flag_types["confidence_low"], 1)
        self.assertEqual(analysis.flag_types["pattern_blocked"], 1)
        self.assertGreater(analysis.detailed_analysis["average_severity"], 0.7)
        self.assertGreater(analysis.detailed_analysis["average_confidence"], 0.8)

    def test_update_agent_performance(self):
        """Test updating agent performance."""
        self.system.update_agent_performance("test_agent", True, 0.8)
        
        stats = self.system.get_performance_stats()
        self.assertIn("test_agent", stats)
        self.assertEqual(stats["test_agent"]["success"], 1)
        self.assertEqual(stats["test_agent"]["total"], 1)
        self.assertGreater(stats["test_agent"]["avg_confidence"], 0.5)


class TestMDAPRedFlaggingSystem(unittest.TestCase):
    """Test MDAP-specific red-flagging system."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = RedFlagConfig()
        self.system = MDAPRedFlaggingSystem(self.config)

    def test_flag_mdap_node(self):
        """Test flagging an MDAP node."""
        node = Mock()
        node.confidence = 0.2  # Low confidence
        
        is_flagged, flags = self.system.flag_mdap_node(node)
        
        self.assertTrue(is_flagged)
        self.assertGreater(len(flags), 0)

    def test_flag_mdap_action(self):
        """Test flagging an MDAP action."""
        is_flagged, flags = self.system.flag_mdap_action(
            action="simp",
            agent_id="test_agent",
            confidence=0.1
        )
        
        self.assertTrue(is_flagged)
        self.assertGreater(len(flags), 0)

    def test_flag_mdap_proof(self):
        """Test flagging an MDAP proof."""
        proof = Mock()
        proof.lean_code = "theorem test : True := by sorry"
        proof.confidence = 0.5
        
        is_flagged, flags = self.system.flag_mdap_proof(proof)
        
        self.assertTrue(is_flagged)
        self.assertGreater(len(flags), 0)


class TestMCTSRedFlaggingSystem(unittest.TestCase):
    """Test MCTS-specific red-flagging system."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = RedFlagConfig()
        self.system = MCTSRedFlaggingSystem(self.config)

    def test_flag_mcts_node(self):
        """Test flagging an MCTS node."""
        node = Mock()
        node.N = 10  # Visit count
        node.W = 1   # Total reward
        node.Q = 0.1 # Average reward
        
        is_flagged, flags = self.system.flag_mcts_node(node)
        
        self.assertFalse(is_flagged)  # Should not be flagged by default
        self.assertEqual(len(flags), 0)

    def test_flag_mcts_path(self):
        """Test flagging an MCTS path."""
        path = [Mock() for _ in range(150)]  # Long path
        
        is_flagged, flags = self.system.flag_mcts_path(path)
        
        self.assertTrue(is_flagged)
        self.assertGreater(len(flags), 0)


class TestMAKERRedFlaggingSystem(unittest.TestCase):
    """Test MAKER-specific red-flagging system."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = RedFlagConfig()
        self.system = MAKERRedFlaggingSystem(self.config)

    def test_flag_maker_vote(self):
        """Test flagging a MAKER vote."""
        vote = Mock()
        vote.confidence = 0.1
        vote.voter_id = "test_voter"
        
        is_flagged, flags = self.system.flag_maker_vote(vote)
        
        self.assertTrue(is_flagged)
        self.assertGreater(len(flags), 0)

    def test_flag_maker_aggregation(self):
        """Test flagging a MAKER aggregation."""
        votes = [Mock(confidence=0.1), Mock(confidence=0.9)]
        
        is_flagged, flags = self.system.flag_maker_aggregation(votes, "result")
        
        self.assertTrue(is_flagged)
        self.assertGreater(len(flags), 0)


class TestIntegratedRedFlaggingSystem(unittest.TestCase):
    """Test integrated red-flagging system."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = RedFlagConfig()
        self.system = IntegratedRedFlaggingSystem(self.config)

    def test_flag_mdap_mcts_item(self):
        """Test flagging different types of items."""
        # Test action flagging
        is_flagged, flags = self.system.flag_mdap_mcts_item(
            item="simp",
            item_type="action",
            context={"agent_id": "test", "confidence": 0.1}
        )
        self.assertTrue(is_flagged)
        
        # Test proof flagging
        is_flagged, flags = self.system.flag_mdap_mcts_item(
            item="theorem test : True := by sorry",
            item_type="proof"
        )
        self.assertTrue(is_flagged)
        
        # Test vote flagging
        vote = Mock()
        vote.confidence = 0.1
        is_flagged, flags = self.system.flag_mdap_mcts_item(
            item=vote,
            item_type="vote"
        )
        self.assertTrue(is_flagged)

    def test_analyze_system_flags(self):
        """Test system-wide flag analysis."""
        flags = [
            RedFlag(RedFlagType.CONFIDENCE_LOW, "Low confidence", 0.9, 0.8),
            RedFlag(RedFlagType.PATTERN_BLOCKED, "Blocked pattern", 0.7, 0.9)
        ]
        
        analysis = self.system.analyze_system_flags(flags)
        
        self.assertEqual(analysis["total_flags"], 2)
        self.assertIsNotNone(analysis["mdap_analysis"])
        self.assertIsNotNone(analysis["mcts_analysis"])
        self.assertIsNotNone(analysis["maker_analysis"])

    def test_get_system_recommendations(self):
        """Test getting system recommendations."""
        flags = [
            RedFlag(RedFlagType.CONFIDENCE_LOW, "Low confidence", 0.9, 0.8),
            RedFlag(RedFlagType.PERFORMANCE_POOR, "Poor performance", 0.7, 0.9)
        ]
        
        recommendations = self.system.get_system_recommendations(flags)
        
        self.assertGreater(len(recommendations), 0)
        self.assertTrue(any("confidence" in rec.lower() for rec in recommendations))


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""

    def test_create_integrated_system(self):
        """Test creating integrated system."""
        system = create_integrated_red_flagging_system()
        
        self.assertIsInstance(system, IntegratedRedFlaggingSystem)

    def test_flag_mdap_mcts_item_convenience(self):
        """Test convenience function for flagging."""
        is_flagged, flags = flag_mdap_mcts_item(
            item="theorem test : True := by sorry",
            item_type="proof"
        )
        
        self.assertTrue(is_flagged)
        self.assertGreater(len(flags), 0)


def run_comprehensive_tests():
    """Run all tests."""
    print("Running comprehensive tests for LeanAide Red-Flagging System...")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add tests
    test_suite.addTest(unittest.makeSuite(TestRedFlagConfig))
    test_suite.addTest(unittest.makeSuite(TestRedFlag))
    test_suite.addTest(unittest.makeSuite(TestRedFlagAnalysis))
    test_suite.addTest(unittest.makeSuite(TestRedFlaggingSystem))
    test_suite.addTest(unittest.makeSuite(TestMDAPRedFlaggingSystem))
    test_suite.addTest(unittest.makeSuite(TestMCTSRedFlaggingSystem))
    test_suite.addTest(unittest.makeSuite(TestMAKERRedFlaggingSystem))
    test_suite.addTest(unittest.makeSuite(TestIntegratedRedFlaggingSystem))
    test_suite.addTest(unittest.makeSuite(TestConvenienceFunctions))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success: {result.testsRun - len(result.failures) - len(result.errors)}/{result.testsRun}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_tests()
    exit(0 if success else 1)