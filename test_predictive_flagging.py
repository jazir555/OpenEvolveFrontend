"""
Test Suite for LeanAide Predictive Flagging System

This module provides comprehensive tests for the predictive flagging system
that integrates with MDAP-MCTS-MAKER.
"""

import asyncio
import unittest
from unittest.mock import Mock, AsyncMock, patch
import json
import time

from leanaide_predictive_flagging import (
    PredictiveFlagConfig,
    Prediction,
    PredictionType,
    PredictiveAnalysis,
    FeatureExtractor,
    PredictionModel,
    SimpleEnsembleModel,
    PredictiveFlaggingSystem,
    MDAPPredictiveFlaggingSystem,
    MCTSPredictiveFlaggingSystem,
    MAKERPredictiveFlaggingSystem,
    IntegratedPredictiveFlaggingSystem,
    create_integrated_predictive_system,
    predict_item_quality,
    provide_early_warning
)


class TestPredictiveFlagConfig(unittest.TestCase):
    """Test predictive flagging configuration."""

    def test_config_creation(self):
        """Test creating predictive flagging configuration."""
        config = PredictiveFlagConfig(
            prediction_confidence_threshold=0.7,
            prediction_accuracy_threshold=0.8,
            prediction_horizon=5,
            min_historical_samples=10,
            historical_window_days=30,
            feature_weights={
                "agent_performance": 0.3,
                "confidence_trend": 0.25,
                "pattern_frequency": 0.2,
                "context_similarity": 0.15,
                "structural_indicators": 0.1
            },
            enable_ml_prediction=True,
            ml_model_type="ensemble",
            enable_feature_engineering=True,
            enable_context_awareness=True,
            enable_quality_prediction=True,
            enable_performance_prediction=True,
            enable_pattern_prediction=True,
            enable_agent_behavior_prediction=True,
            enable_predictive_flagging=True,
            enable_early_warning=True,
            enable_preemptive_pruning=False,
            enable_prediction_feedback=True,
            feedback_learning_rate=0.1
        )

        self.assertEqual(config.prediction_confidence_threshold, 0.7)
        self.assertEqual(config.prediction_accuracy_threshold, 0.8)
        self.assertEqual(config.prediction_horizon, 5)
        self.assertEqual(config.min_historical_samples, 10)
        self.assertEqual(config.historical_window_days, 30)
        self.assertEqual(config.feature_weights["agent_performance"], 0.3)
        self.assertTrue(config.enable_ml_prediction)
        self.assertEqual(config.ml_model_type, "ensemble")
        self.assertTrue(config.enable_feature_engineering)
        self.assertTrue(config.enable_context_awareness)
        self.assertTrue(config.enable_quality_prediction)
        self.assertTrue(config.enable_performance_prediction)
        self.assertTrue(config.enable_pattern_prediction)
        self.assertTrue(config.enable_agent_behavior_prediction)
        self.assertTrue(config.enable_predictive_flagging)
        self.assertTrue(config.enable_early_warning)
        self.assertFalse(config.enable_preemptive_pruning)
        self.assertTrue(config.enable_prediction_feedback)
        self.assertEqual(config.feedback_learning_rate, 0.1)


class TestPrediction(unittest.TestCase):
    """Test prediction data class."""

    def test_prediction_creation(self):
        """Test creating a prediction."""
        prediction = Prediction(
            prediction_type=PredictionType.QUALITY_LOW,
            predicted_item="test_item",
            confidence=0.8,
            probability=0.7,
            severity=0.6,
            features={"feature1": 1.0, "feature2": 0.5},
            model_used="test_model"
        )

        self.assertEqual(prediction.prediction_type, PredictionType.QUALITY_LOW)
        self.assertEqual(prediction.predicted_item, "test_item")
        self.assertEqual(prediction.confidence, 0.8)
        self.assertEqual(prediction.probability, 0.7)
        self.assertEqual(prediction.severity, 0.6)
        self.assertEqual(prediction.features["feature1"], 1.0)
        self.assertEqual(prediction.model_used, "test_model")

    def test_prediction_to_dict(self):
        """Test converting prediction to dictionary."""
        prediction = Prediction(
            prediction_type=PredictionType.PERFORMANCE_POOR,
            predicted_item="test_item",
            confidence=0.8,
            probability=0.7,
            severity=0.6
        )

        pred_dict = prediction.to_dict()
        self.assertEqual(pred_dict["prediction_type"], "performance_poor")
        self.assertEqual(pred_dict["predicted_item"], "test_item")
        self.assertEqual(pred_dict["confidence"], 0.8)
        self.assertEqual(pred_dict["probability"], 0.7)
        self.assertEqual(pred_dict["severity"], 0.6)


class TestFeatureExtractor(unittest.TestCase):
    """Test feature extraction system."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = PredictiveFlagConfig()
        self.extractor = FeatureExtractor(self.config)

    def test_extract_basic_features(self):
        """Test extracting basic features."""
        item = "theorem test : True := by simp"
        context = {"agent_id": "test_agent", "confidence": 0.8}
        
        features = self.extractor._extract_basic_features(item, context)
        
        self.assertGreaterEqual(features["item_length"], 0)
        self.assertGreaterEqual(features["item_word_count"], 0)
        self.assertGreaterEqual(features["item_line_count"], 0)
        self.assertTrue(isinstance(features["has_agent_id"], bool))
        self.assertTrue(isinstance(features["has_confidence"], bool))

    def test_extract_agent_performance_features(self):
        """Test extracting agent performance features."""
        context = {
            "agent_id": "test_agent",
            "agent_performance_history": [
                {"confidence": 0.8, "success_rate": 0.7},
                {"confidence": 0.6, "success_rate": 0.5}
            ]
        }
        
        features = self.extractor._extract_agent_performance_features(context)
        
        self.assertIn("agent_test_agent_avg_confidence", features)
        self.assertIn("agent_test_agent_avg_success_rate", features)

    def test_extract_confidence_trend_features(self):
        """Test extracting confidence trend features."""
        history = [
            Mock(confidence=0.8),
            Mock(confidence=0.7),
            Mock(confidence=0.6)
        ]
        
        features = self.extractor._extract_confidence_trend_features(history)
        
        self.assertIn("confidence_trend", features)
        self.assertIn("confidence_volatility", features)
        self.assertIn("confidence_declining", features)

    def test_extract_pattern_features(self):
        """Test extracting pattern features."""
        item = "theorem test : True := by sorry"
        
        features = self.extractor._extract_pattern_features(item)
        
        # Should have pattern counts for blocked patterns
        self.assertIn("pattern_sorry_count", features)

    def test_extract_structural_features(self):
        """Test extracting structural features."""
        item = "theorem test (n m : Nat) : n + m = m + n := by\n  intro n\n  intro m\n  rw [add_comm]"
        
        features = self.extractor._extract_structural_features(item)
        
        self.assertIn("has_quantifiers", features)
        self.assertIn("has_implications", features)

    def test_extract_contextual_features(self):
        """Test extracting contextual features."""
        context = {
            "depth": 10,
            "node_count": 50,
            "branch_factor": 2.5,
            "remaining_goals": ["goal1", "goal2"],
            "proof_state_complexity": 1.5,
            "time_elapsed": 5.0,
            "iterations": 100
        }
        
        features = self.extractor._extract_contextual_features(context)
        
        self.assertEqual(features["context_depth"], 10)
        self.assertEqual(features["context_node_count"], 50)
        self.assertEqual(features["context_remaining_goals"], 2)

    def test_extract_features(self):
        """Test extracting all features."""
        item = "theorem test : True := by simp"
        context = {"agent_id": "test_agent"}
        history = [Mock(confidence=0.8)]
        
        features = self.extractor.extract_features(item, context, history)
        
        self.assertGreater(len(features), 0)
        self.assertIn("item_length", features)


class TestSimpleEnsembleModel(unittest.TestCase):
    """Test simple ensemble prediction model."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = PredictiveFlagConfig()
        self.model = SimpleEnsembleModel("test_ensemble", self.config)

    def test_model_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.model_id, "test_ensemble")
        self.assertFalse(self.model.is_trained)

    def test_train_model(self):
        """Test training the model."""
        training_data = [
            ({"feature1": 1.0, "feature2": 0.5}, True),
            ({"feature1": 0.2, "feature2": 0.8}, False)
        ]
        
        self.model.train(training_data)
        
        self.assertTrue(self.model.is_trained)
        self.assertEqual(len(self.model.training_data), 2)

    def test_predict(self):
        """Test making predictions."""
        features = {"feature1": 0.5, "feature2": 0.3}
        
        # Even without training, should return defaults
        prob, conf = self.model.predict(features)
        
        self.assertIsInstance(prob, float)
        self.assertIsInstance(conf, float)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)


class TestPredictiveFlaggingSystem(unittest.TestCase):
    """Test predictive flagging system."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = PredictiveFlagConfig(
            prediction_confidence_threshold=0.5,
            min_historical_samples=1
        )
        self.system = PredictiveFlaggingSystem(self.config)

    def test_system_initialization(self):
        """Test system initialization."""
        self.assertEqual(self.config, self.system.config)
        self.assertIsNotNone(self.system.feature_extractor)
        self.assertIsNotNone(self.system.models)
        self.assertEqual(len(self.system.prediction_history), 0)

    def test_predict_item_quality(self):
        """Test predicting item quality."""
        item = "theorem test : True := by simp"
        context = {"agent_id": "test_agent", "confidence": 0.8}
        history = [Mock(confidence=0.7)]
        
        predictions = self.system.predict_item_quality(item, "test_id", context, history)
        
        # May or may not have predictions depending on features, but should not error
        self.assertIsInstance(predictions, list)

    def test_predict_agent_behavior(self):
        """Test predicting agent behavior."""
        predictions = self.system.predict_agent_behavior("test_agent")
        
        self.assertIsInstance(predictions, list)

    def test_predict_confidence_decline(self):
        """Test predicting confidence decline."""
        item = "test_item"
        history = [Mock(confidence=0.8), Mock(confidence=0.7), Mock(confidence=0.6)]
        
        predictions = self.system.predict_confidence_decline(item, history)
        
        self.assertIsInstance(predictions, list)

    def test_get_prediction_analysis(self):
        """Test getting prediction analysis."""
        analysis = self.system.get_prediction_analysis()
        
        self.assertIsInstance(analysis, PredictiveAnalysis)
        self.assertEqual(analysis.total_predictions, 0)

    def test_provide_early_warning(self):
        """Test providing early warning."""
        item = "theorem test : True := by simp"
        context = {"agent_id": "test_agent"}
        history = [Mock(confidence=0.8)]
        
        needs_attention, predictions, message = self.system.provide_early_warning(
            item, "test_id", context, history
        )
        
        self.assertIsInstance(needs_attention, bool)
        self.assertIsInstance(predictions, list)
        self.assertIsInstance(message, str)

    def test_record_outcome(self):
        """Test recording outcome."""
        # Add a prediction to history first
        pred = Prediction(
            prediction_type=PredictionType.QUALITY_LOW,
            predicted_item="test_item",
            confidence=0.8,
            probability=0.7,
            severity=0.6
        )
        self.system.prediction_history.append(
            Mock(
                prediction_id="test_pred_id",
                prediction=pred,
                actual_outcome=None,
                actual_severity=None,
                prediction_accuracy=None,
                feedback_timestamp=None,
                metadata={}
            )
        )
        
        success = self.system.record_outcome("test_pred_id", True, 0.8)
        
        self.assertTrue(success)


class TestMDAPPredictiveFlaggingSystem(unittest.TestCase):
    """Test MDAP-specific predictive flagging system."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = PredictiveFlagConfig()
        self.system = MDAPPredictiveFlaggingSystem(self.config)

    def test_predict_mdap_node_quality(self):
        """Test predicting MDAP node quality."""
        node = Mock()
        node.hash = "test_node_hash"
        node.state = Mock()
        node.state.hash = "test_state_hash"
        node.state.goals = ["goal1"]
        node.state.depth = 5
        
        predictions = self.system.predict_mdap_node_quality(node)
        
        self.assertIsInstance(predictions, list)

    def test_predict_mdap_action_quality(self):
        """Test predicting MDAP action quality."""
        predictions = self.system.predict_mdap_action_quality(
            action="simp",
            agent_id="test_agent",
            confidence=0.7
        )
        
        self.assertIsInstance(predictions, list)

    def test_predict_mdap_proof_quality(self):
        """Test predicting MDAP proof quality."""
        proof = Mock()
        proof.theorem_name = "test_theorem"
        proof.lean_code = "theorem test : True := by simp"
        proof.confidence = 0.8
        
        predictions = self.system.predict_mdap_proof_quality(proof)
        
        self.assertIsInstance(predictions, list)

    def test_record_agent_outcome(self):
        """Test recording agent outcome."""
        self.system.record_agent_outcome(
            agent_id="test_agent",
            action="simp",
            outcome=True,
            confidence=0.8,
            prediction_successful=True
        )
        
        self.assertIn("test_agent", self.system.agent_prediction_history)
        self.assertGreater(len(self.system.agent_prediction_history["test_agent"]), 0)


class TestMCTSPredictiveFlaggingSystem(unittest.TestCase):
    """Test MCTS-specific predictive flagging system."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = PredictiveFlagConfig()
        self.system = MCTSPredictiveFlaggingSystem(self.config)

    def test_predict_mcts_node_quality(self):
        """Test predicting MCTS node quality."""
        node = Mock()
        node.N = 10  # Visit count
        node.W = 5   # Total reward
        node.Q = 0.5 # Average reward
        node.hash = "test_node_hash"
        
        predictions = self.system.predict_mcts_node_quality(node)
        
        self.assertIsInstance(predictions, list)

    def test_predict_mcts_path_quality(self):
        """Test predicting MCTS path quality."""
        path = [Mock() for _ in range(10)]
        
        predictions = self.system.predict_mcts_path_quality(path)
        
        self.assertIsInstance(predictions, list)

    def test_record_node_outcome(self):
        """Test recording node outcome."""
        self.system.record_node_outcome(
            node_hash="test_node",
            outcome=True,
            visit_count=10,
            reward=5.0,
            prediction_successful=True
        )
        
        self.assertIn("test_node", self.system.node_prediction_history)
        self.assertGreater(len(self.system.node_prediction_history["test_node"]), 0)


class TestMAKERPredictiveFlaggingSystem(unittest.TestCase):
    """Test MAKER-specific predictive flagging system."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = PredictiveFlagConfig()
        self.system = MAKERPredictiveFlaggingSystem(self.config)

    def test_predict_maker_vote_quality(self):
        """Test predicting MAKER vote quality."""
        vote = Mock()
        vote.confidence = 0.7
        vote.voter_id = "test_voter"
        vote.tactic = "simp"
        
        predictions = self.system.predict_maker_vote_quality(vote)
        
        self.assertIsInstance(predictions, list)

    def test_predict_maker_aggregation_quality(self):
        """Test predicting MAKER aggregation quality."""
        votes = [Mock(confidence=0.7), Mock(confidence=0.8)]
        result = "selected_tactic"
        
        predictions = self.system.predict_maker_aggregation_quality(votes, result)
        
        self.assertIsInstance(predictions, list)

    def test_record_voter_outcome(self):
        """Test recording voter outcome."""
        self.system.record_voter_outcome(
            voter_id="test_voter",
            vote_accepted=True,
            confidence=0.8,
            prediction_successful=True
        )
        
        self.assertIn("test_voter", self.system.voter_prediction_history)
        self.assertGreater(len(self.system.voter_prediction_history["test_voter"]), 0)


class TestIntegratedPredictiveFlaggingSystem(unittest.TestCase):
    """Test integrated predictive flagging system."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = PredictiveFlagConfig()
        self.system = IntegratedPredictiveFlaggingSystem(self.config)

    def test_predict_quality(self):
        """Test predicting quality for different item types."""
        # Test action prediction
        predictions = self.system.predict_quality(
            item="simp",
            item_type="action",
            context={"agent_id": "test_agent", "confidence": 0.7}
        )
        self.assertIsInstance(predictions, list)
        
        # Test proof prediction
        proof = Mock()
        proof.theorem_name = "test_theorem"
        predictions = self.system.predict_quality(
            item=proof,
            item_type="proof"
        )
        self.assertIsInstance(predictions, list)
        
        # Test node prediction
        node = Mock()
        node.hash = "test_node"
        predictions = self.system.predict_quality(
            item=node,
            item_type="node",
            context={"system": "mdap"}
        )
        self.assertIsInstance(predictions, list)

    def test_provide_early_warning(self):
        """Test providing early warning."""
        needs_attention, predictions, message = self.system.provide_early_warning(
            item="test_item",
            item_type="action",
            context={"agent_id": "test_agent", "confidence": 0.3}  # Low confidence
        )
        
        self.assertIsInstance(needs_attention, bool)
        self.assertIsInstance(predictions, list)
        self.assertIsInstance(message, str)

    def test_analyze_predictions(self):
        """Test analyzing predictions."""
        analysis = self.system.analyze_predictions()
        
        self.assertIn("total_predictions", analysis)
        self.assertIn("mdap_analysis", analysis)
        self.assertIn("mcts_analysis", analysis)
        self.assertIn("maker_analysis", analysis)

    def test_record_outcome(self):
        """Test recording outcome."""
        success = self.system.record_outcome(
            system_type="mdap",
            item_id="test_item",
            outcome=True,
            actual_severity=0.8
        )
        
        self.assertTrue(success)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""

    def test_create_integrated_system(self):
        """Test creating integrated system."""
        system = create_integrated_predictive_system()
        
        self.assertIsInstance(system, IntegratedPredictiveFlaggingSystem)

    def test_predict_item_quality_convenience(self):
        """Test convenience function for predicting item quality."""
        predictions = predict_item_quality(
            item="theorem test : True := by simp",
            item_type="action",
            context={"agent_id": "test_agent", "confidence": 0.7}
        )
        
        self.assertIsInstance(predictions, list)

    def test_provide_early_warning_convenience(self):
        """Test convenience function for early warning."""
        needs_attention, predictions, message = provide_early_warning(
            item="test_item",
            item_type="action",
            context={"agent_id": "test_agent", "confidence": 0.3}
        )
        
        self.assertIsInstance(needs_attention, bool)
        self.assertIsInstance(predictions, list)
        self.assertIsInstance(message, str)


def run_comprehensive_tests():
    """Run all tests."""
    print("Running comprehensive tests for LeanAide Predictive Flagging System...")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add tests
    test_suite.addTest(unittest.makeSuite(TestPredictiveFlagConfig))
    test_suite.addTest(unittest.makeSuite(TestPrediction))
    test_suite.addTest(unittest.makeSuite(TestFeatureExtractor))
    test_suite.addTest(unittest.makeSuite(TestSimpleEnsembleModel))
    test_suite.addTest(unittest.makeSuite(TestPredictiveFlaggingSystem))
    test_suite.addTest(unittest.makeSuite(TestMDAPPredictiveFlaggingSystem))
    test_suite.addTest(unittest.makeSuite(TestMCTSPredictiveFlaggingSystem))
    test_suite.addTest(unittest.makeSuite(TestMAKERPredictiveFlaggingSystem))
    test_suite.addTest(unittest.makeSuite(TestIntegratedPredictiveFlaggingSystem))
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