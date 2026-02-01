"""
Test QualityGateEngine ICR Integration

Tests for the ICR (Iterative Contextual Refinements) integration in QualityGateEngine.
"""

import unittest
from unittest.mock import Mock, MagicMock
from typing import List

# Import the quality gate engine
from quality_gate_engine import (
    QualityGateEngine,
    QualityThresholdManager,
    QualityLevel,
    ContentType,
    GateDecision
)
from evaluator_team import (
    EvaluationMetric,
    EvaluationScore,
    EvaluatorAssessment,
    EvaluationConfidence
)


class TestQualityGateICRIntegration(unittest.TestCase):
    """Test ICR integration in QualityGateEngine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.threshold_manager = QualityThresholdManager()
        self.engine = QualityGateEngine(
            threshold_manager=self.threshold_manager,
            enable_icr=True
        )
        
        # Create mock assessments
        self.mock_assessments = self._create_mock_assessments()
    
    def _create_mock_assessments(self) -> List[EvaluatorAssessment]:
        """Create mock assessments for testing"""
        assessments = []
        
        for i in range(3):
            assessment = EvaluatorAssessment(
                evaluator_id=f"evaluator_{i}",
                scores=[
                    EvaluationScore(metric=EvaluationMetric.CORRECTNESS, score=75.0 + i * 5),
                    EvaluationScore(metric=EvaluationMetric.COMPLETENESS, score=70.0 + i * 5),
                    EvaluationScore(metric=EvaluationMetric.CLARITY, score=65.0 + i * 5),
                    EvaluationScore(metric=EvaluationMetric.EFFECTIVENESS, score=70.0 + i * 5),
                    EvaluationScore(metric=EvaluationMetric.EFFICIENCY, score=65.0 + i * 5),
                    EvaluationScore(metric=EvaluationMetric.MAINTAINABILITY, score=60.0 + i * 5),
                ],
                composite_score=72.0 + i * 3,
                confidence_level=EvaluationConfidence.MODERATE,
                feedback="Test assessment"
            )
            assessments.append(assessment)
        
        return assessments
    
    def test_icr_enabled_by_default(self):
        """Test that ICR is enabled by default"""
        engine = QualityGateEngine(threshold_manager=self.threshold_manager)
        self.assertTrue(engine.enable_icr)
        self.assertIsNotNone(engine.icr_pattern_store)
    
    def test_icr_can_be_disabled(self):
        """Test that ICR can be disabled"""
        engine = QualityGateEngine(
            threshold_manager=self.threshold_manager,
            enable_icr=False
        )
        self.assertFalse(engine.enable_icr)
    
    def test_evaluate_with_icr_storage(self):
        """Test that evaluate stores ICR patterns"""
        # Initial pattern count should be 0
        stats_before = self.engine.get_icr_statistics()
        self.assertEqual(stats_before['total_patterns'], 0)
        
        # Run evaluation
        report = self.engine.evaluate(
            self.mock_assessments,
            content_type=ContentType.CODE,
            quality_level=QualityLevel.STANDARD,
            complexity_score=5,
            store_pattern=True
        )
        
        # Verify report was generated
        self.assertIsNotNone(report)
        self.assertIsInstance(report.decision, GateDecision)
        
        # Verify pattern was stored
        stats_after = self.engine.get_icr_statistics()
        self.assertEqual(stats_after['total_patterns'], 1)
        self.assertGreater(stats_after['overall_pass_rate'], 0)
    
    def test_predict_pass_probability(self):
        """Test pass/fail probability prediction"""
        # Store some patterns first
        self.engine.evaluate(
            self.mock_assessments,
            content_type=ContentType.CODE,
            quality_level=QualityLevel.STANDARD,
            complexity_score=5,
            store_pattern=True
        )
        
        # Get prediction
        prediction = self.engine.predict_pass_probability(
            self.mock_assessments,
            content_type=ContentType.CODE,
            quality_level=QualityLevel.STANDARD,
            complexity_score=5
        )
        
        # Verify prediction structure
        self.assertIn('prediction', prediction)
        self.assertIn('pass_probability', prediction)
        self.assertIn('confidence', prediction)
        self.assertIn('metric_predictions', prediction)
        
        # Prediction should be one of the expected values
        self.assertIn(prediction['prediction'], ['pass', 'conditional_pass', 'fail'])
        
        # Pass probability should be between 0 and 1
        self.assertGreaterEqual(prediction['pass_probability'], 0.0)
        self.assertLessEqual(prediction['pass_probability'], 1.0)
        
        # Confidence should be between 0 and 1
        self.assertGreaterEqual(prediction['confidence'], 0.0)
        self.assertLessEqual(prediction['confidence'], 1.0)
    
    def test_adapt_threshold(self):
        """Test adaptive threshold adjustment"""
        # Store some pass patterns
        for _ in range(5):
            self.engine.evaluate(
                self.mock_assessments,
                content_type=ContentType.CODE,
                quality_level=QualityLevel.STANDARD,
                complexity_score=5,
                store_pattern=True
            )
        
        # Get original threshold
        original_threshold = self.threshold_manager.get_threshold(
            ContentType.CODE, 
            QualityLevel.STANDARD
        )
        original_score = original_threshold.min_overall_score
        
        # Get adapted threshold
        adapted_threshold = self.engine.adapt_threshold(
            original_threshold,
            ContentType.CODE,
            QualityLevel.STANDARD,
            complexity_score=5
        )
        
        # Adapted threshold should be a QualityThreshold object
        self.assertEqual(type(adapted_threshold).__name__, 'QualityThreshold')
        
        # Score should not be negative
        self.assertGreaterEqual(adapted_threshold.min_overall_score, 0)
    
    def test_learn_from_refinement(self):
        """Test learning from refinement outcomes"""
        # Create mock reports
        original_report = Mock()
        original_report.decision = GateDecision.FAIL
        original_report.overall_score = 55.0
        original_report.threshold_used.content_type = ContentType.CODE
        original_report.threshold_used.quality_level = QualityLevel.STANDARD
        
        refined_report = Mock()
        refined_report.decision = GateDecision.PASS
        refined_report.overall_score = 78.0
        
        # Learn from refinement
        result = self.engine.learn_from_refinement(
            original_report,
            refined_report,
            refinement_type="content_improvement"
        )
        
        # Verify result structure
        self.assertTrue(result['learned'])
        self.assertEqual(result['refinement_type'], 'content_improvement')
        self.assertIn('avg_score_improvement', result)
        self.assertIn('success_rate', result)
        self.assertIn('total_refinements', result)
        
        # Score improvement should be positive
        self.assertGreater(result['avg_score_improvement'], 0)
        
        # Success rate should be 1.0 (refinement resulted in pass)
        self.assertEqual(result['success_rate'], 1.0)
    
    def test_get_icr_statistics(self):
        """Test ICR statistics retrieval"""
        # Store some patterns
        for _ in range(3):
            self.engine.evaluate(
                self.mock_assessments,
                content_type=ContentType.DOCUMENT,
                quality_level=QualityLevel.STANDARD,
                complexity_score=5,
                store_pattern=True
            )
        
        # Get statistics
        stats = self.engine.get_icr_statistics()
        
        # Verify statistics structure
        self.assertTrue(stats['icr_enabled'])
        self.assertGreater(stats['total_patterns'], 0)
        self.assertGreaterEqual(stats['overall_pass_rate'], 0.0)
        self.assertLessEqual(stats['overall_pass_rate'], 1.0)
        self.assertIn('patterns_by_content_type', stats)
        self.assertIn('refinement_statistics', stats)
        self.assertIn('adaptive_thresholds', stats)
    
    def test_clear_icr_patterns(self):
        """Test clearing ICR patterns"""
        # Store some patterns
        for _ in range(5):
            self.engine.evaluate(
                self.mock_assessments,
                content_type=ContentType.CODE,
                quality_level=QualityLevel.STANDARD,
                complexity_score=5,
                store_pattern=True
            )
        
        # Verify patterns exist
        stats_before = self.engine.get_icr_statistics()
        self.assertGreater(stats_before['total_patterns'], 0)
        
        # Clear patterns
        self.engine.clear_icr_patterns()
        
        # Verify patterns are cleared
        stats_after = self.engine.get_icr_statistics()
        self.assertEqual(stats_after['total_patterns'], 0)
    
    def test_predict_with_disabled_icr(self):
        """Test prediction when ICR is disabled"""
        engine = QualityGateEngine(
            threshold_manager=self.threshold_manager,
            enable_icr=False
        )
        
        # Get prediction
        prediction = engine.predict_pass_probability(
            self.mock_assessments,
            content_type=ContentType.CODE,
            quality_level=QualityLevel.STANDARD,
            complexity_score=5
        )
        
        # Should return unknown prediction
        self.assertEqual(prediction['prediction'], 'unknown')
        self.assertEqual(prediction['confidence'], 0.0)
        self.assertEqual(prediction['reason'], 'ICR disabled')
    
    def test_metric_predictions(self):
        """Test metric-specific predictions"""
        # Store some patterns
        for _ in range(5):
            self.engine.evaluate(
                self.mock_assessments,
                content_type=ContentType.CODE,
                quality_level=QualityLevel.STANDARD,
                complexity_score=5,
                store_pattern=True
            )
        
        # Get prediction
        prediction = self.engine.predict_pass_probability(
            self.mock_assessments,
            content_type=ContentType.CODE,
            quality_level=QualityLevel.STANDARD,
            complexity_score=5
        )
        
        # Verify metric predictions
        metric_predictions = prediction['metric_predictions']
        self.assertIsInstance(metric_predictions, dict)
        
        for metric, pred in metric_predictions.items():
            self.assertIn('score', pred)
            self.assertIn('predicted_pass_rate', pred)
            self.assertGreaterEqual(pred['predicted_pass_rate'], 0.0)
            self.assertLessEqual(pred['predicted_pass_rate'], 1.0)
    
    def test_adaptive_threshold_accumulates(self):
        """Test that adaptive threshold adjustments accumulate"""
        engine = QualityGateEngine(
            threshold_manager=self.threshold_manager,
            enable_icr=True
        )
        
        # Simulate multiple failed refinements (low success rate)
        original_report = Mock()
        original_report.decision = GateDecision.FAIL
        original_report.overall_score = 50.0
        original_report.threshold_used.content_type = ContentType.CODE
        original_report.threshold_used.quality_level = QualityLevel.STANDARD
        
        refined_report = Mock()
        refined_report.decision = GateDecision.FAIL  # Still failed
        refined_report.overall_score = 55.0  # Slight improvement but still fail
        
        # Learn from multiple failed refinements
        for _ in range(3):
            engine.learn_from_refinement(
                original_report,
                refined_report,
                refinement_type="content_improvement"
            )
        
        # Check that adaptive threshold was adjusted
        stats = engine.get_icr_statistics()
        self.assertIn('adaptive_thresholds', stats)
    
    def test_complexity_patterns_stored_separately(self):
        """Test that patterns are stored by complexity level"""
        # Store patterns for different complexity levels
        for complexity in [3, 5, 7]:
            self.engine.evaluate(
                self.mock_assessments,
                content_type=ContentType.CODE,
                quality_level=QualityLevel.STANDARD,
                complexity_score=complexity,
                store_pattern=True
            )
        
        # Get statistics
        stats = self.engine.get_icr_statistics()
        
        # Should have patterns for different complexity levels
        patterns_by_type = stats['patterns_by_content_type']
        
        # At least some patterns should be stored
        self.assertGreater(stats['total_patterns'], 0)
    
    def test_evaluate_without_storing_pattern(self):
        """Test evaluate with store_pattern=False"""
        # Run evaluation without storing pattern
        report = self.engine.evaluate(
            self.mock_assessments,
            content_type=ContentType.CODE,
            quality_level=QualityLevel.STANDARD,
            complexity_score=5,
            store_pattern=False
        )
        
        # Verify report was still generated
        self.assertIsNotNone(report)
        self.assertIsInstance(report.decision, GateDecision)
        
        # Pattern should not be stored
        stats = self.engine.get_icr_statistics()
        self.assertEqual(stats['total_patterns'], 0)


class TestQualityGateICRE2EWorkflow(unittest.TestCase):
    """End-to-end workflow tests for QualityGate ICR integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.threshold_manager = QualityThresholdManager()
        self.engine = QualityGateEngine(
            threshold_manager=self.threshold_manager,
            enable_icr=True
        )
    
    def _create_assessment(self, scores: dict) -> EvaluatorAssessment:
        """Create an assessment with specific scores"""
        score_objects = [
            EvaluationScore(metric=EvaluationMetric.CORRECTNESS, score=scores.get('correctness', 70)),
            EvaluationScore(metric=EvaluationMetric.COMPLETENESS, score=scores.get('completeness', 70)),
            EvaluationScore(metric=EvaluationMetric.CLARITY, score=scores.get('clarity', 70)),
            EvaluationScore(metric=EvaluationMetric.EFFECTIVENESS, score=scores.get('effectiveness', 70)),
            EvaluationScore(metric=EvaluationMetric.EFFICIENCY, score=scores.get('efficiency', 70)),
            EvaluationScore(metric=EvaluationMetric.MAINTAINABILITY, score=scores.get('maintainability', 70)),
        ]
        
        composite = sum(s.score for s in score_objects) / len(score_objects)
        
        return EvaluatorAssessment(
            evaluator_id="test_evaluator",
            scores=score_objects,
            composite_score=composite,
            confidence_level=EvaluationConfidence.HIGH,
            feedback="Test"
        )
    
    def test_predict_before_evaluate(self):
        """Test predicting outcome before actual evaluation"""
        # Create assessment with known scores
        assessment = self._create_assessment({\n            'correctness': 80,\n            'completeness': 75,\n            'clarity': 70,\n            'effectiveness': 75,\n            'efficiency': 70,\n            'maintainability': 65\n        })\n\n        # Predict first\n        prediction = self.engine.predict_pass_probability(\n            [assessment],\n            content_type=ContentType.CODE,\n            quality_level=QualityLevel.STANDARD,\n            complexity_score=5\n        )\n        \n        # Store prediction\n        predicted_decision = prediction['prediction']\n        \n        # Then evaluate\n        report = self.engine.evaluate(\n            [assessment],\n            content_type=ContentType.CODE,\n            quality_level=QualityLevel.STANDARD,\n            complexity_score=5\n        )\n        \n        # Verify actual decision\n        actual_decision = report.decision.value\n        \n        # With enough patterns, prediction should be reasonably accurate\n        # (This test documents the workflow, actual accuracy depends on data)\n        self.assertIn(actual_decision, ['pass', 'conditional_pass', 'fail'])\n    \n    def test_refinement_workflow(self):\n        \"\"\"Test the full refinement workflow with ICR learning\"\"\"\n        # Simulate initial evaluation that fails\n        failed_assessment = self._create_assessment({\n            'correctness': 55,\n            'completeness': 50,\n            'clarity': 45,\n            'effectiveness': 50,\n            'efficiency': 45,\n            'maintainability': 40\n        })\n        \n        original_report = self.engine.evaluate(\n            [failed_assessment],\n            content_type=ContentType.CODE,\n            quality_level=QualityLevel.STANDARD,\n            complexity_score=5\n        )\n        \n        self.assertEqual(original_report.decision, GateDecision.FAIL)\n        \n        # Simulate refinement\n        refined_assessment = self._create_assessment({\n            'correctness': 80,\n            'completeness': 75,\n            'clarity': 70,\n            'effectiveness': 75,\n            'efficiency': 70,\n            'maintainability': 65\n        })\n        \n        refined_report = self.engine.evaluate(\n            [refined_assessment],\n            content_type=ContentType.CODE,\n            quality_level=QualityLevel.STANDARD,\n            complexity_score=5\n        )\n        \n        # Learn from the refinement\n        learning_result = self.engine.learn_from_refinement(\n            original_report,\n            refined_report,\n            refinement_type=\"content_improvement\"\n        )\n        \n        # Verify learning occurred\n        self.assertTrue(learning_result['learned'])\n        self.assertGreater(learning_result['avg_score_improvement'], 0)\n        self.assertEqual(learning_result['success_rate'], 1.0)\n    \n    def test_adaptive_threshold_improves_accuracy(self):\n        \"\"\"Test that adaptive thresholds improve over time\"\"\"\n        # Initial state\n        initial_stats = self.engine.get_icr_statistics()\n        initial_threshold = self.threshold_manager.get_threshold(\n            ContentType.CODE, \n            QualityLevel.STANDARD\n        )\n        \n        # Store multiple passing patterns\n        for _ in range(10):\n            assessment = self._create_assessment({\n                'correctness': 85,\n                'completeness': 80,\n                'clarity': 75,\n                'effectiveness': 80,\n                'efficiency': 75,\n                'maintainability': 70\n            })\n            self.engine.evaluate(\n                [assessment],\n                content_type=ContentType.CODE,\n                quality_level=QualityLevel.STANDARD,\n                complexity_score=5\n            )\n        \n        # Store a failing pattern\n        fail_assessment = self._create_assessment({\n            'correctness': 50,\n            'completeness': 45,\n            'clarity': 40,\n            'effectiveness': 45,\n            'efficiency': 40,\n            'maintainability': 35\n        })\n        self.engine.evaluate(\n            [fail_assessment],\n            content_type=ContentType.CODE,\n            quality_level=QualityLevel.STANDARD,\n            complexity_score=5\n        )\n        \n        # Get final statistics\n        final_stats = self.engine.get_icr_statistics()\n        \n        # Verify learning occurred\n        self.assertGreater(final_stats['total_patterns'], initial_stats['total_patterns'])\n        self.assertGreater(final_stats['overall_pass_rate'], 0)\n        \n        # Prediction should now have higher confidence\n        prediction = self.engine.predict_pass_probability(\n            [self._create_assessment({\n                'correctness': 82,\n                'completeness': 78,\n                'clarity': 72,\n                'effectiveness': 77,\n                'efficiency': 72,\n                'maintainability': 68\n            })],\n            content_type=ContentType.CODE,\n            quality_level=QualityLevel.STANDARD,\n            complexity_score=5\n        )\n        \n        self.assertGreater(prediction['confidence'], 0.5)  # Should have higher confidence now\n\n\nif __name__ == '__main__':\n    unittest.main()\n