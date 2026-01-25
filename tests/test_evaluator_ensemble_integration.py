"""
Tests for Evaluator Team Ensemble Integration

Tests the integration of OpenEvolve's ensemble functionality with
the Evaluator Team coordination system.

Test Coverage:
- Ensemble initialization and configuration
- Ensemble-weighted evaluator assignment
- Consensus building with ensemble weights
- Ensemble status and metrics tracking
- Fallback mode when ensemble unavailable
- Performance tracking with ensemble
"""

import pytest
import unittest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluator_team_coordinator import (
    EvaluatorTeamCoordinator,
    EvaluationTask,
    EvaluationTaskStatus,
    EvaluationTaskPriority,
    ConsensusMethod,
    EvaluationSession,
    LoadBalancingStrategy
)

try:
    from evaluator_team import (
        EvaluatorTeam,
        EvaluatorMember,
        EvaluationMetric,
        EvaluationThreshold
    )
    EVALUATOR_TEAM_AVAILABLE = True
except ImportError:
    EVALUATOR_TEAM_AVAILABLE = False

try:
    from openevolve.llm.ensemble import LLMEnsemble
    from openevolve.config import LLMModelConfig
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False


class TestEnsembleIntegration(unittest.TestCase):
    """Test ensemble integration with evaluator team coordinator"""

    def setUp(self):
        """Set up test fixtures"""
        if not EVALUATOR_TEAM_AVAILABLE:
            self.skipTest("Evaluator Team not available")

        # Create mock evaluator team
        self.evaluator_team = Mock(spec=EvaluatorTeam)
        self.evaluator_team.team_members = []

        # Create mock evaluators with varying expertise
        for i in range(3):
            evaluator = Mock(spec=EvaluatorMember)
            evaluator.evaluator_id = f"evaluator_{i}"
            evaluator.expertise_level = 7 + i  # 7, 8, 9
            evaluator.evaluation_philosophy = "balanced"
            evaluator.specializations = [EvaluationMetric.OVERALL_QUALITY]
            evaluator.evaluate_content = MagicMock(return_value=self._mock_assessment(i))
            self.evaluator_team.team_members.append(evaluator)

    def _mock_assessment(self, index):
        """Create mock assessment"""
        assessment = MagicMock()
        assessment.evaluator_id = f"evaluator_{index}"
        assessment.composite_score = 75.0 + index * 5
        assessment.scores = []
        assessment.detailed_feedback = {"improvement_suggestions": []}
        assessment.confidence_level = "HIGH"
        assessment.time_taken = 60.0
        return assessment

    @pytest.mark.skipif(not ENSEMBLE_AVAILABLE, reason="Ensemble not available")
    def test_ensemble_initialization(self):
        """Test ensemble is initialized correctly"""
        coordinator = EvaluatorTeamCoordinator(
            evaluator_team=self.evaluator_team,
            use_ensemble=True
        )

        self.assertTrue(coordinator.use_ensemble)
        self.assertTrue(hasattr(coordinator, 'ensemble'))
        self.assertTrue(hasattr(coordinator, 'ensemble_weights'))

    def test_fallback_initialization(self):
        """Test fallback when ensemble not available"""
        coordinator = EvaluatorTeamCoordinator(
            evaluator_team=self.evaluator_team,
            use_ensemble=False
        )

        self.assertFalse(coordinator.use_ensemble)
        self.assertTrue(hasattr(coordinator, 'executor'))

    @pytest.mark.skipif(not ENSEMBLE_AVAILABLE, reason="Ensemble not available")
    def test_custom_ensemble_config(self):
        """Test custom ensemble configuration"""
        ensemble_config = [
            LLMModelConfig(
                name="model1",
                weight=0.5,
                model_id="model1",
                temperature=0.7,
                max_tokens=4096
            ),
            LLMModelConfig(
                name="model2",
                weight=0.5,
                model_id="model2",
                temperature=0.7,
                max_tokens=4096
            )
        ]

        coordinator = EvaluatorTeamCoordinator(
            evaluator_team=self.evaluator_team,
            use_ensemble=True,
            ensemble_config=ensemble_config
        )

        self.assertTrue(coordinator.use_ensemble)
        self.assertIsNotNone(coordinator.ensemble)
        self.assertEqual(len(coordinator.ensemble_weights), 2)

    @pytest.mark.skipif(not ENSEMBLE_AVAILABLE, reason="Ensemble not available")
    def test_ensemble_execution(self):
        """Test task execution with ensemble"""
        coordinator = EvaluatorTeamCoordinator(
            evaluator_team=self.evaluator_team,
            use_ensemble=True,
            max_concurrent_evaluations=2
        )

        # Create test tasks
        sub_problems = [
            {
                "id": "sp_001",
                "description": "Test sub-problem 1",
                "priority": 7
            },
            {
                "id": "sp_002",
                "description": "Test sub-problem 2",
                "priority": 7
            }
        ]

        solutions = {
            "sp_001": "Test solution 1",
            "sp_002": "Test solution 2"
        }

        # Execute evaluations
        session = coordinator.coordinate_solution_evaluations(
            problem_statement="Test problem",
            sub_problems=sub_problems,
            solutions=solutions
        )

        # Verify results
        self.assertEqual(len(session.tasks), 2)
        self.assertGreater(session.completed_tasks, 0)

    def test_fallback_execution(self):
        """Test task execution with fallback executor"""
        coordinator = EvaluatorTeamCoordinator(
            evaluator_team=self.evaluator_team,
            use_ensemble=False,
            max_concurrent_evaluations=2
        )

        # Create test tasks
        sub_problems = [
            {
                "id": "sp_001",
                "description": "Test sub-problem 1",
                "priority": 7
            }
        ]

        solutions = {
            "sp_001": "Test solution 1"
        }

        # Execute evaluations
        session = coordinator.coordinate_solution_evaluations(
            problem_statement="Test problem",
            sub_problems=sub_problems,
            solutions=solutions
        )

        # Verify results
        self.assertEqual(len(session.tasks), 1)
        self.assertGreater(session.completed_tasks, 0)

    @pytest.mark.skipif(not ENSEMBLE_AVAILABLE, reason="Ensemble not available")
    def test_ensemble_weighted_assignment(self):
        """Test ensemble-weighted evaluator assignment"""
        coordinator = EvaluatorTeamCoordinator(
            evaluator_team=self.evaluator_team,
            use_ensemble=True,
            load_balancing_strategy=LoadBalancingStrategy.SPECIALIZATION_BASED
        )

        # Create a test task
        task = EvaluationTask(
            task_id="test_task",
            sub_problem_id="sp_001",
            sub_problem_description="Test",
            solution_content="Test solution"
        )

        # Assign evaluators
        assigned = coordinator._assign_evaluators_with_ensemble_weights(task)

        # Verify assignment
        self.assertIsNotNone(assigned)
        self.assertGreater(len(assigned), 0)
        self.assertLessEqual(len(assigned), coordinator.max_evaluators_per_task)

    @pytest.mark.skipif(not ENSEMBLE_AVAILABLE, reason="Ensemble not available")
    def test_ensemble_status(self):
        """Test ensemble status reporting"""
        coordinator = EvaluatorTeamCoordinator(
            evaluator_team=self.evaluator_team,
            use_ensemble=True
        )

        status = coordinator.get_ensemble_status()

        # Verify status structure
        self.assertIn("use_ensemble", status)
        self.assertIn("ensemble_available", status)
        self.assertIn("coordination_mode", status)
        self.assertTrue(status["use_ensemble"])
        self.assertEqual(status["coordination_mode"], "ensemble")

    @pytest.mark.skipif(not ENSEMBLE_AVAILABLE, reason="Ensemble not available")
    def test_ensemble_weight_update(self):
        """Test dynamic ensemble weight updates"""
        coordinator = EvaluatorTeamCoordinator(
            evaluator_team=self.evaluator_team,
            use_ensemble=True
        )

        # Get initial weights
        initial_weights = coordinator.ensemble_weights.copy()

        # Update weights
        new_weights = {eval_id: 1.0 for eval_id in initial_weights.keys()}
        coordinator.update_ensemble_weights(new_weights)

        # Verify weights were updated
        self.assertNotEqual(coordinator.ensemble_weights, initial_weights)

    @pytest.mark.skipif(not ENSEMBLE_AVAILABLE, reason="Ensemble not available")
    def test_consensus_with_ensemble_weights(self):
        """Test consensus building using ensemble weights"""
        coordinator = EvaluatorTeamCoordinator(
            evaluator_team=self.evaluator_team,
            use_ensemble=True
        )

        # Create mock assessments
        assessments = [self._mock_assessment(i) for i in range(3)]

        # Build consensus
        consensus = coordinator._consensus_weighted_average(
            assessments=assessments,
            content="Test content",
            content_type="code",
            threshold=EvaluationThreshold.STANDARD_APPROVAL
        )

        # Verify consensus
        self.assertIsNotNone(consensus)
        self.assertIn("uses_ensemble_weights", consensus.variance_analysis)
        self.assertTrue(consensus.variance_analysis["uses_ensemble_weights"])
        self.assertIn("ensemble_integration", consensus.evaluation_metadata)

    def test_backward_compatibility(self):
        """Test backward compatibility with non-ensemble mode"""
        # Should work with use_ensemble=False
        coordinator = EvaluatorTeamCoordinator(
            evaluator_team=self.evaluator_team,
            use_ensemble=False
        )

        # Create and execute task
        sub_problems = [{"id": "sp_001", "description": "Test"}]
        solutions = {"sp_001": "Test solution"}

        session = coordinator.coordinate_solution_evaluations(
            problem_statement="Test",
            sub_problems=sub_problems,
            solutions=solutions
        )

        # Should complete successfully
        self.assertEqual(len(session.tasks), 1)

    @pytest.mark.skipif(not ENSEMBLE_AVAILABLE, reason="Ensemble not available")
    def test_dual_mode_operation(self):
        """Test that both ensemble and fallback modes can coexist"""
        # Create ensemble coordinator
        ensemble_coordinator = EvaluatorTeamCoordinator(
            evaluator_team=self.evaluator_team,
            use_ensemble=True
        )

        # Create fallback coordinator
        fallback_coordinator = EvaluatorTeamCoordinator(
            evaluator_team=self.evaluator_team,
            use_ensemble=False
        )

        # Both should be functional
        self.assertTrue(ensemble_coordinator.use_ensemble)
        self.assertFalse(fallback_coordinator.use_ensemble)

        # Both should have the same interface
        self.assertTrue(hasattr(ensemble_coordinator, 'coordinate_solution_evaluations'))
        self.assertTrue(hasattr(fallback_coordinator, 'coordinate_solution_evaluations'))


if __name__ == '__main__':
    unittest.main()

    @pytest.mark.skipif(not ENSEMBLE_AVAILABLE, reason="Ensemble not available")
    def test_consensus_algorithms_preserved(self):
        """Test that consensus algorithms still work with ensemble"""
        for consensus_method in [
            ConsensusMethod.MAJORITY_VOTE,
            ConsensusMethod.WEIGHTED_AVERAGE,
            ConsensusMethod.MEDIAN,
            ConsensusMethod.BATESIAN,
            ConsensusMethod.DEMPSTER_SHAFER,
            ConsensusMethod.DELPHI
        ]:
            coordinator = EvaluatorTeamCoordinator(
                evaluator_team=self.evaluator_team,
                use_ensemble=True,
                consensus_method=consensus_method
            )

            # Create test task
            sub_problems = [
                {
                    "id": f"sp_{consensus_method.value}",
                    "description": "Test sub-problem",
                    "priority": 7
                }
            ]

            solutions = {
                f"sp_{consensus_method.value}": "Test solution"
            }

            # Execute evaluation
            session = coordinator.coordinate_solution_evaluations(
                problem_statement="Test problem",
                sub_problems=sub_problems,
                solutions=solutions
            )

            # Verify consensus was built
            self.assertTrue(len(session.tasks) > 0)
            if session.tasks[0].integrated_evaluation:
                self.assertIsNotNone(session.tasks[0].integrated_evaluation.consensus_score)

    def test_shutdown_with_ensemble(self):
        """Test clean shutdown with ensemble"""
        coordinator = EvaluatorTeamCoordinator(
            evaluator_team=self.evaluator_team,
            use_ensemble=ENSEMBLE_AVAILABLE  # Test with whatever is available
        )

        # Should not raise exception
        coordinator.shutdown()

    def test_evaluator_metrics_with_ensemble(self):
        """Test that evaluator metrics track ensemble usage"""
        from evaluator_analytics import EvaluatorAnalytics, EvaluationRecord, EvaluationStage

        analytics = EvaluatorAnalytics()

        # Create test record
        record = EvaluationRecord(
            evaluator_id="evaluator_0",
            evaluation_id="eval_001",
            stage=EvaluationStage.SOLUTION_GENERATION,
            timestamp=datetime.now(),
            score=85.0,
            confidence=0.9,
            time_taken=60.0,
            criteria_scores={}
        )

        # Add record
        analytics.add_evaluation_record(record)

        # Get metrics
        metrics = analytics.get_evaluator_metrics("evaluator_0")

        # Verify ensemble metrics exist
        self.assertIsNotNone(metrics)
        self.assertTrue(hasattr(metrics, 'ensemble_selection_count'))
        self.assertTrue(hasattr(metrics, 'ensemble_weight'))
        self.assertTrue(hasattr(metrics, 'ensemble_utilization'))


class TestConsensusAlgorithms(unittest.TestCase):
    """Test that all consensus algorithms work correctly"""

    def setUp(self):
        """Set up test fixtures"""
        if not EVALUATOR_TEAM_AVAILABLE:
            self.skipTest("Evaluator Team not available")

        # Create mock assessments
        self.assessments = []
        for i in range(3):
            assessment = MagicMock()
            assessment.evaluator_id = f"evaluator_{i}"
            assessment.composite_score = 70.0 + i * 10  # 70, 80, 90
            assessment.scores = []
            assessment.detailed_feedback = {"improvement_suggestions": []}
            assessment.confidence_level = "HIGH"
            self.assessments.append(assessment)

    def test_majority_vote_consensus(self):
        """Test majority vote consensus algorithm"""
        if not EVALUATOR_TEAM_AVAILABLE:
            self.skipTest("Evaluator Team not available")

        coordinator = EvaluatorTeamCoordinator(use_ensemble=False)
        result = coordinator._consensus_majority_vote(
            self.assessments,
            "test content",
            "general",
            EvaluationThreshold.STANDARD_APPROVAL
        )

        self.assertIsNotNone(result)
        self.assertGreater(result.consensus_score, 0)

    def test_weighted_average_consensus(self):
        """Test weighted average consensus algorithm"""
        if not EVALUATOR_TEAM_AVAILABLE:
            self.skipTest("Evaluator Team not available")

        coordinator = EvaluatorTeamCoordinator(use_ensemble=False)
        result = coordinator._consensus_weighted_average(
            self.assessments,
            "test content",
            "general",
            EvaluationThreshold.STANDARD_APPROVAL
        )

        self.assertIsNotNone(result)
        self.assertGreater(result.consensus_score, 0)

    def test_median_consensus(self):
        """Test median consensus algorithm"""
        if not EVALUATOR_TEAM_AVAILABLE:
            self.skipTest("Evaluator Team not available")

        coordinator = EvaluatorTeamCoordinator(use_ensemble=False)
        result = coordinator._consensus_median(
            self.assessments,
            "test content",
            "general",
            EvaluationThreshold.STANDARD_APPROVAL
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.consensus_score, 80.0)  # Median of 70, 80, 90


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add tests
    suite.addTests(loader.loadTestsFromTestCase(TestEnsembleIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestConsensusAlgorithms))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("="*70)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
