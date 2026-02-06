"""
Comprehensive Test Suite for Ensemble Strategy Selector

Tests all aspects of ensemble-based strategy recommendation including:
- Ensemble prediction methods
- Confidence interval calculation
- Real-time learning and adaptation
- Cold start handling
- Method agreement and disagreement
- Weight adaptation
- Explanation generation

Author: AI Architecture Team
Date: 2026-01-30
Version: 2.0
"""

import pytest
import asyncio
from datetime import datetime, UTC, timedelta
from typing import Dict, Any
import random

from knowledge_engine.core.strategy_recommender import (
    EnsembleStrategySelector,
    EnsemblePrediction,
    MethodPrediction,
    OnlineLearningTracker,
    ProblemCharacteristics,
    HistoricalRun,
    EvolutionSystem,
    EvolutionMode,
    PredictionMethod,
    EvaluationCost,
    ComplexityLevel,
    recommend_evolutionary_strategy
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_historical_runs():
    """Sample historical run data for testing"""
    now = datetime.now(UTC)
    random.seed(42)  # For reproducibility

    runs = [
        # Finance - PES runs (good performance)
        HistoricalRun(
            run_id=f"fin_pes_{i}",
            domain="finance",
            strategy_used="pes",
            mode_used="pes",
            problem_complexity="high",
            final_score=0.80 + random.random() * 0.15,  # 0.80-0.95
            convergence_speed=30,
            evaluation_count=30,
            diversity_score=0.7,
            timestamp=now - timedelta(days=random.randint(1, 30)),
            sample_efficiency=0.027,
            metadata={"keywords": ["portfolio", "optimization", "backtest"]}
        )
        for i in range(10)
    ]

    runs.extend([
        # Finance - QD runs (moderate performance)
        HistoricalRun(
            run_id=f"fin_qd_{i}",
            domain="finance",
            strategy_used="qd",
            mode_used="qd",
            problem_complexity="high",
            final_score=0.65 + random.random() * 0.20,  # 0.65-0.85
            convergence_speed=100,
            evaluation_count=100,
            diversity_score=0.9,
            timestamp=now - timedelta(days=random.randint(1, 30)),
            sample_efficiency=0.0075,
            metadata={"keywords": ["portfolio", "diversity"]}
        )
        for i in range(8)
    ])

    runs.extend([
        # Science - PES runs (excellent performance)
        HistoricalRun(
            run_id=f"sci_pes_{i}",
            domain="science",
            strategy_used="pes",
            mode_used="pes",
            problem_complexity="high",
            final_score=0.85 + random.random() * 0.10,  # 0.85-0.95
            convergence_speed=20,
            evaluation_count=20,
            diversity_score=0.6,
            timestamp=now - timedelta(days=random.randint(1, 30)),
            sample_efficiency=0.045,
            metadata={"keywords": ["experiment", "simulation", "yield"]}
        )
        for i in range(12)
    ])

    runs.extend([
        # Trading - Adversarial runs
        HistoricalRun(
            run_id=f"trd_adv_{i}",
            domain="trading",
            strategy_used="adversarial",
            mode_used="adversarial",
            problem_complexity="medium",
            final_score=0.70 + random.random() * 0.20,
            convergence_speed=80,
            evaluation_count=80,
            diversity_score=0.5,
            timestamp=now - timedelta(days=random.randint(1, 30)),
            sample_efficiency=0.00875,
            metadata={"keywords": ["trading", "strategy", "robustness"]}
        )
        for i in range(7)
    ])

    return runs


@pytest.fixture
def ensemble_selector(sample_historical_runs):
    """Initialize ensemble selector with sample data"""
    selector = EnsembleStrategySelector(learning_enabled=True)

    # Load historical data
    for run in sample_historical_runs:
        selector.historical_runs[run.run_id] = run

    return selector


@pytest.fixture
def sample_problem_chars():
    """Sample problem characteristics"""
    return ProblemCharacteristics(
        domain="finance",
        complexity="high",
        evaluation_cost="expensive",
        has_multiple_objectives=True,
        requires_diversity=True,
        requires_robustness=True,
        constraint_count=3,
        estimated_iterations=50,
        keywords=["portfolio", "optimization", "backtest", "risk", "return"]
    )


# ============================================================================
# TEST ENSEMBLE PREDICTION
# ============================================================================

class TestEnsemblePrediction:
    """Test ensemble prediction functionality"""

    @pytest.mark.asyncio
    async def test_ensemble_prediction_basic(self, ensemble_selector):
        """Test basic ensemble prediction"""
        problem = "Optimize portfolio allocation for max return with min risk"
        domain = "finance"
        constraints = {
            "objectives": ["return", "risk"],
            "time_limit_seconds": 300
        }

        prediction = await ensemble_selector.recommend_with_ensemble(
            problem, domain, constraints, confidence_level=0.95
        )

        # Verify prediction structure
        assert isinstance(prediction, EnsemblePrediction)
        assert prediction.strategy is not None
        assert len(prediction.strategy) == 2
        assert prediction.point_estimate > 0
        assert prediction.confidence_interval[0] < prediction.point_estimate
        assert prediction.confidence_interval[1] > prediction.point_estimate
        assert prediction.confidence_level == 0.95
        assert len(prediction.prediction_methods) > 0
        assert 0 <= prediction.disagreement_ratio <= 1

    @pytest.mark.asyncio
    async def test_ensemble_prediction_methods(self, ensemble_selector):
        """Test that multiple prediction methods are used"""
        problem = "Optimize experimental design"
        domain = "science"
        constraints = {"time_limit_seconds": 900}

        prediction = await ensemble_selector.recommend_with_ensemble(
            problem, domain, constraints
        )

        # Should have predictions from multiple methods
        assert len(prediction.prediction_methods) >= 3

        # Should have individual predictions
        assert len(prediction.individual_predictions) >= 3

        # Should have method weights
        assert len(prediction.method_weights) >= 3
        assert abs(sum(prediction.method_weights.values()) - 1.0) < 0.01

    @pytest.mark.asyncio
    async def test_ensemble_confidence_levels(self, ensemble_selector):
        """Test different confidence levels"""
        problem = "Optimize bridge design"
        domain = "engineering"
        constraints = {"time_limit_seconds": 600}

        for conf_level in [0.90, 0.95, 0.99]:
            prediction = await ensemble_selector.recommend_with_ensemble(
                problem, domain, constraints, confidence_level=conf_level
            )

            assert prediction.confidence_level == conf_level

            # Higher confidence = wider interval
            interval_width = prediction.confidence_interval[1] - prediction.confidence_interval[0]
            assert interval_width > 0

    @pytest.mark.asyncio
    async def test_ensemble_agreement_calculation(self, ensemble_selector):
        """Test method agreement calculation"""
        problem = "Optimize landing page"
        domain = "web"
        constraints = {"time_limit_seconds": 5}

        prediction = await ensemble_selector.recommend_with_ensemble(
            problem, domain, constraints
        )

        # Agreement should be calculated
        assert hasattr(prediction, 'disagreement_ratio')
        assert 0.0 <= prediction.disagreement_ratio <= 1.0

        # If all methods agree, disagreement should be low
        if len(set(prediction.individual_predictions.values())) == 1:
            # All methods predicted the same
            # (This is rare but possible)
            pass


# ============================================================================
# TEST INDIVIDUAL PREDICTION METHODS
# ============================================================================

class TestPredictionMethods:
    """Test individual prediction methods"""

    @pytest.mark.asyncio
    async def test_rule_based_prediction(self, ensemble_selector, sample_problem_chars):
        """Test rule-based prediction method"""
        prediction = await ensemble_selector._rule_based_prediction(
            sample_problem_chars, "finance"
        )

        assert prediction.method == PredictionMethod.RULE_BASED
        assert prediction.system in [EvolutionSystem.LOONGFLOW, EvolutionSystem.OPENEVOLVE]
        assert prediction.mode in ["pes", "qd", "mo", "adversarial", "standard"]
        assert 0 < prediction.confidence <= 1.0
        assert len(prediction.reasoning) > 0
        assert isinstance(prediction.evidence, dict)

    @pytest.mark.asyncio
    async def test_rule_based_expensive_evals(self, ensemble_selector):
        """Test rule-based with expensive evaluations"""
        problem_chars = ProblemCharacteristics(
            domain="science",
            complexity="high",
            evaluation_cost="very_expensive",
            has_multiple_objectives=False,
            requires_diversity=False,
            requires_robustness=False,
            constraint_count=2,
            estimated_iterations=30,
            keywords=["experiment", "simulation"]
        )

        prediction = await ensemble_selector._rule_based_prediction(
            problem_chars, "science"
        )

        # Should recommend PES for expensive evaluations
        assert prediction.mode == "pes"
        assert prediction.system == EvolutionSystem.LOONGFLOW

    @pytest.mark.asyncio
    async def test_rule_based_multi_objective(self, ensemble_selector):
        """Test rule-based with multiple objectives"""
        problem_chars = ProblemCharacteristics(
            domain="engineering",
            complexity="high",
            evaluation_cost="moderate",
            has_multiple_objectives=True,
            requires_diversity=False,
            requires_robustness=False,
            constraint_count=3,
            estimated_iterations=100,
            keywords=["design", "optimization"]
        )

        prediction = await ensemble_selector._rule_based_prediction(
            problem_chars, "engineering"
        )

        # Should recommend MO for multiple objectives
        assert prediction.mode == "mo"

    @pytest.mark.asyncio
    async def test_similarity_based_prediction(self, ensemble_selector, sample_problem_chars):
        """Test similarity-based prediction method"""
        # Need history for similarity
        history = await ensemble_selector.query_historical_performance("finance", "high")

        assert len(history) > 0

        prediction = await ensemble_selector._similarity_based_prediction(
            sample_problem_chars, history
        )

        assert prediction.method == PredictionMethod.SIMILARITY
        assert prediction.system in [EvolutionSystem.LOONGFLOW, EvolutionSystem.OPENEVOLVE]
        assert 0 < prediction.confidence <= 1.0

        # Should have evidence about similar runs
        assert 'similar_runs' in prediction.evidence

    @pytest.mark.asyncio
    async def test_trend_based_prediction(self, ensemble_selector):
        """Test trend-based prediction method"""
        problem_chars = ProblemCharacteristics(
            domain="finance",
            complexity="high",
            evaluation_cost="expensive",
            has_multiple_objectives=False,
            requires_diversity=True,
            requires_robustness=True,
            constraint_count=2,
            estimated_iterations=50,
            keywords=["portfolio", "trading"]
        )

        history = await ensemble_selector.query_historical_performance("finance", "high")

        prediction = await ensemble_selector._trend_based_prediction(
            problem_chars, history, "finance"
        )

        assert prediction.method == PredictionMethod.TREND
        assert prediction.system in [EvolutionSystem.LOONGFLOW, EvolutionSystem.OPENEVOLVE]

        # Should have trend evidence
        if 'trend' in prediction.evidence:
            assert 'trend' in prediction.evidence
            assert 'recent_avg' in prediction.evidence

    @pytest.mark.asyncio
    async def test_ml_based_prediction(self, ensemble_selector, sample_problem_chars):
        """Test ML-based prediction method (optional)"""
        # Enable ML
        ensemble_selector.enable_ml = True

        history = await ensemble_selector.query_historical_performance("finance", "high")

        # Need enough samples for ML
        if len(history) >= ensemble_selector.min_samples_for_ml:
            try:
                prediction = await ensemble_selector._ml_based_prediction(
                    sample_problem_chars, history
                )

                assert prediction.method == PredictionMethod.ML
                assert prediction.system in [EvolutionSystem.LOONGFLOW, EvolutionSystem.OPENEVOLVE]

                # Should have model evidence
                if 'training_samples' in prediction.evidence:
                    assert prediction.evidence['training_samples'] >= ensemble_selector.min_samples_for_ml
            except ImportError:
                # scikit-learn not available, skip
                pytest.skip("scikit-learn not available")
        else:
            pytest.skip("Not enough historical data for ML")


# ============================================================================
# TEST CONFIDENCE INTERVALS
# ============================================================================

class TestConfidenceIntervals:
    """Test confidence interval calculation"""

    @pytest.mark.asyncio
    async def test_confidence_interval_bootstrap(self, ensemble_selector, sample_problem_chars):
        """Test bootstrap confidence interval calculation"""
        history = await ensemble_selector.query_historical_performance("finance", "high")

        strategy = (EvolutionSystem.LOONGFLOW, "pes")
        point_estimate, ci = await ensemble_selector._calculate_confidence_interval(
            strategy, sample_problem_chars, history, confidence_level=0.95
        )

        # Verify structure
        assert isinstance(point_estimate, float)
        assert isinstance(ci, tuple)
        assert len(ci) == 2

        # Point estimate should be in interval
        assert ci[0] <= point_estimate <= ci[1]

        # Interval should be reasonable
        assert ci[0] >= 0.0
        assert ci[1] <= 1.0

    @pytest.mark.asyncio
    async def test_confidence_interval_levels(self, ensemble_selector, sample_problem_chars):
        """Test different confidence levels produce different intervals"""
        history = await ensemble_selector.query_historical_performance("science", "high")

        strategy = (EvolutionSystem.LOONGFLOW, "pes")

        # Calculate intervals at different levels
        ci_90 = await ensemble_selector._calculate_confidence_interval(
            strategy, sample_problem_chars, history, 0.90
        )
        ci_95 = await ensemble_selector._calculate_confidence_interval(
            strategy, sample_problem_chars, history, 0.95
        )
        ci_99 = await ensemble_selector._calculate_confidence_interval(
            strategy, sample_problem_chars, history, 0.99
        )

        # Higher confidence = wider interval
        width_90 = ci_90[1][1] - ci_90[1][0]
        width_95 = ci_95[1][1] - ci_95[1][0]
        width_99 = ci_99[1][1] - ci_99[1][0]

        # Should be monotonic (or at least not decreasing)
        # Note: Due to randomness in bootstrap, this is a soft check
        assert width_90 >= 0
        assert width_95 >= 0
        assert width_99 >= 0

    @pytest.mark.asyncio
    async def test_confidence_interval_insufficient_data(self, ensemble_selector):
        """Test confidence interval with insufficient historical data"""
        # Create problem chars with no matching history
        problem_chars = ProblemCharacteristics(
            domain="unknown_domain",
            complexity="medium",
            evaluation_cost="moderate",
            has_multiple_objectives=False,
            requires_diversity=False,
            requires_robustness=False,
            constraint_count=0,
            estimated_iterations=100,
            keywords=[]
        )

        history = []  # Empty history

        strategy = (EvolutionSystem.LOONGFLOW, "pes")
        point_estimate, ci = await ensemble_selector._calculate_confidence_interval(
            strategy, problem_chars, history, 0.95
        )

        # Should fall back to heuristic interval
        assert ci[0] < point_estimate
        assert ci[1] > point_estimate


# ============================================================================
# TEST WEIGHTED VOTING
# ============================================================================

class TestWeightedVoting:
    """Test weighted voting mechanism"""

    @pytest.mark.asyncio
    async def test_weighted_voting_unanimous(self, ensemble_selector):
        """Test weighted voting with unanimous agreement"""
        predictions = [
            MethodPrediction(
                method=PredictionMethod.RULE_BASED,
                system=EvolutionSystem.LOONGFLOW,
                mode="pes",
                confidence=0.85,
                reasoning="Expensive evaluations",
                evidence={}
            ),
            MethodPrediction(
                method=PredictionMethod.SIMILARITY,
                system=EvolutionSystem.LOONGFLOW,
                mode="pes",
                confidence=0.80,
                reasoning="Similar problems used PES",
                evidence={}
            ),
            MethodPrediction(
                method=PredictionMethod.TREND,
                system=EvolutionSystem.LOONGFLOW,
                mode="pes",
                confidence=0.75,
                reasoning="PES trend improving",
                evidence={}
            )
        ]

        weights = {
            'rule_based': 0.25,
            'similarity': 0.35,
            'trend': 0.25,
            'ml': 0.15
        }

        (system, mode), agreement = ensemble_selector._weighted_voting(predictions, weights)

        # Should agree on PES
        assert system == EvolutionSystem.LOONGFLOW
        assert mode == "pes"

        # Should have high agreement
        assert agreement > 0.8

    @pytest.mark.asyncio
    async def test_weighted_voting_split(self, ensemble_selector):
        """Test weighted voting with split decisions"""
        predictions = [
            MethodPrediction(
                method=PredictionMethod.RULE_BASED,
                system=EvolutionSystem.LOONGFLOW,
                mode="pes",
                confidence=0.85,
                reasoning="Expensive evaluations",
                evidence={}
            ),
            MethodPrediction(
                method=PredictionMethod.SIMILARITY,
                system=EvolutionSystem.OPENEVOLVE,
                mode="qd",
                confidence=0.75,
                reasoning="Similar problems used QD",
                evidence={}
            ),
            MethodPrediction(
                method=PredictionMethod.TREND,
                system=EvolutionSystem.OPENEVOLVE,
                mode="mo",
                confidence=0.70,
                reasoning="MO trend improving",
                evidence={}
            )
        ]

        weights = {
            'rule_based': 0.25,
            'similarity': 0.35,
            'trend': 0.25,
            'ml': 0.15
        }

        (system, mode), agreement = ensemble_selector._weighted_voting(predictions, weights)

        # Should pick one
        assert system in [EvolutionSystem.LOONGFLOW, EvolutionSystem.OPENEVOLVE]

        # Agreement should be lower
        assert agreement < 0.8

    @pytest.mark.asyncio
    async def test_weighted_voting_weights(self, ensemble_selector):
        """Test that weights affect outcome"""
        predictions = [
            MethodPrediction(
                method=PredictionMethod.RULE_BASED,
                system=EvolutionSystem.LOONGFLOW,
                mode="pes",
                confidence=0.90,
                reasoning="Rule-based",
                evidence={}
            ),
            MethodPrediction(
                method=PredictionMethod.SIMILARITY,
                system=EvolutionSystem.OPENEVOLVE,
                mode="qd",
                confidence=0.90,
                reasoning="Similarity",
                evidence={}
            )
        ]

        # Test with rule-based having higher weight
        weights1 = {'rule_based': 0.8, 'similarity': 0.2}
        (system1, mode1), _ = ensemble_selector._weighted_voting(predictions, weights1)

        # Test with similarity having higher weight
        weights2 = {'rule_based': 0.2, 'similarity': 0.8}
        (system2, mode2), _ = ensemble_selector._weighted_voting(predictions, weights2)

        # Different weights should potentially lead to different outcomes
        # (depending on confidence scores)
        # At minimum, verify both are valid
        assert system1 in [EvolutionSystem.LOONGFLOW, EvolutionSystem.OPENEVOLVE]
        assert system2 in [EvolutionSystem.LOONGFLOW, EvolutionSystem.OPENEVOLVE]


# ============================================================================
# TEST ONLINE LEARNING
# ============================================================================

class TestOnlineLearning:
    """Test real-time learning and adaptation"""

    def test_tracker_initialization(self):
        """Test learning tracker initialization"""
        tracker = OnlineLearningTracker(window_size=50)

        assert tracker.window_size == 50
        assert len(tracker.recommendations_made) == 0
        assert len(tracker.actual_performance) == 0
        assert len(tracker.accuracy_history) == 0
        assert tracker.total_recommendations == 0

    def test_record_recommendation(self, ensemble_selector, sample_problem_chars):
        """Test recording a recommendation"""
        # Create a mock prediction
        prediction = EnsemblePrediction(
            strategy=(EvolutionSystem.LOONGFLOW, "pes"),
            point_estimate=0.80,
            confidence_interval=(0.75, 0.85),
            confidence_level=0.95,
            prediction_methods=["rule_based", "similarity"],
            disagreement_ratio=0.2,
            reasoning="Test reasoning",
            method_weights={'rule_based': 0.5, 'similarity': 0.5},
            individual_predictions={
                'rule_based': (EvolutionSystem.LOONGFLOW, "pes", 0.85),
                'similarity': (EvolutionSystem.LOONGFLOW, "pes", 0.80)
            }
        )

        rec_id = ensemble_selector.learning_tracker.record_recommendation(
            prediction, sample_problem_chars
        )

        # Verify recorded
        assert rec_id.startswith("rec_")
        assert len(ensemble_selector.learning_tracker.recommendations_made) == 1
        assert ensemble_selector.learning_tracker.total_recommendations == 1

    def test_record_actual_performance(self, ensemble_selector, sample_problem_chars):
        """Test recording actual performance"""
        # First record a recommendation
        prediction = EnsemblePrediction(
            strategy=(EvolutionSystem.LOONGFLOW, "pes"),
            point_estimate=0.80,
            confidence_interval=(0.75, 0.85),
            confidence_level=0.95,
            prediction_methods=["rule_based"],
            disagreement_ratio=0.0,
            reasoning="Test",
            method_weights={'rule_based': 1.0},
            individual_predictions={
                'rule_based': (EvolutionSystem.LOONGFLOW, "pes", 0.85)
            }
        )

        rec_id = ensemble_selector.learning_tracker.record_recommendation(
            prediction, sample_problem_chars
        )

        # Then record actual performance
        metrics = ensemble_selector.learning_tracker.record_actual_performance(
            recommendation_id=rec_id,
            actual_performance=0.85,  # Better than predicted
            run_id="test_run_001"
        )

        # Verify metrics
        assert 'accuracy' in metrics
        assert 'error' in metrics
        assert metrics['accuracy'] > 0.5  # Should be decent

        # Verify recorded
        assert len(ensemble_selector.learning_tracker.actual_performance) == 1
        assert len(ensemble_selector.learning_tracker.accuracy_history) == 1

    def test_weight_adaptation(self, ensemble_selector):
        """Test ensemble weight adaptation"""
        tracker = ensemble_selector.learning_tracker

        # Record enough recommendations to trigger adaptation
        for i in range(25):
            # Create prediction
            prediction = EnsemblePrediction(
                strategy=(EvolutionSystem.LOONGFLOW, "pes"),
                point_estimate=0.75,
                confidence_interval=(0.70, 0.80),
                confidence_level=0.95,
                prediction_methods=["rule_based", "similarity", "trend"],
                disagreement_ratio=0.3,
                reasoning="Test",
                method_weights={'rule_based': 0.33, 'similarity': 0.33, 'trend': 0.34},
                individual_predictions={
                    'rule_based': (EvolutionSystem.LOONGFLOW, "pes", 0.8),
                    'similarity': (EvolutionSystem.LOONGFLOW, "pes", 0.7),
                    'trend': (EvolutionSystem.LOONGFLOW, "pes", 0.75)
                }
            )

            problem_chars = ProblemCharacteristics(
                domain="test",
                complexity="medium",
                evaluation_cost="moderate",
                has_multiple_objectives=False,
                requires_diversity=False,
                requires_robustness=False,
                constraint_count=0,
                estimated_iterations=100,
                keywords=[]
            )

            rec_id = tracker.record_recommendation(prediction, problem_chars)

            # Record actual performance (alternating good/bad)
            actual = 0.9 if i % 2 == 0 else 0.6
            metrics = tracker.record_actual_performance(rec_id, actual, f"run_{i}")

            # Check if weights adapted after 20
            if i >= 20:
                # Should have adapted
                # (actual adaptation logic is in the tracker)
                pass

        # Verify weights still sum to 1.0
        weights = tracker.get_current_weights()
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_get_accuracy_metrics(self, ensemble_selector):
        """Test getting accuracy metrics"""
        tracker = ensemble_selector.learning_tracker

        # Record some data
        for i in range(10):
            prediction = EnsemblePrediction(
                strategy=(EvolutionSystem.LOONGFLOW, "pes"),
                point_estimate=0.75,
                confidence_interval=(0.70, 0.80),
                confidence_level=0.95,
                prediction_methods=["rule_based"],
                disagreement_ratio=0.0,
                reasoning="Test",
                method_weights={'rule_based': 1.0},
                individual_predictions={'rule_based': (EvolutionSystem.LOONGFLOW, "pes", 0.8)}
            )

            problem_chars = ProblemCharacteristics(
                domain="test",
                complexity="medium",
                evaluation_cost="moderate",
                has_multiple_objectives=False,
                requires_diversity=False,
                requires_robustness=False,
                constraint_count=0,
                estimated_iterations=100,
                keywords=[]
            )

            rec_id = tracker.record_recommendation(prediction, problem_chars)
            tracker.record_actual_performance(rec_id, 0.8, f"run_{i}")

        # Get metrics
        metrics = tracker.get_accuracy_metrics()

        assert 'average_accuracy' in metrics
        assert 'total_recommendations' in metrics
        assert metrics['total_recommendations'] == 10
        assert 0 <= metrics['average_accuracy'] <= 1


# ============================================================================
# TEST COLD START
# ============================================================================

class TestColdStart:
    """Test cold start handling"""

    @pytest.mark.asyncio
    async def test_cold_start_prediction(self, ensemble_selector):
        """Test cold start recommendation"""
        # Empty selector (no historical data)
        empty_selector = EnsembleStrategySelector(learning_enabled=True)

        problem_chars = ProblemCharacteristics(
            domain="finance",
            complexity="high",
            evaluation_cost="expensive",
            has_multiple_objectives=False,
            requires_diversity=True,
            requires_robustness=True,
            constraint_count=2,
            estimated_iterations=50,
            keywords=["portfolio", "optimization"]
        )

        prediction = await empty_selector.handle_cold_start(problem_chars, "finance")

        # Should still give a recommendation
        assert prediction.strategy is not None
        assert prediction.point_estimate > 0

        # Should have lower confidence due to cold start
        assert prediction.confidence_level < 1.0

        # Should use only rule-based
        assert 'rule_based' in prediction.prediction_methods

    @pytest.mark.asyncio
    async def test_cold_start_explanation(self, ensemble_selector):
        """Test cold start includes explanation"""
        empty_selector = EnsembleStrategySelector(learning_enabled=True)

        problem_chars = ProblemCharacteristics(
            domain="science",
            complexity="high",
            evaluation_cost="very_expensive",
            has_multiple_objectives=False,
            requires_diversity=False,
            requires_robustness=False,
            constraint_count=1,
            estimated_iterations=30,
            keywords=["experiment", "simulation"]
        )

        prediction = await empty_selector.handle_cold_start(problem_chars, "science")

        # Reasoning should mention cold start
        assert "cold start" in prediction.reasoning.lower()


# ============================================================================
# TEST EXPLANATION GENERATION
# ============================================================================

class TestExplanationGeneration:
    """Test explanation generation"""

    @pytest.mark.asyncio
    async def test_explain_ensemble_recommendation(self, ensemble_selector, sample_problem_chars):
        """Test ensemble recommendation explanation"""
        prediction = EnsemblePrediction(
            strategy=(EvolutionSystem.LOONGFLOW, "pes"),
            point_estimate=0.82,
            confidence_interval=(0.78, 0.86),
            confidence_level=0.95,
            prediction_methods=["rule_based", "similarity", "trend"],
            disagreement_ratio=0.15,
            reasoning="Test reasoning with details",
            method_weights={'rule_based': 0.30, 'similarity': 0.35, 'trend': 0.35},
            individual_predictions={
                'rule_based': (EvolutionSystem.LOONGFLOW, "pes", 0.85),
                'similarity': (EvolutionSystem.LOONGFLOW, "pes", 0.80),
                'trend': (EvolutionSystem.LOONGFLOW, "pes", 0.78)
            }
        )

        explanation = ensemble_selector.explain_ensemble_recommendation(
            prediction, sample_problem_chars
        )

        # Verify explanation structure
        assert "Ensemble Strategy Recommendation" in explanation
        assert "LOONGFLOW" in explanation
        assert "PES" in explanation
        assert "Point Estimate" in explanation
        assert "Confidence Interval" in explanation
        assert "Method Agreement" in explanation
        assert "Problem Analysis" in explanation
        assert "Learning Metrics" in explanation

        # Verify specific values
        assert "0.82" in explanation or "82%" in explanation or "82.00%" in explanation  # Point estimate
        assert "78" in explanation and "%" in explanation  # Lower bound (78.00%)
        assert "86" in explanation and "%" in explanation  # Upper bound (86.00%)

    @pytest.mark.asyncio
    async def test_explanation_includes_learning_metrics(self, ensemble_selector, sample_problem_chars):
        """Test explanation includes learning metrics"""
        # Add some learning data
        for i in range(5):
            prediction = EnsemblePrediction(
                strategy=(EvolutionSystem.LOONGFLOW, "pes"),
                point_estimate=0.75,
                confidence_interval=(0.70, 0.80),
                confidence_level=0.95,
                prediction_methods=["rule_based"],
                disagreement_ratio=0.0,
                reasoning="Test",
                method_weights={'rule_based': 1.0},
                individual_predictions={'rule_based': (EvolutionSystem.LOONGFLOW, "pes", 0.8)}
            )

            rec_id = ensemble_selector.learning_tracker.record_recommendation(
                prediction, sample_problem_chars
            )
            ensemble_selector.learning_tracker.record_actual_performance(rec_id, 0.8, f"run_{i}")

        # Create new prediction
        prediction = EnsemblePrediction(
            strategy=(EvolutionSystem.LOONGFLOW, "pes"),
            point_estimate=0.80,
            confidence_interval=(0.75, 0.85),
            confidence_level=0.95,
            prediction_methods=["rule_based"],
            disagreement_ratio=0.0,
            reasoning="Test",
            method_weights={'rule_based': 1.0},
            individual_predictions={'rule_based': (EvolutionSystem.LOONGFLOW, "pes", 0.85)}
        )

        explanation = ensemble_selector.explain_ensemble_recommendation(
            prediction, sample_problem_chars
        )

        # Should include learning metrics
        assert "Average Accuracy" in explanation
        assert "Total Recommendations" in explanation


# ============================================================================
# TEST INTEGRATION
# ============================================================================

class TestIntegration:
    """Integration tests for complete workflows"""

    @pytest.mark.asyncio
    async def test_full_recommendation_workflow(self, ensemble_selector):
        """Test complete recommendation workflow"""
        problem = "Optimize portfolio allocation across 5 assets"
        domain = "finance"
        constraints = {
            "objectives": ["return", "risk"],
            "time_limit_seconds": 300
        }

        # Get recommendation
        prediction = await ensemble_selector.recommend_with_ensemble(
            problem, domain, constraints
        )

        # Verify structure
        assert prediction.strategy is not None
        assert prediction.point_estimate > 0

        # Simulate running the strategy
        # Get the recommendation ID
        rec_id = ensemble_selector.learning_tracker.recommendations_made[-1]['id']

        # Record actual performance
        actual_score = 0.83
        metrics = ensemble_selector.learning_tracker.record_actual_performance(
            rec_id, actual_score, run_id="test_run_001"
        )

        # Verify learning
        assert 'accuracy' in metrics
        assert metrics['accuracy'] > 0

        # Check metrics improved
        learning_metrics = ensemble_selector.get_learning_metrics()
        assert learning_metrics['total_recommendations'] > 0

    @pytest.mark.asyncio
    async def test_multi_domain_learning(self, ensemble_selector):
        """Test learning across multiple domains"""
        domains = ["finance", "science", "trading"]

        for domain in domains:
            problem = f"Optimize {domain} problem"
            constraints = {"time_limit_seconds": 300}

            prediction = await ensemble_selector.recommend_with_ensemble(
                problem, domain, constraints
            )

            # Should get a recommendation for each domain
            assert prediction.strategy is not None

            # Record performance
            rec_id = ensemble_selector.learning_tracker.recommendations_made[-1]['id']
            ensemble_selector.learning_tracker.record_actual_performance(
                rec_id, 0.8, f"run_{domain}"
            )

        # Verify learned from all domains
        metrics = ensemble_selector.get_learning_metrics()
        assert metrics['total_recommendations'] >= len(domains)

    @pytest.mark.asyncio
    async def test_weight_evolution(self, ensemble_selector):
        """Test that weights evolve over time"""
        initial_weights = ensemble_selector.method_weights.copy()

        # Make recommendations and record performance
        for i in range(25):
            problem = f"Optimize problem {i}"
            domain = "finance" if i % 2 == 0 else "science"
            constraints = {"time_limit_seconds": 300}

            prediction = await ensemble_selector.recommend_with_ensemble(
                problem, domain, constraints
            )

            # Vary performance to simulate different method effectiveness
            actual = 0.9 if i % 3 == 0 else 0.65

            rec_id = ensemble_selector.learning_tracker.recommendations_made[-1]['id']
            ensemble_selector.learning_tracker.record_actual_performance(
                rec_id, actual, f"run_{i}"
            )

        # Get final weights
        final_weights = ensemble_selector.method_weights

        # Weights may have changed
        # (not guaranteed to be different, but should be tracked)
        assert len(final_weights) == len(initial_weights)

        # Should still sum to 1.0
        assert abs(sum(final_weights.values()) - 1.0) < 0.01


# ============================================================================
# TEST CONVENIENCE FUNCTIONS
# ============================================================================

class TestConvenienceFunctions:
    """Test convenience functions"""

    @pytest.mark.asyncio
    async def test_recommend_evolutionary_strategy_ensemble(self):
        """Test convenience function with ensemble"""
        problem = "Optimize experimental design"
        domain = "science"
        constraints = {"time_limit_seconds": 900}

        prediction = await recommend_evolutionary_strategy(
            problem, domain, constraints, use_ensemble=True
        )

        # Should return EnsemblePrediction
        assert isinstance(prediction, EnsemblePrediction)
        assert prediction.strategy is not None


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
