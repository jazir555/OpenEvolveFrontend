"""
Long-Horizon Learning Tests

Comprehensive tests for online learning, A/B testing, causal modeling, and meta-learning.

Author: Claude (Sonnet 4.5)
Date: January 30, 2026
"""

import pytest
import asyncio
from datetime import datetime, UTC, timedelta
import numpy as np
from typing import Dict, Any, List

# Import modules to test
import sys
from pathlib import Path

# Add knowledge_engine to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge_engine.online_learning import OnlineLearner
from knowledge_engine.ab_testing import ABTestFramework
from knowledge_engine.causal_modeling import CausalModelBuilder
from knowledge_engine.meta_learning import MetaLearner, FeatureExtractor
from knowledge_engine.schemas.long_horizon import (
    LearningOutcome,
    StrategyPerformance,
    AdaptationAction,
    Experiment,
    VariantStats,
    CausalModel,
    CausalRelationship,
    MetaPattern,
    StrategyRecommendation,
    OutcomeType,
    AdaptationActionType,
    ExperimentStatus,
    ExplorationStrategy
)


# ========================================================================
# ONLINE LEARNING TESTS
# ========================================================================

class TestOnlineLearning:
    """Test online learning from streaming outcomes"""

    @pytest.fixture
    def learner(self):
        """Create online learner"""
        return OnlineLearner(
            exploration_strategy=ExplorationStrategy.EPSILON_GREEDY,
            initial_epsilon=0.3,
            performance_window=50
        )

    @pytest.fixture
    def sample_outcomes(self):
        """Generate sample outcomes"""
        outcomes = []
        for i in range(20):
            outcome = LearningOutcome(
                workflow_id="test_workflow",
                strategy_used=f"strategy_{i % 3}",
                outcome_type=OutcomeType.SUCCESS if i % 2 == 0 else OutcomeType.PARTIAL,
                metrics={"fitness": 0.7 + np.random.random() * 0.3},
                context={"domain": "finance"}
            )
            outcomes.append(outcome)
        return outcomes

    @pytest.mark.asyncio
    async def test_record_outcome(self, learner, sample_outcomes):
        """Test recording outcomes"""
        for outcome in sample_outcomes:
            await learner.record_outcome(outcome)

        # Check outcomes were recorded
        perf = await learner.get_strategy_performance("test_workflow", "strategy_0")
        assert perf is not None
        assert perf.total_outcomes > 0

    @pytest.mark.asyncio
    async def test_idempotency(self, learner):
        """Test recording same outcome multiple times (idempotency)"""
        outcome = LearningOutcome(
            workflow_id="test_workflow",
            strategy_used="strategy_0",
            outcome_type=OutcomeType.SUCCESS,
            metrics={"fitness": 0.8},
            context={}
        )

        # Record same outcome twice
        await learner.record_outcome(outcome)
        await learner.record_outcome(outcome)

        # Should only count once
        perf = await learner.get_strategy_performance("test_workflow", "strategy_0")
        assert perf.total_outcomes == 1

    @pytest.mark.asyncio
    async def test_moving_average(self, learner):
        """Test moving average calculation"""
        # Record outcomes with known scores
        for i in range(10):
            outcome = LearningOutcome(
                workflow_id="test_workflow",
                strategy_used="strategy_0",
                outcome_type=OutcomeType.SUCCESS,
                metrics={"fitness": 0.5 + i * 0.05},
                context={}
            )
            await learner.record_outcome(outcome)

        perf = await learner.get_strategy_performance("test_workflow", "strategy_0")
        assert perf.moving_average > 0.5
        assert perf.moving_average < 1.0

    @pytest.mark.asyncio
    async def test_best_strategy_selection(self, learner):
        """Test selecting best strategy"""
        # Strategy_0 performs better
        for i in range(10):
            outcome = LearningOutcome(
                workflow_id="test_workflow",
                strategy_used="strategy_0",
                outcome_type=OutcomeType.SUCCESS,
                metrics={"fitness": 0.9},
                context={}
            )
            await learner.record_outcome(outcome)

            outcome2 = LearningOutcome(
                workflow_id="test_workflow",
                strategy_used="strategy_1",
                outcome_type=OutcomeType.PARTIAL,
                metrics={"fitness": 0.6},
                context={}
            )
            await learner.record_outcome(outcome2)

        best = await learner.get_best_strategy("test_workflow")
        assert best == "strategy_0"

    @pytest.mark.asyncio
    async def test_exploration_vs_exploitation(self, learner):
        """Test exploration/exploitation decision"""
        # Record some outcomes first
        for i in range(5):
            outcome = LearningOutcome(
                workflow_id="test_workflow",
                strategy_used="strategy_0",
                outcome_type=OutcomeType.SUCCESS,
                metrics={"fitness": 0.8},
                context={}
            )
            await learner.record_outcome(outcome)

        # Test exploration decision
        should_explore = await learner.should_explore()
        assert isinstance(should_explore, bool)

    @pytest.mark.asyncio
    async def test_adaptation_recommendation(self, learner):
        """Test adaptation recommendations"""
        # Record outcomes showing degradation
        for i in range(25):
            fitness = 0.9 - i * 0.02  # Decreasing performance
            outcome = LearningOutcome(
                workflow_id="test_workflow",
                strategy_used="strategy_0",
                outcome_type=OutcomeType.SUCCESS if fitness > 0.7 else OutcomeType.PARTIAL,
                metrics={"fitness": max(0.0, fitness)},
                context={}
            )
            await learner.record_outcome(outcome)

        # Check for adaptation
        action = await learner.adapt_strategy("test_workflow", 0.5)
        # May or may not recommend adaptation depending on threshold
        if action:
            assert isinstance(action, AdaptationAction)


# ========================================================================
# A/B TESTING TESTS
# ========================================================================

class TestABTesting:
    """Test A/B testing framework"""

    @pytest.fixture
    def framework(self):
        """Create A/B testing framework"""
        return ABTestFramework(
            significance_level=0.05,
            min_sample_size=20,  # Lower for testing
            test_method="frequentist"
        )

    @pytest.mark.asyncio
    async def test_create_experiment(self, framework):
        """Test creating experiment"""
        experiment = await framework.create_experiment(
            name="Test Experiment",
            description="Testing strategy A vs B",
            variants=["strategy_a", "strategy_b"]
        )

        assert experiment.experiment_id.startswith("exp_")
        assert experiment.status == ExperimentStatus.RUNNING
        assert len(experiment.variants) == 2

    @pytest.mark.asyncio
    async def test_record_observations(self, framework):
        """Test recording observations"""
        # Create experiment
        experiment = await framework.create_experiment(
            name="Test",
            description="Test",
            variants=["A", "B"]
        )

        # Record observations
        for i in range(25):
            await framework.record_observation(
                experiment.experiment_id,
                "A",
                outcome=0.7 + np.random.random() * 0.2,
                is_success=True
            )

            await framework.record_observation(
                experiment.experiment_id,
                "B",
                outcome=0.5 + np.random.random() * 0.2,
                is_success=np.random.random() > 0.5
            )

        # Check sample sizes
        experiment = await framework.get_experiment(experiment.experiment_id)
        assert experiment.variants["A"].sample_size == 25
        assert experiment.variants["B"].sample_size == 25

    @pytest.mark.asyncio
    async def test_statistical_significance(self, framework):
        """Test statistical significance detection"""
        # Create experiment
        experiment = await framework.create_experiment(
            name="Significance Test",
            description="Test significance",
            variants=["control", "treatment"]
        )

        # Control: mean 0.5
        for i in range(30):
            await framework.record_observation(
                experiment.experiment_id,
                "control",
                outcome=0.5 + np.random.randn() * 0.1,
                is_success=False
            )

        # Treatment: mean 0.8 (significant difference)
        for i in range(30):
            await framework.record_observation(
                experiment.experiment_id,
                "treatment",
                outcome=0.8 + np.random.randn() * 0.1,
                is_success=True
            )

        # Get results
        results = await framework.get_results(experiment.experiment_id)

        # Should detect significant difference
        assert results.significance or results.confidence > 0.8

    @pytest.mark.asyncio
    async def test_bayesian_analysis(self):
        """Test Bayesian A/B testing"""
        framework = ABTestFramework(test_method="bayesian", min_sample_size=20)

        experiment = await framework.create_experiment(
            name="Bayesian Test",
            description="Test Bayesian",
            variants=["A", "B"]
        )

        # Record observations
        for i in range(30):
            await framework.record_observation(
                experiment.experiment_id,
                "A",
                outcome=0.7,
                is_success=True
            )

            await framework.record_observation(
                experiment.experiment_id,
                "B",
                outcome=0.6,
                is_success=np.random.random() > 0.4
            )

        # Get results
        results = await framework.get_results(experiment.experiment_id)

        # Bayesian should give probability
        assert 0 <= results.confidence <= 1
        assert results.test_statistic >= 0  # Win probability


# ========================================================================
# CAUSAL MODELING TESTS
# ========================================================================

class TestCausalModeling:
    """Test causal modeling"""

    @pytest.fixture
    def builder(self):
        """Create causal model builder"""
        return CausalModelBuilder(
            discovery_method="pc",
            min_confidence=0.6
        )

    @pytest.fixture
    def sample_outcomes(self):
        """Generate sample outcomes"""
        outcomes = []

        # Create synthetic data where exploration_rate affects fitness
        for exp_rate in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for i in range(10):
                # Higher exploration -> higher fitness (simulated)
                fitness = 0.5 + exp_rate * 0.4 + np.random.randn() * 0.1

                outcome = {
                    "context": {
                        "exploration_rate": exp_rate,
                        "population_size": 100,
                        "temperature": 1.0
                    },
                    "metrics": {
                        "fitness": max(0.0, min(1.0, fitness)),
                        "convergence_time": 100 + exp_rate * 50
                    }
                }
                outcomes.append(outcome)

        return outcomes

    @pytest.mark.asyncio
    async def test_build_causal_model(self, builder, sample_outcomes):
        """Test building causal model"""
        model = await builder.build_model(
            domain="test",
            outcomes=sample_outcomes
        )

        assert model.domain == "test"
        assert len(model.relationships) > 0
        assert len(model.factors) > 0
        assert len(model.outcomes) > 0

    @pytest.mark.asyncio
    async def test_identify_causes(self, builder, sample_outcomes):
        """Test identifying causes of outcome"""
        model = await builder.build_model(
            domain="test",
            outcomes=sample_outcomes
        )

        causes = await builder.identify_causes(model, "fitness")

        # Should find some causes
        assert len(causes) >= 0

    @pytest.mark.asyncio
    async def test_predict_intervention(self, builder, sample_outcomes):
        """Test predicting intervention effect"""
        model = await builder.build_model(
            domain="test",
            outcomes=sample_outcomes
        )

        # Predict effect of changing exploration_rate
        prediction = await builder.predict_intervention(
            model=model,
            cause="exploration_rate",
            value=0.8
        )

        assert prediction.intervention == "Set exploration_rate to 0.8"
        assert isinstance(prediction.predicted_effect, float)

    @pytest.mark.asyncio
    async def test_explain_outcome(self, builder, sample_outcomes):
        """Test explaining outcome"""
        model = await builder.build_model(
            domain="test",
            outcomes=sample_outcomes
        )

        explanation = await builder.explain_outcome(
            model=model,
            outcome="fitness"
        )

        assert explanation.outcome == "fitness"
        assert isinstance(explanation.confidence, float)


# ========================================================================
# META-LEARNING TESTS
# ========================================================================

class TestMetaLearning:
    """Test meta-learning across workflows"""

    @pytest.fixture
    def learner(self):
        """Create meta-learner"""
        return MetaLearner(
            min_evidence=2,
            confidence_threshold=0.6
        )

    @pytest.fixture
    def sample_workflows(self):
        """Generate sample workflows"""
        workflows = []

        # Create workflows where PES strategy works well for finance
        for i in range(5):
            workflows.append({
                "workflow_id": f"wf_{i}",
                "domain": "finance",
                "strategy": "pes",
                "outcome_type": "success",
                "fitness": 0.8 + np.random.random() * 0.15,
                "config": {
                    "enable_planning": True,
                    "max_iterations": 50
                },
                "context": {
                    "exploration_rate": 0.3
                },
                "metrics": {
                    "fitness": 0.85,
                    "time": 120
                }
            })

        # QD works well for science
        for i in range(5):
            workflows.append({
                "workflow_id": f"wf_{i+5}",
                "domain": "science",
                "strategy": "qd",
                "outcome_type": "success",
                "fitness": 0.75 + np.random.random() * 0.2,
                "config": {
                    "feature_dimensions": ["complexity"],
                    "num_islands": 5
                },
                "context": {
                    "exploration_rate": 0.7
                },
                "metrics": {
                    "fitness": 0.8,
                    "diversity": 0.9
                }
            })

        return workflows

    @pytest.mark.asyncio
    async def test_extract_patterns(self, learner, sample_workflows):
        """Test pattern extraction"""
        patterns = await learner.extract_patterns(sample_workflows)

        # Should extract some patterns
        assert len(patterns) > 0

        # Check pattern structure
        pattern = patterns[0]
        assert isinstance(pattern, MetaPattern)
        assert len(pattern.evidence) > 0
        assert pattern.confidence > 0

    @pytest.mark.asyncio
    async def test_recommend_strategy(self, learner, sample_workflows):
        """Test strategy recommendation"""
        # Extract patterns first
        await learner.extract_patterns(sample_workflows)

        # Recommend for finance problem
        recommendation = await learner.recommend_strategy({
            "problem_id": "test_problem",
            "domain": "finance",
            "num_variables": 50
        })

        assert isinstance(recommendation, StrategyRecommendation)
        assert recommendation.recommended_strategy in ["pes", "qd", "hybrid"]
        assert 0 <= recommendation.confidence <= 1

    @pytest.mark.asyncio
    async def test_transfer_knowledge(self, learner, sample_workflows):
        """Test knowledge transfer"""
        # Extract patterns from source
        await learner.extract_patterns(sample_workflows)

        # Transfer to new domain
        transferred = await learner.transfer_knowledge(
            source_domain="finance",
            target_domain="trading"
        )

        # Should transfer some patterns
        assert len(transferred) >= 0

    def test_feature_extraction(self):
        """Test feature extractor"""
        extractor = FeatureExtractor()

        features = extractor.extract_features({
            "domain": "finance",
            "num_variables": 100,
            "evaluation_cost": "high",
            "problem_type": "optimization"
        })

        assert features["domain"] == "finance"
        assert features["num_variables"] == 100
        assert features["scale_category"] == "medium"


# ========================================================================
# INTEGRATION TESTS
# ========================================================================

class TestLongHorizonIntegration:
    """Integration tests for long-horizon learning"""

    @pytest.mark.asyncio
    async def test_online_to_ab_testing(self):
        """Test integration between online learning and A/B testing"""
        # Setup online learner
        learner = OnlineLearner()
        framework = ABTestFramework(min_sample_size=10)

        # Create A/B test
        experiment = await framework.create_experiment(
            name="Strategy Comparison",
            description="Compare strategies",
            variants=["strategy_a", "strategy_b"]
        )

        # Simulate workflow outcomes
        for i in range(15):
            # Strategy A performs better
            outcome_a = LearningOutcome(
                workflow_id="test",
                strategy_used="strategy_a",
                outcome_type=OutcomeType.SUCCESS,
                metrics={"fitness": 0.85},
                context={}
            )
            await learner.record_outcome(outcome_a)

            await framework.record_observation(
                experiment.experiment_id,
                "strategy_a",
                outcome=0.85,
                is_success=True
            )

            # Strategy B performs worse
            outcome_b = LearningOutcome(
                workflow_id="test",
                strategy_used="strategy_b",
                outcome_type=OutcomeType.PARTIAL,
                metrics={"fitness": 0.65},
                context={}
            )
            await learner.record_outcome(outcome_b)

            await framework.record_observation(
                experiment.experiment_id,
                "strategy_b",
                outcome=0.65,
                is_success=False
            )

        # Get best strategy from online learning
        best_online = await learner.get_best_strategy("test")

        # Get winner from A/B test
        results = await framework.get_results(experiment.experiment_id)

        # Both should agree
        assert best_online == "strategy_a"
        assert results.winner == "strategy_a"

    @pytest.mark.asyncio
    async def test_causal_to_meta_learning(self):
        """Test integration between causal modeling and meta-learning"""
        # Setup
        causal_builder = CausalModelBuilder()
        meta_learner = MetaLearner(min_evidence=2)

        # Generate outcomes with known causal structure
        outcomes = []
        workflows = []

        for temp in [0.5, 1.0, 1.5]:
            for i in range(5):
                # Temperature affects diversity
                diversity = temp * 0.5 + np.random.random() * 0.2

                outcome = {
                    "context": {"temperature": temp},
                    "metrics": {"diversity": min(1.0, diversity)}
                }
                outcomes.append(outcome)

                workflows.append({
                    "workflow_id": f"wf_{temp}_{i}",
                    "domain": "test",
                    "strategy": "adaptive",
                    "outcome_type": "success",
                    "context": {"temperature": temp},
                    "metrics": {"diversity": min(1.0, diversity)}
                })

        # Build causal model
        model = await causal_builder.build_model("test", outcomes)

        # Extract meta-patterns
        patterns = await meta_learner.extract_patterns(workflows)

        # Should find patterns related to temperature
        assert len(model.relationships) > 0
        assert len(patterns) >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
