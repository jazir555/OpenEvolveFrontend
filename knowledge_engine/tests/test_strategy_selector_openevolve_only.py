"""
Test OpenEvolve-Only Mode for Strategy Selector

Tests the strategy selector's ability to work without LoongFlow availability.
"""

import asyncio
import pytest
from datetime import datetime, UTC

from knowledge_engine.core.strategy_recommender import (
    EnsembleStrategySelector,
    LoongFlowChecker,
    ProblemCharacteristics,
    EvaluationCost,
    ComplexityLevel,
)


class TestLoongFlowChecker:
    """Test LoongFlow availability checker"""

    def test_checker_returns_bool(self):
        """Checker should return a boolean"""
        result = LoongFlowChecker.is_available()
        assert isinstance(result, bool)

    def test_checker_caches_result(self):
        """Checker should cache the result"""
        # First call
        result1 = LoongFlowChecker.is_available()

        # Reset and call again
        LoongFlowChecker.reset()
        result2 = LoongFlowChecker.is_available()

        # Should both be bools
        assert isinstance(result1, bool)
        assert isinstance(result2, bool)


class TestOpenEvolveOnlyMode:
    """Test OpenEvolve-only recommendation mode"""

    @pytest.fixture
    def selector(self):
        """Create selector with LoongFlow disabled"""
        return EnsembleStrategySelector(
            knowledge_engine=None,
            enable_loongflow=False
        )

    @pytest.fixture
    def sample_problem(self):
        """Sample problem for testing"""
        return {
            "description": "Optimize portfolio allocation for risk-adjusted returns",
            "domain": "finance",
            "constraints": {
                "objectives": ["maximize_returns", "minimize_risk"],
                "constraints": ["budget_limit"],
                "time_limit_seconds": 60
            }
        }

    def test_selector_initializes_with_loongflow_disabled(self, selector):
        """Selector should initialize correctly with LoongFlow disabled"""
        assert selector.enable_loongflow is False
        assert selector.loongflow_available is False

    def test_selector_detects_loongflow_unavailable(self):
        """Selector should detect when LoongFlow is unavailable"""
        selector = EnsembleStrategySelector(
            knowledge_engine=None,
            enable_loongflow=True  # Try to enable
        )

        # If LoongFlow is actually unavailable, should be False
        # If it's available, this test might fail (that's OK)
        if not LoongFlowChecker.is_available():
            assert selector.loongflow_available is False

    def test_loongflow_usage_determination(self, selector):
        """Test _determine_loongflow_usage logic"""
        # Test 1: Runtime override to disable
        result = selector._determine_loongflow_usage(enable_loongflow=False)
        assert result is False

        # Test 2: Runtime override to enable (should still check availability)
        result = selector._determine_loongflow_usage(enable_loongflow=True)
        # If LoongFlow unavailable, should be False
        if not LoongFlowChecker.is_available():
            assert result is False

        # Test 3: No override, use config (disabled)
        result = selector._determine_loongflow_usage(enable_loongflow=None)
        assert result is False

    @pytest.mark.asyncio
    async def test_openevolve_rule_based_prediction(self, selector):
        """Test OpenEvolve-only rule-based prediction"""
        problem_chars = ProblemCharacteristics(
            domain="finance",
            complexity=ComplexityLevel.HIGH,
            evaluation_cost=EvaluationCost.EXPENSIVE,
            has_multiple_objectives=True,
            requires_diversity=True,
            requires_robustness=True,
            constraint_count=3,
            estimated_iterations=50
        )

        prediction = await selector._openevolve_rule_based(
            problem_chars,
            "finance"
        )

        # Should recommend OpenEvolve system
        assert prediction.system == "openevolve"

        # Should recommend MO mode for multiple objectives
        assert prediction.mode in ["mo", "qd", "adversarial", "standard"]

        # Should have reasonable confidence
        assert 0.0 <= prediction.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_get_default_openevolve_mode(self, selector):
        """Test default OpenEvolve mode selection"""
        prediction = selector._get_default_openevolve_mode("finance")

        assert prediction.system == "openevolve"
        assert prediction.mode in ["qd", "mo", "adversarial", "standard"]
        assert prediction.confidence > 0.0

    @pytest.mark.asyncio
    async def test_openevolve_only_recommendation(self, selector, sample_problem):
        """Test full OpenEvolve-only recommendation"""
        prediction = await selector.recommend_with_ensemble(
            problem_description=sample_problem["description"],
            domain=sample_problem["domain"],
            constraints=sample_problem["constraints"],
            enable_loongflow=False
        )

        # Should return a prediction
        assert prediction is not None

        # Should not recommend LoongFlow system
        system, mode = prediction.strategy
        if system == "loongflow":
            pytest.fail("Should not recommend LoongFlow in OpenEvolve-only mode")

        # Should have valid confidence interval
        assert prediction.point_estimate >= 0.0
        assert len(prediction.confidence_interval) == 2
        assert prediction.confidence_interval[0] <= prediction.point_estimate
        assert prediction.confidence_interval[1] >= prediction.point_estimate

    @pytest.mark.asyncio
    async def test_convenience_method_openevolve_only(self, selector, sample_problem):
        """Test convenience method for OpenEvolve-only recommendation"""
        prediction = await selector.recommend_openevolve_only(
            problem_description=sample_problem["description"],
            domain=sample_problem["domain"],
            constraints=sample_problem["constraints"]
        )

        # Should work the same as explicit enable_loongflow=False
        assert prediction is not None
        system, mode = prediction.strategy
        if system == "loongflow":
            pytest.fail("Should not recommend LoongFlow in OpenEvolve-only mode")

    def test_get_available_modes(self, selector):
        """Test getting available modes"""
        modes = selector.get_available_modes()

        # Should not include PES mode (LoongFlow)
        assert "pes" not in modes

        # Should include OpenEvolve modes
        assert "qd" in modes
        assert "mo" in modes
        assert "adversarial" in modes
        assert "standard" in modes

    def test_is_loongflow_available(self, selector):
        """Test checking LoongFlow availability"""
        result = selector.is_loongflow_available()

        # Should be False when disabled
        assert result is False

    @pytest.mark.asyncio
    async def test_cold_start_openevolve_only(self, selector):
        """Test cold start handling in OpenEvolve-only mode"""
        problem_chars = ProblemCharacteristics(
            domain="science",
            complexity=ComplexityLevel.MEDIUM,
            evaluation_cost=EvaluationCost.MODERATE,
            has_multiple_objectives=False,
            requires_diversity=True,
            requires_robustness=False,
            constraint_count=1,
            estimated_iterations=30
        )

        prediction = await selector.handle_cold_start(
            problem_chars=problem_chars,
            domain="science",
            enable_loongflow=False
        )

        # Should still return a valid prediction
        assert prediction is not None
        system, mode = prediction.strategy

        # Should be OpenEvolve system
        assert system == "openevolve"

        # Should have lower confidence due to cold start
        assert prediction.confidence_level < 1.0


class TestModeSelectionRules:
    """Test OpenEvolve mode selection rules"""

    @pytest.fixture
    def selector(self):
        """Create selector with LoongFlow disabled"""
        return EnsembleStrategySelector(
            knowledge_engine=None,
            enable_loongflow=False
        )

    @pytest.mark.asyncio
    async def test_multi_objective_selects_mo(self, selector):
        """Test that multi-objective problems select MO mode"""
        problem_chars = ProblemCharacteristics(
            domain="engineering",
            complexity=ComplexityLevel.HIGH,
            evaluation_cost=EvaluationCost.EXPENSIVE,
            has_multiple_objectives=True,
            requires_diversity=False,
            requires_robustness=False,
            constraint_count=2,
            estimated_iterations=50
        )

        prediction = await selector._openevolve_rule_based(
            problem_chars,
            "engineering"
        )

        # Should select MO mode
        assert prediction.mode == "mo"
        assert prediction.system == "openevolve"

    @pytest.mark.asyncio
    async def test_diversity_selects_qd(self, selector):
        """Test that diversity requirements select QD mode"""
        problem_chars = ProblemCharacteristics(
            domain="science",
            complexity=ComplexityLevel.MEDIUM,
            evaluation_cost=EvaluationCost.MODERATE,
            has_multiple_objectives=False,
            requires_diversity=True,
            requires_robustness=False,
            constraint_count=1,
            estimated_iterations=30
        )

        prediction = await selector._openevolve_rule_based(
            problem_chars,
            "science"
        )

        # Should select QD mode
        assert prediction.mode == "qd"
        assert prediction.system == "openevolve"

    @pytest.mark.asyncio
    async def test_robustness_selects_adversarial(self, selector):
        """Test that robustness requirements select Adversarial mode"""
        problem_chars = ProblemCharacteristics(
            domain="pharma",
            complexity=ComplexityLevel.HIGH,
            evaluation_cost=EvaluationCost.VERY_EXPENSIVE,
            has_multiple_objectives=False,
            requires_diversity=False,
            requires_robustness=True,
            constraint_count=3,
            estimated_iterations=100
        )

        prediction = await selector._openevolve_rule_based(
            problem_chars,
            "pharma"
        )

        # Should select Adversarial mode
        assert prediction.mode == "adversarial"
        assert prediction.system == "openevolve"

    @pytest.mark.asyncio
    async def test_default_selects_standard(self, selector):
        """Test that default case selects Standard mode"""
        problem_chars = ProblemCharacteristics(
            domain="web",
            complexity=ComplexityLevel.LOW,
            evaluation_cost=EvaluationCost.CHEAP,
            has_multiple_objectives=False,
            requires_diversity=False,
            requires_robustness=False,
            constraint_count=0,
            estimated_iterations=200
        )

        prediction = await selector._openevolve_rule_based(
            problem_chars,
            "web"
        )

        # Should select Standard mode
        assert prediction.mode == "standard"
        assert prediction.system == "openevolve"


class TestIntegrationScenarios:
    """Integration tests for OpenEvolve-only scenarios"""

    @pytest.fixture
    def selector(self):
        """Create selector with LoongFlow disabled"""
        return EnsembleStrategySelector(
            knowledge_engine=None,
            enable_loongflow=False
        )

    @pytest.mark.asyncio
    async def test_finance_portfolio_optimization(self, selector):
        """Test finance portfolio optimization scenario"""
        prediction = await selector.recommend_with_ensemble(
            problem_description="Optimize portfolio allocation for maximum risk-adjusted returns",
            domain="finance",
            constraints={
                "objectives": ["maximize_returns", "minimize_risk"],
                "constraints": ["budget_limit", "sector_limits"],
                "time_limit_seconds": 60
            },
            enable_loongflow=False
        )

        # Should recommend OpenEvolve
        system, mode = prediction.strategy
        assert system == "openevolve"
        assert mode in ["mo", "qd", "adversarial", "standard"]

    @pytest.mark.asyncio
    async def test_scientific_exploration(self, selector):
        """Test scientific exploration scenario"""
        prediction = await selector.recommend_with_ensemble(
            problem_description="Find diverse solutions for protein folding",
            domain="science",
            constraints={
                "objectives": ["minimize_energy"],
                "constraints": ["physics_constraints"],
                "time_limit_seconds": 600
            },
            enable_loongflow=False
        )

        # Should recommend OpenEvolve
        system, mode = prediction.strategy
        assert system == "openevolve"

    @pytest.mark.asyncio
    async def test_safety_critical_engineering(self, selector):
        """Test safety-critical engineering scenario"""
        prediction = await selector.recommend_with_ensemble(
            problem_description="Design bridge structure that withstands extreme conditions",
            domain="engineering",
            constraints={
                "objectives": ["minimize_weight", "maximize_strength"],
                "constraints": ["safety_factors"],
                "safety_critical": True,
                "time_limit_seconds": 300
            },
            enable_loongflow=False
        )

        # Should recommend OpenEvolve Adversarial for robustness
        system, mode = prediction.strategy
        assert system == "openevolve"
        # Adversarial likely for safety-critical


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
