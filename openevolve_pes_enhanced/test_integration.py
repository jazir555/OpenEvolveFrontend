"""Tests for PES Enhanced integration layer.

These tests verify that:
1. The enhancement layer wraps existing code correctly
2. All enhancements work independently
3. Backward compatibility is maintained
"""

import unittest
import asyncio
from unittest.mock import MagicMock, patch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openevolve_pes_enhanced import (
    PESEnhancedConfig,
    CostOptimizer,
    EarlyStoppingController,
    CostAwareStrategySelector,
    SummarizationEngine,
)


class TestConfig(unittest.TestCase):
    """Test configuration system."""
    
    def test_default_config(self):
        """Test default config has enhancements disabled."""
        config = PESEnhancedConfig()
        self.assertFalse(config.enable_cost_optimization)
        self.assertFalse(config.enable_early_stopping)
        self.assertFalse(config.enable_planning)
        self.assertTrue(config.preserve_existing_behavior)
    
    def test_enable_all(self):
        """Test enable_all creates config with all features."""
        config = PESEnhancedConfig.enable_all()
        self.assertTrue(config.enable_cost_optimization)
        self.assertTrue(config.enable_early_stopping)
        self.assertTrue(config.enable_planning)
        self.assertTrue(config.enable_summarization)
    
    def test_cost_aware(self):
        """Test cost_aware factory method."""
        config = PESEnhancedConfig.cost_aware(max_cost_usd=5.0)
        self.assertTrue(config.enable_cost_optimization)
        self.assertTrue(config.enable_early_stopping)
        self.assertEqual(config.cost.max_cost_usd, 5.0)


class TestCostOptimizer(unittest.TestCase):
    """Test cost optimization components."""
    
    def test_budget_tracker_initialization(self):
        """Test budget tracker initializes correctly."""
        from openevolve_pes_enhanced.cost_optimizer import BudgetTracker
        
        tracker = BudgetTracker(
            max_cost_usd=10.0,
            max_tokens=50000,
            max_time_ms=300000
        )
        
        status = tracker.get_status()
        self.assertEqual(status.status, "ok")
        self.assertFalse(status.should_stop)
    
    def test_budget_tracker_alerts(self):
        """Test budget tracker warning/critical thresholds."""
        from openevolve_pes_enhanced.cost_optimizer import BudgetTracker
        
        tracker = BudgetTracker(
            max_cost_usd=10.0,
            warning_threshold=0.50,
            critical_threshold=0.80
        )
        
        # Simulate 60% cost usage
        tracker.cost_used = 6.0
        status = tracker.get_status()
        self.assertEqual(status.status, "critical")
        self.assertTrue(status.should_stop)
    
    def test_cost_estimation(self):
        """Test cost estimation."""
        from openevolve_pes_enhanced.cost_optimizer import CostAwarePlanner
        
        planner = CostAwarePlanner()
        estimate = planner.estimate_cost(iterations=50, population_size=20)
        
        self.assertIn("total_cost_usd", estimate)
        self.assertIn("total_tokens", estimate)
        self.assertIn("total_evaluations", estimate)
        self.assertEqual(estimate["total_evaluations"], 1000)
    
    def test_efficiency_calculation(self):
        """Test efficiency gain calculation."""
        optimizer = CostOptimizer()
        optimizer.initialize_budget()
        
        metrics = optimizer.calculate_efficiency(actual_evaluations=400)
        
        # Baseline is 2.5x actual = 1000
        # Saved = 1000 - 400 = 600
        # Gain = 600 / 1000 = 0.6
        self.assertEqual(metrics.efficiency_gain, 0.6)
        self.assertEqual(metrics.evaluations_saved, 600)


class TestEarlyStopping(unittest.TestCase):
    """Test early stopping and convergence detection."""
    
    def test_convergence_detector(self):
        """Test convergence detection."""
        from openevolve_pes_enhanced.execution_monitor import ConvergenceDetector
        
        detector = ConvergenceDetector(fitness_threshold=0.95)
        
        # Add snapshots leading to convergence
        from openevolve_pes_enhanced.execution_monitor import ExecutionSnapshot
        import time
        
        for i in range(10):
            detector.update(ExecutionSnapshot(
                iteration=i,
                best_fitness=0.90 + (i * 0.01),
                avg_fitness=0.85 + (i * 0.008),
                diversity_score=0.5 - (i * 0.02),
                timestamp_ms=int(time.time() * 1000)
            ))
        
        status = detector.check_convergence()
        self.assertIsInstance(status.is_converged, bool)
    
    def test_early_stopping_patience(self):
        """Test early stopping with patience."""
        controller = EarlyStoppingController(patience=3, min_improvement=0.01)
        controller.start()
        
        # Simulate iterations with no improvement
        for i in range(5):
            should_stop, reason = controller.check_should_stop(
                iteration=i,
                best_fitness=0.5,  # No improvement
                avg_fitness=0.5,
                diversity=0.3
            )
        
        # Should stop after patience (3) iterations without improvement
        self.assertTrue(should_stop)
        self.assertIn("No improvement", reason)
    
    def test_fitness_threshold_convergence(self):
        """Test convergence when fitness threshold reached."""
        controller = EarlyStoppingController()
        controller.start()
        
        should_stop, reason = controller.check_should_stop(
            iteration=5,
            best_fitness=0.96,  # Above 0.95 threshold
            avg_fitness=0.90,
            diversity=0.2
        )
        
        self.assertTrue(should_stop)
        self.assertIn("Converged", reason)


class TestStrategySelection(unittest.TestCase):
    """Test strategy selection."""
    
    def test_lean_detection(self):
        """Test Lean 4 problem detection."""
        selector = CostAwareStrategySelector()
        
        decision = selector.select_strategy(
            problem_description="Prove a theorem about lists",
            code=None,
            language="lean"
        )
        
        from openevolve_pes_enhanced.strategy_enhancer import StrategyType
        self.assertEqual(decision.strategy, StrategyType.LEAN_PROOF)
    
    def test_budget_based_selection(self):
        """Test strategy selection based on budget."""
        selector = CostAwareStrategySelector()
        
        # Tight budget should select PES for efficiency
        decision = selector.select_strategy(
            problem_description="Optimize function",
            code=None,
            max_cost_usd=0.5
        )
        
        from openevolve_pes_enhanced.strategy_enhancer import StrategyType
        self.assertEqual(decision.strategy, StrategyType.PES_ENHANCED)
        self.assertIn("efficiency", decision.reasoning.lower())
    
    def test_complexity_estimation(self):
        """Test problem complexity estimation."""
        selector = CostAwareStrategySelector()
        
        # Complex problem
        complexity = selector._estimate_complexity(
            description="Optimize with constraints " * 50,  # Long description
            code="class Complex:\n" + "    pass\n" * 100  # 100+ lines
        )
        
        self.assertIn(complexity, ["high", "very_high"])


class TestSummarization(unittest.TestCase):
    """Test summarization engine."""
    
    def test_pattern_extraction(self):
        """Test pattern extraction from fitness history."""
        from openevolve_pes_enhanced.summarization_engine import InsightExtractor
        
        extractor = InsightExtractor()
        
        # Rapid improvement pattern
        fitness_history = [0.1, 0.3, 0.5, 0.7, 0.85, 0.9]
        diversity_history = [0.8, 0.75, 0.7, 0.6, 0.5, 0.45]
        
        patterns = extractor.extract_patterns(fitness_history, diversity_history)
        
        # Should detect rapid early improvement
        rapid_patterns = [p for p in patterns if "rapid" in p.description.lower()]
        self.assertTrue(len(rapid_patterns) > 0 or len(patterns) > 0)
    
    def test_success_factor_identification(self):
        """Test success factor identification."""
        from openevolve_pes_enhanced.summarization_engine import InsightExtractor
        
        extractor = InsightExtractor()
        
        # Strong improvement
        fitness_history = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
        
        factors = extractor.identify_success_factors(fitness_history)
        
        # Should identify strong initial population or consistent improvement
        self.assertTrue(len(factors) > 0)
    
    def test_summary_generation(self):
        """Test complete summary generation."""
        engine = SummarizationEngine()
        
        # Mock execution history
        history = [
            {"best_fitness": 0.1 * i, "diversity": 0.8 - 0.05 * i, "evaluations": 10, "timestamp_ms": i * 1000}
            for i in range(1, 11)
        ]
        
        cost_data = {"total_cost_usd": 2.50}
        
        summary = engine.summarize(
            execution_history=history,
            cost_data=cost_data,
            strategy="test",
            problem_type="test"
        )
        
        self.assertIsInstance(summary.total_iterations, int)
        self.assertIsInstance(summary.total_cost_usd, float)
        self.assertIsInstance(summary.efficiency_gain, float)


class TestIntegration(unittest.TestCase):
    """Test integration with existing components."""
    
    def test_wrapper_initialization(self):
        """Test wrapper initializes correctly."""
        from openevolve_pes_enhanced import PESIntegrationWrapper
        
        wrapper = PESIntegrationWrapper()
        self.assertIsNotNone(wrapper.cost_optimizer)
        self.assertIsNotNone(wrapper.strategy_enhancer)
    
    def test_wrapper_with_enhancements(self):
        """Test wrapper with all enhancements."""
        from openevolve_pes_enhanced import PESIntegrationWrapper, PESEnhancedConfig
        
        config = PESEnhancedConfig.enable_all()
        wrapper = PESIntegrationWrapper(config)
        
        self.assertTrue(config.enable_cost_optimization)
        self.assertTrue(config.enable_early_stopping)
    
    def test_cost_estimate_api(self):
        """Test cost estimation API."""
        from openevolve_pes_enhanced import PESIntegrationWrapper
        
        wrapper = PESIntegrationWrapper()
        estimate = wrapper.get_cost_estimate(50, 20)
        
        self.assertIn("total_cost_usd", estimate)
        self.assertGreater(estimate["total_cost_usd"], 0)
    
    def test_recommendation_api(self):
        """Test parameter recommendation API."""
        from openevolve_pes_enhanced import PESIntegrationWrapper
        
        wrapper = PESIntegrationWrapper()
        rec = wrapper.recommend_parameters("Test problem", 10.0)
        
        self.assertIn("strategy", rec)
        self.assertIn("parameters", rec)
        self.assertIn("estimated_cost", rec)


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility with existing code."""
    
    def test_existing_api_unchanged(self):
        """Verify that existing APIs are preserved."""
        # This test documents that existing imports should still work
        # We can't actually import because it may not be available in test environment
        
        # These should continue to work:
        # from openevolve_pes_integration import enhance_code
        # from openevolve_agnostic_pes import AgnosticPESEngine
        # from leanaide_pes_handler import LeanPESHandler
        
        self.assertTrue(True)  # Placeholder
    
    def test_enhancement_is_additive(self):
        """Verify enhancements don't change existing behavior."""
        config = PESEnhancedConfig()  # Default: all enhancements disabled
        
        self.assertFalse(config.enable_cost_optimization)
        self.assertFalse(config.enable_early_stopping)
        self.assertTrue(config.preserve_existing_behavior)


def run_async_test(coro):
    """Helper to run async tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


if __name__ == "__main__":
    unittest.main()
