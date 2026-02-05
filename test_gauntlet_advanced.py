"""
Comprehensive Tests for Advanced Gauntlet Types

Tests all 8+ gauntlet types:
1. Adversarial Gauntlet
2. Formal Verification Gauntlet
3. Statistical Gauntlet
4. Domain-Specific Gauntlets (Physics, Finance, Chemistry, Engineering)
5. Multi-Objective Gauntlet
6. Evolutionary Gauntlet
7. Temporal Gauntlet
8. Cross-Validation Gauntlet

Plus orchestration and scoring tests.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import time
import numpy as np
from typing import Dict, List, Any
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gauntlet_types import (
    GauntletType, GauntletResult, BaseGauntlet,
    AdversarialGauntlet, FormalVerificationGauntlet, StatisticalGauntlet,
    DomainSpecificGauntlet, MultiObjectiveGauntlet, EvolutionaryGauntlet,
    TemporalGauntlet, CrossValidationGauntlet,
    create_gauntlet, list_available_gauntlets
)

from gauntlet_orchestrator import (
    OrchestrationMode, GauntletOrchestrator, GauntletScoringSystem,
    run_sequential_gauntlets, run_parallel_gauntlets, run_adaptive_gauntlets
)


class MockSolution:
    """Mock solution for testing."""
    def __init__(self, content: str, solution_id: str = "test_solution"):
        self.id = solution_id
        self.content = content
        self.content_type = "code"


class TestAdversarialGauntlet(unittest.TestCase):
    """Tests for Adversarial Gauntlet."""
    
    def setUp(self):
        self.gauntlet = AdversarialGauntlet("test_adversarial")
        self.solution = MockSolution("def test(): return 42")
    
    def test_create_adversarial_gauntlet(self):
        """Test adversarial gauntlet creation."""
        self.assertEqual(self.gauntlet.gauntlet_type, GauntletType.ADVERSARIAL)
        self.assertEqual(self.gauntlet.name, "test_adversarial")
        self.assertIn("systematic", self.gauntlet.attack_modes)
    
    def test_execute_basic(self):
        """Test basic adversarial execution."""
        context = {
            "content": "def test(): return 42",
            "content_type": "code"
        }
        
        result = self.gauntlet.execute(self.solution, context)
        
        self.assertIsInstance(result, GauntletResult)
        self.assertEqual(result.gauntlet_type, GauntletType.ADVERSARIAL)
        self.assertIn("score", result.details or {})
    
    def test_robustness_scoring(self):
        """Test robustness score calculation."""
        # Test with no issues
        red_team_result = {"issues": [], "confidence": 0.9}
        score = self.gauntlet._calculate_robustness_score(red_team_result, None)
        self.assertEqual(score, 1.0)
        
        # Test with issues
        red_team_result = {
            "issues": [
                {"severity": "high"},
                {"severity": "medium"}
            ],
            "confidence": 0.8
        }
        score = self.gauntlet._calculate_robustness_score(red_team_result, None)
        self.assertLess(score, 1.0)
        self.assertGreater(score, 0.0)


class TestFormalVerificationGauntlet(unittest.TestCase):
    """Tests for Formal Verification Gauntlet."""
    
    def setUp(self):
        self.gauntlet = FormalVerificationGauntlet("test_formal")
        self.solution = MockSolution("def verified(): pass")
    
    def test_create_formal_gauntlet(self):
        """Test formal verification gauntlet creation."""
        self.assertEqual(self.gauntlet.gauntlet_type, GauntletType.FORMAL_VERIFICATION)
        self.assertEqual(self.gauntlet.timeout, 30)
    
    def test_execute_no_properties(self):
        """Test execution with no properties."""
        context = {"properties": []}
        
        result = self.gauntlet.execute(self.solution, context)
        
        self.assertTrue(result.passed)
        self.assertEqual(result.score, 1.0)
    
    def test_execute_with_properties(self):
        """Test execution with properties."""
        context = {
            "properties": [
                {"name": "null_safety"},
                {"name": "bounds_check"}
            ]
        }
        
        result = self.gauntlet.execute(self.solution, context)
        
        self.assertIsInstance(result, GauntletResult)
        self.assertIn("verification_results", result.details or {})
    
    def test_heuristic_verification(self):
        """Test heuristic verification without Z3."""
        property_spec = {"name": "null_safety"}
        code = "if x is not None: return x"
        
        result = self.gauntlet._heuristic_verification(code, property_spec)
        
        self.assertIn("verified", result)
        self.assertIn("property", result)


class TestStatisticalGauntlet(unittest.TestCase):
    """Tests for Statistical Gauntlet."""
    
    def setUp(self):
        self.gauntlet = StatisticalGauntlet("test_statistical")
        self.solution = MockSolution("statistical solution")
    
    def test_create_statistical_gauntlet(self):
        """Test statistical gauntlet creation."""
        self.assertEqual(self.gauntlet.gauntlet_type, GauntletType.STATISTICAL)
        self.assertEqual(self.gauntlet.num_samples, 1000)
    
    def test_execute_with_data(self):
        """Test execution with test data."""
        context = {
            "test_data": [1.0, 2.0, 3.0, 4.0, 5.0],
            "expected_distribution": {"mean": 3.0, "variance": 2.0}
        }
        
        result = self.gauntlet.execute(self.solution, context)
        
        self.assertIsInstance(result, GauntletResult)
        self.assertIn("test_results", result.details or {})
    
    def test_mean_test(self):
        """Test mean hypothesis test."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        expected = {"mean": 3.0}
        
        result = self.gauntlet._test_mean(data, expected)
        
        self.assertIn("passed", result)
        self.assertIn("p_value", result)
    
    def test_variance_test(self):
        """Test variance hypothesis test."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        expected = {"variance": 2.0}
        
        result = self.gauntlet._test_variance(data, expected)
        
        self.assertIn("passed", result)
        self.assertIn("p_value", result)


class TestDomainSpecificGauntlet(unittest.TestCase):
    """Tests for Domain-Specific Gauntlets."""
    
    def test_physics_gauntlet(self):
        """Test physics domain gauntlet."""
        gauntlet = DomainSpecificGauntlet("physics", "test_physics")
        solution = MockSolution("Calculate force with F=ma")
        
        context = {"parameters": {"mass": 10, "acceleration": 2}}
        result = gauntlet.execute(solution, context)
        
        self.assertEqual(result.gauntlet_type, GauntletType.DOMAIN_PHYSICS)
        self.assertIn("domain", result.details or {})
    
    def test_finance_gauntlet(self):
        """Test finance domain gauntlet."""
        gauntlet = DomainSpecificGauntlet("finance", "test_finance")
        solution = MockSolution("Calculate portfolio risk")
        
        context = {"risk_tolerance": "medium"}
        result = gauntlet.execute(solution, context)
        
        self.assertEqual(result.gauntlet_type, GauntletType.DOMAIN_FINANCE)
    
    def test_chemistry_gauntlet(self):
        """Test chemistry domain gauntlet."""
        gauntlet = DomainSpecificGauntlet("chemistry", "test_chemistry")
        solution = MockSolution("Balance chemical equation")
        
        result = gauntlet.execute(solution, {})
        
        self.assertEqual(result.gauntlet_type, GauntletType.DOMAIN_CHEMISTRY)
    
    def test_engineering_gauntlet(self):
        """Test engineering domain gauntlet."""
        gauntlet = DomainSpecificGauntlet("engineering", "test_engineering")
        solution = MockSolution("Calculate stress with safety factor")
        
        result = gauntlet.execute(solution, {})
        
        self.assertEqual(result.gauntlet_type, GauntletType.DOMAIN_ENGINEERING)
    
    def test_domain_check(self):
        """Test domain-specific checks."""
        gauntlet = DomainSpecificGauntlet("physics", "test")
        rule = {"name": "unit_check", "check": "units", "severity": "critical"}
        solution = MockSolution("10 kg * m/s^2")
        
        result = gauntlet._run_domain_check(rule, solution, {})
        
        self.assertIn("passed", result)
        self.assertIn("severity", result)


class TestMultiObjectiveGauntlet(unittest.TestCase):
    """Tests for Multi-Objective Gauntlet."""
    
    def setUp(self):
        self.gauntlet = MultiObjectiveGauntlet("test_multi")
        self.solution = MockSolution("multi-objective solution")
    
    def test_create_multi_objective_gauntlet(self):
        """Test multi-objective gauntlet creation."""
        self.assertEqual(self.gauntlet.gauntlet_type, GauntletType.MULTI_OBJECTIVE)
    
    def test_execute_with_objectives(self):
        """Test execution with multiple objectives."""
        context = {
            "objective_values": {
                "cost": 0.8,
                "performance": 0.9,
                "reliability": 0.7
            }
        }
        
        result = self.gauntlet.execute(self.solution, context)
        
        self.assertIsInstance(result, GauntletResult)
        self.assertIn("weighted_score", result.details or {})
    
    def test_pareto_optimality(self):
        """Test Pareto optimality checking."""
        values = [0.8, 0.9, 0.7]
        reference_front = [
            [0.7, 0.8, 0.6],
            [0.6, 0.7, 0.5]
        ]
        
        is_optimal, dominated = self.gauntlet._check_pareto_optimality(values, reference_front)
        
        self.assertIsInstance(is_optimal, bool)
        self.assertIsInstance(dominated, int)
    
    def test_hypervolume_calculation(self):
        """Test hypervolume indicator calculation."""
        front = [[0.8, 0.9], [0.7, 0.8]]
        reference = [1.0, 1.0]
        
        volume = self.gauntlet._calculate_hypervolume(front, reference)
        
        self.assertGreaterEqual(volume, 0.0)


class TestEvolutionaryGauntlet(unittest.TestCase):
    """Tests for Evolutionary Gauntlet."""
    
    def setUp(self):
        self.gauntlet = EvolutionaryGauntlet("test_evolutionary")
        self.solution = MockSolution("evolutionary solution")
    
    def test_create_evolutionary_gauntlet(self):
        """Test evolutionary gauntlet creation."""
        self.assertEqual(self.gauntlet.gauntlet_type, GauntletType.EVOLUTIONARY)
        self.assertEqual(self.gauntlet.population_size, 50)
    
    def test_execute_basic(self):
        """Test basic evolutionary execution."""
        context = {}
        
        result = self.gauntlet.execute(self.solution, context)
        
        self.assertIsInstance(result, GauntletResult)
        self.assertIn("raw_fitness", result.details or {})
    
    def test_fitness_calculation(self):
        """Test fitness calculation."""
        solution = MockSolution("def good_function(): pass  # Well documented")
        context = {}
        
        fitness = self.gauntlet._default_fitness(solution, context)
        
        self.assertGreater(fitness, 0.0)
        self.assertLessEqual(fitness, 1.0)
    
    def test_mutation(self):
        """Test solution mutation."""
        solution = "original code"
        mutated = self.gauntlet._mutate_solution(solution)
        
        self.assertIsInstance(mutated, str)


class TestTemporalGauntlet(unittest.TestCase):
    """Tests for Temporal Gauntlet."""
    
    def setUp(self):
        self.gauntlet = TemporalGauntlet("test_temporal")
        self.solution = MockSolution("temporal solution")
    
    def test_create_temporal_gauntlet(self):
        """Test temporal gauntlet creation."""
        self.assertEqual(self.gauntlet.gauntlet_type, GauntletType.TEMPORAL)
        self.assertEqual(self.gauntlet.time_steps, 100)
    
    def test_execute_with_time_series(self):
        """Test execution with time series data."""
        context = {
            "time_series_data": [1.0, 1.1, 1.05, 1.08, 1.02]
        }
        
        result = self.gauntlet.execute(self.solution, context)
        
        self.assertIsInstance(result, GauntletResult)
        self.assertIn("stability", result.details or {})
        self.assertIn("convergence", result.details or {})
    
    def test_stability_check(self):
        """Test stability checking."""
        # Stable series
        stable_data = [1.0, 1.01, 0.99, 1.02, 1.0]
        result = self.gauntlet._check_stability(stable_data)
        
        self.assertIn("stable", result)
        self.assertIn("coefficient_of_variation", result)
    
    def test_convergence_check(self):
        """Test convergence checking."""
        # Converging series
        converging = [1.0, 0.8, 0.6, 0.51, 0.501, 0.5001, 0.5, 0.5, 0.5, 0.5]
        result = self.gauntlet._check_convergence(converging)
        
        self.assertIn("converged", result)
        self.assertIn("final_mean", result)
    
    def test_trend_analysis(self):
        """Test trend analysis."""
        # Improving trend
        improving = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        result = self.gauntlet._analyze_trend(improving)
        
        self.assertIn("direction", result)
        self.assertIn("slope", result)


class TestCrossValidationGauntlet(unittest.TestCase):
    """Tests for Cross-Validation Gauntlet."""
    
    def setUp(self):
        self.gauntlet = CrossValidationGauntlet("test_cv")
        self.solution = MockSolution("cv solution")
    
    def test_create_cross_validation_gauntlet(self):
        """Test cross-validation gauntlet creation."""
        self.assertEqual(self.gauntlet.gauntlet_type, GauntletType.CROSS_VALIDATION)
        self.assertEqual(self.gauntlet.k_folds, 5)
    
    def test_execute_with_data(self):
        """Test execution with dataset."""
        context = {
            "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        }
        
        result = self.gauntlet.execute(self.solution, context)
        
        self.assertIsInstance(result, GauntletResult)
        self.assertIn("fold_results", result.details or {})
        self.assertIn("mean_score", result.details or {})
    
    def test_k_fold_split(self):
        """Test K-fold data splitting."""
        data = list(range(10))
        eval_fn = lambda s, d: len(d) / 10  # Simple evaluator
        
        results = self.gauntlet._k_fold_validation(self.solution, data, eval_fn)
        
        self.assertEqual(len(results), 5)  # 5 folds


class TestGauntletOrchestrator(unittest.TestCase):
    """Tests for Gauntlet Orchestrator."""
    
    def setUp(self):
        self.orchestrator = GauntletOrchestrator(max_workers=2)
        self.solution = MockSolution("test solution")
        self.gauntlets = [
            StatisticalGauntlet("stat_1"),
            StatisticalGauntlet("stat_2")
        ]
    
    def tearDown(self):
        self.orchestrator.shutdown()
    
    def test_sequential_orchestration(self):
        """Test sequential gauntlet execution."""
        context = {"test_data": [1.0, 2.0, 3.0]}
        config = {"stop_on_failure": False}
        
        result = self.orchestrator.orchestrate(
            OrchestrationMode.SEQUENTIAL,
            self.gauntlets,
            self.solution,
            context,
            config
        )
        
        self.assertEqual(result.mode, OrchestrationMode.SEQUENTIAL)
        self.assertEqual(len(result.individual_results), 2)
        self.assertIsNotNone(result.overall_score)
    
    def test_parallel_orchestration(self):
        """Test parallel gauntlet execution."""
        context = {"test_data": [1.0, 2.0, 3.0]}
        
        result = self.orchestrator.orchestrate(
            OrchestrationMode.PARALLEL,
            self.gauntlets,
            self.solution,
            context
        )
        
        self.assertEqual(result.mode, OrchestrationMode.PARALLEL)
        self.assertEqual(len(result.individual_results), 2)
    
    def test_adaptive_orchestration(self):
        """Test adaptive gauntlet selection."""
        context = {"test_data": [1.0, 2.0, 3.0]}
        
        result = self.orchestrator.orchestrate(
            OrchestrationMode.ADAPTIVE,
            self.gauntlets,
            self.solution,
            context
        )
        
        self.assertEqual(result.mode, OrchestrationMode.ADAPTIVE)
    
    def test_hierarchical_orchestration(self):
        """Test hierarchical gauntlet execution."""
        # Create gauntlets at different levels
        gauntlets = [
            DomainSpecificGauntlet("physics", "physics_1"),
            StatisticalGauntlet("stat_1")
        ]
        context = {"test_data": [1.0, 2.0, 3.0]}
        
        result = self.orchestrator.orchestrate(
            OrchestrationMode.HIERARCHICAL,
            gauntlets,
            self.solution,
            context
        )
        
        self.assertEqual(result.mode, OrchestrationMode.HIERARCHICAL)


class TestGauntletScoringSystem(unittest.TestCase):
    """Tests for Gauntlet Scoring System."""
    
    def setUp(self):
        self.scoring = GauntletScoringSystem()
        self.results = [
            GauntletResult(
                gauntlet_type=GauntletType.STATISTICAL,
                gauntlet_name="test1",
                solution_id="s1",
                passed=True,
                score=0.8,
                confidence=0.9,
                execution_time=1.0,
                timestamp=MagicMock()
            ),
            GauntletResult(
                gauntlet_type=GauntletType.ADVERSARIAL,
                gauntlet_name="test2",
                solution_id="s1",
                passed=False,
                score=0.6,
                confidence=0.8,
                execution_time=2.0,
                timestamp=MagicMock()
            )
        ]
    
    def test_multi_dimensional_scoring(self):
        """Test multi-dimensional score calculation."""
        score = self.scoring.calculate_multi_dimensional_score(
            self.results,
            dimensions=["correctness", "robustness"],
            weights=[0.5, 0.5]
        )
        
        self.assertIn("dimensions", score)
        self.assertIn("overall_score", score)
        self.assertGreaterEqual(score["overall_score"], 0.0)
        self.assertLessEqual(score["overall_score"], 1.0)
    
    def test_confidence_interval(self):
        """Test confidence interval calculation."""
        ci = self.scoring.calculate_confidence_interval(self.results)
        
        self.assertIn("mean", ci)
        self.assertIn("ci_lower", ci)
        self.assertIn("ci_upper", ci)
        self.assertLessEqual(ci["ci_lower"], ci["mean"])
        self.assertGreaterEqual(ci["ci_upper"], ci["mean"])
    
    def test_aggregation_statistics(self):
        """Test statistical aggregation."""
        stats = self.scoring.aggregate_statistics(self.results)
        
        self.assertIn("scores", stats)
        self.assertIn("pass_rate", stats)
        self.assertEqual(stats["total_gauntlets"], 2)


class TestGauntletFactory(unittest.TestCase):
    """Tests for Gauntlet Factory."""
    
    def test_create_all_gauntlet_types(self):
        """Test creating all gauntlet types via factory."""
        types = [
            "adversarial",
            "formal",
            "statistical",
            "physics",
            "finance",
            "multi_objective",
            "evolutionary",
            "temporal",
            "cross_validation"
        ]
        
        for gauntlet_type in types:
            with self.subTest(gauntlet_type=gauntlet_type):
                gauntlet = create_gauntlet(gauntlet_type)
                self.assertIsInstance(gauntlet, BaseGauntlet)
    
    def test_list_available_gauntlets(self):
        """Test listing available gauntlet types."""
        available = list_available_gauntlets()
        
        self.assertIsInstance(available, dict)
        self.assertIn("adversarial", available)
        self.assertIn("statistical", available)
        self.assertIn("evolutionary", available)
    
    def test_invalid_gauntlet_type(self):
        """Test error on invalid gauntlet type."""
        with self.assertRaises(ValueError):
            create_gauntlet("invalid_type")


class TestConvenienceFunctions(unittest.TestCase):
    """Tests for convenience functions."""
    
    def test_run_sequential(self):
        """Test sequential convenience function."""
        gauntlets = [StatisticalGauntlet(f"stat_{i}") for i in range(2)]
        solution = MockSolution("test")
        context = {"test_data": [1.0, 2.0, 3.0]}
        
        result = run_sequential_gauntlets(gauntlets, solution, context)
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result.individual_results), 2)
    
    def test_run_parallel(self):
        """Test parallel convenience function."""
        gauntlets = [StatisticalGauntlet(f"stat_{i}") for i in range(2)]
        solution = MockSolution("test")
        context = {"test_data": [1.0, 2.0, 3.0]}
        
        result = run_parallel_gauntlets(gauntlets, solution, context, max_workers=2)
        
        self.assertIsNotNone(result)


class TestGauntletManagerIntegration(unittest.TestCase):
    """Tests for GauntletManager integration with advanced types."""
    
    def setUp(self):
        try:
            from gauntlet_manager import GauntletManager
            self.manager = GauntletManager()
        except ImportError:
            self.skipTest("GauntletManager not available")
    
    def test_list_advanced_types(self):
        """Test listing advanced gauntlet types."""
        types = self.manager.list_advanced_gauntlet_types()
        
        self.assertIsInstance(types, dict)
        self.assertGreater(len(types), 0)
    
    def test_create_adversarial_via_manager(self):
        """Test creating adversarial gauntlet via manager."""
        solution = MockSolution("test code")
        
        result = self.manager.create_adversarial_gauntlet(
            "test_adv",
            solution,
            attack_modes=["systematic"]
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn("score", result)
    
    def test_create_statistical_via_manager(self):
        """Test creating statistical gauntlet via manager."""
        solution = MockSolution("test")
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        result = self.manager.create_statistical_gauntlet(
            "test_stat",
            solution,
            test_data=data
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn("score", result)
        self.assertIn("test_results", result)


def run_all_tests():
    """Run all gauntlet tests and report results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestAdversarialGauntlet,
        TestFormalVerificationGauntlet,
        TestStatisticalGauntlet,
        TestDomainSpecificGauntlet,
        TestMultiObjectiveGauntlet,
        TestEvolutionaryGauntlet,
        TestTemporalGauntlet,
        TestCrossValidationGauntlet,
        TestGauntletOrchestrator,
        TestGauntletScoringSystem,
        TestGauntletFactory,
        TestConvenienceFunctions,
        TestGauntletManagerIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("GAUNTLET SYSTEM TEST SUMMARY")
    print("="*60)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"Success Rate: {success_rate:.1f}%")
    
    if result.wasSuccessful():
        print("\n[OK] ALL GAUNTLET TESTS PASSED!")
    else:
        print("\n[FAIL] SOME TESTS FAILED")
    
    return result


if __name__ == "__main__":
    run_all_tests()
